from abc import abstractmethod
from math import ceil, floor
from typing import List, Optional

import numpy as np
from src.envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment

from src.vsl_algorithms.utils import JSD
from src.vsl_policies import VAlignedDiscreteSpaceActionPolicy, ValueSystemLearningPolicy
from src.vsl_reward_functions import AbstractVSLRewardFunction, LinearVSLRewardFunction, ConvexTensorModule, ProabilisticProfiledRewardFunction, TrainingModes

from imitation.algorithms import base
from imitation.data import types
from imitation.util import logger as imit_logger
from imitation.util import networks
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Type,
    Union,
)
from imitation.util import util

import torch as th

from imitation.data import types, rollout


class BaseVSLAlgorithm(base.DemonstrationAlgorithm):
    def set_demonstrations(self, demonstrations: Union[Iterable[types.Trajectory], Iterable[types.TransitionMapping], types.TransitionsMinimal]) -> None:
        pass

    @property
    def policy(self):
        return self.learned_policy_per_va

    def __init__(
        self,
        env: Union[TabularVAMDP, ValueAlignedEnvironment],
        reward_net: AbstractVSLRewardFunction,
        vgl_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        vsi_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        vgl_optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        vsi_optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        discount: float = 1.0,
        log_interval: Optional[int] = 100,
        vgl_expert_policy: Optional[ValueSystemLearningPolicy] = None,
        vsi_expert_policy: Optional[ValueSystemLearningPolicy] = None,

        # A Society or other mechanism might return different alignment functions at different times.
        target_align_func_sampler=lambda *args: args[0],


        vsi_target_align_funcs=[],

        vgl_target_align_funcs=[],
        learn_stochastic_policy=True,
        stochastic_expert=True,  # solely for prediction purposes
        environment_is_stochastic=False,
        training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,

    ) -> None:
        self.discount = discount
        self.env = env

        self.learn_stochastic_policy = learn_stochastic_policy
        self.stochastic_expert = stochastic_expert
        self.environment_is_stochastic = environment_is_stochastic
        # self.vgl_expert_sampler = vgl_expert_sampler # list of expert trajectories just to see different origin destinations and align_funcs
        # self.vsi_expert_sampler = vsi_expert_sampler # list of expert target align_func trajectories

        # self.initial_state_distribution_train = initial_state_distribution_train if initial_state_distribution_train is not None else env.initial_state_dist
        # self.initial_state_distribution_test= initial_state_distribution_test if initial_state_distribution_test is not None else env.initial_state_dist

        self.vgl_target_align_funcs = vgl_target_align_funcs
        self.vsi_target_align_funcs = vsi_target_align_funcs
        self.all_targets = set()
        self.all_targets.update(self.vgl_target_align_funcs)
        self.all_targets.update(self.vsi_target_align_funcs)
        self.all_targets = list(self.all_targets)
        self.vgl_expert_policy: VAlignedDiscreteSpaceActionPolicy = vgl_expert_policy
        self.vsi_expert_policy: VAlignedDiscreteSpaceActionPolicy = vsi_expert_policy

        self.target_align_function_sampler = target_align_func_sampler

        super().__init__(
            demonstrations=None,
            custom_logger=custom_logger,
        )

        self.reward_net = reward_net
        self.probabilistic_reward_net = None
        self.current_net = reward_net

        self.vgl_optimizer_cls = vgl_optimizer_cls
        self.vsi_optimizer_cls = vsi_optimizer_cls

        vgl_optimizer_kwargs = vgl_optimizer_kwargs or {"lr": 1e-1}
        vsi_optimizer_kwargs = vsi_optimizer_kwargs or {"lr": 2e-1}
        self.vsi_optimizer_kwargs = vsi_optimizer_kwargs
        self.vgl_optimizer_kwargs = vgl_optimizer_kwargs

        self.probabilistic_vsi_optimizer_cls = vsi_optimizer_cls
        self.probabilistic_vsi_optimizer_kwargs = vsi_optimizer_kwargs

        self.log_interval = log_interval

        self.learned_policy_per_va = None

        self.training_mode = training_mode
        self.reward_net = reward_net
        self.reward_net.set_mode(self.training_mode)
        self.__previous_next_states = None

    def set_reward_net(self, reward_net: AbstractVSLRewardFunction):
        self.reward_net = reward_net

    def set_probabilistic_net(self, probabilistic_net: ProabilisticProfiledRewardFunction):
        self.probabilistic_reward_net = probabilistic_net

    def get_reward_net(self):
        return self.reward_net

    def get_probabilistic_net(self):
        return self.probabilistic_reward_net

    def get_current_reward_net(self):
        return self.current_net

    def get_metrics(self):
        return {'learned_rewards': self.state_action_callable_reward_from_reward_net_per_target_align_func(self.get_reward_net())}

    def train(self, max_iter: int = 1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING,
              assumed_grounding=None,
              n_seeds_for_sampled_trajectories=None,
              n_sampled_trajs_per_seed=1,
              use_probabilistic_reward=False,
              n_reward_reps_if_probabilistic_reward=10,
              custom_optimizer_kwargs=None,
              custom_prob_optimizer_kwargs=None,
              **kwargs) -> np.ndarray:

        self.training_mode = mode
        self.assumed_grounding = assumed_grounding

        if use_probabilistic_reward:
            if self.probabilistic_reward_net is None:
                assert isinstance(self.reward_net, LinearVSLRewardFunction)
                self.probabilistic_reward_net = ProabilisticProfiledRewardFunction(
                    environment=self.env, use_action=self.reward_net.use_action, use_done=self.reward_net.use_done, use_next_state=self.reward_net.use_next_state, use_one_hot_state_action=self.reward_net.use_one_hot_state_action, use_state=self.reward_net.use_state,
                    activations=self.reward_net.activations, hid_sizes=self.reward_net.hid_sizes,
                    basic_layer_classes=[*self.reward_net.basic_layer_classes[:len(
                        self.reward_net.basic_layer_classes)-1], ConvexTensorModule],
                    mode=mode, negative_grounding_layer=self.reward_net.negative_grounding_layer, use_bias=self.reward_net.use_bias
                )
                self.probabilistic_reward_net.values_net = self.reward_net.values_net
                self.probabilistic_reward_net.set_alignment_function(
                    self.reward_net.get_learned_align_function())

            self.current_net = self.probabilistic_reward_net
        else:
            self.current_net = self.reward_net

        self.current_net.set_mode(mode)
        if assumed_grounding is not None and mode in [TrainingModes.EVAL, TrainingModes.VALUE_SYSTEM_IDENTIFICATION]:
            self.current_net.set_grounding_function(assumed_grounding)
        if mode in [TrainingModes.VALUE_GROUNDING_LEARNING, TrainingModes.SIMULTANEOUS]:
            self.current_net.reset_learned_grounding_function(
                None)
        target_align_funcs = self.vgl_target_align_funcs if mode == TrainingModes.VALUE_GROUNDING_LEARNING else self.vsi_target_align_funcs
        self.target_align_funcs_to_learned_align_funcs = dict()

        if mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            optimizer_kwargs = self.vgl_optimizer_kwargs if custom_optimizer_kwargs is None else custom_optimizer_kwargs
            self.vgl_optimizer_kwargs = optimizer_kwargs
            self.vgl_optimizer = self.vgl_optimizer_cls(
                self.current_net.parameters(), **optimizer_kwargs)
            self.target_align_funcs_to_learned_align_funcs = {
                al: al for al in self.vgl_target_align_funcs}
            with networks.training(self.current_net):
                reward_nets_per_target_align_func = self.train_vgl(
                    max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)

        elif mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            optimizer_kwargs = self.vsi_optimizer_kwargs if custom_optimizer_kwargs is None else custom_optimizer_kwargs
            self.vsi_optimizer_kwargs = optimizer_kwargs

            reward_nets_per_target_align_func = dict()
            self.target_align_funcs_to_learned_align_funcs = dict()

            for ti, target_align_func in enumerate(target_align_funcs):
                self.current_net.reset_learned_alignment_function()
                self.current_net.set_mode(
                    TrainingModes.VALUE_SYSTEM_IDENTIFICATION)
                self.env.set_align_func(target_align_func)

                if use_probabilistic_reward:
                    prob_optimizer_kwargs = self.probabilistic_vsi_optimizer_kwargs if custom_prob_optimizer_kwargs is None else custom_prob_optimizer_kwargs
                    self.vsi_optimizer_kwargs = prob_optimizer_kwargs
                    self.probabilistic_vsi_optimizer = self.probabilistic_vsi_optimizer_cls(
                        self.current_net.parameters(), **prob_optimizer_kwargs)
                    with networks.training(self.current_net):
                        self.target_align_funcs_to_learned_align_funcs[target_align_func] = self.train_vsl_probabilistic(
                            max_iter=max_iter, n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed,
                            n_reward_reps_if_probabilistic_reward=n_reward_reps_if_probabilistic_reward,
                            target_align_func=target_align_func)

                else:
                    self.vsi_optimizer = self.vsi_optimizer_cls(
                        self.current_net.parameters(), **optimizer_kwargs)
                    with networks.training(self.current_net):
                        self.target_align_funcs_to_learned_align_funcs[target_align_func] = self.train_vsl(
                            max_iter=max_iter, n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories,
                            n_sampled_trajs_per_seed=n_sampled_trajs_per_seed, target_align_func=target_align_func)

                reward_nets_per_target_align_func[target_align_func] = self.current_net.copy(
                )

        else:
            # TODO: Simultaneous learning?
            raise NotImplementedError(
                "Simultaneous learning mode is not implemented")

        # Organizing learned content:

        self.learned_policy_per_va = self.calculate_learned_policies(
            target_align_funcs)

        if mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            return self.target_align_funcs_to_learned_align_funcs, reward_nets_per_target_align_func, self.get_metrics()
        elif mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            return self.current_net.get_learned_grounding(), reward_nets_per_target_align_func, self.get_metrics()
        else:
            return self.current_net.get_learned_grounding(), self.target_align_funcs_to_learned_align_funcs, reward_nets_per_target_align_func, self.get_metrics()

    def test_accuracy_for_align_funcs(self, learned_rewards_per_round: List[np.ndarray],
                                      testing_policy_per_round: List[ValueSystemLearningPolicy],
                                      target_align_funcs_to_learned_align_funcs: Dict,
                                      expert_policy: ValueSystemLearningPolicy,
                                      random_policy: ValueSystemLearningPolicy,
                                      ratios_expert_random=[
                                          1, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.0],
                                      n_seeds=100,
                                      n_samples_per_seed=1,
                                      seed=26,
                                      epsilon_for_undecided_preference=0.05,
                                      testing_align_funcs=[],
                                      initial_state_distribution_for_expected_alignment_estimation=None,
                                      basic_profiles=None):

        basic_profiles = [tuple(v)
                          for v in np.eye(self.reward_net.hid_sizes[-1])]

        metrics_per_ratio = dict()

        prev_initial_distribution = self.env.initial_state_dist
        if initial_state_distribution_for_expected_alignment_estimation is not None:
            self.env.set_initial_state_distribution(
                initial_state_distribution_for_expected_alignment_estimation)

        expert_trajs_for_al_estimation = {rep: {al: expert_policy.obtain_trajectories(n_seeds=n_seeds*10, repeat_per_seed=n_samples_per_seed,
                                                                                      seed=(seed+2352)*rep, stochastic=self.stochastic_expert,
                                                                                      end_trajectories_when_ended=True,
                                                                                      align_funcs_in_policy=[al,], with_reward=True, alignments_in_env=[al,]) for al in testing_align_funcs}
                                          for rep in range(len(testing_policy_per_round))}
        policy_trajs_for_al_estimation = {rep: {al: testing_policy_per_round[rep].obtain_trajectories(n_seeds=n_seeds*10, repeat_per_seed=n_samples_per_seed,
                                                                                                      seed=(seed+74571)*rep, stochastic=self.stochastic_expert,
                                                                                                      end_trajectories_when_ended=True,
                                                                                                      align_funcs_in_policy=[al,], with_reward=True, alignments_in_env=[al,]) for al in testing_align_funcs}
                                          for rep in range(len(testing_policy_per_round))}
        self.env.set_initial_state_distribution(prev_initial_distribution)

        expert_trajs = {rep: {al: expert_policy.obtain_trajectories(n_seeds=n_seeds, repeat_per_seed=n_samples_per_seed,
                                                                    seed=(seed+2352)*rep, stochastic=self.stochastic_expert,
                                                                    end_trajectories_when_ended=True,
                                                                    align_funcs_in_policy=[al,], with_reward=True, alignments_in_env=[al,]) for al in testing_align_funcs}
                        for rep in range(len(testing_policy_per_round))}

        random_trajs = {rep: {al: random_policy.obtain_trajectories(n_seeds=n_seeds, repeat_per_seed=n_samples_per_seed,
                                                                    seed=(seed+34355)*rep, stochastic=True,
                                                                    end_trajectories_when_ended=True,
                                                                    align_funcs_in_policy=[al,], with_reward=True, alignments_in_env=[al,]) for al in testing_align_funcs}
                        for rep in range(len(testing_policy_per_round))}

        real_matrix = {al: self.env.reward_matrix_per_align_func(
            al) for al in testing_align_funcs}
        value_expectations = {al: [] for al in testing_align_funcs}
        value_expectations_expert = {al: [] for al in testing_align_funcs}
        for ratio in ratios_expert_random:

            qualitative_loss_per_al_func = {al: []
                                            for al in testing_align_funcs}
            jsd_per_al_func = {al: [] for al in testing_align_funcs}
            n_repescados_per_al_func = {al: [] for al in testing_align_funcs}

            for rep, reward_rep in enumerate(learned_rewards_per_round):
                for al in testing_align_funcs:
                    real_matrix_al = real_matrix[al]
                    all_trajs = [*((np.random.permutation(np.asarray(expert_trajs[rep][al]))[0:floor(len(expert_trajs[rep][al])*ratio)]).tolist()),
                                 *((np.random.permutation(np.asarray(random_trajs[rep][al]))[0:ceil(len(random_trajs[rep][al])*(1.0-ratio))]).tolist())]

                    returns_expert = []
                    returns_estimated = []
                    returns_real_from_learned_policy = {
                        alb: [] for alb in basic_profiles}
                    returns_real_from_expert_policy = {
                        alb: [] for alb in basic_profiles}

                    for ti in all_trajs:
                        if isinstance(reward_rep[al], Callable):
                            estimated_return_i = rollout.discounted_sum(
                                reward_rep[al](ti.obs[:-1], ti.acts), gamma=self.discount)
                        else:
                            assert isinstance(reward_rep[al], np.ndarray)
                            estimated_return_i = rollout.discounted_sum(
                                reward_rep[al][ti.obs[:-1], ti.acts], gamma=self.discount)
                        real_return_i = rollout.discounted_sum(
                            real_matrix_al[ti.obs[:-1], ti.acts], gamma=self.discount)
                        if self.discount == 1.0:
                            assert np.sum(ti.rews) == np.sum(real_return_i)
                        returns_expert.append(real_return_i)
                        returns_estimated.append(estimated_return_i)
                    returns_expert = np.asarray(returns_expert)
                    returns_estimated = np.asarray(returns_estimated)
                    if float(ratio) == 1.0:
                        for al_basic in basic_profiles:
                            rb = real_matrix[al_basic]
                            for lti in policy_trajs_for_al_estimation[rep][al]:
                                real_return_in_learned_pol = rollout.discounted_sum(
                                    rb[lti.obs[:-1], lti.acts],
                                    gamma=self.discount)

                                returns_real_from_learned_policy[al_basic].append(
                                    real_return_in_learned_pol)
                            for exp in expert_trajs_for_al_estimation[rep][al]:

                                real_return_basic_expert = rollout.discounted_sum(
                                    rb[exp.obs[:-1], exp.acts],
                                    gamma=self.discount)
                                returns_real_from_expert_policy[al_basic].append(
                                    real_return_basic_expert)

                    # Equals the -tn parameter. (default 100)
                    N = len(all_trajs)

                    i_j = np.random.choice(N, size=(N*10, 2), replace=True)
                    i_indices, j_indices = i_j[:, 0], i_j[:, 1]

                    estimated_diffs = np.clip(
                        returns_estimated[i_indices] - returns_estimated[j_indices], -50.0, 50.0)
                    real_diffs = np.clip(
                        returns_expert[i_indices] - returns_expert[j_indices], -50.0, 50.0)

                    probs_estimated = 1 / (1 + np.exp(estimated_diffs))
                    probs_real = 1 / (1 + np.exp(real_diffs))

                    probs_real = np.array(probs_real)
                    probs_estimated = np.array(probs_estimated)

                    todos = np.where(probs_real >= 0.0)[0]
                    epsilons = [0.0, 0.01, 0.05]
                    accuracy_per_epsilon = {eps: None for eps in epsilons}
                    n_repescados_per_epsilon = {eps: 0 for eps in epsilons}
                    for epsilon in epsilons:
                        exito_por_mayor = np.intersect1d(np.where(probs_estimated > 0.5)[
                                                         0], np.where(probs_real > 0.5)[0])
                        exito_por_menor = np.intersect1d(np.where(probs_estimated < 0.5)[
                                                         0], np.where(probs_real < 0.5)[0])
                        exito_por_igual = np.intersect1d(np.where(probs_estimated == 0.5)[0],
                                                         np.where(probs_real == 0.5)[0])

                        exito_por_aprox_igual = np.intersect1d(np.where(np.abs(probs_estimated - 0.5) <= epsilon)[0],
                                                               np.where(np.abs(probs_real - 0.5) <= epsilon)[0])

                        acertados = np.union1d(
                            exito_por_mayor, exito_por_menor)
                        acertados = np.union1d(acertados, exito_por_igual)

                        fallados = np.setdiff1d(todos, acertados)

                        acertados_con_margen = np.intersect1d(acertados, np.where(
                            np.abs(probs_real - 0.5) > epsilon)[0])
                        fallados_con_margen = np.intersect1d(fallados, np.where(
                            np.abs(probs_real - 0.5) > epsilon)[0])

                        acertados_difussos = np.intersect1d(acertados, np.where(
                            np.abs(probs_real - 0.5) <= epsilon)[0])
                        acertados_difusos_con_estimacion_correcta = np.intersect1d(acertados_difussos, np.where(
                            np.abs(probs_estimated - 0.5) <= epsilon)[0])
                        acertados_difusos_sobreestimados = np.intersect1d(acertados_difussos, np.where(
                            np.abs(probs_estimated - 0.5) > epsilon)[0])

                        fallados_difusos = np.intersect1d(fallados, np.where(
                            np.abs(probs_real - 0.5) <= epsilon)[0])
                        fallados_considerados_aciertos_por_margen_de_error = np.intersect1d(fallados_difusos, np.where(
                            np.abs(probs_estimated - 0.5) <= epsilon)[0])
                        fallados_totalmente = np.intersect1d(fallados_difusos, np.where(
                            np.abs(probs_estimated - 0.5) > epsilon)[0])
                        all_cases = [acertados_con_margen, fallados_con_margen, acertados_difusos_con_estimacion_correcta,
                                     acertados_difusos_sobreestimados, fallados_considerados_aciertos_por_margen_de_error, fallados_totalmente]

                        total = acertados_con_margen
                        intersec = todos
                        # print("a, b, c11, c12, c2,c3")
                        for id, set_ in enumerate(all_cases):
                            total = np.union1d(total, set_)
                            intersec = np.intersect1d(intersec, set_)
                            # print("LEN ", id, ": ", len(set_))
                        len_total_disj = np.sum(
                            [len(set_) for set_ in all_cases])

                        exitos = np.union1d(acertados, exito_por_aprox_igual)
                        # print("TOTAL", len(total), "TOTAL_DISJ", len_total_disj)
                        assert len(total) == len_total_disj
                        # print("INTERSEC", len(intersec))
                        assert len(intersec) == 0
                        assert len(exitos) == len(
                            np.union1d(acertados, fallados_considerados_aciertos_por_margen_de_error))
                        accuracy = len(exitos)/len(todos)
                        accuracy_per_epsilon[epsilon] = accuracy
                        n_repescados_per_epsilon[epsilon] = len(
                            fallados_considerados_aciertos_por_margen_de_error)

                    is_better_estimated = probs_estimated > (
                        0.5 + epsilon_for_undecided_preference)
                    is_better_real = probs_real > (
                        0.5 + epsilon_for_undecided_preference)
                    is_worse_estimated = probs_estimated < (
                        0.5 - epsilon_for_undecided_preference)
                    is_worse_real = probs_real < (
                        0.5 - epsilon_for_undecided_preference)

                    is_equal_estimated = np.abs(
                        probs_estimated-0.5) <= epsilon_for_undecided_preference
                    is_equal_real = np.abs(
                        probs_real - 0.5) <= epsilon_for_undecided_preference

                    real_labels = np.column_stack(
                        (is_better_real, is_equal_real, is_worse_real))
                    estimated_labels = np.column_stack(
                        (is_better_estimated, is_equal_estimated, is_worse_estimated))

                    # print(real_labels,estimated_labels)
                    # qualitative_loss = qualitative_loss_score(real_labels, estimated_labels, multi_class="ovr")
                    # ACC average (dumb). qualitative_loss = np.mean([np.mean(np.array(real_labels[ri]==estimated_labels[ri], dtype=np.float32)) for ri in range(len(real_labels)) ])
                    # F1 score. (Not exactly okey)
                    # qualitative_loss = f1_score(real_labels, estimated_labels, average='weighted',zero_division=np.nan)
                    # Accuracy with new method adding tolerance for equal cases.

                    qualitative_loss_per_al_func[al].append(
                        accuracy_per_epsilon)

                    n_repescados_per_al_func[al].append(
                        n_repescados_per_epsilon)
                    # ce_per_al_func[al].append(th.nn.functional.binary_cross_entropy(th.tensor(probs_real), th.tensor(probs_estimated)).detach().numpy())
                    # JSD (Not used)
                    jsd = JSD(probs_real, probs_estimated)
                    jsd_per_al_func[al].append({eps: jsd for eps in epsilons})
                    # Instead, use the number of artificially misclassified cases inside the interval +-epsilon

                    if float(ratio) == 1.0:
                        value_expectations[al].append({
                            alb: np.mean(returns_real_from_learned_policy[alb]) for alb in basic_profiles
                        })

                        value_expectations_expert[al].append({
                            alb: np.mean(returns_real_from_expert_policy[alb]) for alb in basic_profiles
                        })

            # metrics_per_ratio[ratio]={'f1': qualitative_loss_per_al_func, 'jsd': jsd_per_al_func}
            metrics_per_ratio[ratio] = {'acc': qualitative_loss_per_al_func,
                                        'repescados': n_repescados_per_al_func, 'jsd': jsd_per_al_func}

        return metrics_per_ratio, value_expectations, value_expectations_expert

    @abstractmethod
    def get_tabular_policy_from_reward_per_align_func(self, align_funcs, *args, **kwargs):
        pass

    def _resample_next_states(self):
        n_actions = self.env.action_dim
        n_states = self.env.state_dim

        next_state_mat = np.zeros((n_states, n_actions))
        if not self.environment_is_stochastic:
            if self.__previous_next_states is not None:
                return self.__previous_next_states.clone().detach()

        for a in range(n_actions):
            for s in range(n_states):
                tr = self.env.transition_matrix[s, a]
                if np.allclose(tr, 0.0):
                    ns = s
                else:
                    ns = np.random.choice(n_states, size=1, p=tr)[0]
                next_state_mat[s, a] = ns

        """next_states = sample_next_states(self.env.transition_matrix)
        next_obs_mat = self.env.observation_matrix[next_states]"""
        torch_next_state_mat = th.as_tensor(
            next_state_mat, dtype=th.long).requires_grad_(False).detach()
        self.__previous_next_states = torch_next_state_mat
        return torch_next_state_mat

    def state_action_callable_reward_from_reward_net_per_target_align_func(self, reward_net: AbstractVSLRewardFunction, targets=None):
        if hasattr(self.env, 'state_dim'):
            one_hot_observations = th.eye(
                self.env.state_dim*self.env.action_dim)
            rewards_per_target_al = dict()

            for target, learned in (self.target_align_funcs_to_learned_align_funcs.items() if self.training_mode != TrainingModes.VALUE_GROUNDING_LEARNING else zip(self.all_targets, self.all_targets)):
                if targets is not None:
                    if target not in targets:
                        continue
                reward_matrix = th.zeros(
                    (self.env.state_dim, self.env.action_dim))

                if not self.environment_is_stochastic or not reward_net.use_next_state:
                    repetitions_for_reward_estimation = 1
                else:
                    # TODO new parameter possibly.
                    repetitions_for_reward_estimation = 10
                assert repetitions_for_reward_estimation == 1

                for r in range(repetitions_for_reward_estimation):
                    next_state_mat = self._resample_next_states()
                    observations = th.arange(
                        self.env.state_dim, dtype=reward_net.dtype, device=reward_net.device).long()
                    for action in range(self.env.action_dim):
                        action_array = util.safe_to_tensor(np.array(
                            [action]*len(observations)),  dtype=reward_net.dtype, device=reward_net.device).long()
                        # print("OBS", observations)
                        # print("ACTS", action_array)
                        next_state_array = util.safe_to_tensor(next_state_mat[observations.numpy(
                        ), action_array.numpy()],  dtype=reward_net.dtype, device=reward_net.device)
                        # print("NEXTS", next_state_array)
                        reward_matrix[observations, action_array] += self.calculate_rewards(
                            align_func=learned if self.training_mode != TrainingModes.VALUE_GROUNDING_LEARNING else target,
                            # Should be: assumed_grounding if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION else self.current_net.get_learned_grounding(),
                            grounding=self.assumed_grounding,
                            obs_mat=observations,
                            action_mat=action_array,
                            next_state_obs_mat=next_state_array,
                            obs_action_mat=one_hot_observations,
                            reward_mode=TrainingModes.EVAL,
                            use_probabilistic_reward=False,  # TODO ?,
                            n_reps_if_probabilistic_reward=1,  
                            requires_grad=False
                        )[1]
                        # print("REWARD", reward_matrix)
                        assert reward_matrix.shape == (
                            self.env.state_dim, self.env.action_dim)
                rewards_per_target_al[target] = reward_matrix
                assert rewards_per_target_al[target].shape == (
                    self.env.state_dim, self.env.action_dim)
            return lambda target: (lambda state=None, action=None: (
                rewards_per_target_al[target] if (state is None and action is None)
                else rewards_per_target_al[target][state, :] if action is None and state is not None
                else rewards_per_target_al[target][:, action] if action is not None and state is None
                else rewards_per_target_al[target][state, action]))
        else:
            # TODO: test this if needed... Untested

            def state_action_reward(target, learned, state, action):

                obs = util.safe_to_tensor([state] if isinstance(state, int) else np.asarray(
                    state) if isinstance(state, np.ndarray) else state.detach().numpy())
                act = util.safe_to_tensor([action] if isinstance(action, int) else np.asarray(
                    action) if isinstance(action, np.ndarray) else action.detach().numpy())
                rew = 0
                if not self.environment_is_stochastic or not reward_net.use_next_state:
                    repetitions_for_reward_estimation = 1
                else:
                    # TODO new parameter possibly.
                    repetitions_for_reward_estimation = 10
                assert repetitions_for_reward_estimation == 1

                for r in range(repetitions_for_reward_estimation):
                    next_obs = util.safe_to_tensor(self.env.transition(
                        state, act),  dtype=reward_net.dtype, device=reward_net.device)
                    rew += self.calculate_rewards(
                        align_func=learned if self.training_mode != TrainingModes.VALUE_GROUNDING_LEARNING else target,
                        # Should be: assumed_grounding if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION else self.current_net.get_learned_grounding(),
                        grounding=self.assumed_grounding,
                        obs_mat=obs,
                        action_mat=act,
                        next_state_obs_mat=next_obs,
                        obs_action_mat=None,
                        reward_mode=TrainingModes.EVAL,
                        use_probabilistic_reward=False,  # TODO ?,
                        n_reps_if_probabilistic_reward=1,  
                        requires_grad=False
                    )[1][0]
                return rew
            return lambda target: lambda state, action: state_action_reward(target, self.target_align_funcs_to_learned_align_funcs[target] if self.training_mode != TrainingModes.VALUE_GROUNDING_LEARNING else None, state, action)

    @abstractmethod
    def calculate_learned_policies(self, target_align_funcs) -> ValueSystemLearningPolicy:
        ...

    @abstractmethod
    def train_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func) -> Dict[Any, AbstractVSLRewardFunction]:
        ...

    @abstractmethod
    def train_vsl_probabilistic(self, max_iter,
                                n_seeds_for_sampled_trajectories,
                                n_sampled_trajs_per_seed,
                                n_reward_reps_if_probabilistic_reward,
                                target_align_func):
        ...

    @abstractmethod
    def train_vgl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed) -> Dict[Any, AbstractVSLRewardFunction]:
        ...

    def calculate_rewards(self, align_func=None, grounding=None, obs_mat=None, next_state_obs_mat=None, action_mat=None, obs_action_mat=None,
                          reward_mode=TrainingModes.EVAL, recover_previous_config_after_calculation=True,
                          use_probabilistic_reward=False, n_reps_if_probabilistic_reward=10, requires_grad=True):

        if self.current_net.use_one_hot_state_action:
            if obs_action_mat is None:
                obs_action_mat = th.as_tensor(
                    np.identity(self.env.state_dim*self.env.action_dim),
                    dtype=self.current_net.dtype,
                    device=self.current_net.device,
                )
            obs_action_mat.requires_grad_(requires_grad)

        if recover_previous_config_after_calculation:
            previous_rew_mode = self.current_net.mode
            previous_rew_ground = self.current_net.cur_value_grounding
            previous_rew_alignment = self.current_net.cur_align_func

        if requires_grad is False:
            if obs_mat is not None and isinstance(obs_mat, th.Tensor):
                obs_mat = obs_mat.detach()
            if action_mat is not None and isinstance(action_mat, th.Tensor):
                action_mat = action_mat.detach()
            if obs_action_mat is not None and isinstance(obs_action_mat, th.Tensor):
                obs_action_mat = obs_action_mat.detach()
            if next_state_obs_mat is not None and isinstance(next_state_obs_mat, th.Tensor):
                next_state_obs_mat = next_state_obs_mat.detach()

        self.current_net.set_mode(reward_mode)
        self.current_net.set_grounding_function(grounding)
        self.current_net.set_alignment_function(align_func)

        assert self.current_net.mode == reward_mode

        if use_probabilistic_reward is False:
            predicted_r, used_align_func, _ = self.calculation_rew(
                align_func=align_func, obs_mat=obs_mat, action_mat=action_mat,
                obs_action_mat=obs_action_mat, next_state_obs=next_state_obs_mat,
                use_probabilistic_reward=use_probabilistic_reward)

            predicted_r_np = predicted_r.detach().cpu().numpy()

            ret = predicted_r, predicted_r_np
        else:
            list_of_reward_calculations = []
            align_func_used_in_each_repetition = []
            prob_of_each_repetition = []
            for _ in range(n_reps_if_probabilistic_reward):
                predicted_r, used_align_func, probability = self.calculation_rew(
                    align_func=align_func, obs_mat=obs_mat, action_mat=action_mat,
                    obs_action_mat=obs_action_mat, next_state_obs=next_state_obs_mat,
                    use_probabilistic_reward=use_probabilistic_reward)

                list_of_reward_calculations.append(predicted_r)
                align_func_used_in_each_repetition.append(used_align_func)
                prob_of_each_repetition.append(probability)
            predicted_rs = th.stack(list_of_reward_calculations)
            prob_of_each_repetition_th = th.stack(prob_of_each_repetition)
            predicted_rs_np = predicted_rs.detach().cpu().numpy()

            ret = predicted_rs, predicted_rs_np, align_func_used_in_each_repetition, prob_of_each_repetition_th

        if recover_previous_config_after_calculation:
            self.current_net.set_mode(previous_rew_mode)
            self.current_net.set_grounding_function(previous_rew_ground)
            self.current_net.set_alignment_function(previous_rew_alignment)

        return ret

    def calculation_rew(self, align_func, obs_mat, action_mat=None, obs_action_mat=None, next_state_obs=None, use_probabilistic_reward=False):
        if use_probabilistic_reward:
            self.current_net.fix_alignment_function()

        next_state_observations = None

        if self.current_net.use_next_state:
            next_state_observations = next_state_obs

        if self.current_net.use_one_hot_state_action:
            if self.current_net.use_next_state:
                next_state_observations = next_state_observations.view(
                    *obs_action_mat.shape)

            predicted_r = th.reshape(self.current_net(
                obs_action_mat, None, next_state_observations, None), (self.env.state_dim, self.env.action_dim))
        elif self.current_net.use_action or self.current_net.use_next_state:
            if self.current_net.use_action:
                assert action_mat is not None
                # assert action_mat.size() == (self.env.action_dim, obs_mat.shape[0], self.env.action_dim)
            if self.current_net.use_next_state:
                assert next_state_observations is not None

            predicted_r = self.current_net(
                obs_mat,
                action_mat,
                next_state_observations if self.current_net.use_next_state else None,
                None)

        used_alignment_func, probability, _ = self.current_net.get_next_align_func_and_its_probability(
            align_func)

        if use_probabilistic_reward:
            self.current_net.free_alignment_function()

        state_action_with_predefined_reward_mask = self.env.get_state_actions_with_known_reward(
            used_alignment_func)

        if state_action_with_predefined_reward_mask is not None:
            if len(predicted_r.size()) == 1:
                lobs = obs_mat.long().to(predicted_r.device)
                lacts = th.argmax(action_mat, dim=1).long() if len(
                    action_mat.shape) > 1 else action_mat.long()
                lacts = lacts.to(device=predicted_r.device)
                lobs = lobs.to(device=predicted_r.device)
                real_reward_th = th.tensor(self.env.reward_matrix_per_align_func(used_alignment_func),
                                           dtype=predicted_r.dtype, device=predicted_r.device, requires_grad=False)

                mask = state_action_with_predefined_reward_mask[lobs, lacts]
                # Gather indices for the states and actions that satisfy the mask

                indices = mask.nonzero()

                # Use advanced indexing to update the predicted rewards
                predicted_r[indices] = real_reward_th[lobs[indices],
                                                      lacts[indices]]

                """ DEBUG: for i, (s,a, ns) in enumerate(zip(lobs, lacts, next_state_observations)):
                    if state_action_with_predefined_reward_mask[int(s),int(a)] == True:
                        assert predicted_r[i] == real_reward_th[s,a]
                        predicted_r[i] = real_reward_th[s,a]"""

            else:

                predicted_r[state_action_with_predefined_reward_mask] = th.as_tensor(
                    self.env.reward_matrix_per_align_func(used_alignment_func)[
                        state_action_with_predefined_reward_mask],
                    dtype=predicted_r.dtype, device=predicted_r.device)

        return predicted_r, used_alignment_func, probability
