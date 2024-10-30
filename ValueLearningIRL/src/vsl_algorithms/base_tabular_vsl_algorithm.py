from copy import deepcopy
import enum
import itertools
from math import ceil, floor
from typing import Callable, Dict, List, Optional, Tuple
import scipy.linalg
import scipy.special
from sklearn.metrics import f1_score, log_loss

from gymnasium import Env
import imitation
import imitation.algorithms
import numpy as np
import scipy
import scipy.sparse
from src.envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment
from src.vsl_algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.vsl_policies import VAlignedDictSpaceActionPolicy, ValueSystemLearningPolicy
from src.vsl_reward_functions import AbstractVSLRewardFunction, TrainingModes, squeeze_r

from seals import base_envs


from imitation.data import types
from imitation.util import logger as imit_logger
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Type,
    Union,
)
import torch as th


from imitation.data import types, rollout

class PolicyApproximators(enum.Enum):
    MCE_ORIGINAL = 'mce_original'
    SOFT_VALUE_ITERATION = 'value_iteration'
    
def dict_metrics(**kwargs):
    return dict(kwargs)

from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np
def JSD(P, Q):
    n = len(P)
    avg_jsd = 0.0
    for p,q in zip(P,Q):
        dist_p = np.array([p,1.0-p])
        dist_q = np.array([q,1.0-q])
        M = 0.5 * (dist_p + dist_q)
        avg_jsd += ((0.5 * (entropy(dist_p, M) + entropy(dist_q,M)))/n)
    return avg_jsd

def concentrate_on_max_policy(pi, distribute_probability_on_max_prob_actions=False):
    """
    Modify the policy matrix pi such that for each state s, the policy is concentrated 
    on the action with the maximum probability, or equally among the actions with maximum probability

    Parameters:
    - pi: A 2D numpy array of shape (n_states, n_actions), 
          representing the policy matrix.
    - distribute_probability_on_max_prob_actions: boolean, optional

    Returns:
    - A modified policy matrix with the distributions concentrated on the 
      action with the maximum probability.
    """
    if distribute_probability_on_max_prob_actions:
        max_values = np.max(pi, axis=1, keepdims=True)

        # Create a boolean matrix where True indicates a maximum value
        is_max = (pi == max_values)

        # Count the number of maximum values per row
        num_max_values = np.sum(is_max, axis=1, keepdims=True)

        # Create the final matrix where each maximum value is divided by the number of max values
        pi_new = is_max / num_max_values
    else:
        # Find the action with the maximum probability for each state
        max_action_indices = np.argmax(pi, axis=-1)  # Shape: (n_states,)

        # Create a new matrix of zeros
        pi_new = np.zeros_like(pi)

        # Use advanced indexing to set the maximum action probability to 1
        n_states, n_actions = pi.shape
        pi_new[np.arange(n_states), max_action_indices] = 1

    return pi_new



def mce_partition_fh(
    env: base_envs.TabularModelPOMDP,
    *,
    reward: Optional[np.ndarray] = None,
    discount: float = 1.0,
    horizon=None,
    approximator_kwargs={
        'value_iteration_tolerance': 0.00001, 'iterations': 100},
    policy_approximator: Union[PolicyApproximators, Callable[[
        Env, np.ndarray, float], np.ndarray]] = PolicyApproximators.MCE_ORIGINAL,
    deterministic=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Performs the soft Bellman backup for a finite-horizon MDP.

    Calculates V^{soft}, Q^{soft}, and pi using recurrences (9.1), (9.2), and
    (9.3) from Ziebart (2010).

    Args:
        env: a tabular, known-dynamics MDP.
        reward: a reward matrix. Defaults to env.reward_matrix.
        discount: discount rate.
        value_iteration_tolerance: if using PolicyAproximators.SOFT_VALUE_ITERATION, error tolerance to stop the algorithm.
        policy_approximator: Policy approximation method. Defaults to PolicyApproximators.MCE_ORIGINAL. Can also be set from a callable of the form: V, Q, pi = policy_approximator(env, reward, discount, **approximator_kwargs) 
        deterministic: whether the resulting policy to be treated as deterministic (True) or stochastic (False).

    Returns:
        (V, Q, \pi) corresponding to the soft values, Q-values and MCE policy.
        V is a 2d array, indexed V[t,s]. Q is a 3d array, indexed Q[t,s,a].
        \pi is a 3d array, indexed \pi[t,s,a]. (or also a 2d array \pi[s,a] depending on the approximation method)

    Raises:
        ValueError: if ``env.horizon`` is None (infinite horizon).
    """
    # shorthand
    if horizon is None:
        horizon = env.horizon
    if horizon is None:
        raise ValueError("Only finite-horizon environments are supported.")
    n_states = env.state_dim
    n_actions = env.action_dim
    T = env.transition_matrix
    if reward is None:
        reward = env.reward_matrix

    if len(reward.shape) == 1:
        broad_R = reward[:, None]
    else:
        broad_R = reward
        assert len(reward.shape) == 2

    if policy_approximator == PolicyApproximators.MCE_ORIGINAL:
        # Initialization
        # indexed as V[t,s]
        V = np.full((horizon, n_states), -np.inf)
        # indexed as Q[t,s,a]
        Q = np.zeros((horizon, n_states, n_actions))
        # Base case: final timestep
        # final Q(s,a) is just reward
        Q[horizon - 1, :, :] = broad_R

        # V(s) is always normalising constant
        V[horizon - 1,
            :] = scipy.special.logsumexp(Q[horizon - 1, :, :], axis=1)

        # Recursive case
        for t in reversed(range(horizon - 1)):
            next_values_s_a = T @ V[t + 1, :]
            Q[t, :, :] = broad_R + discount * next_values_s_a
            V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

        pi = np.exp(Q - V[:, :, None])[0]
    elif policy_approximator == PolicyApproximators.SOFT_VALUE_ITERATION:
        # Initialization
        # indexed as V[t,s]
        V = np.full((n_states), -1)
        # indexed as Q[t,s,a]
        Q = broad_R
        err = np.inf
        iterations = 0
        value_iteration_tolerance = approximator_kwargs['value_iteration_tolerance']
        max_iterations = horizon
        if 'iterations' in approximator_kwargs.keys():
            max_iterations = approximator_kwargs['iterations']
        
        while err > value_iteration_tolerance and (iterations < max_iterations):
            values_prev = V.copy()
            values_prev[env.goal_states] = 0.0
            next_values_s_a = T @ values_prev
            Q = broad_R + discount * next_values_s_a
            V = np.max(Q, axis=1)
            err = np.max(np.abs(V-values_prev))
            iterations += 1
        pi = scipy.special.softmax(Q - V[:, None], axis=1)

    else:
        V, Q, pi = policy_approximator(
            env, reward, discount, **approximator_kwargs)

    if deterministic:

        if len(pi.shape) == 2:
            pi = concentrate_on_max_policy(pi)
            for i in range(pi.shape[0]):
                assert np.allclose(np.sum(pi[i]), 1)
        else:
            for time_t in range(pi.shape[0]):
                pi[time_t] = concentrate_on_max_policy(pi[time_t])
                for i in range(pi[time_t].shape[0]):
                    assert np.allclose(np.sum(pi[time_t, i]), 1)

    return V, Q, pi
class BaseTabularMDPVSLAlgorithm(BaseVSLAlgorithm):
    def set_demonstrations(self, demonstrations: Union[Iterable[types.Trajectory], Iterable[types.TransitionMapping], types.TransitionsMinimal]) -> None:
        pass

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
        environment_is_stochastic=False,
        training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
        stochastic_expert=True,
        approximator_kwargs = {},
        policy_approximator=PolicyApproximators.MCE_ORIGINAL,
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,

    ) -> None:
        super().__init__(env=env, reward_net=reward_net, vgl_optimizer_cls=vgl_optimizer_cls,
                         vsi_optimizer_cls=vsi_optimizer_cls, vsi_optimizer_kwargs=vsi_optimizer_kwargs,
                         vgl_optimizer_kwargs=vgl_optimizer_kwargs, discount=discount, log_interval=log_interval, vgl_expert_policy=vgl_expert_policy, vsi_expert_policy=vsi_expert_policy,
                         target_align_func_sampler=target_align_func_sampler, vsi_target_align_funcs=vsi_target_align_funcs,
                         vgl_target_align_funcs=vgl_target_align_funcs, training_mode=training_mode, custom_logger=custom_logger, learn_stochastic_policy=learn_stochastic_policy)
        self.rewards_per_target_align_func_callable = None
        self.environment_is_stochastic = environment_is_stochastic
        self.stochastic_expert= stochastic_expert
        self.approximator_kwargs = approximator_kwargs
        self.policy_approximator=policy_approximator
        self.__previous_next_states = None

    def get_metrics(self):
        return dict_metrics(learned_rewards=self.rewards_per_target_align_func_callable)

    def train_callback(self, t):
        # pass
        return

    @property
    def policy(self):
        return self.learned_policy_per_va

    def _resample_next_observations(self):
        n_actions = self.env.action_dim
        n_states = self.env.state_dim
        obs_dim = self.env.obs_dim

        next_obs_mat = np.zeros((n_actions, n_states, obs_dim))
        for a in range(n_actions):
            for s in range(n_states):
                try:
                    ns = np.random.choice(
                        n_states, size=1, p=self.env.transition_matrix[s, a])
                except ValueError:
                    ns = s
                next_obs_mat[a, s, :] = self.env.observation_matrix[ns]

        """next_states = sample_next_states(self.env.transition_matrix)
        next_obs_mat = self.env.observation_matrix[next_states]"""
        torch_next_obs_mat = th.as_tensor(
            next_obs_mat, dtype=self.current_net.dtype, device=self.current_net.device)
        return torch_next_obs_mat

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

    def calculation_rew(self, align_func, obs_mat, action_mat=None, obs_action_mat=None, next_state_obs=None, use_probabilistic_reward=False):
        if use_probabilistic_reward:
            self.current_net.fix_alignment_function()

        next_state_observations = None

        if self.current_net.use_next_state:
            if next_state_obs is None:
                next_states = self._resample_next_states()

                next_state_observations = obs_mat[next_states]
            else:
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
                assert action_mat.size() == (self.env.action_dim,
                                             obs_mat.shape[0], self.env.action_dim)
            if self.current_net.use_next_state:
                assert next_state_observations is not None

            predicted_r = th.stack([
                squeeze_r(
                    self.current_net(
                        obs_mat,
                        (action_mat[i]
                         if self.current_net.use_action else None),
                        next_state_observations[:, i,
                                                :] if self.current_net.use_next_state else None,
                        None)
                ) for i in range(self.env.action_dim)], dim=1)

        assert predicted_r.size() == (obs_mat.shape[0], self.env.action_dim)

        used_alignment_func, probability, _ = self.current_net.get_next_align_func_and_its_probability(
            align_func)

        if use_probabilistic_reward:
            self.current_net.free_alignment_function()

        state_actions_with_special_reward = self.env.get_state_actions_with_known_reward(
            used_alignment_func)
        

        if state_actions_with_special_reward is not None:
            predicted_r[state_actions_with_special_reward] = th.as_tensor(
                self.env.reward_matrix_per_align_func(used_alignment_func)[
                    state_actions_with_special_reward],
                dtype=predicted_r.dtype, device=predicted_r.device)

        return predicted_r, used_alignment_func, probability

    def calculate_rewards(self, align_func=None, grounding=None, obs_mat=None, next_state_obs_mat=None, action_mat=None, obs_action_mat=None,
                          reward_mode=TrainingModes.EVAL, recover_previous_config_after_calculation=True,
                          use_probabilistic_reward=False, n_reps_if_probabilistic_reward=10, requires_grad=True):

        if obs_mat is None:
            obs_mat = self.env.observation_matrix
            obs_mat = th.as_tensor(
                obs_mat,
                dtype=self.current_net.dtype,
                device=self.current_net.device,
            )
            obs_mat.requires_grad_(requires_grad)

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

        if requires_grad is False and action_mat is not None:
            action_mat = action_mat.detach()

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

    def train(self, max_iter: int = 1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None, n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_probabilistic_reward=False, n_reward_reps_if_probabilistic_reward=10) -> np.ndarray:
        obs_mat = self.env.observation_matrix

        self.torch_obs_mat = th.as_tensor(
            obs_mat,
            dtype=self.reward_net.dtype,
            device=self.reward_net.device,
        )
        self.torch_obs_mat.requires_grad_(True)

        self.torch_action_mat = None
        if self.reward_net.use_action:
            actions_one_hot = th.eye(
                self.env.action_space.n, requires_grad=True)
            self.torch_action_mat = th.stack([actions_one_hot[i].repeat(
                obs_mat.shape[0], 1) for i in range(self.env.action_space.n)], dim=0)

        self.torch_obs_action_mat = th.as_tensor(
            np.identity(self.env.state_dim*self.env.action_dim),
            dtype=self.reward_net.dtype,
            device=self.reward_net.device,
        )
        self.torch_obs_action_mat.requires_grad_(True)

        self.rewards_per_target_align_func = None

        self.rewards_per_target_align_func_callable = lambda target: lambda: self.prob_reward_matrix_setter(
            target, use_probabilistic_reward=use_probabilistic_reward, assumed_grounding=assumed_grounding)

        return super().train(max_iter=max_iter, mode=mode, assumed_grounding=assumed_grounding,
                             n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories,
                             use_probabilistic_reward=use_probabilistic_reward, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed, n_reward_reps_if_probabilistic_reward=n_reward_reps_if_probabilistic_reward)

    def prob_reward_matrix_setter(self, target, use_probabilistic_reward, numpy=True, requires_grad=False, assumed_grounding=None):
        if use_probabilistic_reward:
            rewards_th, rewards_np, _, _ = self.calculate_rewards(align_func=self.target_align_funcs_to_learned_align_funcs[target],
                                                                  grounding=assumed_grounding,
                                                                  obs_mat=self.torch_obs_mat,
                                                                  action_mat=self.torch_action_mat,
                                                                  obs_action_mat=self.torch_obs_action_mat,
                                                                  reward_mode=TrainingModes.EVAL, requires_grad=requires_grad,
                                                                  use_probabilistic_reward=True, n_reps_if_probabilistic_reward=1)
            return rewards_th[0] if not numpy else rewards_np[0]
        else:
            if self.rewards_per_target_align_func is None:
                self.rewards_per_target_align_func = dict()
            if target not in self.rewards_per_target_align_func.keys():
                rewards_th, rewards_np = self.calculate_rewards(align_func=self.target_align_funcs_to_learned_align_funcs[target],
                                                                grounding=assumed_grounding,
                                                                obs_mat=self.torch_obs_mat,
                                                                action_mat=self.torch_action_mat,
                                                                obs_action_mat=self.torch_obs_action_mat,
                                                                reward_mode=TrainingModes.EVAL, requires_grad=requires_grad, use_probabilistic_reward=False,
                                                                n_reps_if_probabilistic_reward=1)

                self.rewards_per_target_align_func[target] = self.state_action_reward_from_computed_reward(
                    rewards_th if not numpy else rewards_np)
            return self.rewards_per_target_align_func[target]

    def state_action_reward_from_computed_reward(self, rewards):
        return rewards
    def get_policy_from_reward_per_align_func(self, align_funcs, reward_net_per_al: Dict[tuple, AbstractVSLRewardFunction], expert=False, random=False, use_custom_grounding=None, 
                                              target_to_learned =None, use_probabilistic_reward=False, n_reps_if_probabilistic_reward=10,
                                              state_encoder = None, expose_state=True):
        reward_matrix_per_al = dict()
        profile_to_assumed_matrix = {}
        if random:
            profile_to_assumed_matrix = {pr: np.ones(
                (self.env.state_dim, self.env.action_dim))/self.env.action_dim for pr in align_funcs}
            # TODO: random only in feasible states...
        else:
            prev_net = self.current_net
            for w in align_funcs: 
                learned_w_or_real_w = w
                if expert:
                    deterministic = not self.stochastic_expert
                    reward = self.env.reward_matrix_per_align_func(w)
                else:
                    deterministic = not self.learn_stochastic_policy
                    if use_custom_grounding is not None:
                        assumed_grounding = use_custom_grounding
                    else:
                        assumed_grounding = reward_net_per_al[w].get_learned_grounding()
                    
                    self.current_net = reward_net_per_al[w]
                    if target_to_learned is not None and w in target_to_learned.keys():
                        learned_w_or_real_w = target_to_learned[w]
                        
                    else:
                        learned_w_or_real_w = w
                    
                    ret = self.calculate_rewards(learned_w_or_real_w, grounding=assumed_grounding, 
                                                    obs_mat=self.torch_obs_mat,
                                                    action_mat=self.torch_action_mat,
                                                    obs_action_mat=self.torch_obs_action_mat,
                                                    reward_mode=TrainingModes.EVAL, recover_previous_config_after_calculation=True, 
                                            use_probabilistic_reward=use_probabilistic_reward, 
                                            n_reps_if_probabilistic_reward=n_reps_if_probabilistic_reward, requires_grad=False)
                    if not use_probabilistic_reward:
                        _,reward = ret
                    else:
                        raise NotImplementedError("Probabilistic reward is yet to be tested")
                 
                _,_, assumed_expert_pi = mce_partition_fh(self.env, discount=self.discount,
                                                    reward=reward,
                                                    approximator_kwargs=self.approximator_kwargs,
                                                    policy_approximator=self.policy_approximator,
                                                    deterministic= deterministic )
                profile_to_assumed_matrix[w] = assumed_expert_pi
                
                reward_matrix_per_al[w] = reward

            self.current_net = prev_net
        policy = VAlignedDictSpaceActionPolicy(policy_per_va_dict = profile_to_assumed_matrix, env = self.env, state_encoder=state_encoder, expose_state=expose_state)
        return policy, reward_matrix_per_al
    def test_accuracy_for_align_funcs(self, learned_rewards_per_round: List[np.ndarray],
                                        testing_policy_per_round: List[VAlignedDictSpaceActionPolicy],
                                        target_align_funcs_to_learned_align_funcs: Dict,
                                        expert_policy = VAlignedDictSpaceActionPolicy,
                                        random_policy = VAlignedDictSpaceActionPolicy,
                                        ratios_expert_random = [1, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.0],
                                        n_seeds = 100,
                                        n_samples_per_seed = 5,
                                        seed=26,
                                        epsilon_for_undecided_preference = 0.05,
                                        testing_align_funcs=[],
                                        initial_state_distribution_for_expected_alignment_estimation = None,
                                        basic_profiles=None):
        
        basic_profiles = [tuple(v) for v in np.eye(self.reward_net.hid_sizes[-1])]
        
        metrics_per_ratio = dict()
        
        prev_initial_distribution = self.env.initial_state_dist
        if initial_state_distribution_for_expected_alignment_estimation is not None:
            self.env.set_initial_state_distribution(initial_state_distribution_for_expected_alignment_estimation)
            
        expert_trajs_for_al_estimation = {rep: {al: expert_policy.obtain_trajectories(n_seeds=n_seeds, repeat_per_seed=n_samples_per_seed*10, 
                                                         seed=(seed+2352)*rep,stochastic=self.stochastic_expert,
                                                         end_trajectories_when_ended=True,
                                                         with_alignfunctions=[al,],with_reward=True, alignments_in_env=[al,]) for al in testing_align_funcs}
                    for rep in range(len(testing_policy_per_round))}
        policy_trajs_for_al_estimation = {rep: {al: testing_policy_per_round[rep].obtain_trajectories(n_seeds=n_seeds, repeat_per_seed=n_samples_per_seed*10, 
                                                         seed=(seed+74571)*rep,stochastic=self.stochastic_expert,
                                                         end_trajectories_when_ended=True,
                                                         with_alignfunctions=[al,],with_reward=True,alignments_in_env=[al,]) for al in testing_align_funcs}
                        for rep in range(len(testing_policy_per_round))}
        self.env.set_initial_state_distribution( prev_initial_distribution)

        expert_trajs = {rep: {al: expert_policy.obtain_trajectories(n_seeds=n_seeds, repeat_per_seed=n_samples_per_seed, 
                                                         seed=(seed+2352)*rep,stochastic=self.stochastic_expert,
                                                         end_trajectories_when_ended=True,
                                                         with_alignfunctions=[al,],with_reward=True, alignments_in_env=[al,]) for al in testing_align_funcs}
                    for rep in range(len(testing_policy_per_round))}
        
        
        random_trajs = {rep: {al: random_policy.obtain_trajectories(n_seeds=n_seeds, repeat_per_seed=n_samples_per_seed, 
                                                                seed=(seed+34355)*rep,stochastic=self.stochastic_expert,
                                                                end_trajectories_when_ended=True,
                                                                with_alignfunctions=[al,],with_reward=True, alignments_in_env=[al,]) for al in testing_align_funcs}
                for rep in range(len(testing_policy_per_round))}
        
        real_matrix = {al: self.env.reward_matrix_per_align_func(al) for al in testing_align_funcs}
        value_expectations = {al: [] for al in testing_align_funcs}
        value_expectations_expert = {al: [] for al in testing_align_funcs}
        for ratio in ratios_expert_random:
            
            
            
            qualitative_loss_per_al_func = {al: [] for al in testing_align_funcs}
            jsd_per_al_func = {al: [] for al in testing_align_funcs}
            
            for rep, reward_rep in enumerate(learned_rewards_per_round):
                
                for al in testing_align_funcs:
                    real_matrix_al = real_matrix[al]
                    all_trajs = [*((np.random.permutation(np.asarray(expert_trajs[rep][al]))[0:floor(n_seeds*ratio)]).tolist()), 
                         *((np.random.permutation(np.asarray(random_trajs[rep][al]))[0:ceil(n_seeds*(1.0-ratio))]).tolist())]
                    
                    returns_expert = []
                    returns_estimated = []
                    returns_real_from_learned_policy = {alb: [] for alb in basic_profiles}
                    returns_real_from_expert_policy = {alb: [] for alb in basic_profiles}
                    for ti in all_trajs:
                        
                        
                        estimated_return_i = rollout.discounted_sum(reward_rep[al][ti.obs[:-1], ti.acts], gamma=self.discount)
                        real_return_i = rollout.discounted_sum(real_matrix_al[ti.obs[:-1], ti.acts], gamma=self.discount)
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
                                
                                returns_real_from_learned_policy[al_basic].append(real_return_in_learned_pol)
                            for exp in expert_trajs_for_al_estimation[rep][al]:
                                
                                real_return_basic_expert = rollout.discounted_sum(
                                    rb[exp.obs[:-1], exp.acts], 
                                    gamma=self.discount)
                                returns_real_from_expert_policy[al_basic].append(real_return_basic_expert)
                        
                    
                    N = len(all_trajs)
                    i_j = np.random.choice(N, size=(N*10, 2), replace=True)
                    i_indices, j_indices = i_j[:, 0], i_j[:, 1]

                    estimated_diffs = np.clip(returns_estimated[i_indices] - returns_estimated[j_indices], -50.0, 50.0)
                    real_diffs = np.clip(returns_expert[i_indices] - returns_expert[j_indices], -50.0, 50.0)

                    probs_estimated = 1 / (1 + np.exp(estimated_diffs))
                    probs_real = 1 / (1 + np.exp(real_diffs))

                    probs_real = np.array(probs_real) 
                    probs_estimated = np.array(probs_estimated) 
                
                    is_better_estimated = np.asarray(probs_estimated) > (0.5 + epsilon_for_undecided_preference)
                    is_better_real = np.asarray(probs_real) > (0.5 + epsilon_for_undecided_preference)
                    is_worse_estimated = np.asarray(probs_estimated)< (0.5 - epsilon_for_undecided_preference)
                    is_worse_real = np.asarray(probs_real) < (0.5 - epsilon_for_undecided_preference)

                    is_equal_estimated = np.abs(np.asarray(probs_estimated)-0.5) <= epsilon_for_undecided_preference
                    is_equal_real = np.abs(np.asarray(probs_real) - 0.5) <= epsilon_for_undecided_preference
                    
                    real_labels = np.column_stack((is_better_real, is_equal_real, is_worse_real))
                    estimated_labels = np.column_stack((is_better_estimated, is_equal_estimated, is_worse_estimated))
                    

                    #print(real_labels,estimated_labels)
                    #qualitative_loss = qualitative_loss_score(real_labels, estimated_labels, multi_class="ovr")
                    # ACC average (dumb). qualitative_loss = np.mean([np.mean(np.array(real_labels[ri]==estimated_labels[ri], dtype=np.float32)) for ri in range(len(real_labels)) ])
                    # F1 score.
                    qualitative_loss = f1_score(real_labels, estimated_labels, average='weighted',zero_division=np.nan)
                    # F1 score.

                    qualitative_loss_per_al_func[al].append(qualitative_loss)
                    
                    #ce_per_al_func[al].append(th.nn.functional.binary_cross_entropy(th.tensor(probs_real), th.tensor(probs_estimated)).detach().numpy())
                    jsd_per_al_func[al].append(JSD(probs_real, probs_estimated))
                    if float(ratio) == 1.0:
                        value_expectations[al].append({
                            alb: np.mean(returns_real_from_learned_policy[alb]) for alb in basic_profiles
                            })
                        
                        value_expectations_expert[al].append({
                            alb: np.mean(returns_real_from_expert_policy[alb]) for alb in basic_profiles
                            })

            metrics_per_ratio[ratio]={'f1': qualitative_loss_per_al_func, 'jsd': jsd_per_al_func}
            
        return metrics_per_ratio, value_expectations, value_expectations_expert