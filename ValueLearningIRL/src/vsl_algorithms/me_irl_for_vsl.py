"""
Based on https://imitation.readthedocs.io/en/latest/algorithms/mce_irl.html
Adapted for the RoadWorld environment
"""

import enum
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)
import gymnasium as gym
import numpy as np
import scipy
import scipy.special
import torch as th
from seals import base_envs
from imitation.data import types, rollout
from imitation.util import logger as imit_logger
from imitation.util import util


from src.envs.tabularVAenv import TabularVAMDP
from src.vsl_algorithms.base_tabular_vsl_algorithm import BaseTabularMDPVSLAlgorithm
from src.vsl_algorithms.base_vsl_algorithm import dict_metrics
from src.vsl_reward_functions import AbstractVSLRewardFunction, ProabilisticProfiledRewardFunction, TrainingModes


from src.vsl_policies import VAlignedDictDiscreteStateActionPolicyTabularMDP, VAlignedDiscreteSpaceActionPolicy, ValueSystemLearningPolicy


class PolicyApproximators(enum.Enum):
    MCE_ORIGINAL = 'mce_original'
    SOFT_VALUE_ITERATION = 'value_iteration'


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
    approximator_kwargs={
        'value_iteration_tolerance': 0.00001, 'iterations': 100},
    policy_approximator: Union[PolicyApproximators, Callable[[
        gym.Env, np.ndarray, float], np.ndarray]] = PolicyApproximators.MCE_ORIGINAL,
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


def mce_occupancy_measures(
    env: base_envs.TabularModelPOMDP,
    *,
    reward: Optional[np.ndarray] = None,
    pi: Optional[np.ndarray] = None,
    discount: float = 1.0,

    deterministic=False,
    policy_approximator=PolicyApproximators.MCE_ORIGINAL,
    approximator_kwargs={},
    initial_state_distribution=None,
    use_action_visitations=True
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate state visitation frequency Ds for each state s under a given policy pi.

    You can get pi from `mce_partition_fh`.

    Args:
        env: a tabular MDP.
        reward: reward matrix. Defaults is env.reward_matrix.
        pi: policy to simulate. Defaults to soft-optimal policy w.r.t reward
            matrix.
        discount: rate to discount the cumulative occupancy measure D.

        deterministic: Whether the resulting policy to concentrate all its mass in the best action (True) or
            distribute according to a softmax of the Q-Values (False)
        policy_approximator: A methd to calculate or approximate the optimal policy. Can be a callable or one of the 
            supported options in PolicyApproximators.
        approximator_kwargs: Arguments for the policy approximator
        initial_state_distribution: If None, assume random initial state distribution, else support a distribution
            over initial states in the MDP
        use_action_visitations: Whether to return state-action visitation counts instead of state-only visitations (default in ME IRL). 
            Defaults to True, indicating that the algorithm counts the state-action visitations. 

    Returns:
        Tuple of ``D`` (ndarray) and ``Dcum`` (ndarray). ``D`` is of shape
        ``(env.horizon, env.n_states)`` and records the probability of being in a
        given state at a given timestep. ``Dcum`` is of shape ``(env.n_states,)``
        and records the expected discounted number of times each state is visited.

    Raises:
        ValueError: if ``env.horizon`` is None (infinite horizon).
    """
    # shorthand
    horizon = env.horizon
    if horizon is None:
        raise ValueError("Only finite-horizon environments are supported.")
    n_states = env.state_dim
    n_actions = env.action_dim
    T = env.transition_matrix
    if reward is None:
        reward = env.reward_matrix
    if pi is None:
        _, _, pi = mce_partition_fh(env, reward=reward, policy_approximator=policy_approximator,
                                    discount=discount, approximator_kwargs=approximator_kwargs, deterministic=deterministic)

    initial_state_distribution = env.initial_state_dist if initial_state_distribution is None else initial_state_distribution

    D = np.zeros((horizon + 1, n_states))

    D[0, :] = initial_state_distribution
    for t in range(horizon):
        for a in range(n_actions):
            if len(pi.shape) == 3:
                E = D[t] * pi[t, :, a]
            elif len(pi.shape) == 2:
                E = D[t] * pi[:, a]
            else:
                E = D[t] * (pi == a)  # Multiarmed bandit variant...?
            D[t + 1, :] += E @ T[:, a, :]

    Dcum = rollout.discounted_sum(D, discount)

    if use_action_visitations:
        if len(pi.shape) == 3:
            D = np.zeros((horizon+1, n_states, n_actions))
            D_s = np.zeros((horizon + 1, n_states))

            D_s[0, :] = initial_state_distribution
            Dcum = np.zeros((n_states, n_actions))
            for a in range(n_actions):
                if len(pi.shape) == 3:
                    D[0, :, a] = initial_state_distribution*pi[0, :, a]
                elif len(pi.shape) == 2:
                    D[0, :, a] = initial_state_distribution*pi[:, a]
            for t in range(horizon):

                for a in range(n_actions):
                    for a_ in range(n_actions):
                        if len(pi.shape) == 3:
                            E = D[t, :, a] * pi[t, :, a_]
                        elif len(pi.shape) == 2:
                            E = D[t, :, a] * pi[:, a_]
                        else:
                            # Multiarmed bandit variant...?
                            E = D[t, :, a] * (pi == a_)
                        D[t + 1, :, a_] += E @ T[:, a, :]
                """for a_ in range(n_actions):
                    D[t+1, :, a_] = D[t+1, : ,a_] * (pi[t+1, :, a_] if t < horizon-1 else 1)"""

            for a in range(n_actions):
                Dcum[:, a] = rollout.discounted_sum(D[:, :, a], discount)
        else:
            Dcum = np.multiply(pi, Dcum[:, np.newaxis])
            assert Dcum.shape == (n_states, n_actions)
        if __debug__:
            D = np.zeros((horizon + 1, n_states))

            D[0, :] = initial_state_distribution
            for t in range(horizon):
                for a in range(n_actions):
                    if len(pi.shape) == 3:
                        E = D[t] * pi[t, :, a]
                    elif len(pi.shape) == 2:
                        E = D[t] * pi[:, a]
                    else:
                        E = D[t] * (pi == a)  # Multiarmed bandit variant...?
                    D[t + 1, :] += E @ T[:, a, :]

            Dstates = rollout.discounted_sum(D, discount)
            # print(np.max(Dstates - np.sum(Dcum, axis=-1)))
            # assert np.allclose(np.sum(Dcum, axis=-1), Dstates)
        assert isinstance(Dcum, np.ndarray)
    return D, Dcum


def get_demo_oms_from_trajectories(trajs: Iterable[types.Trajectory], state_dim, discount, groupby_al_func=True):
    if groupby_al_func:
        num_demos_per_al = dict()
        demo_state_om = dict()

        for traj in trajs:
            al = traj.infos[0]['align_func']
            if al not in demo_state_om:
                num_demos_per_al[al] = 0
                demo_state_om[al] = np.zeros((state_dim,))

            cum_discount = 1.0
            for obs in types.assert_not_dictobs(traj.obs):
                demo_state_om[al][obs] += cum_discount
                cum_discount *= discount
            num_demos_per_al[al] += 1
        for al in demo_state_om.keys():
            demo_state_om[al] /= num_demos_per_al[al]
    else:
        num_demos = 0
        demo_state_om = np.zeros((state_dim,))

        for traj in trajs:
            cum_discount = 1.0
            for obs in types.assert_not_dictobs(traj.obs):
                demo_state_om[obs] += cum_discount
                cum_discount *= discount
            num_demos += 1
        demo_state_om /= num_demos

    return demo_state_om


class MaxEntropyIRLForVSL(BaseTabularMDPVSLAlgorithm):
    """
    Based on https://imitation.readthedocs.io/en/latest/algorithms/mce_irl.html
    """

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

    def __init__(
        self,
        env: Union[TabularVAMDP],
        reward_net: AbstractVSLRewardFunction,
        vgl_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        vsi_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        vgl_optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        vsi_optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        discount: float = 1.0,
        vc_diff_epsilon: float = 1e-3,
        gradient_norm_epsilon: float = 1e-4,
        log_interval: Optional[int] = 100,
        vgl_expert_policy: Optional[ValueSystemLearningPolicy] = None,
        vsi_expert_policy: Optional[ValueSystemLearningPolicy] = None,
        vgl_expert_sampler=None,
        vsi_expert_sampler=None,
        # A Society or other mechanism might return different alignment functions at different times.
        target_align_func_sampler=lambda *args: args[0],


        vsi_target_align_funcs=[],

        vgl_target_align_funcs=[],

        demo_om_from_policy=True,
        policy_approximator=PolicyApproximators.MCE_ORIGINAL,

        approximator_kwargs={
            'value_iteration_tolerance': 0.0000001, 'iterations': 100},

        initial_state_distribution_train=None,
        initial_state_distribution_test=None,
        training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
        learn_stochastic_policy=True,
        expert_is_stochastic=True,
        environment_is_stochastic=False,
        use_feature_expectations_for_vsi=False,
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,

    ) -> None:
        super().__init__(
            env=env,
            reward_net=reward_net,
            vgl_optimizer_cls=vgl_optimizer_cls,
            vsi_optimizer_cls=vsi_optimizer_cls,
            vgl_optimizer_kwargs=vgl_optimizer_kwargs,
            vsi_optimizer_kwargs=vsi_optimizer_kwargs,
            discount=discount,
            log_interval=log_interval,
            vgl_expert_policy=vgl_expert_policy,
            vsi_expert_policy=vsi_expert_policy,
            # A Society or other mechanism might return different alignment functions at different times.
            target_align_func_sampler=target_align_func_sampler,


            vsi_target_align_funcs=vsi_target_align_funcs,
            environment_is_stochastic=environment_is_stochastic,
            vgl_target_align_funcs=vgl_target_align_funcs,

            training_mode=training_mode,
            learn_stochastic_policy=learn_stochastic_policy,
            custom_logger=custom_logger,
        )

        self.policy_approximator = policy_approximator
        self.approximator_kwargs = approximator_kwargs
        self.expert_is_stochastic = expert_is_stochastic
        self.use_feature_expectations = use_feature_expectations_for_vsi

        self.exposed_state_env = base_envs.ExposePOMDPStateWrapper(env)
        self.vgl_demo_state_om = dict()
        self.vsi_demo_state_om = dict()

        # list of expert trajectories just to see different origin destinations and align_funcs
        self.vgl_expert_sampler = vgl_expert_sampler
        # list of expert target align_func trajectories
        self.vsi_expert_sampler = vsi_expert_sampler

        self.initial_state_distribution_train = initial_state_distribution_train if initial_state_distribution_train is not None else env.initial_state_dist
        self.initial_state_distribution_test = initial_state_distribution_test if initial_state_distribution_test is not None else env.initial_state_dist

        self.demo_om_from_policy = demo_om_from_policy
        self.target_align_function_sampler = target_align_func_sampler
        if demo_om_from_policy:
            self._set_vgl_demo_oms_from_policy(self.vgl_expert_policy)
            self._set_vsi_demo_oms_from_policy(self.vsi_expert_policy)

        self.vc_diff_epsilon = vc_diff_epsilon
        self.gradient_norm_epsilon = gradient_norm_epsilon

        # Initialize policy to be uniform random. We don't use this for MCE IRL
        # training, but it gives us something to return at all times with `policy`
        # property, similar to other algorithms.
        if self.env.horizon is None:
            raise ValueError("Only finite-horizon environments are supported.")

        self.learned_policy_per_va = VAlignedDictDiscreteStateActionPolicyTabularMDP(
            {}, env=self.env, state_encoder=lambda exposed_state, info: exposed_state)  # Starts with random policy
        for al_func in self.vgl_target_align_funcs:
            probability_matrix = np.random.rand(
                self.env.state_dim, self.env.action_dim)
            random_pol = probability_matrix / \
                probability_matrix.sum(axis=1, keepdims=True)
            self.learned_policy_per_va.set_policy_for_va(al_func, random_pol)

    def mce_partition_fh_per_align_func(self, align_func, reward_matrix=None, action_mat=None, obs_action_mat=None, reward_mode=TrainingModes.VALUE_GROUNDING_LEARNING, use_probabilistic_reward=False):

        if reward_matrix is None:
            reward_matrix_torch, reward_matrix = self.calculate_rewards(align_func=align_func,
                                                                        obs_mat=self.env.observation_matrix,
                                                                        action_mat=action_mat,
                                                                        obs_action_mat=obs_action_mat,
                                                                        reward_mode=reward_mode, requires_grad=False, use_probabilistic_reward=use_probabilistic_reward)

        V, Q, pi = mce_partition_fh(env=self.env, reward=reward_matrix, discount=self.discount,
                                    policy_approximator=self.policy_approximator,
                                    approximator_kwargs=self.approximator_kwargs,
                                    deterministic=not self.learn_stochastic_policy)

        return pi

    def probabilistic_reward_per_target_align_func(self, target_al):
        self.probabilistic_reward_net.set_alignment_function(
            self.target_align_funcs_to_learned_align_funcs[target_al])

        _, new_reward_per_rep, align_per_rep, prob_per_rep = self.calculate_rewards(self.target_align_funcs_to_learned_align_funcs[target_al],
                                                                                    obs_mat=self.torch_obs_mat,
                                                                                    action_mat=self.torch_action_mat,
                                                                                    obs_action_mat=self.torch_obs_action_mat,
                                                                                    requires_grad=False,
                                                                                    use_probabilistic_reward=True, n_reps_if_probabilistic_reward=1,
                                                                                    reward_mode=TrainingModes.EVAL)
        # print("prob reward", target_al, self.target_align_funcs_to_learned_align_funcs[
        #      target_al], self.probabilistic_reward_net.cur_align_func, align_per_rep, prob_per_rep)
        return new_reward_per_rep[0]

    def train_vsl_probabilistic(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, n_reward_reps_if_probabilistic_reward, target_align_func):
        average_losses = 0.0
        for t in range(max_iter):
            predicted_rs_np, visitations, old_pi, loss, reward, learned_al_function, average_losses = self._train_vsi_distribution_reward(self.torch_obs_mat, target_align_func,
                                                                                                                                          action_mat=self.torch_action_mat, obs_action_mat=self.torch_obs_action_mat,
                                                                                                                                          n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories,
                                                                                                                                          n_reward_reps=n_reward_reps_if_probabilistic_reward,
                                                                                                                                          average_losses=average_losses, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed)

            grad_norm, abs_diff_in_vc = self._train_statistics(
                t, self.vsi_demo_state_om[target_align_func], visitations, loss, reward, target_align_func, learned_al_function)
            self.linf_delta_per_align_func[target_align_func].append(
                np.max(abs_diff_in_vc))
            self.grad_norm_per_align_func[target_align_func].append(grad_norm)

            self.train_callback(t)
            if self.linf_delta_per_align_func[target_align_func][-1] <= self.vc_diff_epsilon or grad_norm <= self.gradient_norm_epsilon:
                self._print_statistics(
                    t, visitations, loss, reward, target_align_func, learned_al_function, grad_norm, abs_diff_in_vc)
                break
        return learned_al_function

    def train_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func):
        for t in range(max_iter):

            predicted_r_np, visitations, expert_visitations, old_pi, loss, reward, learned_al_function = self._train_step_vsi(self.torch_obs_mat,
                                                                                                                              target_align_func=target_align_func,
                                                                                                                              action_mat=self.torch_action_mat,
                                                                                                                              obs_action_mat=self.torch_obs_action_mat,
                                                                                                                              n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories,
                                                                                                                              n_sampled_trajs_per_seed=n_sampled_trajs_per_seed)
            # self.rewards_per_target_align_func[target_align_func] = reward

            grad_norm, abs_diff_in_vc = self._train_statistics(
                t, expert_visitations, visitations, loss, reward, target_align_func, learned_al_function)
            self.linf_delta_per_align_func[target_align_func].append(
                np.max(abs_diff_in_vc))
            self.grad_norm_per_align_func[target_align_func].append(grad_norm)

            self.train_callback(t)
            if self.linf_delta_per_align_func[target_align_func][-1] <= self.vc_diff_epsilon or grad_norm <= self.gradient_norm_epsilon:
                self._print_statistics(t, visitations, expert_visitations, loss, reward,
                                       target_align_func, learned_al_function, grad_norm, abs_diff_in_vc)
                break
        return learned_al_function

    def train(self, max_iter: int = 1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None, n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_probabilistic_reward=False, n_reward_reps_if_probabilistic_reward=10) -> np.ndarray:

        self.linf_delta_per_align_func = {al: [] for al in set(
            self.vsi_target_align_funcs).union(set(self.vgl_target_align_funcs))}
        self.grad_norm_per_align_func = {al: [] for al in set(
            self.vsi_target_align_funcs).union(set(self.vgl_target_align_funcs))}

        # Organizing learned content:"""

        return super().train(max_iter=max_iter, mode=mode,
                             assumed_grounding=assumed_grounding,
                             n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories,
                             use_probabilistic_reward=use_probabilistic_reward,
                             n_sampled_trajs_per_seed=n_sampled_trajs_per_seed,
                             n_reward_reps_if_probabilistic_reward=n_reward_reps_if_probabilistic_reward)

    def train_vgl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed):
        reward_nets_per_target_align_func = dict()
        for t in range(max_iter):
            for align_func in self.vgl_target_align_funcs:
                if t > 0 and (self.linf_delta_per_align_func[align_func][-1] <= self.vc_diff_epsilon or self.grad_norm_per_align_func[align_func][-1] <= self.gradient_norm_epsilon):
                    continue
                old_reward, visitations, expert_visitations, old_pi, loss, reward = self._train_step_vgl(self.torch_obs_mat,
                                                                                                         align_func=align_func,
                                                                                                         action_mat=self.torch_action_mat,
                                                                                                         obs_action_mat=self.torch_obs_action_mat,
                                                                                                         n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed)
                # self.rewards_per_target_align_func[align_func] = reward

                grad_norm, abs_diff_in_vc = self._train_statistics(
                    t, expert_visitations, visitations, loss, reward, align_func, learned_grounding=self.current_net.get_learned_grounding())
                self.linf_delta_per_align_func[align_func].append(
                    np.max(abs_diff_in_vc))
                self.grad_norm_per_align_func[align_func].append(grad_norm)
            # print(self.linf_delta_per_align_func)
            last_max_vc_diff = max(
                [lvlist[-1] for al_fun, lvlist in self.linf_delta_per_align_func.items() if al_fun in self.vgl_target_align_funcs])
            last_max_grad_norm = max(
                [grlist[-1] for al_fun, grlist in self.grad_norm_per_align_func.items() if al_fun in self.vgl_target_align_funcs])

            self.train_callback(t)
            if last_max_vc_diff <= self.vc_diff_epsilon or last_max_grad_norm <= self.gradient_norm_epsilon:
                self._print_statistics(t, visitations, expert_visitations, loss, reward, align_func, None,
                                       grad_norm, abs_diff_in_vc, learned_grounding=self.current_net.get_learned_grounding())
                break

        for align_func in self.vgl_target_align_funcs:
            self.current_net.set_alignment_function(align_func)
            reward_nets_per_target_align_func[align_func] = self.current_net.copy(
            )
            assert len(self.linf_delta_per_align_func[align_func]) > 0
            assert len(self.grad_norm_per_align_func[align_func]) > 0

        return reward_nets_per_target_align_func

    def calculate_learned_policies(self, target_align_funcs) -> ValueSystemLearningPolicy:
        for target_align_func in target_align_funcs:
            learned_al_function = target_align_func if self.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING else self.target_align_funcs_to_learned_align_funcs[
                target_align_func]
            pi = self.mce_partition_fh_per_align_func(learned_al_function,
                                                      reward_matrix=self.rewards_per_target_align_func_callable(
                                                          target_align_func)(),
                                                      action_mat=self.torch_action_mat,
                                                      obs_action_mat=self.torch_obs_action_mat,
                                                      reward_mode=TrainingModes.EVAL)
            # self.learned_policy_per_va.set_policy_for_va(target_align_func, pi)
            self.learned_policy_per_va.set_policy_for_va(
                learned_al_function, pi)
        return self.learned_policy_per_va

    def get_metrics(self):
        metrics = super().get_metrics()
        new_metrics = dict_metrics(
            tvc=self.linf_delta_per_align_func, grad=self.grad_norm_per_align_func)
        metrics.update(new_metrics)
        return metrics

    def _train_statistics(self, t, expert_demo_om, visitations, loss, reward, target_align_func, learned_al_function=None, learned_grounding=None):
        grads = []
        for p in self.current_net.parameters():
            assert p.grad is not None  # for type checker
            grads.append(p.grad)
        grad_norm = util.tensor_iter_norm(grads).item()
        abs_diff_in_visitation_counts = np.abs(expert_demo_om - visitations)
        if self.log_interval is not None and 0 == (t % self.log_interval):
            self._print_statistics(t, visitations, expert_demo_om, loss, reward, target_align_func,
                                   learned_al_function, grad_norm, abs_diff_in_visitation_counts, learned_grounding=learned_grounding)
        return grad_norm, abs_diff_in_visitation_counts

    def _print_statistics(self, t, visitations, expert_visitations, loss, reward, target_align_func, learned_al_function,  grad_norm, abs_diff_in_visitation_counts, learned_grounding=None):
        avg_linf_delta = np.mean(abs_diff_in_visitation_counts)
        norm_reward = np.linalg.norm(reward)
        max_error_state = np.where(
            abs_diff_in_visitation_counts == np.max(abs_diff_in_visitation_counts))
        max_error_state = (max_error_state[0][0], max_error_state[1][0]) if len(
            abs_diff_in_visitation_counts.shape) == 2 else max_error_state[0][0]

        params = self.current_net.parameters()
        weight_norm = util.tensor_iter_norm(params).item()
        self.logger.record("iteration", t)
        self.logger.record("Target align_func", target_align_func)
        if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            self.logger.record("Learned align_func: ", tuple(
                [float("{0:.3f}".format(v)) for v in learned_al_function]))
        else:
            self.logger.record(
                "Learned grounding: ", learned_grounding if learned_grounding is not None else "none")
        self.logger.record("linf_delta", np.max(abs_diff_in_visitation_counts))
        self.logger.record("weight_norm", weight_norm)
        self.logger.record("grad_norm", grad_norm)
        self.logger.record("loss", loss)
        self.logger.record("avg_linf_delta", avg_linf_delta)
        self.logger.record("norm reward", norm_reward)

        if not (self.use_feature_expectations and self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION):
            self.logger.record("state_worse", max_error_state)

            self.logger.record("state_worse_visit",
                               visitations[max_error_state])
            if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
                self.logger.record("state_worse_worig",
                                   expert_visitations[max_error_state])
            else:
                self.logger.record("state_worse_worig",
                                   expert_visitations[max_error_state])
        self.logger.dump(t)

    def state_action_reward_from_computed_reward(self, rewards):
        return transform_state_only_reward_to_state_action_reward(
            rewards, self.env.state_dim, self.env.action_dim, self.env.transition_matrix)

    """
    def state_action_callable_reward_from_computed_rewards_per_target_align_func(self, rewards_per_target_align_func: Union[Dict, Callable]):
        if isinstance(rewards_per_target_align_func, dict):
            for al_f in rewards_per_target_align_func.keys():
                rewards_per_target_align_func[al_f] = transform_state_only_reward_to_state_action_reward(
                    rewards_per_target_align_func[al_f], self.env.state_dim, self.env.action_dim, self.env.transition_matrix)
                                                
            rewards_per_target_align_func_callable = lambda al_f: rewards_per_target_align_func[al_f]
        else:
            rewards_per_target_align_func_callable = lambda al_func: transform_state_only_reward_to_state_action_reward(
                rewards_per_target_align_func(al_func), 
                state_dim=self.env.state_dim,  
                action_dim=self.env.action_dim,
                transition_matrix=self.env.transition_matrix)
        return rewards_per_target_align_func_callable """

    def get_expert_demo_om(self, seed_target_align_func, n_seeds_for_sampled_trajectories=30, n_sampled_trajs_per_seed=3, use_action_visitations=False):

        target_align_func = self.target_align_function_sampler(
            seed_target_align_func)

        if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:

            if self.demo_om_from_policy:
                if target_align_func not in self.vsi_demo_state_om.keys() or use_action_visitations != (self.vsi_demo_state_om[target_align_func].shape == (self.env.state_dim, self.env.action_dim)):
                    _, self.vsi_demo_state_om[target_align_func], _ = self.mce_occupancy_measures(pi=self.vsi_expert_policy.policy_per_va(
                        target_align_func), deterministic=not self.expert_is_stochastic, use_action_visitations=use_action_visitations)

            else:
                if target_align_func not in self.vsi_demo_state_om.keys() or self.expert_is_stochastic:
                    trajs = self.vsi_expert_sampler(
                        [target_align_func], n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)
                    self.vsi_demo_state_om[target_align_func] = get_demo_oms_from_trajectories(
                        trajs, state_dim=self.env.state_dim, discount=self.discount, groupby_al_func=False)
            return self.vsi_demo_state_om[target_align_func]
        else:
            if self.demo_om_from_policy:
                if target_align_func not in self.vgl_demo_state_om.keys() or use_action_visitations != (self.vgl_demo_state_om[target_align_func].shape == (self.env.state_dim, self.env.action_dim)):

                    _, self.vgl_demo_state_om[target_align_func], _ = self.mce_occupancy_measures(pi=self.vgl_expert_policy.policy_per_va(
                        target_align_func), deterministic=not self.expert_is_stochastic, use_action_visitations=use_action_visitations)
            else:
                if target_align_func not in self.vgl_demo_state_om.keys() or self.expert_is_stochastic:
                    trajs = self.vgl_expert_sampler(
                        [target_align_func], n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)
                    self.vgl_demo_state_om[target_align_func] = get_demo_oms_from_trajectories(
                        trajs, state_dim=self.env.state_dim, discount=self.discount, groupby_al_func=False)
            return self.vgl_demo_state_om[target_align_func]

    def mce_occupancy_measures(
            self,
        reward_matrix: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None,
        deterministic=False,
        train_or_test='train',
        use_action_visitations=True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the expected time dependent visitation counts, the accumulated ones according to the discount and the approximated policy

        Args:
            reward_matrix (Optional[np.ndarray], optional): Reward matrix for approximating the policy (not needed if pi is not None). Defaults to None.
            pi (Optional[np.ndarray], optional): Policy matrix from which ti calculate the measures. Defaults to None.
            deterministic (bool, optional): See mce_occuancy_measures. Defaults to False.
            train_or_test (str, optional): To select the initial state distribution. Defaults to 'train'.
            use_action_visitations (bool, optional): See mce_occuancy_measures. Defaults to True.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: time dependent visitation counts, 
            the accumulated ones according to the discount and the approximated policy
        """
        if pi is None:
            V, Q, pi = mce_partition_fh(self.exposed_state_env, reward=reward_matrix, discount=self.discount, policy_approximator=self.policy_approximator,
                                        approximator_kwargs=self.approximator_kwargs,
                                        deterministic=deterministic)

        D, Dcums = mce_occupancy_measures(env=self.exposed_state_env, pi=pi,
                                          discount=self.discount,
                                          policy_approximator=self.policy_approximator,
                                          approximator_kwargs=self.approximator_kwargs,
                                          deterministic=deterministic,
                                          initial_state_distribution=self.initial_state_distribution_train if train_or_test == 'train' else self.initial_state_distribution_test,
                                          use_action_visitations=use_action_visitations)
        return D, Dcums, pi

    def _train_vsi_distribution_reward(self, obs_mat, target_align_func, beta=0.999, action_mat=None, obs_action_mat=None, n_reward_reps=10, n_seeds_for_sampled_trajectories=30, n_sampled_trajs_per_seed=3, n_rep_permutations=10, average_losses=0):

        learned_al_function = self.probabilistic_reward_net.get_learned_align_function()
        use_action_visitations = self.current_net.use_action or self.current_net.use_next_state

        predicted_rs, predicted_rs_np, align_func_used_in_each_repetition, prob_of_each_repetition = self.calculate_rewards(align_func=None, obs_mat=obs_mat, action_mat=action_mat,
                                                                                                                            obs_action_mat=obs_action_mat,
                                                                                                                            reward_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                                                                                                            recover_previous_config_after_calculation=False, use_probabilistic_reward=True,
                                                                                                                            n_reps_if_probabilistic_reward=n_reward_reps, requires_grad=False)
        with th.no_grad():
            policy_per_target = dict()
            policy_matrix_per_target = dict()

            if not use_action_visitations:
                visitations_per_repetition = np.zeros(
                    (n_reward_reps, self.env.state_dim))
                demo_om_per_repetition = np.zeros(
                    (n_reward_reps, self.env.state_dim))  # [[]]*n_reward_reps
            else:
                visitations_per_repetition = np.zeros(
                    (n_reward_reps, self.env.state_dim, self.env.action_dim))
                demo_om_per_repetition = np.zeros(
                    (n_reward_reps, self.env.state_dim, self.env.action_dim))
            om_per_align_func = dict()
            n_seeds = n_seeds_for_sampled_trajectories

            for i, align_func_i in enumerate(align_func_used_in_each_repetition):

                demo_om_per_repetition[i] = self.get_expert_demo_om(
                    target_align_func, n_seeds_for_sampled_trajectories=n_seeds, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed, use_action_visitations=use_action_visitations)

                if align_func_i not in policy_matrix_per_target.keys():
                    policy_matrix_per_target[align_func_i] = self.mce_partition_fh_per_align_func(
                        align_func_i, reward_matrix=predicted_rs_np[i], reward_mode=self.current_net.mode)

                if self.demo_om_from_policy:
                    if align_func_i not in om_per_align_func.keys():
                        _, visitations_per_repetition[i], _ = self.mce_occupancy_measures(
                            reward_matrix=predicted_rs_np[i], pi=policy_matrix_per_target[align_func_i], deterministic=not self.learn_stochastic_policy, use_action_visitations=use_action_visitations)
                    else:
                        visitations_per_repetition[i] = om_per_align_func[align_func_i]
                else:
                    if align_func_i not in policy_per_target.keys():
                        policy_per_target[align_func_i] = VAlignedDictDiscreteStateActionPolicyTabularMDP(
                            policy_per_va_dict={align_func_i: policy_matrix_per_target[align_func_i]}, env=self.env)

                    trajs = policy_per_target[align_func_i].obtain_trajectories(
                        n_seeds=n_seeds, repeat_per_seed=n_sampled_trajs_per_seed, stochastic=self.learn_stochastic_policy, with_alignfunctions=[
                            align_func_i,], t_max=self.env.horizon, exploration=0
                    )
                    visitations_per_repetition[i] = get_demo_oms_from_trajectories(
                        trajs=trajs, state_dim=self.env.state_dim, discount=self.discount, groupby_al_func=False)

                    # print(align_func_i, n_seeds, trajs, n_sampled_trajs_per_seed)
                om_per_align_func[align_func_i] = visitations_per_repetition[i]

        for r in range(1):

            self.probabilistic_vsi_optimizer.zero_grad()
            if False:  #  This is one by one. Does not work correctly always
                visitation_diff_expectation = np.mean([np.abs(
                    visitations_per_repetition - np.random.permutation(demo_om_per_repetition)) for r in range(1)], axis=0)
                visitation_diff_expectation_th = th.as_tensor(visitation_diff_expectation, dtype=self.probabilistic_reward_net.dtype,
                                                              device=self.probabilistic_reward_net.device)

                average_losses = np.float32(
                    (1-beta)*visitation_diff_expectation + beta*average_losses)

                print("VIS TIME: ", s)
                prob_of_each_repetition = prob_of_each_repetition.float()
                weights_th = th.mean(th.matmul(
                    (prob_of_each_repetition), visitation_diff_expectation_th-average_losses), axis=0)
                # print(weights_th.size())

                loss = weights_th
                real_loss = th.mean(loss)
            # This assumes that prob(i) == prob(align_i), and matches expected visitation counts.
            seen_al_funcs = []
            demo_om_per_repetition_th = th.as_tensor(demo_om_per_repetition, dtype=self.probabilistic_reward_net.dtype,
                                                     device=self.probabilistic_reward_net.device).requires_grad_(False)
            visitation_per_repetition_th = th.as_tensor(visitations_per_repetition, dtype=self.probabilistic_reward_net.dtype,
                                                        device=self.probabilistic_reward_net.device).requires_grad_(False)

            expected_vc_th = None
            for i, al_i in enumerate(align_func_used_in_each_repetition):
                if al_i not in seen_al_funcs:
                    if expected_vc_th is None:
                        expected_vc_th = prob_of_each_repetition[i] * \
                            visitation_per_repetition_th[i]
                    else:
                        expected_vc_th += prob_of_each_repetition[i] * \
                            visitation_per_repetition_th[i]
                    seen_al_funcs.append(al_i)

            expected_demo_om_th = th.mean(demo_om_per_repetition_th, axis=0)
            if use_action_visitations:
                assert expected_vc_th.size() == th.Size(
                    [self.env.state_dim, self.env.action_dim])
                assert expected_demo_om_th.size() == th.Size(
                    [self.env.state_dim, self.env.action_dim])
            else:
                assert expected_vc_th.size() == th.Size([self.env.state_dim])
                assert expected_demo_om_th.size() == th.Size(
                    [self.env.state_dim])
            real_loss = th.mean(th.abs(expected_vc_th-expected_demo_om_th))

            real_loss.backward(retain_graph=False)
            # average_losses = (1-beta)*real_loss.detach().numpy() + beta*average_losses

            self.probabilistic_vsi_optimizer.step()

        # average_losses = (1-beta)*np.mean([th.abs(losses_per_r[i]).detach().numpy() for i in range(len(losses_per_r))]) + beta*average_losses

        with th.no_grad():
            learned_al_function = self.probabilistic_reward_net.get_learned_align_function()
            # print("LEARNED ALIG", learned_al_function)

            avg_oldreward = np.mean(predicted_rs_np, axis=0)
            visitations = expected_vc_th.detach().numpy()  # mean or sum(?)
            demo_om_per_repetition = expected_demo_om_th.detach().numpy()
            self.vsi_demo_state_om[target_align_func] = demo_om_per_repetition

        return predicted_rs_np, visitations, policy_matrix_per_target, real_loss, avg_oldreward, learned_al_function, average_losses

    def feature_expectations(self, policy: np.ndarray, features_are_groundings=True, obs_mat: th.Tensor = None, action_mat: th.Tensor = None,
                             obs_action_mat: th.Tensor = None):
        initial_state_dist = self.initial_state_distribution_train

        state_dist = initial_state_dist
        accumulated_feature_expectations = 0
        for t in range(self.env.horizon):
            next_state_observations = None

            if self.current_net.use_next_state:
                next_states = self._resample_next_states()
                next_state_observations = obs_mat[next_states]

            if self.current_net.use_one_hot_state_action:
                if self.current_net.use_next_state:
                    next_state_observations = next_state_observations.view(
                        *obs_action_mat.shape)
                if features_are_groundings:
                    features = self.current_net.forward_value_groundings(
                        obs_action_mat, None, next_state_observations, None)
                else:
                    features = self.current_net.construct_input(
                        obs_action_mat, None, next_state_observations, None)
                features = th.reshape(
                    features, (self.env.state_dim, self.env.action_dim, features.shape[-1]))
            elif self.current_net.use_action or self.current_net.use_next_state:
                if self.current_net.use_action:
                    assert action_mat is not None
                    assert action_mat.size() == (self.env.action_dim,
                                                 obs_mat.shape[0], self.env.action_dim)
                if self.current_net.use_next_state:
                    assert next_state_observations is not None

                if features_are_groundings:
                    features = th.stack([
                        self.current_net.forward_value_groundings(
                            obs_mat,
                            (action_mat[i]
                             if self.current_net.use_action else None),
                            next_state_observations[:, i,
                                                    :] if self.current_net.use_next_state else None,
                            None)
                        for i in range(self.env.action_dim)], dim=1)
                else:
                    features = th.stack([
                        self.current_net.construct_input(
                            obs_mat,
                            (action_mat[i]
                             if self.current_net.use_action else None),
                            next_state_observations[:, i,
                                                    :] if self.current_net.use_next_state else None,
                            None)
                        for i in range(self.env.action_dim)], dim=1)
            assert (features.size()[0], features.size()[1]) == (
                obs_mat.shape[0], self.env.action_dim)
            features = features.detach().numpy()

            pol_t = policy if len(policy.shape) == 2 else policy[0]
            state_action_prob = np.multiply(pol_t, state_dist[:, np.newaxis])

            features_time_t = np.sum(
                features * state_action_prob[:, :, np.newaxis], axis=(0, 1))/self.env.state_dim

            if t == 0:
                accumulated_feature_expectations = features_time_t
            else:
                accumulated_feature_expectations += features_time_t
            # /self.env.state_dim
            state_dist = np.sum(self.env.transition_matrix *
                                state_action_prob[:, :, np.newaxis], axis=(0, 1))
            assert np.allclose(np.sum(state_dist), 1.0)
        return accumulated_feature_expectations

    def mce_vsl_loss_calculation(self, target_align_func, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, predicted_r, predicted_r_np,
                                 obs_mat: th.Tensor = None, action_mat: th.Tensor = None,
                                 obs_action_mat: th.Tensor = None):
        use_action_visitations = self.current_net.use_action or self.current_net.use_next_state

        prev_pi = None
        if self.use_feature_expectations and self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            prev_pi = self.mce_partition_fh_per_align_func(
                target_align_func, reward_matrix=predicted_r_np, reward_mode=self.current_net.mode)
            visitations_or_feature_expectations = self.feature_expectations(
                policy=prev_pi, features_are_groundings=True, obs_mat=obs_mat, action_mat=action_mat, obs_action_mat=obs_action_mat)
            expert_visitations_or_feature_expectation = self.feature_expectations(policy=self.vsi_expert_policy.policy_per_va(
                target_align_func), features_are_groundings=True, obs_mat=obs_mat, action_mat=action_mat, obs_action_mat=obs_action_mat)
        else:
            expert_visitations_or_feature_expectation = self.get_expert_demo_om(
                target_align_func, n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed, use_action_visitations=use_action_visitations)
            if self.demo_om_from_policy:
                _, visitations_or_feature_expectations, prev_pi = self.mce_occupancy_measures(
                    reward_matrix=predicted_r_np,
                    deterministic=not self.learn_stochastic_policy, use_action_visitations=use_action_visitations
                )

            else:
                if prev_pi is None:
                    prev_pi = self.mce_partition_fh_per_align_func(
                        target_align_func, reward_matrix=predicted_r_np, reward_mode=self.current_net.mode)
                policy = VAlignedDictDiscreteStateActionPolicyTabularMDP(
                    policy_per_va_dict={target_align_func: prev_pi}, env=self.env)

                trajs = policy.obtain_trajectories(
                    n_seeds=n_seeds_for_sampled_trajectories, repeat_per_seed=n_sampled_trajs_per_seed, stochastic=self.learn_stochastic_policy, with_alignfunctions=[
                        target_align_func], t_max=self.env.horizon, exploration=0
                )
                visitations_or_feature_expectations = get_demo_oms_from_trajectories(
                    trajs=trajs, state_dim=self.env.state_dim, discount=self.discount)[target_align_func]
        # Forward/back/step (grads are zeroed at the top).
        # weights_th(s) = \pi(s) - D(s)

        if self.use_feature_expectations and self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            # print(visitations_or_feature_expectations)
            loss = th.mean(
                self.current_net.value_system_layer(self.current_net.cur_align_func)(
                    th.as_tensor(visitations_or_feature_expectations.reshape(
                        (1, -1)), dtype=self.current_net.dtype, device=self.current_net.device)
                ) -
                self.current_net.value_system_layer(self.current_net.cur_align_func)(
                    th.as_tensor(expert_visitations_or_feature_expectation.reshape(
                        (1, -1)), device=self.current_net.device, dtype=self.current_net.dtype)
                ))
        else:
            weights_th = th.as_tensor(
                visitations_or_feature_expectations - expert_visitations_or_feature_expectation,
                dtype=self.current_net.dtype,
                device=self.current_net.device,
            )

            if len(predicted_r.shape) == 1:
                # The "loss" is then:
                #   E_\pi[r_\theta(S)] - E_D[r_\theta(S)]
                loss = th.dot(weights_th, predicted_r)
            else:  # Use action in the reward.
                if predicted_r.shape == weights_th.shape:
                    loss = th.sum(th.multiply(predicted_r, weights_th))
                else:
                    next_state_prob = th.as_tensor(self.env.transition_matrix, dtype=self.current_net.dtype,
                                                   device=self.current_net.device)
                    # Calculate the expected visitations on the expected next states
                    # loss = th.sum(th.vstack([th.matmul(predicted_r.t(), th.mul(next_state_prob[:,:,k],weights_th[k])) for k in range(self.env.state_dim)]))
                    # Expand dimensions of `weights_th` to align with `next_state_prob`
                    loss_matrix = predicted_r.unsqueeze(
                        2) * (next_state_prob * weights_th.unsqueeze(0).unsqueeze(0))  # Shape: (N, M, K)
                    loss = loss_matrix.sum()
                # loss = Sum_s,a,s'[R(s,a)*P(s,a,s')*weights_th(s')]

        loss.backward()
        return visitations_or_feature_expectations, expert_visitations_or_feature_expectation, prev_pi, loss

    def _train_step_vsi(self, obs_mat: th.Tensor, target_align_func, action_mat: th.Tensor = None, obs_action_mat: th.Tensor = None, next_obs_mat: th.Tensor = None, n_seeds_for_sampled_trajectories=30, n_sampled_trajs_per_seed=3) -> Tuple[np.ndarray, np.ndarray]:

        assert self.current_net.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION
        self.vsi_optimizer.zero_grad()

        predicted_r, predicted_r_np = self.calculate_rewards(align_func=None, obs_mat=obs_mat, action_mat=action_mat,
                                                             obs_action_mat=obs_action_mat,
                                                             next_state_obs_mat=next_obs_mat,
                                                             reward_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                                             recover_previous_config_after_calculation=False)

        visitations, expert_visitations, prev_pi, loss = self.mce_vsl_loss_calculation(target_align_func, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, predicted_r, predicted_r_np, obs_mat=obs_mat, action_mat=action_mat,
                                                                                       obs_action_mat=obs_action_mat,)
        self.vsi_optimizer.step()

        learned_al_function = self.current_net.get_learned_align_function()
        _, new_reward = self.calculate_rewards(learned_al_function,
                                               obs_mat=obs_mat,
                                               action_mat=action_mat,
                                               obs_action_mat=obs_action_mat,
                                               next_state_obs_mat=next_obs_mat,
                                               reward_mode=TrainingModes.EVAL, requires_grad=False)

        return predicted_r_np, visitations, expert_visitations, prev_pi, loss, new_reward, learned_al_function

    def _train_step_vgl(self, obs_mat: th.Tensor, align_func: Any, action_mat: th.Tensor = None, obs_action_mat: th.Tensor = None, n_seeds_for_sampled_trajectories=30, n_sampled_trajs_per_seed=3) -> Tuple[np.ndarray, np.ndarray]:
        assert self.current_net.mode == TrainingModes.VALUE_GROUNDING_LEARNING
        self.vgl_optimizer.zero_grad()
        self.current_net.set_alignment_function(align_func)

        predicted_r, predicted_r_np = self.calculate_rewards(align_func=align_func, obs_mat=obs_mat, action_mat=action_mat,
                                                             obs_action_mat=obs_action_mat,
                                                             reward_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                                                             recover_previous_config_after_calculation=False)
        assert self.current_net.mode == TrainingModes.VALUE_GROUNDING_LEARNING
        visitations, expert_visitations, prev_pi, loss = self.mce_vsl_loss_calculation(align_func, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, predicted_r, predicted_r_np,
                                                                                       obs_mat=obs_mat, action_mat=action_mat,
                                                                                       obs_action_mat=obs_action_mat,)
        self.vgl_optimizer.step()

        _, new_reward = self.calculate_rewards(align_func,
                                               obs_mat=obs_mat,
                                               action_mat=action_mat,
                                               obs_action_mat=obs_action_mat,
                                               reward_mode=TrainingModes.EVAL)

        return predicted_r_np, visitations, expert_visitations, prev_pi, loss, new_reward

    def _set_vgl_demo_oms_from_policy(self, policy: VAlignedDiscreteSpaceActionPolicy) -> None:
        pass

    def _set_vsi_demo_oms_from_policy(self, policy: VAlignedDiscreteSpaceActionPolicy, ) -> None:
        pass

    def set_demonstrations(self, demonstrations: Union[Iterable[types.Trajectory], Iterable[types.TransitionMapping], types.TransitionsMinimal]) -> None:
        pass


def check_coherent_rewards(max_entropy_algo: MaxEntropyIRLForVSL, align_funcs_to_test=[], real_grounding=th.nn.Identity(), policy_approx_method=PolicyApproximators.MCE_ORIGINAL, stochastic_expert=False, stochastic_learner=False):
    previous_mode = max_entropy_algo.reward_net.mode
    previous_al = max_entropy_algo.reward_net.cur_align_func

    max_entropy_algo.reward_net.set_mode(TrainingModes.EVAL)
    max_entropy_algo.reward_net.set_grounding_function(real_grounding)
    env_real = max_entropy_algo.env

    for al in align_funcs_to_test:
        max_entropy_algo.reward_net.set_alignment_function(al)

        # print(max_entropy_algo.reward_net(None, None, th.tensor([[0.1,0.2,0.4]]), None))
        # assert max_entropy_algo.reward_net(None, None, th.tensor([[0.1,0.2,0.4]]),None) == th.tensor([-0.1])
        # assert max_entropy_algo.reward_net(None, None, th.tensor([[0.8998,0.2,0.4]]),None) == th.tensor([-0.8998])
        torch_action_mat = None
        obs_mat = th.tensor(max_entropy_algo.env.observation_matrix,
                            dtype=max_entropy_algo.reward_net.dtype, requires_grad=False)

        if max_entropy_algo.reward_net.use_action:
            actions_one_hot = th.eye(
                max_entropy_algo.env.action_space.n, requires_grad=True)
            torch_action_mat = th.stack([actions_one_hot[i].repeat(
                obs_mat.shape[0], 1) for i in range(max_entropy_algo.env.action_space.n)], dim=0)

        torch_obs_action_mat = None
        if max_entropy_algo.reward_net.use_one_hot_state_action:
            torch_obs_action_mat = th.as_tensor(
                np.identity(max_entropy_algo.env.state_dim *
                            max_entropy_algo.env.action_dim),
                dtype=max_entropy_algo.reward_net.dtype,
                device=max_entropy_algo.reward_net.device,
            )

        reward = env_real.reward_matrix_per_align_func(al)

        _, _, assumed_expert_pi = mce_partition_fh(env_real, discount=max_entropy_algo.discount,
                                                   reward=reward,
                                                   approximator_kwargs={
                                                       'value_iteration_tolerance': 0.0000001, 'iterations': 100},
                                                   policy_approximator=policy_approx_method, deterministic=not stochastic_expert)

        rews = reward

        _, rews = max_entropy_algo.calculate_rewards(al, obs_mat=obs_mat,
                                                     action_mat=torch_action_mat,
                                                     obs_action_mat=torch_obs_action_mat, requires_grad=False)

        _, _, assumed_pi = mce_partition_fh(env_real, discount=max_entropy_algo.discount,
                                            reward=rews,
                                            approximator_kwargs={
                                                'value_iteration_tolerance': 0.0000001, 'iterations': 100},
                                            policy_approximator=policy_approx_method, deterministic=not stochastic_learner)

        if __debug__:
            print(al)
            print(reward.shape, rews.shape)
        assert rews.shape == reward.shape
        if __debug__:
            print(np.where(reward != rews))
            print(rews[np.where(reward != rews)])
            print(reward[np.where(reward != rews)])

        assert np.allclose(rews, reward)

        try:
            getattr(max_entropy_algo, 'mce_occupancy_measures')
            if __debug__:
                _, visits, pii = max_entropy_algo.mce_occupancy_measures(
                    rews, deterministic=not stochastic_learner, use_action_visitations=True)

                _, visits_ok, piii = max_entropy_algo.mce_occupancy_measures(
                    reward, assumed_expert_pi, deterministic=not stochastic_expert, use_action_visitations=True)

                print(np.where(visits_ok != visits))
                print(visits[np.where(visits != visits_ok)])
                print(visits_ok[np.where(visits != visits_ok)])
                assert np.allclose(visits, visits_ok)
        except:
            pass

    max_entropy_algo.reward_net.set_mode(previous_mode)
    max_entropy_algo.reward_net.set_alignment_function(previous_al)


def transform_state_only_reward_to_state_action_reward(possibly_state_only_reward, state_dim, action_dim, transition_matrix=None):
    if possibly_state_only_reward.shape == (state_dim,):
        rewards_per_state_action = np.zeros((state_dim, action_dim))
        for s in range(state_dim):
            for a in range(action_dim):
                if transition_matrix is not None:
                    ns_probs = transition_matrix[s, a, :]
                    rewards_per_state_action[s, a] = np.dot(
                        ns_probs, possibly_state_only_reward)
                else:
                    rewards_per_state_action[s,
                                             a] = possibly_state_only_reward[s]
    elif possibly_state_only_reward.shape == (state_dim, action_dim):
        return possibly_state_only_reward
    else:
        raise ValueError("Unexpected reward form: %s" %
                         possibly_state_only_reward)