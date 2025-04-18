import enum
from typing import Callable, Dict, Optional, Tuple
import scipy.linalg
import scipy.special

from gymnasium import Env
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


from imitation.data import types


class PolicyApproximators(enum.Enum):
    MCE_ORIGINAL = 'mce_original'
    SOFT_VALUE_ITERATION = 'value_iteration'
    NEW_SOFT_VALUE_ITERATION = 'new_value_iteration'


def concentrate_on_max_policy(pi, distribute_probability_on_max_prob_actions=False, valid_action_checker=None):
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
    n_states, n_actions = pi.shape
    pi_new = np.zeros_like(pi)

    for state in range(n_states):
        # Get the valid actions for this state, or all actions if no valid_action_checker is provided
        if valid_action_checker is not None:
            valid_actions = valid_action_checker(state)
        else:
            valid_actions = np.arange(n_actions)

        # Extract the probabilities for valid actions only
        valid_probs = pi[state, valid_actions]

        if distribute_probability_on_max_prob_actions:
            # Find the maximum probability among valid actions
            max_value = np.max(valid_probs)

            # Determine which valid actions have the maximum probability
            max_valid_actions = valid_actions[valid_probs == max_value]

            # Distribute the probability equally among the max valid actions
            pi_new[state, max_valid_actions] = 1 / len(max_valid_actions)
        else:
            # Concentrate on a single max probability valid action
            max_valid_action = valid_actions[np.argmax(valid_probs)]
            pi_new[state, max_valid_action] = 1

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
            # values_prev[env.goal_states] = 0.0
            next_values_s_a = T @ values_prev
            Q = broad_R + discount * next_values_s_a
            V = np.max(Q, axis=1)
            err = np.max(np.abs(V-values_prev))
            iterations += 1
        pi = scipy.special.softmax(Q - V[:, None], axis=1)
    elif policy_approximator == PolicyApproximators.NEW_SOFT_VALUE_ITERATION:
        value_iteration_tolerance = np.min(np.abs(broad_R[broad_R != 0.0]))

        # Initialization
        # indexed as V[t,s]
        V = np.full((n_states), -1)
        V[env.goal_states] = 0.0
        Q = broad_R
        # we assume the MDP is correct, but illegal state should respond to this rule. By design broad_R should be very negative in illegal states.
        illegal_reward = np.max(np.abs(broad_R))*-1.0

        inv_states = env.invalid_states
        if inv_states is not None:
            V[inv_states] = illegal_reward
            # Q[inv_states,:] = V[inv_states]
        # indexed as Q[t,s,a]

        valid_state_actions = np.full_like(
            broad_R, fill_value=False, dtype=np.bool_)
        for s in range(len(V)):
            valid_state_actions[s, env.valid_actions(s, None)] = True

        broad_R[valid_state_actions == False] = illegal_reward
        err = np.inf
        iterations = 0
        # value_iteration_tolerance = approximator_kwargs['value_iteration_tolerance']
        max_iterations = horizon
        if 'iterations' in approximator_kwargs.keys():
            max_iterations = approximator_kwargs['iterations']

        while err >= value_iteration_tolerance and iterations < max_iterations:
            values_prev = V.copy()
            next_values_s_a = T @ values_prev
            Q = broad_R + discount * next_values_s_a

            # Q[valid_state_actions==False] = illegal_reward
            V = np.max(Q, axis=1)
            V[env.goal_states] = 0.0
            if inv_states is not None:

                V[inv_states] = values_prev[inv_states]
            err = np.max(np.abs(V-values_prev))
            if iterations > 0.9*max_iterations:
                print("VALUE ITERATION IS HARD HERE", err,
                      iterations, np.mean(np.abs(V-values_prev)))

            iterations += 1
        pi = scipy.special.softmax(Q - V[:, None], axis=1)
    else:
        V, Q, pi = policy_approximator(
            env, reward, discount, **approximator_kwargs)

    if deterministic:

        if len(pi.shape) == 2:
            pi = concentrate_on_max_policy(
                pi, valid_action_checker=lambda s: env.valid_actions(s, None))
            if __debug__:
                for i in range(pi.shape[0]):
                    assert np.allclose(np.sum(pi[i]), 1)
        else:
            for time_t in range(pi.shape[0]):
                pi[time_t] = concentrate_on_max_policy(
                    pi[time_t], valid_action_checker=lambda s: env.valid_actions(s, None))
                if __debug__:
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
        approximator_kwargs={},
        policy_approximator=PolicyApproximators.MCE_ORIGINAL,
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,

    ) -> None:
        super().__init__(env=env, reward_net=reward_net, vgl_optimizer_cls=vgl_optimizer_cls,
                         stochastic_expert=stochastic_expert, environment_is_stochastic=environment_is_stochastic,
                         vsi_optimizer_cls=vsi_optimizer_cls, vsi_optimizer_kwargs=vsi_optimizer_kwargs,
                         vgl_optimizer_kwargs=vgl_optimizer_kwargs, discount=discount, log_interval=log_interval, vgl_expert_policy=vgl_expert_policy, vsi_expert_policy=vsi_expert_policy,
                         target_align_func_sampler=target_align_func_sampler, vsi_target_align_funcs=vsi_target_align_funcs,
                         vgl_target_align_funcs=vgl_target_align_funcs, training_mode=training_mode, custom_logger=custom_logger, learn_stochastic_policy=learn_stochastic_policy)
        self.rewards_per_target_align_func_callable = None
        self.approximator_kwargs = approximator_kwargs
        self.policy_approximator = policy_approximator

    def get_metrics(self):
        return dict(learned_rewards=self.rewards_per_target_align_func_callable)

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

        if obs_mat.shape[0] == self.torch_obs_mat.shape[0]:
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

    def train(self, max_iter: int = 1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None,
              n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_probabilistic_reward=False,
              n_reward_reps_if_probabilistic_reward=10, **kwargs) -> np.ndarray:
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
                             use_probabilistic_reward=use_probabilistic_reward, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed,
                             n_reward_reps_if_probabilistic_reward=n_reward_reps_if_probabilistic_reward, **kwargs)

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

    def get_tabular_policy_from_reward_per_align_func(self, align_funcs, reward_net_per_al: Dict[tuple, AbstractVSLRewardFunction], expert=False, random=False, use_custom_grounding=None,
                                                      target_to_learned=None, use_probabilistic_reward=False, n_reps_if_probabilistic_reward=10,
                                                      state_encoder=None, expose_state=True, precise_deterministic=False):
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
                        assumed_grounding = reward_net_per_al[w].get_learned_grounding(
                        )

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
                        _, reward = ret
                    else:
                        raise NotImplementedError(
                            "Probabilistic reward is yet to be tested")

                if precise_deterministic and expert:
                    assumed_expert_pi = np.load(
                        f'roadworld_env_use_case/expert_policy_{w}.npy')

                else:
                    _, _, assumed_expert_pi = mce_partition_fh(self.env, discount=self.discount,
                                                               reward=reward,
                                                               approximator_kwargs=self.approximator_kwargs,
                                                               policy_approximator=self.policy_approximator,
                                                               deterministic=deterministic)
                profile_to_assumed_matrix[w] = assumed_expert_pi

                reward_matrix_per_al[w] = reward

            self.current_net = prev_net
        policy = VAlignedDictSpaceActionPolicy(
            policy_per_va_dict=profile_to_assumed_matrix, env=self.env, state_encoder=state_encoder, expose_state=expose_state)
        return policy, reward_matrix_per_al
