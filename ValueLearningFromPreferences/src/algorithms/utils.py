import enum
import random
from typing import Callable, Iterable, Optional, Tuple, Union
from gymnasium import Env
import numpy as np
import scipy
from seals import base_envs
from imitation.data import types, rollout


from colorama import Fore, Style, init

def convert_nested_list_to_tuple(nested_list):
    if isinstance(nested_list, list):
        return tuple(convert_nested_list_to_tuple(item) for item in nested_list)
    return nested_list



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
    
