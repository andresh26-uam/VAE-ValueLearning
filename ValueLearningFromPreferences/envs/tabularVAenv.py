
from abc import abstractmethod
from typing import Any, Callable, SupportsFloat, Union
import torch
from typing_extensions import override
from numpy import ndarray
from seals import base_envs
import numpy as np

import gymnasium as gym


def encrypt_state(state, original_state_space):

    new_number = 0
    total_states = 1

    for i in range(original_state_space.shape[0]):
        new_number += state[i]*total_states
        total_states *= original_state_space[i].n

    return int(new_number)


def translate_state(state, original_state_space):

    new_state = np.zeros(original_state_space.shape[0], dtype=np.int64)

    for i in range(len(new_state)):
        new_modulo = original_state_space[i].n
        new_state[i] = state % new_modulo

        state -= new_state[i]
        state /= new_modulo

    return new_state


class ValueAlignedEnvironment(gym.Wrapper):

    def __init__(self, env: gym.Env, n_values: int, horizon: int = None, done_when_horizon_is_met: bool = False,  trunc_when_horizon_is_met: bool = True, **kwargs):
        super().__init__(env)
        self.horizon = horizon
        self.done_when_horizon_is_met = done_when_horizon_is_met
        self.trunc_when_horizon_is_met = trunc_when_horizon_is_met
        self._cur_align_func = None
        self.current_assumed_grounding = None
        self.n_values = n_values
        self.basic_profiles = [tuple(t) for t in np.eye(self.n_values)]
        # self.action_space = self.env.action_space
        # self.observation_space = self.observation_space

        # self.metadata = self.env.metadata
        # self.reward_range = self.env.reward_range

        # self.spec = self.env.spec

    def align_func_yielder(self, a, ns=None, prev_align_func=None, info=None):
        return self._cur_align_func


    def get_reward_per_value(self, vindex, obs=None, action=None, next_obs=None, info=None, custom_grounding=None) -> SupportsFloat:
        return self.get_reward_per_align_func(align_func=self.basic_profiles[vindex], obs=obs, action=action, next_obs=next_obs, info=info, custom_grounding=custom_grounding)
    @abstractmethod
    def get_reward_per_align_func(self, align_func, obs=None, action=None, next_obs=None, info=None, custom_grounding=None) -> SupportsFloat:
        ...

    def set_align_func(self, align_func):
        self._cur_align_func = align_func

    def get_align_func(self):
        return self._cur_align_func

    def reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:

        s, info = self._reset(seed=seed, options=options)
        info['align_func'] = self.get_align_func()
        self.prev_observation, self.prev_info = s, info
        self.time = 0
        return s, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        ns, original_rew, done, trunc, info = self._step(action)
        self.set_align_func(self.align_func_yielder(
            action, ns=ns, prev_align_func=self.get_align_func(), info=self.prev_info))
        info['align_func'] = self.get_align_func()

        r = self.get_reward_per_align_func(
            self.get_align_func(), self.prev_observation, action, ns, info, custom_grounding=self.assumed_grounding)
        self.time += 1
        if self.horizon is not None and self.time >= self.horizon:
            trunc = self.trunc_when_horizon_is_met or trunc
            done = self.done_when_horizon_is_met or done
        self.prev_observation = ns
        self.prev_info = info
        return ns, r, done, trunc, info

    def _reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        return self.unwrapped.reset(seed=seed, options=options)

    def _step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.unwrapped.step(action)
    
    @abstractmethod
    def calculate_assumed_grounding(self, **kwargs) -> tuple[Union[torch.nn.Module, np.ndarray], np.ndarray]:
        ...

    @property
    def assumed_grounding(self):
        return self.current_assumed_grounding
    
class TabularVAMDP(ValueAlignedEnvironment, base_envs.TabularModelPOMDP):

    def __init__(self, n_values: int, transition_matrix: ndarray, observation_matrix: ndarray,
                 reward_matrix_per_va: Callable[[Any], ndarray],
                 default_reward_matrix: ndarray, initial_state_dist: ndarray = None,
                 horizon: int = None, done_when_horizon_is_met: bool = False,  trunc_when_horizon_is_met: bool = True, **kwargs):
        self._real_env = base_envs.TabularModelPOMDP(transition_matrix=transition_matrix, observation_matrix=observation_matrix,
                                                     reward_matrix=default_reward_matrix, horizon=horizon, initial_state_dist=initial_state_dist)

        self.reward_matrix_per_align_func = reward_matrix_per_va
        self.done_when_horizon_is_met = done_when_horizon_is_met
        self.trunc_when_horizon_is_met = trunc_when_horizon_is_met

        super().__init__(self._real_env, n_values=n_values, horizon=horizon, done_when_horizon_is_met=done_when_horizon_is_met,
                         trunc_when_horizon_is_met=trunc_when_horizon_is_met)
        self.env: base_envs.TabularModelPOMDP
        self.transition_matrix = self.unwrapped.transition_matrix
        #self.current_assumed_grounding = self.calculate_assumed_grounding()

    @property
    def unwrapped(self):
        return self._real_env

    def valid_actions(self, state, align_func=None):
        return np.arange(self.reward_matrix.shape[1])

    def get_reward_per_align_func(self, align_func, obs=None, action=None, next_obs=None, info=None, custom_grounding=None):
        return self.reward_matrix_per_align_func(align_func, custom_grounding=custom_grounding)[info['state'], action]

    def get_state_actions_with_known_reward(self, align_func):
        return None

    def set_initial_state_distribution(self, dist):
        self.unwrapped.initial_state_dist = dist
        self.initial_state_dist = dist

    @property
    @abstractmethod
    def goal_states(self):
        ...

    def _reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        s, i = self.unwrapped.reset(seed=seed, options=options)
        i['state'] = self.state
        i['next_state'] = self.state
        return s, i

    def _step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        prev_state = self.state
        ns, r, d, t, i = self.unwrapped.step(action)
        # d = d or self.env.state in self.goal_states
        i['state'] = prev_state
        i['next_state'] = self.state
        # d = True if d is True else self.state in self.goal_states
        return ns, r, d, t, i

    @property
    def invalid_states(self):
        return None

    @override
    def obs_from_state(self, state: np.int64) -> np.ndarray[Any, np.dtype]:
        return self.unwrapped.observation_matrix[state]

    