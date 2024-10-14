
from abc import abstractmethod, abstractproperty
from typing import Any, Callable, Iterator, SupportsFloat
from typing_extensions import override
from numpy import ndarray
from seals import base_envs
import numpy as np
from seals.base_envs import DiscreteSpaceInt
from itertools import cycle

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

    def __init__(self, env: gym.Env, horizon: int = None, done_when_horizon_is_met: bool = False,  trunc_when_horizon_is_met: bool = True, **kwargs):
        super().__init__(env)
        self.horizon=horizon
        self.done_when_horizon_is_met = done_when_horizon_is_met
        self.trunc_when_horizon_is_met = trunc_when_horizon_is_met
        self.cur_align_func = None
        #self.action_space = self.env.action_space
        #self.observation_space = self.observation_space
        
        #self.metadata = self.env.metadata
        #self.reward_range = self.env.reward_range
        
        #self.spec = self.env.spec
    

    def align_func_yielder(self, a, ns=None, prev_align_func=None, info=None):
        return self.cur_align_func
    
    @abstractmethod
    def step_reward_per_va(self, align_func, action) -> SupportsFloat:
        ...

    def reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        s,info= self._reset(seed=seed, options=options)
        self.time=0
        info['align_func'] = self.cur_align_func
        return s, info
    

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        ns, original_rew, done, trunc, info = self._step(action)
        self.cur_align_func = self.align_func_yielder(action, ns=ns, prev_align_func=self.cur_align_func, info=info)
        info['align_func'] = self.cur_align_func
        r = self.step_reward_per_va(self.cur_align_func, action)
        self.time+=1
        if self.horizon is not None and self.time >= self.horizon:
            trunc = self.trunc_when_horizon_is_met or trunc
            done = self.done_when_horizon_is_met or done
        return ns, r, done, trunc, info
    
    def _reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        return self.unwrapped.reset(seed=seed, options=options)

    def _step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.unwrapped.step(action)
        
    
    
class TabularVAMDP(ValueAlignedEnvironment,base_envs.TabularModelPOMDP):
     
    def __init__(self, transition_matrix: ndarray, observation_matrix: ndarray, 
                 reward_matrix_per_va: Callable[[Any], ndarray], 
                 default_reward_matrix: ndarray, initial_state_dist: ndarray = None,
                 horizon: int = None, done_when_horizon_is_met: bool = False,  trunc_when_horizon_is_met: bool = True, **kwargs):
        self._real_env = base_envs.TabularModelPOMDP(transition_matrix=transition_matrix, observation_matrix=observation_matrix, 
                         reward_matrix=default_reward_matrix, horizon=horizon, initial_state_dist=initial_state_dist)
        
        self.reward_matrix_per_align_func = reward_matrix_per_va
        self.done_when_horizon_is_met=done_when_horizon_is_met
        self.trunc_when_horizon_is_met=trunc_when_horizon_is_met


        super().__init__(self._real_env, horizon=horizon, done_when_horizon_is_met=done_when_horizon_is_met, trunc_when_horizon_is_met=trunc_when_horizon_is_met)
        self.env: base_envs.TabularModelPOMDP
        self.transition_matrix = self.unwrapped.transition_matrix
    @property
    def unwrapped(self):
        return self._real_env

    def step_reward_per_va(self, align_func, action):
        
        return self.reward_matrix_per_align_func(align_func)[self.state, action]
    
    
    def get_state_actions_with_known_reward(self, align_func):
        return None

    @property
    @abstractmethod
    def goal_states(self):
        ...
    
    def _reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        return self.unwrapped.reset(seed=seed, options=options)

    def _step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        
        ns,r,d,t,i = self.unwrapped.step(action)
        #d = d or self.env.state in self.goal_states
        return ns,r,d,t,i
    
    @override
    def obs_from_state(self, state: np.int64) -> np.ndarray[Any, np.dtype]:
        return self.unwrapped.observation_matrix[state]

    