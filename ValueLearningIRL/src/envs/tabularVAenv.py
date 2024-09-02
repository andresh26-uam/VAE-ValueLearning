
from abc import abstractmethod, abstractproperty
from typing import Any, Callable, Iterator, SupportsFloat
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

class ValueAlignedEnvironment(gym.Env):

    def __init__(self, env: gym.Env, horizon: int = None, done_when_horizon_is_met: bool = False,  trunc_when_horizon_is_met: bool = True, **kwargs):
        super().__init__()
        self.horizon=horizon
        self.done_when_horizon_is_met = done_when_horizon_is_met
        self.trunc_when_horizon_is_met = trunc_when_horizon_is_met
        self.env=env
        self.action_space = self.env.action_space
        self.observation_space = self.observation_space
        
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        
        self.spec = self.env.spec

    @property
    def unwrapped(self):
        return self.env
    
    def set_render_mode(self, render_mode):
        self.env.render_mode = render_mode
        self.render_mode = self.env.render_mode
        
    def render(self):
        self.env.render()

    def align_func_yielder(self, a, ns=None, prev_align_func=None, info=None):
        return self.cur_align_func
    
    @abstractmethod
    def step_reward_per_va(self, align_func, action):
        ...

    def reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        r,info= self.env.reset(seed=seed, options=options)
        self.time=0
        info['align_func'] = self.cur_align_func

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        ns, original_rew, done, trunc, info = self.env.step(action)
        self.cur_align_func = self.align_func_yielder(action, ns=ns, prev_align_func=self.cur_align_func, info=info)
        info['align_func'] = self.cur_align_func
        r = self.step_reward_per_va(self.cur_align_func, a)
        self.time+=1
        if self.horizon is not None and self.time >= self.horizon:
            trunc = self.trunc_when_horizon_is_met or trunc
            done = self.done_when_horizon_is_met or done
        return ns, r, done, trunc, info
        
    
class TabularVAPOMDP(base_envs.TabularModelPOMDP,ValueAlignedEnvironment):
     
    def __init__(self, transition_matrix: ndarray, observation_matrix: ndarray, 
                 reward_matrix_per_va: Callable[[Any], ndarray], 
                 default_reward_matrix: ndarray, initial_state_dist: ndarray = None,
                 horizon: int = None, done_when_horizon_is_met: bool = False,  trunc_when_horizon_is_met: bool = True, **kwargs):

        super().__init__(transition_matrix=transition_matrix, observation_matrix=observation_matrix, 
                         reward_matrix=default_reward_matrix, horizon=horizon, initial_state_dist=initial_state_dist)
        
        self.reward_matrix_per_va = reward_matrix_per_va
        ValueAlignedEnvironment.__init__(self, env=self, horizon=horizon, done_when_horizon_is_met=done_when_horizon_is_met, trunc_when_horizon_is_met=trunc_when_horizon_is_met)
    
    def step_reward_per_va(self, align_func, action):
        return self.reward_matrix_per_va(align_func)[self._cur_state, action]
    
    
    @abstractmethod   
    def obs_from_state(self, state: np.int64) -> ndarray[Any, np.dtype]:
        pass
    
    @property
    @abstractmethod
    def goal_states(self):
        return []
        


    