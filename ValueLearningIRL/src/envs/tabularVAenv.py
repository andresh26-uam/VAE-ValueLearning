
from abc import abstractmethod, abstractproperty
from typing import Any, Callable
from numpy import ndarray
from seals import base_envs
import numpy as np
from seals.base_envs import DiscreteSpaceInt
class TabularVAPOMDP(base_envs.TabularModelPOMDP):
    def __init__(self, transition_matrix: ndarray, observation_matrix: ndarray, reward_matrix_per_va: Callable[[Any], ndarray], reward_matrix_per_va_complete: Callable[[Any], ndarray], default_reward_matrix: ndarray, horizon: int = None, initial_state_dist: ndarray = None):
        

        super().__init__(transition_matrix=transition_matrix, observation_matrix=observation_matrix, reward_matrix=default_reward_matrix, horizon=horizon, initial_state_dist=initial_state_dist)
        self.reward_matrix_per_va = reward_matrix_per_va
        self.reward_matrix_per_va_complete = reward_matrix_per_va_complete
        

    @abstractmethod   
    def obs_from_state(self, state: np.int64) -> ndarray[Any, np.dtype]:
        pass
    
    @property
    @abstractmethod
    def goal_states(self):
        return []