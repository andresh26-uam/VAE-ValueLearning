import enum
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from seals.base_envs import DiscreteSpaceInt
from firefighters_use_case.env import HighRiseFireEnv
from firefighters_use_case.constants import ACTION_AGGRESSIVE_FIRE_SUPPRESSION


from gymnasium import spaces

from src.envs.tabularVAenv import TabularVAPOMDP

class FeatureSelection(enum.Enum):

    ONE_HOT_OBSERVATIONS = 'observations_one_hot'
    ONE_HOT_FEATURES = 'features_one_hot'
    ORIGINAL_OBSERVATIONS = 'observations'
    ENCRYPTED_OBSERVATIONS = 'encrypted_observations'
    DEFAULT = None

class FireFightersEnv(TabularVAPOMDP):
    """
    A simplified two-objective MDP environment for an urban high-rise fire scenario.
    Objectives: Professionalism and Proximity
    """
    metadata = {'render.modes': ['human']}

    def render(self):
        return self.real_env.render()
    def __init__(self, feature_selection=FeatureSelection.ONE_HOT_FEATURES, horizon=100):
        self.real_env = HighRiseFireEnv()
        
        if feature_selection == FeatureSelection.ORIGINAL_OBSERVATIONS:
            self.observation_space = self.real_env.state_space
        elif feature_selection == FeatureSelection.ONE_HOT_OBSERVATIONS:
            self.observation_space = spaces.MultiBinary(n=self.real_env.n_states)    
        elif feature_selection == FeatureSelection.ONE_HOT_FEATURES:
            self.observation_space = spaces.MultiBinary(n=sum([nv for nv in self.real_env.state_space.nvec]))
        elif feature_selection == FeatureSelection.ENCRYPTED_OBSERVATIONS:
            self.observation_space = spaces.Discrete(self.real_env.n_states)
        else:
            self.observation_space = self.real_env.state_space
        # No, states as indexes, observations in one-hot encoding.

        self.state_space = spaces.Discrete(self.real_env.n_states)
        self.action_space = self.real_env.action_space
        #self.action_dim = self.real_env.action_space.n
        #         self.state_dim = self.real_env.n_states
        self.n_states = self.real_env.n_states

        transition_matrix = np.zeros((self.n_states, self.action_space.n, self.n_states))
        reward_matrix_per_va = dict()
        reward_matrix_per_va_complete = dict()


        reward_matrix_per_va[(1.0, 0.0)] = np.zeros((self.n_states, self.action_space.n))
        reward_matrix_per_va[(0.0, 1.0)] = np.zeros((self.n_states, self.action_space.n))

        reward_matrix_per_va_complete[(1.0, 0.0)] = np.zeros((self.n_states, self.action_space.n, self.n_states))
        reward_matrix_per_va_complete[(0.0, 1.0)] = np.zeros((self.n_states, self.action_space.n, self.n_states))

        _goal_states = list()

        observation_matrix = np.zeros((self.n_states, *self.observation_space.shape))
        for s in range(self.n_states):
            s_trans = self.real_env.translate(s)
            if feature_selection == FeatureSelection.ORIGINAL_OBSERVATIONS:
                    observation_matrix[s,:] = s_trans 
            elif feature_selection == FeatureSelection.ONE_HOT_OBSERVATIONS:
                    observation_matrix[s,:] = np.eye(self.real_env.n_states)[s]
            elif feature_selection == FeatureSelection.ENCRYPTED_OBSERVATIONS:
                observation_matrix[s,:] = s
            elif feature_selection == FeatureSelection.ONE_HOT_FEATURES:
                vec = np.concatenate([np.eye(self.real_env.state_space.nvec[i])[s_trans[i]] for i in range(self.real_env.state_space.nvec.shape[0])], -1)
                observation_matrix[s,:] = vec
            if not self.real_env.is_done(s_trans):
                for a in range(self.action_space.n):
                    ns_trans = self.real_env.transition(s_trans,a)
                    ns = self.real_env.encrypt(ns_trans)
                    transition_matrix[s,a,ns] = 1.0
                    reward_matrix_per_va_complete[(1.0, 0.0)][s,a,ns], reward_matrix_per_va_complete[(0.0, 1.0)][s,a,ns] = self.real_env.calculate_rewards(s_trans, a, ns_trans)
                    reward_matrix_per_va[(1.0, 0.0)][s,a], reward_matrix_per_va[(0.0, 1.0)][s,a] = reward_matrix_per_va_complete[(1.0, 0.0)][s,a,ns], reward_matrix_per_va_complete[(0.0, 1.0)][s,a,ns]
            else:
                _goal_states.append(s_trans)
                for a in range(self.action_space.n):  
                    ns_trans = self.real_env.transition(s_trans,a)    
                                 
                    ns = self.real_env.encrypt(ns_trans)
                    reward_matrix_per_va_complete[(1.0, 0.0)][s,a,ns], reward_matrix_per_va_complete[(0.0, 1.0)][s,a,ns] = self.real_env.calculate_rewards(s_trans, a, ns_trans)
                    reward_matrix_per_va[(1.0, 0.0)][s,a], reward_matrix_per_va[(0.0, 1.0)][s,a] = reward_matrix_per_va_complete[(1.0, 0.0)][s,a,ns], reward_matrix_per_va_complete[(0.0, 1.0)][s,a,ns]

                    transition_matrix[s,a,ns] = 0.0
                    

        initial_state_dist = np.zeros(self.n_states)
        initial_state_dist[self.real_env.encrypt(np.array([0, 3, 4, 0, 0, 3]))] = 1.0
        #initial_state_dist = np.ones(self.n_states)/self.n_states
        self._cur_state = self.real_env.state
        self._goal_states = np.asarray(_goal_states)

        
        super(FireFightersEnv, self).__init__(transition_matrix=transition_matrix, observation_matrix=observation_matrix, 
                                              reward_matrix_per_va=lambda va: reward_matrix_per_va[va], 
                                              reward_matrix_per_va_complete=lambda va: reward_matrix_per_va_complete[va], 
                                              default_reward_matrix=reward_matrix_per_va[(0.0, 1.0)], horizon=horizon, initial_state_dist=initial_state_dist)
    @property
    def state(self) -> np.dtype:
        """Data type of observation vectors (e.g. np.float32)."""
        return self._cur_state
    @property
    def observation(self) -> np.dtype:
        """Data type of observation vectors (e.g. np.float32)."""
        return self.real_env.state
    
    def obs_from_state(self, state: np.int64) -> np.ndarray[Any, np.dtype]:
        return self.real_env.translate(state)
    
    def step(self, action):
        # Simulate state transitions and compute rewards
        
        next_obs = self.real_env.transition(self.real_env.state, action)
        rewards = self.real_env.calculate_rewards(self.real_env.state, action, next_obs)
        self.real_env.state = next_obs

        # Check if the episode is done
        done = self.real_env.is_done(self.real_env.state)
        self._cur_state = self.real_env.encrypt(self.real_env.state)
        # In case
        info = {'state': self._cur_state, 'obs': next_obs}
        return next_obs, rewards, done, done, info
    @property
    def goal_states(self):
        return self._goal_states
    
    def reset(self, seed=None, options=None, force_new_state=None):
        s = self.real_env.reset(force_new_state=force_new_state)
        self._cur_state = self.real_env.encrypt(s)
        self.real_env.state = s
        i = {'state': self._cur_state, 'obs': s}
        return s, {'real_state': self._cur_state}
    

