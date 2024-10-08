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

class FeatureSelectionFFEnv(enum.Enum):

    
    ONE_HOT_FEATURES = 'features_one_hot' # Each feature is one-hot encoded
    ORIGINAL_OBSERVATIONS = 'observations' # Use the original observations of the form [0,1,4,3,0]
    ENCRYPTED_OBSERVATIONS = 'encrypted_observations' # Use State unique identifier given by its encryption, e.g. 320
    ONE_HOT_OBSERVATIONS = 'observations_one_hot' # Use State encrption, but one hot encoded
    ORDINAL_AND_ONE_HOT_FEATURES = 'ordinal_and_one_hot' # TODO: Ordinal Features may remain as 0, 1, 2 but categorical ones are one hot encoded. 
    DEFAULT = None

class FireFightersEnv(TabularVAPOMDP):
    """
    A simplified two-objective MDP environment for an urban high-rise fire scenario.
    Objectives: Professionalism and Proximity
    """
    metadata = {'render.modes': ['human']}

    def render(self):
        return self.real_env.render()
    def __init__(self, feature_selection=FeatureSelectionFFEnv.ONE_HOT_FEATURES, horizon=100, initial_state_distribution='uniform'):
        self.real_env = HighRiseFireEnv()
        
        if feature_selection == FeatureSelectionFFEnv.ORIGINAL_OBSERVATIONS:
            self.observation_space = self.real_env.state_space
        elif feature_selection == FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS:
            self.observation_space = spaces.MultiBinary(n=self.real_env.n_states)    
        elif feature_selection == FeatureSelectionFFEnv.ONE_HOT_FEATURES:
            self.observation_space = spaces.MultiBinary(n=sum([nv for nv in self.real_env.state_space.nvec]))
        elif feature_selection == FeatureSelectionFFEnv.ENCRYPTED_OBSERVATIONS:
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


        reward_matrix_per_va[(1.0, 0.0)] = np.zeros((self.n_states, self.action_space.n))
        reward_matrix_per_va[(0.0, 1.0)] = np.zeros((self.n_states, self.action_space.n))

        _goal_states = list()

        observation_matrix = np.zeros((self.n_states, *self.observation_space.shape))

        self._states_with_known_reward = np.zeros((self.n_states,self.action_space.n), dtype=np.bool_) 
        for s in range(self.n_states):
            s_trans = self.real_env.translate(s)
            if feature_selection == FeatureSelectionFFEnv.ORIGINAL_OBSERVATIONS:
                    observation_matrix[s,:] = s_trans 
            elif feature_selection == FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS:
                    observation_matrix[s,:] = np.eye(self.real_env.n_states)[s]
            elif feature_selection == FeatureSelectionFFEnv.ENCRYPTED_OBSERVATIONS:
                observation_matrix[s,:] = s
            elif feature_selection == FeatureSelectionFFEnv.ONE_HOT_FEATURES:
                vec = np.concatenate([np.eye(self.real_env.state_space.nvec[i])[s_trans[i]] for i in range(self.real_env.state_space.nvec.shape[0])], -1)
                observation_matrix[s,:] = vec
            if not self.real_env.is_done(s_trans):
                for a in range(self.action_space.n):
                    ns_trans = self.real_env.transition(s_trans,a)
                    ns = self.real_env.encrypt(ns_trans)
                    transition_matrix[s,a,ns] = 1.0
                    
                    reward_matrix_per_va[(1.0, 0.0)][s,a], reward_matrix_per_va[(0.0, 1.0)][s,a] = self.real_env.calculate_rewards(s_trans, a, ns_trans)
            else:
                _goal_states.append(s)
                self._states_with_known_reward[s,:] = True
                for a in range(self.action_space.n):  
                    ns_trans = self.real_env.transition(s_trans,a)    
                                 
                    ns = self.real_env.encrypt(ns_trans)
                    transition_matrix[s,a,s] = 1.0
                    reward_matrix_per_va[(1.0, 0.0)][s,a], reward_matrix_per_va[(0.0, 1.0)][s,a] = 0,0
        
        self._goal_states = np.asarray(_goal_states)

        if isinstance(initial_state_distribution, np.ndarray):

            assert np.allclose(np.sum(initial_state_distribution), 1.0)

            self.initial_state_dist = initial_state_distribution

        
        elif initial_state_distribution == 'uniform':
            self.initial_state_dist = np.ones(self.n_states)/self.n_states
            
        else:
            self.initial_state_dist = np.zeros(self.n_states)
            self.initial_state_dist[self.real_env.encrypt(np.array([0, 3, 4, 0, 0, 3]))] = 1.0

        #self._cur_state = self.real_env.state
        
        self.reward_matrix_per_va_dict = reward_matrix_per_va
        super(FireFightersEnv, self).__init__(transition_matrix=transition_matrix, observation_matrix=observation_matrix, 
                                              reward_matrix_per_va=self._get_reward_matrix_per_va, 
                                              default_reward_matrix=reward_matrix_per_va[(0.0, 1.0)], horizon=horizon, initial_state_dist=self.initial_state_dist)
        self.cur_align_func = (1.0, 0.0)
    def _get_reward_matrix_per_va(self, align_func):
        return self.reward_matrix_per_va_dict[(1.0,0.0)]*align_func[0] + self.reward_matrix_per_va_dict[(0.0,1.0)]*align_func[1] 
    
    def get_state_actions_with_known_reward(self, align_func):
        return self._states_with_known_reward
    
    @property
    def state(self) -> np.dtype:
        """Data type of state vectors (must be np.int64)."""
        return self._cur_state
    
    
    
    """def _step(self, action):
        # Simulate state transitions and compute rewards
        
        next_obs = self.real_env.transition(self.real_env.state, action)
        rewards = self.real_env.calculate_rewards(self.real_env.state, action, next_obs)
        self.real_env.state = next_obs

        # Check if the episode is done
        done = self.real_env.is_done(self.real_env.state)
        self._cur_state = self.real_env.encrypt(self.real_env.state)
        # In case
        info = {'state': self._cur_state}
        return next_obs, rewards, done, done, info"""
    @property
    def goal_states(self):
        return self._goal_states
    
    """def _reset(self, seed=None, options=None, force_new_state=None):
        s = self.real_env.reset(force_new_state=force_new_state)
        self._cur_state = self.real_env.encrypt(s)
        self.real_env.state = s
        i = {'state': self._cur_state}
        return s, i"""
    

