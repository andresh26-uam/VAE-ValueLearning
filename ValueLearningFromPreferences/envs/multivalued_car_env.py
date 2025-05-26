from copy import deepcopy
import enum
import os
from typing import Any, SupportsFloat
import numpy as np
from gymnasium import spaces
import torch
from use_cases.multivalue_car_use_case import ADS_Environment
from use_cases.firefighters_use_case.env import HighRiseFireEnv


from gymnasium import spaces

from envs.tabularVAenv import TabularVAMDP, encrypt_state

ADS_Environment.Environment(seed=-1, obstacles=0)

class FireFightersEnv(TabularVAMDP):
    """
    A simplified two-objective MDP environment for an urban high-rise fire scenario.
    Objectives: Professionalism and Proximity
    """
    metadata = {'render.modes': ['human']}

    def render(self):
        return self.real_env.render()

    def __init__(self, horizon=100, initial_state_distribution='uniform'):
        self.real_env = ADS_Environment.Environment()

        self.observation_space = None #??????????
        # No, states as indexes, observations in one-hot encoding.

        self.state_space = spaces.Discrete(self.real_env.n_states)
        self.action_space = self.real_env.action_space
        self.feature_selection = feature_selection
        # self.action_dim = self.real_env.action_space.n
        #         self.state_dim = self.real_env.n_states
        self.n_states = self.real_env.n_states

        transition_matrix = np.zeros(
            (self.n_states, self.action_space.n, self.n_states))
        reward_matrix_per_va = dict()

        reward_matrix_per_va[(1.0, 0.0)] = np.zeros(
            (self.n_states, self.action_space.n))
        reward_matrix_per_va[(0.0, 1.0)] = np.zeros(
            (self.n_states, self.action_space.n))

        _goal_states = list()
        _invalid_states = list()
        observation_matrix = np.zeros(
            (self.n_states, *self.observation_space.shape))

        self._states_with_known_reward = np.zeros(
            (self.n_states, self.action_space.n), dtype=np.bool_)
        for s in range(self.n_states):
            s_trans = self.real_env.translate(s)
            if feature_selection == FeatureSelectionFFEnv.ORIGINAL_OBSERVATIONS:
                observation_matrix[s, :] = s_trans
            elif feature_selection == FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS:
                observation_matrix[s, :] = np.eye(self.real_env.n_states)[s]
            elif feature_selection == FeatureSelectionFFEnv.ENCRYPTED_OBSERVATIONS:
                observation_matrix[s, :] = s
            elif feature_selection == FeatureSelectionFFEnv.ONE_HOT_FEATURES:
                vec = np.concatenate([np.eye(self.real_env.state_space.nvec[i])[
                                     s_trans[i]] for i in range(self.real_env.state_space.nvec.shape[0])], -1)
                observation_matrix[s, :] = vec
            if not self.real_env.is_done(s_trans):
                for a in range(self.action_space.n):
                    ns_trans = self.real_env.transition(s_trans, a)
                    ns = self.real_env.encrypt(ns_trans)
                    transition_matrix[s, a, ns] = 1.0

                    reward_matrix_per_va[(1.0, 0.0)][s, a], reward_matrix_per_va[(
                        0.0, 1.0)][s, a] = self.real_env.calculate_rewards(s_trans, a, ns_trans)
            else:
                if s_trans[STATE_MEDICAL] != 0:

                    _goal_states.append(s)
                else:

                    _invalid_states.append(s)
                self._states_with_known_reward[s, :] = True
                for a in range(self.action_space.n):
                    ns_trans = self.real_env.transition(s_trans, a)

                    ns = self.real_env.encrypt(ns_trans)
                    transition_matrix[s, a, s] = 1.0
                    reward_matrix_per_va[(1.0, 0.0)][s, a], reward_matrix_per_va[(
                        0.0, 1.0)][s, a] = self.real_env.calculate_rewards(s_trans, a, ns_trans) if ns_trans[STATE_MEDICAL] != 0 else [-1.0, -1.0]

        self._goal_states = np.asarray(_goal_states)
        self._invalid_states = np.asarray(_invalid_states)

        if isinstance(initial_state_distribution, np.ndarray):

            assert np.allclose(np.sum(initial_state_distribution), 1.0)

            self.initial_state_dist = initial_state_distribution

        elif initial_state_distribution == 'uniform' or initial_state_distribution == 'random':
            self.initial_state_dist = np.ones(self.n_states)/self.n_states

        else:
            self.initial_state_dist = np.zeros(self.n_states)
            self.initial_state_dist[self.real_env.encrypt(
                np.array([0, 3, 4, 0, 0, 3]))] = 1.0
        self.initial_state_dist[self.invalid_states] = 0.0
        self.initial_state_dist /= np.sum(self.initial_state_dist)
        assert np.allclose(np.sum(self.initial_state_dist), 1.0)
        # self._cur_state = self.real_env.state

        self.reward_matrix_per_va_dict = reward_matrix_per_va
        super(FireFightersEnv, self).__init__(n_values=2,
                                              transition_matrix=transition_matrix, observation_matrix=observation_matrix,
                                              reward_matrix_per_va=self._get_reward_matrix_per_va,
                                              default_reward_matrix=reward_matrix_per_va[(0.0, 1.0)], horizon=horizon, initial_state_dist=self.initial_state_dist)
        self.set_align_func((1.0, 0.0))

    def _get_reward_matrix_per_va(self, align_func, custom_grounding=None):
        
        assert isinstance(align_func, tuple)
        assert isinstance(float(align_func[0]), float)
        assert isinstance(float(align_func[1]), float)
        if custom_grounding is None:
            v = self.reward_matrix_per_va_dict[(
                1.0, 0.0)]*align_func[0] + self.reward_matrix_per_va_dict[(0.0, 1.0)]*align_func[1]
        else:
        # custom grounding 0 might is only used for trajectories
            if isinstance(custom_grounding, tuple):
                custom_grounding = custom_grounding[1]
            if custom_grounding.shape == (400,5,2):
                v = custom_grounding[:,:,0]*align_func[0] + custom_grounding[:,:,1]*align_func[1]
            
        # assert np.max(v) <= 1.00001
        # assert np.min(v) >= -1.00001
        return v