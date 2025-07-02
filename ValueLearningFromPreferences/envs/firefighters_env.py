from copy import deepcopy
import enum
from functools import partial
import os
from typing import Any, SupportsFloat
import numpy as np
from gymnasium import spaces
import torch
from use_cases.firefighters_use_case.constants import STATE_MEDICAL
from use_cases.firefighters_use_case.env import HighRiseFireEnv


from gymnasium import spaces

from envs.tabularVAenv import TabularVAMDP, encrypt_state, grounding_func_from_matrix

def calculate_s_trans_ONE_HOT_FEATURES(vec, state_space, action_space):

    splits = np.cumsum([*state_space.nvec, action_space.n][:-1])
    split_vec = np.split(vec, splits, axis=-1)
    # Use argmax to find the indices of 1 in each segment for all rows
    s_trans = np.stack([np.argmax(segment, axis=-1)
                       for segment in split_vec], axis=-1)
    encrypted_s_trans = np.apply_along_axis(
        lambda row: encrypt_state(row[:-1], state_space), axis=1, arr=s_trans)
    actions = np.apply_along_axis(lambda row: row[-1], axis=1, arr=s_trans)
    return encrypted_s_trans, actions

class FeatureSelectionFFEnv(enum.Enum):

    ONE_HOT_FEATURES = 'features_one_hot'  # Each feature is one-hot encoded
    # Use the original observations of the form [0,1,4,3,0]
    ORIGINAL_OBSERVATIONS = 'observations'
    # Use State unique identifier given by its encryption, e.g. 320
    ENCRYPTED_OBSERVATIONS = 'encrypted_observations'
    # Use State encrption, but one hot encoded
    ONE_HOT_OBSERVATIONS = 'observations_one_hot'
    # TODO: Ordinal Features may remain as 0, 1, 2 but categorical ones are one hot encoded for now.
    ORDINAL_AND_ONE_HOT_FEATURES = 'ordinal_and_one_hot'
    DEFAULT = None


class FireFightersEnv(TabularVAMDP):
    """
    A simplified two-objective MDP environment for an urban high-rise fire scenario.
    Objectives: Professionalism and Proximity
    """
    metadata = {'render.modes': ['human']}

    def render(self):
        return self.real_env.render()

    def __init__(self, env_name= 'FireFighters-v0', feature_selection=FeatureSelectionFFEnv.ONE_HOT_FEATURES, horizon=100, initial_state_distribution='uniform'):
        self.init__kwargs = locals()
        self.init__kwargs.pop('self', None)
        self.init__kwargs.pop('__class__', None)
        self.real_env = HighRiseFireEnv()

        if feature_selection == FeatureSelectionFFEnv.ORIGINAL_OBSERVATIONS:
            self.observation_space = self.real_env.state_space
        elif feature_selection == FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS:
            self.observation_space = spaces.MultiBinary(
                n=self.real_env.n_states)
        elif feature_selection == FeatureSelectionFFEnv.ONE_HOT_FEATURES:
            self.observation_space = spaces.MultiBinary(
                n=sum([nv for nv in self.real_env.state_space.nvec]))
        elif feature_selection == FeatureSelectionFFEnv.ENCRYPTED_OBSERVATIONS:
            self.observation_space = spaces.Discrete(self.real_env.n_states)
        else:
            self.observation_space = self.real_env.state_space
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
        super(FireFightersEnv, self).__init__(env_name=env_name,n_values=2,
                                              transition_matrix=transition_matrix, observation_matrix=observation_matrix,
                                              reward_matrix_per_va=self._get_reward_matrix_per_va,
                                              default_reward_matrix=reward_matrix_per_va[(0.0, 1.0)], horizon=horizon, initial_state_dist=self.initial_state_dist)
        self.set_align_func((1.0, 0.0))
        self.is_stochastic = False

    def _get_reward_matrix_per_va(self, align_func, custom_grounding=None):
        if isinstance(align_func[0], str):
            align_func = align_func[1]
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
            custom_grounding = custom_grounding() # Call the function to get the matrix
            if custom_grounding.shape == (400,5,2):
                v = custom_grounding[:,:,0]*align_func[0] + custom_grounding[:,:,1]*align_func[1]
            
        # assert np.max(v) <= 1.00001
        # assert np.min(v) >= -1.00001
        return v

    def get_state_actions_with_known_reward(self, align_func):
        return  self._states_with_known_reward

   

    @property
    def goal_states(self):
        return self._goal_states

    @property
    def invalid_states(self):
        return self._invalid_states

    def calculate_assumed_grounding(self, variants=None, variants_save_files=None, save_folder=None, recalculate=False, **kwargs):

        if self.feature_selection == FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS:
            assumed_grounding = np.zeros(
                (self.state_dim*self.action_dim, 2), dtype=np.float64)
            assumed_grounding[:, 0] = np.reshape(self.obtain_grounding( variant=variants[0] if variants is not None else None, 
                file_save=os.path.join(save_folder, variants_save_files[0]) if variants_save_files is not None else None, recalculate=recalculate), 
                (self.state_dim*self.action_dim,))
            assumed_grounding[:, 1] = np.reshape(self.obtain_grounding( variant=variants[1] if variants is not None else None, 
                file_save=os.path.join(save_folder, variants_save_files[1]) if variants_save_files is not None else None, recalculate=recalculate)
, (self.state_dim*self.action_dim,))
            self.set_grounding_func(grounding_func_from_matrix(assumed_grounding))

            return self.get_grounding_func()
        elif self.feature_selection == FeatureSelectionFFEnv.ONE_HOT_FEATURES:
            assumed_grounding = np.zeros(
                (self.state_dim, self.action_dim, 2), dtype=np.float64)
            assumed_grounding[:, :, 0] = self.obtain_grounding( variant=variants[0] if variants is not None else None, 
                file_save=os.path.join(save_folder, variants_save_files[0]) if variants_save_files is not None else None, recalculate=recalculate)
            assumed_grounding[:, :, 1] = self.obtain_grounding( variant=variants[1] if variants is not None else None, 
                file_save=os.path.join(save_folder, variants_save_files[1]) if variants_save_files is not None else None, recalculate=recalculate)

            #t_assumed_grounding = torch.tensor(assumed_grounding, dtype=torch.float32).requires_grad_(False)

            """def processing_obs(torch_obs):
                states, actions = calculate_s_trans_ONE_HOT_FEATURES(
                    torch_obs, self.real_env.state_space, self.real_env.action_space)

                ret = t_assumed_grounding[states, actions]
                return ret"""
            self.set_grounding_func(grounding_func_from_matrix(assumed_grounding))


            return self.get_grounding_func()
        else:
            raise ValueError(f"Feature selection not registered {self.feature_selection}")
        
    def obtain_grounding(self, variant=None, file_save=None, recalculate=True):
        if variant is None or 'default' in variant:
            return self.reward_matrix_per_align_func(
                (0.0, 1.0) if 'proximity' in variant else (1.0, 0.0) if 'professionalism' in variant else None)
        elif variant == 'professionalist':
            with open(file_save, 'wb' if recalculate else 'r') as fsave:
                if recalculate or os.stat(file_save).st_size == 0: 
                    reward_matrix = deepcopy(self.reward_matrix_per_align_func((1.0,0.0)))
                    reward_matrix[reward_matrix > 0] = 1.0 # Always go for professionalism
                    np.save(fsave, reward_matrix)
                else:
                    reward_matrix = np.load(fsave)
        elif variant == 'proximitier':
            with open(file_save, 'wb' if recalculate else 'r') as fsave:
                if recalculate or os.stat(file_save).st_size == 0: 
                    reward_matrix = deepcopy(self.reward_matrix_per_align_func((0.0,1.0)))
                    reward_matrix[reward_matrix > 0] = 1.0 # Always go for proximity
                    np.save(fsave, reward_matrix)
                else:
                    reward_matrix = np.load(fsave)
        
        return reward_matrix