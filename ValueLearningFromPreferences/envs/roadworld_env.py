from copy import deepcopy
from functools import partial
import os
from typing_extensions import override
from envs.tabularVAenv import ContextualEnv, TabularVAMDP
from use_cases.roadworld_env_use_case.network_env import DATA_FOLDER, FeaturePreprocess, FeatureSelection, RoadWorldGymPOMDP
import numpy as np


from use_cases.roadworld_env_use_case.utils.load_data import ini_od_dist
from use_cases.roadworld_env_use_case.values_and_costs import BASIC_PROFILES



class FixedDestRoadWorldGymPOMDP(TabularVAMDP):
    metadata = {'render.modes': ['human']}

    def _base_reward_matrix_per_align_func(self, profile):
        
        if profile in self._base_rw_per_al.keys() and self._base_rw_per_al[profile].shape == (self.real_environ.state_dim, self.real_environ.action_dim):
            #np.testing.assert_allclose(self._base_rw_per_al[profile][self.real_environ.cur_des], 0)
            
            return self._base_rw_per_al[profile]
        elif profile in BASIC_PROFILES and profile not in self._base_rw_per_al.keys():

            reward_matrix = np.zeros(
                (self.real_environ.state_dim, self.real_environ.action_dim), dtype=np.float32)
            for s in range(self.real_environ.state_dim):
                for a in range(self.real_environ.action_dim):
                    reward_matrix[s, a] = self.real_environ.get_reward(
                        s, a, None, profile=profile)

            self._base_rw_per_al[profile] = reward_matrix
            return self._base_rw_per_al[profile]
        else:
            return np.sum([profile[i]*self._base_reward_matrix_per_align_func(bp) for i, bp in enumerate(BASIC_PROFILES)], axis=0)

    def obtain_grounding(self, variant=None, file_save=None, recalculate=True):
        if variant is None or 'default' in variant:
            mat = self._base_reward_matrix_per_align_func(
                (1.0, 0.0, 0.0) if 'sus' in variant else (0.0, 1.0, 0.0) if 'com' in variant else (0.0, 0.0, 1.0) if 'eff' in variant else None)
            
            return mat
        """elif variant == 'professionalist':
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
        
        return reward_matrix"""
    
    def calculate_assumed_grounding(self, variants=None, variants_save_files=None, save_folder=None, recalculate=False, **kwargs):

        if self.real_environ.feature_selection == FeatureSelection.ONLY_COSTS:
            assumed_grounding = np.zeros(
                (self.state_dim,self.action_dim, 3), dtype=np.float64)
            assumed_grounding[:, :, 0] = self.obtain_grounding( variant=variants[0] if variants is not None else None, 
                file_save=os.path.join(save_folder, variants_save_files[0]) if variants_save_files is not None else None, recalculate=recalculate)
            assumed_grounding[:, :, 1] = self.obtain_grounding( variant=variants[1] if variants is not None else None, 
                file_save=os.path.join(save_folder, variants_save_files[1]) if variants_save_files is not None else None, recalculate=recalculate)
            assumed_grounding[:,:, 2] = self.obtain_grounding( variant=variants[2] if variants is not None else None, 
                file_save=os.path.join(save_folder, variants_save_files[2]) if variants_save_files is not None else None, recalculate=recalculate)
            self.current_assumed_grounding = assumed_grounding, assumed_grounding

            return assumed_grounding, assumed_grounding
        
        else:
            raise ValueError(f"Feature selection not registered {self.real_environ.feature_selection}")

    def _get_reward_matrix_for_profile(self, profile: tuple, custom_grounding=None):
        if isinstance(profile[0], str):
            profile = profile[1]
        if custom_grounding is None:
            if profile in self.reward_matrix_dict.keys() and self.reward_matrix_dict[profile].shape == (self.real_environ.state_dim, self.real_environ.action_dim):
                np.testing.assert_allclose(self.reward_matrix_dict[profile][self.real_environ.cur_des], 0)
                
                return self.reward_matrix_dict[profile]
            elif profile in BASIC_PROFILES and profile not in self.reward_matrix_dict.keys():

                reward_matrix = np.zeros(
                    (self.real_environ.state_dim, self.real_environ.action_dim), dtype=np.float32)
                for s in range(self.real_environ.state_dim):
                    for a in range(self.real_environ.action_dim):
                        reward_matrix[s, a] = self.real_environ.get_reward(
                            s, a, self.real_environ.cur_des, profile=profile)

                self.reward_matrix_dict[profile] = reward_matrix
                return self.reward_matrix_dict[profile]
            else:
                return np.sum([profile[i]*self._get_reward_matrix_for_profile(bp) for i, bp in enumerate(BASIC_PROFILES)], axis=0)
            
        else:
        # custom grounding 0 might is only used for trajectories
            if isinstance(custom_grounding, tuple):
                custom_grounding=custom_grounding[1]
            if custom_grounding.shape == (self.real_environ.state_dim,self.real_environ.action_dim,3):
                v = custom_grounding[:,:,0]*profile[0] + custom_grounding[:,:,1]*profile[1] + custom_grounding[:,:,2]*profile[2]
            
            assert v.shape == (self.real_environ.state_dim,self.real_environ.action_dim)
            v[self.get_state_actions_with_known_reward(profile)] = self.reward_matrix_per_align_func(profile)[self.get_state_actions_with_known_reward(profile)]
            np.testing.assert_allclose(v[self.goal_states], 0.0)
            return v

    def _create_env(self, feature_selection=FeatureSelection.ONLY_COSTS,
                              feature_preprocessing=FeaturePreprocess.NORMALIZATION, fixed_dests=[64, ]):
        cv = 0  # cross validation process [0, 1, 2, 3, 4]
        size = 100  # size of training data [100, 1000, 10000]

        """environment"""
        edge_p = f"{DATA_FOLDER}/edge.txt"
        network_p = f"{DATA_FOLDER}/transit.npy"
        path_feature_p = f"{DATA_FOLDER}/feature_od.npy"
        train_p = f"{DATA_FOLDER}/cross_validation/train_CV%d_size%d.csv" % (
            cv, size)
        test_p = f"{DATA_FOLDER}/cross_validation/test_CV%d.csv" % cv
        node_p = f"{DATA_FOLDER}/node.txt"

        od_list, od_dist = ini_od_dist(train_p)


        env_creator = partial(RoadWorldGymPOMDP, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path=path_feature_p,
                              pre_reset=(od_list, od_dist), profile=(1.0, 0.0, 0.0), visualize_example=False, horizon=self.horizon,
                              feature_selection=feature_selection,
                              feature_preprocessing=feature_preprocessing,
                              use_optimal_reward_per_profile=False)
        env = env_creator()

        if fixed_dests is not None:
            od_list = [str(state) + '_' + str(dest)
                    for state in env.valid_edges for dest in fixed_dests]

            env_creator = partial(RoadWorldGymPOMDP, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path=path_feature_p,
                                pre_reset=(od_list, od_dist),
                                profile=(1.0, 0.0, 0.0), visualize_example=True, horizon=self.horizon,
                                destinations=fixed_dests,
                                feature_selection=feature_selection,
                                feature_preprocessing=feature_preprocessing,
                                use_optimal_reward_per_profile=False)
            env = env_creator()

        return env


    def __init__(self, horizon=50, with_destination=None, done_when_horizon_is_met=False, trunc_when_horizon_is_met=True, env_kwargs={
        'feature_selection': FeatureSelection.ONLY_COSTS, 
        'feature_preprocessing': FeaturePreprocess.NORMALIZATION}):
        self.reward_matrix_dict = dict()

        self._base_rw_per_al = dict()
        
        self.horizon = horizon
        
        env_kwargs['fixed_dests'] = [with_destination, ] if with_destination is not None else None
        
        env = self._create_env(**env_kwargs)
        if with_destination is not None:
            assert env.cur_des == with_destination

        reward_matrix_per_va = self._get_reward_matrix_for_profile
        self.real_environ = env
        
        self._invalid_states = [s for s in range(
            env.n_states) if s not in env.valid_edges]
        
        super().__init__(horizon=self.horizon
                         ,n_values= len(BASIC_PROFILES),
                         initial_state_dist=env.initial_state_dist,
                         transition_matrix=env.transition_matrix,
                         observation_matrix=env.observation_matrix,
                         reward_matrix_per_va=reward_matrix_per_va,
                         default_reward_matrix=reward_matrix_per_va(env.last_profile),
                         
                         done_when_horizon_is_met=done_when_horizon_is_met, 
                         trunc_when_horizon_is_met=trunc_when_horizon_is_met)
        self.set_align_func(env.last_profile)

    def step(self, action):
        s, r, d, t, i = super().step(action)
        d = self.state in self.goal_states
        return s, r, d, t, i

    def valid_actions(self, state, align_func=None):
        return self.real_environ.get_action_list(state)

    def get_state_actions_with_known_reward(self, align_func):
        return self.real_environ.state_actions_with_known_reward

    @property
    def goal_states(self):
        return [self.real_environ.cur_des,]

    @property
    def invalid_states(self):
        return self._invalid_states

    @override
    def render(self, caminos_by_value={'sus': [], 'sec': [], 'eff': []}, file='dummy.png', show=True, show_edge_weights=False, custom_weights: dict = None, custom_weights_dest: int = None):
        self.real_environ.render(caminos_by_value=caminos_by_value, file=file, show=show, show_edge_weights=show_edge_weights,
                                 custom_weights=custom_weights, custom_weights_dest=custom_weights_dest)

    
    def real_reset(self, *, seed = None, options = None, to_destination=None,contextualize=False):
        print("RESET FROM", self.real_environ.cur_des)
        if contextualize:
            self.contextualize(context=to_destination, seed=seed, options=options)
        else:
            self.real_environ.reset(seed=seed, options=options, des=None, profile=self.real_environ.last_profile, full_random=True)
        print("RESET TO", self.real_environ.cur_des)
        shouldbe_, i_should_be = super().real_reset(seed=seed, options=options)
        print("S", "O", "NEW", self.state, shouldbe_, self.real_environ.cur_des)
        return shouldbe_, i_should_be
class VariableDestRoadWorldGymPOMDP(FixedDestRoadWorldGymPOMDP, ContextualEnv):


    def __init__(self, horizon=50, done_when_horizon_is_met=False, trunc_when_horizon_is_met=True, env_kwargs={
        'feature_selection': FeatureSelection.ONLY_COSTS, 
        'feature_preprocessing': FeaturePreprocess.NORMALIZATION}):
        super().__init__(horizon=horizon, with_destination=None, done_when_horizon_is_met=done_when_horizon_is_met, trunc_when_horizon_is_met=trunc_when_horizon_is_met, env_kwargs=env_kwargs)
        self.cached_reward_by_context = dict()

    def contextualize(self, context, seed=None, options=None):
        return self._set_destination(context, seed=seed, options=options)
    @property
    def context(self):
        return self.real_environ.cur_des
    
    def _set_destination(self, destination=None, seed = None, options = None):
        #if destination is None or destination != self.real_environ.cur_des:
        self.cached_reward_by_context[ self.real_environ.cur_des] = self.reward_matrix_dict
        
        notused, notused2 = self.real_environ.reset(seed=seed, options=options, des=destination, profile=self.real_environ.last_profile, full_random=True)
        
        if destination is not None:
            assert self.real_environ.cur_des == destination
        else:
            destination = self.real_environ.cur_des

        self.transition_matrix = self.real_environ.transition_matrix
        self.observation_matrix = self.real_environ.observation_matrix
        
        self.reward_matrix_dict = self.cached_reward_by_context.get(destination, dict())

    
    def real_reset(self, *, seed = None, options = None, to_destination=None,contextualize=True):
        
        return super().real_reset(seed=seed, options=options, to_destination=to_destination, contextualize=True)
    
    