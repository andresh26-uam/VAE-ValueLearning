"""
Based on https://imitation.readthedocs.io/en/latest/algorithms/mce_irl.html
Adapted for the RoadWorld environment
"""

import collections
import enum
import random
import re
import warnings
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import editdistance
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch as th
from seals import base_envs
from imitation.algorithms import base
from imitation.data import types
from imitation.util import logger as imit_logger
from imitation.util import networks, util

from copy import deepcopy

from src.network_env import FeaturePreprocess, RoadWorldPOMDPStateAsTuple
from src.policies import SimplePolicy, TabularPolicyPerProfile, calculate_expected_cost_and_std, calculate_expected_similarities_and_std, jaccard, jaccard_similarity, profiled_aggregated_similarity_Agourogiannis_et_al_2023
from src.policies import ValueIterationPolicy
from src.reward_functions import TrainingModes, ProfiledRewardFunction, cost_model_from_reward_net, squeeze_r
from src.src_rl.aggregations import SumScore
from src.values_and_costs import BASIC_PROFILE_NAMES, BASIC_PROFILES, PROFILE_COLORS

import pandas as pd

def get_demo_oms_from_trajectories(trajs: Iterable[types.Trajectory], state_dim, discount =1, reference_experts: Dict[tuple, List[types.Trajectory]] = None, groupby='odpr') -> dict:
        demo_om = dict()
        num_demos = dict()
        for traj in trajs:
            
            orig_des_profile = ((traj.infos[0]['orig'], traj.infos[0]['des']), traj.infos[0]['profile'])# if groupby == 'odpr' else (traj.infos[0]['orig'], traj.infos[0]['des'])
           
            if orig_des_profile not in num_demos.keys():
                num_demos[orig_des_profile] = 0
                demo_om[orig_des_profile] = np.zeros((state_dim,))

            obs_relevant = traj.obs
            
            np.add.at(demo_om[orig_des_profile], (
                np.asarray(obs_relevant)[:,0],), 1)
            
            num_demos[orig_des_profile] += 1
        for orig_des_profile, num_demos_od in num_demos.items():
            demo_om[orig_des_profile] /= num_demos_od

        if groupby == 'odpr':
            return demo_om
        
        elif groupby == 'od':
            demos_by_od = dict()
            n_demos_per_od = dict()
            for odpr, demo_om in demo_om.items():
                od, pr = odpr
                if od not in demos_by_od:
                    n_demos_per_od[od] = 1
                    demos_by_od[od] = demo_om
                else:
                    n_demos_per_od[od] += 1
                    demos_by_od[od] += demo_om
            for od in demos_by_od.keys():
                demos_by_od[od] /= n_demos_per_od[od]
            return demos_by_od
        raise ValueError(groupby)


class TrainingSetModes(enum.Enum):
    PROFILED_SOCIETY = 'profiled_society' # Used for profile learning of a society sampling trajectories according to a probabilty distribution of profiles.
    PROFILED_EXPERT = 'cost_model' # Default in Value Learning.
        
class MCEIRL_RoadNetwork(base.DemonstrationAlgorithm[types.TransitionsMinimal]):
    """
    Based on https://imitation.readthedocs.io/en/latest/algorithms/mce_irl.html
    Adapted for the RoadWorld environment
    """
    def set_reward_net(self, reward_net: ProfiledRewardFunction):

        self._reward_net = reward_net
        self.optimizer = self.optimizer_cls(self._reward_net.parameters(), **self.optimizer_kwargs)
    def get_reward_net(self):
        return self._reward_net

    def __init__(
        self,
        env: Union[base_envs.TabularModelPOMDP, RoadWorldPOMDPStateAsTuple],
        reward_net: ProfiledRewardFunction,
        rng: np.random.Generator,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        discount: float = 1.0,
        mean_vc_diff_eps: float = 1e-3,
        grad_l2_eps: float = 1e-4,
        log_interval: Optional[int] = 100,
        expert_policy: Optional[Union[SimplePolicy, np.ndarray]] = None,
        expert_trajectories: Optional[List[types.Trajectory]] = None,
        use_expert_policy_oms_instead_of_monte_carlo: bool = True,
        training_profiles = 'auto',
        overlaping_percentage: float = 1.0,
        n_repeat_per_od_monte_carlo: int = 1,
        render_partial_plots = True,
        fd_lambda = 0.0,
        use_dijkstra = False,
        stochastic_expert = None,
        od_list_train = None,
        od_list_test = None,
        training_mode = TrainingModes.VALUE_GROUNDING_LEARNING,
        training_set_mode = TrainingSetModes.PROFILED_EXPERT,
        name_method='',
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        
        self.discount = discount
        self.fd_lambda = fd_lambda
        self.env: RoadWorldPOMDPStateAsTuple = env
        self.demo_state_om = None
        self.render_partial_plots = render_partial_plots
        self.overlaping_percentage = overlaping_percentage
        self.expert_trajectories = expert_trajectories # list of expert trajectories just to see different origin destinations and profiles
        self.expert_trajectories_per_odpr = dict()
        self.expert_trajectories_per_pr = dict()
        self.use_dijkstra = use_dijkstra
        
        self.od_list_train = od_list_train if od_list_train is not None else env.od_list_int
        self.od_list_test = od_list_test if od_list_test is not None else env.od_list_int


        self.destinations = list(set(od[1] for od in env.od_list_int))
        self.destinations_train = list(set(od[1] for od in self.od_list_train))
        self.destinations_test = list(set(od[1] for od in self.od_list_test))


        if stochastic_expert is None:
            self.stochastic_expert = True if n_repeat_per_od_monte_carlo > 1 else False
        else:
            self.stochastic_expert = stochastic_expert

        for t in self.expert_trajectories:
            pr = t.infos[0]['profile']
            odpr = ((t.infos[0]['orig'], t.infos[0]['des']), pr)
            if odpr not in self.expert_trajectories_per_odpr.keys():

                self.expert_trajectories_per_odpr[odpr] = []
            if pr not in self.expert_trajectories_per_pr.keys():
                self.expert_trajectories_per_pr[pr] = []
                

            self.expert_trajectories_per_odpr[odpr].append(t)
            self.expert_trajectories_per_pr[pr].append(t)
            

        
        if training_profiles == 'auto':
            self.training_profiles = list(set(tuple(t.infos[0]['profile']) for t in self.expert_trajectories))
        else:
            self.training_profiles = list(set(training_profiles))
        self.n_repeat_per_od_monte_carlo = 1 if use_expert_policy_oms_instead_of_monte_carlo else n_repeat_per_od_monte_carlo
        self.use_mce_om = use_expert_policy_oms_instead_of_monte_carlo

       

        super().__init__(
                demonstrations=None,
                custom_logger=custom_logger,
            )
        self.expert_policy: SimplePolicy = expert_policy if isinstance(expert_policy, SimplePolicy) else SimplePolicy.from_policy_matrix(expert_policy, self.env) 
            
        
        if self.use_mce_om: # TODO: UNUSED ?
            assert isinstance(expert_policy, dict)
            self.demo_state_om_per_profile = dict()
            for pr in self.training_profiles:
                assert isinstance(expert_policy[pr], np.ndarray)
                _, demo_state_om = self.mce_occupancy_measures(pi=expert_policy[pr], profile=pr, od_list=self.env.od_list_int)
                for odpr, om_dif in demo_state_om.items():
                    self.demo_state_om_per_profile[odpr] = om_dif
        else:
            
            sampled_expert_trajectories = self.expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=self.training_profiles)
            
            self._set_demo_oms_from_trajectories(sampled_expert_trajectories)
        
        self._reward_net = reward_net
        self.optimizer_cls = optimizer_cls
        optimizer_kwargs = optimizer_kwargs or {"lr": 1e-2}
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer_cls(reward_net.parameters(), **optimizer_kwargs)
        

        self.mean_vc_diff_eps = mean_vc_diff_eps
        self.grad_l2_eps = grad_l2_eps
        self.log_interval = log_interval
        self.rng = rng

        self.name_method = name_method
        # Initialize policy to be uniform random. We don't use this for MCE IRL
        # training, but it gives us something to return at all times with `policy`
        # property, similar to other algorithms.
        if self.env.horizon is None:
            raise ValueError("Only finite-horizon environments are supported.")
        #ones = np.ones((self.env.state_dim, self.env.action_dim))
        #uniform_pi = ones / self.env.action_dim

        self.default_vi_policy = ValueIterationPolicy(self.env) # Starts with random policy
        #self.vi_policy.train(0.001, verbose=True, stochastic=True, custom_reward_function=lambda s,a,d: self.env.reward_matrix[self.env.netconfig[s][a]]) # alpha stands for error tolerance in value_iteration
        self.vi_policy_per_profile = {
            pr: ValueIterationPolicy(self.env) for pr in self.training_profiles
        }
        self._policy = TabularPolicyPerProfile(
            state_space=self.env.state_space,
            action_space=self.env.action_space,
            pi={pr: self.vi_policy_per_profile[pr].pi_from_d for pr in self.training_profiles},
            rng=self.rng,
        )

        self.training_mode = training_mode
        self.training_set_mode = training_set_mode

        self.policies_per_profile = dict()
        self._reward_net.set_mode(self.training_mode)
        
    
    def mce_occupancy_measures(
            self,
        *,
        reward: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None,
        profile = (1.0,0.0,0.0),
        od_list = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        horizon = self.env.horizon
        if horizon is None:
            raise ValueError("Only finite-horizon environments are supported.")
        n_states = self.env.state_dim

        if reward is None:
            reward = self.env.reward_matrix[profile]
        od_list_int = od_list if od_list is not None else self.env.od_list_int
        if pi is None:
            pi = self.mce_partition_fh(reward_matrix=reward, profile=profile, od_list=od_list_int)
        Dcums = {}
        Ds = {}
        num_od = dict()
        for demo in self.expert_trajectories_per_pr[profile]:
            #print("om for demo", demo)

            demo: types.Trajectory
            orig = demo.infos[0]['orig']
            des = demo.infos[0]['des']

            od = (orig, des)
            if od not in od_list_int:
                continue
            #pi[s, :] = 0
            ldemo = min(len(demo), horizon)

            if od not in Ds:
                Ds[od] = np.zeros((ldemo+1, n_states)) # It should never be bigger than horizon expert routes
                
                num_od[(orig, des)] = 0

            Ds[od][0, orig] += 1
            for t in range(ldemo):
                for s in self.env.valid_edges:
                    acts_to_s, sprev = self.env.pre_acts_and_pre_states[s]
                    Ds[od][t + 1, s] += np.dot(Ds[od][t,sprev], pi[des, sprev, acts_to_s])#@ T[sprev, :, s] #pi already accounts for T.
            num_od[od]+=1        
                    
            #assert isinstance(Dcum, np.ndarray)
        for (orig, des), Dsval in Ds.items():
            Dcums[((orig, des),tuple(profile))] =  Dsval.sum(axis=0)/num_od[(orig, des)]

        return Ds, Dcums
        
    def _set_demo_oms_from_trajectories(self, trajs: Iterable[types.Trajectory])  -> None:
        
        self.demo_state_om_per_profile = get_demo_oms_from_trajectories(trajs, state_dim = self.env.state_dim, discount=self.discount)

    def set_demonstrations(self, demonstrations: Union[Iterable[types.Trajectory],Iterable[types.TransitionMapping], types.TransitionsMinimal]) -> None:
        pass

    def adapt_policy_to_profile(self, profile=(0.5,0.5,0), use_cached_policies=False, on_destinations=None):
        #print(np.where(reward_matrix >= 0.0))
        #print(reward_matrix[:,413])
        if self.use_dijkstra:
            if use_cached_policies:
                if profile not in self.policies_per_profile.keys():
                    reward_matrix = self.calculate_rewards_for_destinations_and_profile(self.destinations if on_destinations is None else on_destinations, profile) 
                    sampler = SimplePolicy.from_environment_expert(self.env, profiles=[profile,], 
                                                            custom_cost = lambda st_des, pr: -reward_matrix[st_des[0], st_des[1]])
                    self.policies_per_profile[profile] = sampler
                else:
                    sampler = self.policies_per_profile[profile]
            else:
                reward_matrix = self.calculate_rewards_for_destinations_and_profile(self.destinations if on_destinations is None else on_destinations, profile) 
                    
                sampler = SimplePolicy.from_environment_expert(self.env, profiles=[profile,], 
                                                         custom_cost = lambda st_des, pr: -reward_matrix[st_des[0], st_des[1]])
            
            adapted_policy = sampler
        else:
            reward_matrix = self.calculate_rewards_for_destinations_and_profile(self.destinations if on_destinations is None else on_destinations, profile)
        
            self.default_vi_policy.value_iteration(error=0.1, 
                                            profile=self.env.last_profile if profile is None else tuple(profile),
                                            custom_reward=self.reward_from_matrix(reward_matrix), expert_paths_per_od_profile=self.expert_trajectories_per_odpr,
                                            reset_vi=True, od_list=self.env.od_list_int,
                                            full_match=False, verbose=False)  
            adapted_policy = self.default_vi_policy

        self.policy.set_pi_or_policy(adapted_policy)
        self.policy.set_profile(profile) 

        return adapted_policy

        

    
        
    def mce_partition_fh(self, reward_matrix, profile=None, od_list=None):
        
        if profile in self.training_profiles:
            return self.vi_policy_per_profile[profile].value_iteration(error=1, 
                                            profile=self.env.last_profile if profile is None else tuple(profile),
                                            custom_reward=self.reward_from_matrix(reward_matrix), 
                                            reset_vi=False, expert_paths_per_od_profile=self.expert_trajectories_per_odpr, 
                                            od_list=od_list,
                                            full_match=False, verbose=False)
        else:
            print("EMERGENCY PROFILE NEW: ", profile)
            return self.default_vi_policy.value_iteration(error=1, 
                                            profile=self.env.last_profile if profile is None else tuple(profile),
                                            custom_reward=self.reward_from_matrix(reward_matrix), expert_paths_per_od_profile=self.expert_trajectories_per_odpr,
                                            reset_vi=True, od_list=od_list,
                                            full_match=False, verbose=False)
    
    def reward_from_matrix(self, reward_matrix):
        def _reward_from_matrix(s,a,d):
            next_state = self.env.get_state_des_transition((s,d), a)
            if len(reward_matrix.shape) == 2:
                return reward_matrix[next_state[0], next_state[1]]
            else:
                return reward_matrix[next_state[0]]
        return _reward_from_matrix
    
    def expected_trajectory_cost_calculation(self, on_profiles, learned_profiles, costs_measured_with_profiles=BASIC_PROFILES, stochastic_sampling = False, n_samples_per_od=None, custom_cost_preprocessing=None, repeat_society=10, bins=30, new_test_data=0, name_method='', plot_histograms=False, plot_learned_profiles_as_probabilities=False):
        
        if new_test_data is not None:
            od_list_test_new = set(new_test_data)
            od_list_all = deepcopy(od_list_test_new)
            od_list_all.update(self.od_list_train)
            od_list_all = list(od_list_all)

            od_list_test_new = list(od_list_test_new)
            #expert_policy = SimplePolicy.from_environment_expert(profiles=on_profiles, environment=self.env, on_od_list=od_list_all)
            expert_policy = self.expert_policy
        else:
            expert_policy = self.expert_policy
        
        destinations_all = list(set([od[1] for od in od_list_all]))
        n_samples_per_od = self.n_repeat_per_od_monte_carlo if n_samples_per_od is None else int(n_samples_per_od)
        cost_preprocessing = FeaturePreprocess.NORMALIZATION if custom_cost_preprocessing is None else custom_cost_preprocessing
        

        measure_profiles = []
        measure_profiles.extend(costs_measured_with_profiles)
        #measure_profiles.extend('societal_cost')

        """'expert_profile': {pr: (0,0) for pr in on_profiles_with_societal_cost},'society_profile':{pr: (0,0) for pr in on_profiles_with_societal_cost},"""
        train_rows = {
            'expert': {pr: (0,0) for pr in measure_profiles},
                      'learned': {pr: (0,0) for pr in measure_profiles},
                      #'learned_profile': {pr: (0,0) for pr in on_profiles_with_societal_cost},
                      #'learned_profile_probability': {pr: (0,0) for pr in on_profiles_with_societal_cost}
                      }
        train_rows.update({f'expert_pr_{tuple(float(f"{t:0.2f}") for t in pr)}': {pr_2: (0,0) for pr_2 in measure_profiles} for pr in on_profiles})
        train_rows.update({f'society_pr_{tuple(float(f"{t:0.2f}") for t in pr)}': {pr_2: (0,0) for pr_2 in measure_profiles} for pr in on_profiles})
        train_rows.update({f'learned_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}_result_{tuple(float(f"{t:0.2f}") for t in learned_pr)}': {pr_2: (0,0) for pr_2 in measure_profiles} for learned_pr, target_pr in learned_profiles})
        if plot_learned_profiles_as_probabilities:
            train_rows.update({f'learned_pr_probability_{tuple(float(f"{t:0.2f}") for t in target_pr)}_result_{tuple(float(f"{t:0.2f}") for t in learned_pr)}': {pr_2: (0,0) for pr_2 in measure_profiles} for learned_pr, target_pr in learned_profiles})
        

        test_rows = deepcopy(train_rows)
        box_plots_train = deepcopy(train_rows)
        box_plots_test = deepcopy(test_rows)


        similarity_columns = ['Learned Profile', 'Target Profile', ]
        for pr in measure_profiles:
            name = f'Similarity in terms of {str.capitalize(BASIC_PROFILE_NAMES.get(pr, tuple(float(f"{t:0.2f}") for t in pr)))}'
            similarity_columns.append(name)
        for pr in measure_profiles:
            name = f'Similarity with {str.capitalize(BASIC_PROFILE_NAMES.get(pr, tuple(float(f"{t:0.2f}") for t in pr)))} expert'
            similarity_columns.append(name)
        similarity_columns.append('Similarity with target profile agent')
        similarity_columns.append('Similarity with target profile society')
        
        similarity_metrics = ['agou', 'jaccard', 'visitation_count']
        similarity_data_train = {m: [] for m in similarity_metrics}

        similarity_data_test = deepcopy(similarity_data_train)


        for learned_pr, target_pr in learned_profiles:
        
            #learned_pr = self.get_reward_net().get_learned_profile()
            self.adapt_policy_to_profile(learned_pr, use_cached_policies=True, on_destinations=destinations_all)
            #assert self._reward_net.cur_profile == learned_pr
            # Profile learned.
            sampler_learned_profile: SimplePolicy = SimplePolicy.from_sb3_policy(policy=self.policy, real_env=self.env)
            learned_profile_trajs_train = sampler_learned_profile.sample_trajectories(stochastic=stochastic_sampling, repeat_per_od=n_samples_per_od, with_profiles=[learned_pr,], od_list=self.od_list_train)
            learned_profile_trajs_test = sampler_learned_profile.sample_trajectories(stochastic=stochastic_sampling, repeat_per_od=n_samples_per_od, with_profiles=[learned_pr,], od_list=od_list_test_new)
            learned_profile_as_probs_trajs_train = []
            learned_profile_as_probs_trajs_test = []
            society_trajs_train = []
            society_trajs_test = []


            expert_trajs_train = expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=n_samples_per_od, with_profiles=[target_pr,], od_list=self.od_list_train)
            assert len(expert_trajs_train) > 0
            
            expert_trajs_test = expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=n_samples_per_od, with_profiles=[target_pr,], od_list=od_list_test_new)
            assert len(expert_trajs_test) > 0

            for _ in range(repeat_society):
                od_per_pr_train= self.divide_od_list_as_per_profile(self.od_list_train, target_pr)
                od_per_pr_test = self.divide_od_list_as_per_profile(od_list_test_new, target_pr)
                for bpr in od_per_pr_test.keys():
                    #society_profile_trajs_train.extend(expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=n_samples_per_od, with_profiles=[pr,], od_list=od_per_pr_train[pr]))
                    #society_profile_trajs_test.extend(expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=n_samples_per_od, with_profiles=[pr,], od_list=od_per_pr_test[pr]))
                    society_trajs_train.extend(expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=n_samples_per_od, with_profiles=[bpr,], od_list=od_per_pr_train[bpr]))
                    society_trajs_test.extend(expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=n_samples_per_od, with_profiles=[bpr,], od_list=od_per_pr_test[bpr]))
                    
                if plot_learned_profiles_as_probabilities:
                    for bpr in od_per_pr_test.keys():    
                        self.adapt_policy_to_profile(bpr, use_cached_policies=True, on_destinations=destinations_all)
                        sampler: SimplePolicy = SimplePolicy.from_sb3_policy(policy=self.policy, real_env=self.env)
                    
                        learned_profile_as_probs_trajs_train.extend(sampler.sample_trajectories(stochastic=stochastic_sampling, repeat_per_od=n_samples_per_od, with_profiles=[bpr,], od_list=od_per_pr_train[bpr]))
                        learned_profile_as_probs_trajs_test.extend(sampler.sample_trajectories(stochastic=stochastic_sampling, repeat_per_od=n_samples_per_od, with_profiles=[bpr,], od_list=od_per_pr_test[bpr]))
                        
            #costs_learned_and_expert_train, agou_learned_and_expert_train_target, jensen_learned_and_expert_train_target, jensen_learned_and_expert_train_per_pr = calculate_expected_similarities_and_std(self.env.cost_model, profile_for_cost_model=target_pr, 
            costs_learned_and_expert_train, agou_learned_and_expert_train_target = calculate_expected_similarities_and_std(self.env.cost_model, profile_for_cost_model=target_pr, 
                                                                                                                                                                                                               sample_trajs=learned_profile_trajs_train, 
                                                                                                                                                                                                           expert_trajs=expert_trajs_train, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing)
            #costs_learned_and_expert_test, agou_learned_and_expert_test_target, jensen_learned_and_expert_test_target, jensen_learned_and_expert_test_per_pr = calculate_expected_similarities_and_std(self.env.cost_model,profile_for_cost_model=target_pr, sample_trajs=learned_profile_trajs_test, expert_trajs=expert_trajs_test, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing)
            costs_learned_and_expert_test, agou_learned_and_expert_test_target = calculate_expected_similarities_and_std(self.env.cost_model,profile_for_cost_model=target_pr, sample_trajs=learned_profile_trajs_test, expert_trajs=expert_trajs_test, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing)
            
            """costs_learned_and_society_train, agou_learned_and_society_train_target, jensen_learned_and_society_train_target, jensen_learned_and_society_train_per_pr = calculate_expected_similarities_and_std(self.env.cost_model, profile_for_cost_model=target_pr, 
                                                                                                                                                                                                               sample_trajs=learned_profile_trajs_train, 
                                                                                                                                                                                                               expert_trajs=society_trajs_train, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing)
            """
            
            costs_learned_and_society_train, agou_learned_and_society_train_target = calculate_expected_similarities_and_std(self.env.cost_model, profile_for_cost_model=target_pr, 
                                                                                                                                                                                                               sample_trajs=learned_profile_trajs_train, 
                                                                                                                                                                                                               expert_trajs=society_trajs_train, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing)
            """costs_learned_and_society_test, agou_learned_and_society_test_target, jensen_learned_and_society_test_target, jensen_learned_and_society_test_per_pr = calculate_expected_similarities_and_std(self.env.cost_model, profile_for_cost_model=target_pr,
                                                                                                                                                                                                           sample_trajs=learned_profile_trajs_test, 
                                                                                                                                                                                                           expert_trajs=society_trajs_test, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing)
            """
            costs_learned_and_society_test, agou_learned_and_society_test_target = calculate_expected_similarities_and_std(self.env.cost_model, profile_for_cost_model=target_pr,
                                                                                                                                                                                                           sample_trajs=learned_profile_trajs_test, 
                                                                                                                                                                                                           expert_trajs=society_trajs_test, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing)
            
            jaccard_per_pr_expert_train, jaccard_expert_train_normal, jaccard_expert_train_costs, _ = jaccard_similarity(self.env.cost_model, profile_for_cost_model=target_pr, sample_trajs=learned_profile_trajs_train, expert_trajs=expert_trajs_train, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing,return_per_od=True)
            jaccard_per_pr_expert_test, jaccard_expert_test_normal, jaccard_expert_test_costs, _ = jaccard_similarity(self.env.cost_model, profile_for_cost_model=target_pr, sample_trajs=learned_profile_trajs_test, expert_trajs=expert_trajs_test, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing, return_per_od=True)
            
            jaccard_per_pr_society_train, jaccard_society_train_normal, jaccard_society_train_costs, _ = jaccard_similarity(self.env.cost_model, profile_for_cost_model=target_pr, sample_trajs=learned_profile_trajs_train, expert_trajs=society_trajs_train, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing, return_per_od=True)
            jaccard_per_pr_society_test, jaccard_society_test_normal, jaccard_society_test_costs, _ = jaccard_similarity(self.env.cost_model, profile_for_cost_model=target_pr, sample_trajs=learned_profile_trajs_test, expert_trajs=society_trajs_test, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing, return_per_od=True)
            
            
            demo_oms_learned_profile_trajs = get_demo_oms_from_trajectories(learned_profile_trajs_train,state_dim=self.env.state_dim, groupby='od')
            demo_oms_expert_trajs = get_demo_oms_from_trajectories(expert_trajs_train, state_dim=self.env.state_dim,groupby='od')
            demo_oms_society_trajs = get_demo_oms_from_trajectories(society_trajs_train, state_dim=self.env.state_dim,groupby='od')
            keys_vc = demo_oms_learned_profile_trajs.keys()
            
            visitation_diff_expert_train = np.sum([np.abs(demo_oms_learned_profile_trajs[od] - demo_oms_expert_trajs[od]) for od in keys_vc],axis=1)
            visitation_diff_society_train = np.sum([np.abs(demo_oms_learned_profile_trajs[od] - demo_oms_society_trajs[od]) for od in keys_vc],axis=1)
            
            demo_oms_learned_profile_trajs = get_demo_oms_from_trajectories(learned_profile_trajs_test,state_dim=self.env.state_dim, groupby='od')
            demo_oms_expert_trajs = get_demo_oms_from_trajectories(expert_trajs_test, state_dim=self.env.state_dim,groupby='od')
            demo_oms_society_trajs = get_demo_oms_from_trajectories(society_trajs_test, state_dim=self.env.state_dim,groupby='od')
            keys_vc = demo_oms_learned_profile_trajs.keys()
            visitation_diff_expert_test = np.sum([np.abs(demo_oms_learned_profile_trajs[od] - demo_oms_expert_trajs[od]) for od in keys_vc], axis=1)
            visitation_diff_society_test = np.sum([np.abs(demo_oms_learned_profile_trajs[od] - demo_oms_society_trajs[od]) for od in keys_vc], axis=1)
            

            similarity_agou_now_train = {'Learned Profile': learned_pr, 'Target Profile': target_pr,}
            similarity_agou_now_test = {'Learned Profile': learned_pr, 'Target Profile': target_pr,}
            similarity_agou_now_train['Similarity with target profile agent'] = np.mean(agou_learned_and_expert_train_target),np.std(agou_learned_and_expert_train_target)
            similarity_agou_now_test['Similarity with target profile agent'] = np.mean(agou_learned_and_expert_test_target),np.std(agou_learned_and_expert_test_target)
            similarity_agou_now_train['Similarity with target profile society'] = np.mean(agou_learned_and_society_train_target),np.std(agou_learned_and_society_train_target)
            similarity_agou_now_test['Similarity with target profile society'] = np.mean(agou_learned_and_society_test_target),np.std(agou_learned_and_society_test_target)

            similarity_vc_now_train = {'Learned Profile': learned_pr, 'Target Profile': target_pr,}
            similarity_vc_now_test = {'Learned Profile': learned_pr, 'Target Profile': target_pr,}
            similarity_vc_now_train['Similarity with target profile agent'] = np.mean(visitation_diff_expert_train), np.std(visitation_diff_expert_train)
            similarity_vc_now_test['Similarity with target profile agent'] = np.mean(visitation_diff_expert_test), np.std(visitation_diff_expert_test)
            similarity_vc_now_train['Similarity with target profile society'] = np.mean(visitation_diff_society_train), np.std(visitation_diff_society_train)
            similarity_vc_now_test['Similarity with target profile society'] = np.mean(visitation_diff_society_test), np.std(visitation_diff_society_test)

            similarity_jaccard_now_train = {'Learned Profile': learned_pr, 'Target Profile': target_pr,}
            similarity_jaccard_now_test = {'Learned Profile': learned_pr, 'Target Profile': target_pr,}
            similarity_jaccard_now_train['Similarity with target profile agent'] = np.mean(jaccard_expert_train_costs), np.std(jaccard_expert_train_costs)
            similarity_jaccard_now_test['Similarity with target profile agent'] = np.mean(jaccard_expert_test_costs), np.std(jaccard_expert_test_costs)
            similarity_jaccard_now_train['Similarity with target profile society'] = np.mean(jaccard_society_train_costs), np.std(jaccard_society_train_costs)
            similarity_jaccard_now_test['Similarity with target profile society'] = np.mean(jaccard_society_test_costs), np.std(jaccard_society_test_costs)



            for ipr_2, pr_2 in enumerate(measure_profiles):
                assert np.allclose(costs_learned_and_expert_train['sample'][:,ipr_2], np.dot(costs_learned_and_expert_train['sample'], np.asarray(pr_2)))
                #jensen_pr_2_train, agou_pr_2_train = np.dot(jensen_learned_and_expert_train_per_pr, pr_2), profiled_aggregated_similarity_Agourogiannis_et_al_2023(costs_learned_and_expert_train['sample'], costs_learned_and_expert_train['expert'], preference_weights=pr_2)
                #jensen_pr_2_test, agou_pr_2_test = np.dot(jensen_learned_and_expert_test_per_pr, pr_2), profiled_aggregated_similarity_Agourogiannis_et_al_2023(costs_learned_and_expert_test['sample'], costs_learned_and_expert_test['expert'], preference_weights=pr_2)
                agou_pr_2_train =profiled_aggregated_similarity_Agourogiannis_et_al_2023(costs_learned_and_expert_train['sample'], costs_learned_and_expert_train['expert'], preference_weights=pr_2)
                agou_pr_2_test = profiled_aggregated_similarity_Agourogiannis_et_al_2023(costs_learned_and_expert_test['sample'], costs_learned_and_expert_test['expert'], preference_weights=pr_2)
                
                jaccard_pr_2_train = jaccard_per_pr_expert_train[pr_2]
                jaccard_pr_2_test = jaccard_per_pr_expert_test[pr_2]
            


                similarity_agou_now_train[f'Similarity in terms of {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))}'] = np.mean(agou_pr_2_train),np.std(agou_pr_2_train)
                similarity_agou_now_test[f'Similarity in terms of {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))}'] = np.mean(agou_pr_2_test),np.std(agou_pr_2_test)
                
                #similarity_vc_now_train[f'Similarity in terms of {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))}'] = np.ma
                #similarity_vc_now_test[f'Similarity in terms of {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))}'] = jensen_pr_2_test
               
                similarity_jaccard_now_train[f'Similarity in terms of {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))}'] = np.mean(jaccard_pr_2_train),np.std(jaccard_pr_2_train)
                similarity_jaccard_now_test[f'Similarity in terms of {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))}'] =  np.mean(jaccard_pr_2_test),np.std(jaccard_pr_2_test)
                
                #self.adapt_policy_to_profile(pr_2, use_cached_policies=True, on_destinations=destinations_all)
                if pr_2 in BASIC_PROFILES:
                    expert_mpr_train = expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=n_samples_per_od, with_profiles=[pr_2,], od_list=self.od_list_train)
                    expert_mpr_test = expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=n_samples_per_od, with_profiles=[pr_2,], od_list=od_list_test_new)
                    
                    costs_mpr_train, agou_mpr_train = calculate_expected_similarities_and_std(self.env.cost_model, profile_for_cost_model=target_pr, sample_trajs=learned_profile_trajs_train, expert_trajs=expert_mpr_train, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing)
                    costs_mpr_test, agou_mpr_test = calculate_expected_similarities_and_std(self.env.cost_model, profile_for_cost_model=target_pr,
                                                                                                                                                                                                            sample_trajs=learned_profile_trajs_test, 
                                                                                                                                                                                                            expert_trajs=expert_mpr_test, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing)
                
                    agou_mpr_train =profiled_aggregated_similarity_Agourogiannis_et_al_2023(costs_mpr_train['sample'], costs_mpr_train['expert'], preference_weights=pr_2)
                    agou_mpr_test = profiled_aggregated_similarity_Agourogiannis_et_al_2023(costs_mpr_test['sample'], costs_mpr_test['expert'], preference_weights=pr_2)
                    similarity_agou_now_train[f'Similarity with {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))} expert'] = (np.mean(agou_mpr_train), np.std(agou_mpr_train))
                    similarity_agou_now_test[f'Similarity with {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))} expert'] = (np.mean(agou_mpr_test), np.std(agou_mpr_test))
                    

                    demo_oms_learned_profile_trajs_test = get_demo_oms_from_trajectories(learned_profile_trajs_test,state_dim=self.env.state_dim, groupby='od')
                    demo_oms_learned_profile_trajs_train = get_demo_oms_from_trajectories(learned_profile_trajs_train,state_dim=self.env.state_dim, groupby='od')
                    
                    demo_oms_expert_trajs_train = get_demo_oms_from_trajectories(expert_mpr_train, state_dim=self.env.state_dim,groupby='od')
                    demo_oms_expert_trajs_test = get_demo_oms_from_trajectories(expert_mpr_test, state_dim=self.env.state_dim,groupby='od')
                    keys_vc_train = demo_oms_learned_profile_trajs_train.keys()
                    keys_vc_test = demo_oms_learned_profile_trajs_test.keys()
                    visitation_diff_expert_test = np.sum([np.abs(demo_oms_learned_profile_trajs_test[od] - demo_oms_expert_trajs_test[od]) for od in keys_vc_test],axis=1)
                    visitation_diff_expert_train = np.sum([np.abs(demo_oms_learned_profile_trajs_train[od] - demo_oms_expert_trajs_train[od]) for od in keys_vc_train],axis=1)
                    
                    
                    similarity_vc_now_train[f'Similarity with {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))} expert'] = (np.mean(visitation_diff_expert_train), np.std(visitation_diff_expert_train))
                    similarity_vc_now_test[f'Similarity with {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))} expert'] = (np.mean(visitation_diff_expert_test), np.std(visitation_diff_expert_test))
                    
                    jaccard_per_pr_expert, _, jaccard_expert_cost_train, _= jaccard_similarity(self.env.cost_model, profile_for_cost_model=pr_2, sample_trajs=learned_profile_trajs_train, expert_trajs=expert_mpr_train, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing, return_per_od=True)
                    jaccard_per_pr_expert, _, jaccard_expert_cost_test, _= jaccard_similarity(self.env.cost_model, profile_for_cost_model=pr_2, sample_trajs=learned_profile_trajs_test, expert_trajs=expert_mpr_test, profile_criteria=costs_measured_with_profiles, preprocessing =custom_cost_preprocessing, return_per_od=True)
                    
                    similarity_jaccard_now_train[f'Similarity with {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))} expert'] = (np.mean(jaccard_expert_cost_train), np.std(jaccard_expert_cost_train))
                    similarity_jaccard_now_test[f'Similarity with {str.capitalize(BASIC_PROFILE_NAMES.get(pr_2, tuple(float(f"{t:0.2f}") for t in pr_2)))} expert'] = (np.mean(jaccard_expert_cost_test), np.std(jaccard_expert_cost_test))
                    
                avg_cost_learned_profile_trajs_train, std_learned_profile_trajs_train = np.mean(costs_learned_and_expert_train['sample'][:,ipr_2]), np.std(costs_learned_and_expert_train['sample'][:,ipr_2])
                avg_cost_learned_profile_trajs_test, std_learned_profile_trajs_test = np.mean(costs_learned_and_expert_test['sample'][:,ipr_2]), np.std(costs_learned_and_expert_test['sample'][:,ipr_2])

                """avg_cost_learned_profile_trajs_train_2, std_learned_profile_trajs_train_2, allcosts_lpttr = calculate_expected_cost_and_std(self.env.cost_model, profile_for_cost_model=pr_2 if pr_2 != 'societal_cost' else learned_pr, mode='societal_cost' if pr_2 == 'societal_cost' else 'cost_model', trajectories=learned_profile_trajs_train, preprocessing = cost_preprocessing,
                                                                                                                                        bins=bins, file_name=name_method + f'value_system_identification_profile_{learned_pr}', plot_histogram=plot_histograms)
                """
                #assert np.allclose(avg_cost_learned_profile_trajs_train, avg_cost_learned_profile_trajs_train_2)
                #assert np.allclose(std_learned_profile_trajs_train_2,std_learned_profile_trajs_train)

                """avg_cost_learned_profile_trajs_test, std_learned_profile_trajs_test , allcosts_lptte = calculate_expected_cost_and_std(self.env.cost_model, profile_for_cost_model=pr_2 if pr_2 != 'societal_cost' else learned_pr, mode='societal_cost' if pr_2 == 'societal_cost' else 'cost_model', trajectories=learned_profile_trajs_test, preprocessing = cost_preprocessing,
                                                                                                                    bins=bins, file_name=name_method + f'value_system_identification_profile_{learned_pr}_test', plot_histogram=plot_histograms)"""
                
                box_plots_train[f'learned_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}_result_{tuple(float(f"{t:0.2f}") for t in learned_pr)}'][pr_2] = costs_learned_and_expert_train['sample'][:,ipr_2]
                box_plots_test[f'learned_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}_result_{tuple(float(f"{t:0.2f}") for t in learned_pr)}'][pr_2] = costs_learned_and_expert_test['sample'][:,ipr_2]

                train_rows[f'learned_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}_result_{tuple(float(f"{t:0.2f}") for t in learned_pr)}'][pr_2] = (avg_cost_learned_profile_trajs_train, std_learned_profile_trajs_train)
                test_rows[f'learned_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}_result_{tuple(float(f"{t:0.2f}") for t in learned_pr)}'][pr_2] = (avg_cost_learned_profile_trajs_test, std_learned_profile_trajs_test)

                
                
                avg_train, std_train = np.mean(costs_learned_and_expert_train['expert'][:,ipr_2]), np.std(costs_learned_and_expert_train['expert'][:,ipr_2])
                avg_test, std_test = np.mean(costs_learned_and_expert_test['expert'][:,ipr_2]), np.std(costs_learned_and_expert_test['expert'][:,ipr_2])

                box_plots_train[f'expert_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}'][pr_2] = costs_learned_and_expert_train['expert'][:,ipr_2]
                box_plots_test[f'expert_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}'][pr_2] = costs_learned_and_expert_test['expert'][:,ipr_2]

                train_rows[f'expert_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}'][pr_2] = (avg_train, std_train)
                test_rows[f'expert_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}'][pr_2] = (avg_test, std_test)

                avg_train, std_train = np.mean(costs_learned_and_society_train['expert'][:,ipr_2]), np.std(costs_learned_and_society_train['expert'][:,ipr_2])
                avg_test, std_test = np.mean(costs_learned_and_society_test['expert'][:,ipr_2]), np.std(costs_learned_and_society_test['expert'][:,ipr_2])

                box_plots_train[f'society_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}'][pr_2] = costs_learned_and_society_train['expert'][:,ipr_2]
                box_plots_test[f'society_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}'][pr_2] = costs_learned_and_society_test['expert'][:,ipr_2]

                train_rows[f'society_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}'][pr_2] = (avg_train, std_train)
                test_rows[f'society_pr_{tuple(float(f"{t:0.2f}") for t in target_pr)}'][pr_2] = (avg_test, std_test)


                if plot_learned_profiles_as_probabilities:
                    avg_cost_learned_probs_profile_trajs_train, std_learned_probs_profile_trajs_train , allcosts_lpptr = calculate_expected_cost_and_std(self.env.cost_model, profile_for_cost_model=pr_2 if pr_2 != 'societal_cost' else pr, mode='societal_cost' if pr_2 == 'societal_cost' else 'cost_model', trajectories=learned_profile_as_probs_trajs_train, preprocessing = cost_preprocessing,
                                                                                                                    bins=bins, file_name=name_method + f'learned_profile_as_probs_{learned_pr}', plot_histogram=plot_histograms)
                    avg_cost_learned_probs_profile_trajs_test, std_learned_probs_profile_trajs_test , allcosts_lppte = calculate_expected_cost_and_std(self.env.cost_model, profile_for_cost_model=pr_2 if pr_2 != 'societal_cost' else pr, mode='societal_cost' if pr_2 == 'societal_cost' else 'cost_model', trajectories=learned_profile_as_probs_trajs_test, preprocessing = cost_preprocessing,
                                                                                                                        bins=bins, file_name=name_method + f'learned_profile_as_probs_{learned_pr}_test', plot_histogram=plot_histograms)
                    

                    train_rows[f'learned_pr_probability_{tuple(float(f"{t:0.2f}") for t in target_pr)}_result_{tuple(float(f"{t:0.2f}") for t in learned_pr)}'][pr_2] = (avg_cost_learned_probs_profile_trajs_train, std_learned_probs_profile_trajs_train)
                    test_rows[f'learned_pr_probability_{tuple(float(f"{t:0.2f}") for t in target_pr)}_result_{tuple(float(f"{t:0.2f}") for t in learned_pr)}'][pr_2] = (avg_cost_learned_probs_profile_trajs_test, std_learned_probs_profile_trajs_test)
                    box_plots_train[f'learned_pr_probability_{tuple(float(f"{t:0.2f}") for t in target_pr)}_result_{tuple(float(f"{t:0.2f}") for t in learned_pr)}'][pr_2] = allcosts_lpptr
                    box_plots_test[f'learned_pr_probability_{tuple(float(f"{t:0.2f}") for t in target_pr)}_result_{tuple(float(f"{t:0.2f}") for t in learned_pr)}'][pr_2] = allcosts_lppte
            
            # ENDED learned, target current
            similarity_data_train['agou'].append(similarity_agou_now_train)
            similarity_data_train['visitation_count'].append(similarity_vc_now_train)
            similarity_data_train['jaccard'].append(similarity_jaccard_now_train)

            similarity_data_test['agou'].append(similarity_agou_now_test)
            similarity_data_test['visitation_count'].append(similarity_vc_now_test)
            similarity_data_test['jaccard'].append(similarity_jaccard_now_test)
            
        
        # SOCIETY with target profile (proportions several times)

        for pr in measure_profiles:
            self.adapt_policy_to_profile(pr, use_cached_policies=True, on_destinations=destinations_all)
            sampler: SimplePolicy = SimplePolicy.from_sb3_policy(policy=self.policy, real_env=self.env)
            
            value_learned_trajs_train = sampler.sample_trajectories(stochastic=stochastic_sampling, repeat_per_od=n_samples_per_od, with_profiles=[pr,], od_list=self.od_list_train)
            
            value_learned_trajs_test = sampler.sample_trajectories(stochastic=stochastic_sampling, repeat_per_od=n_samples_per_od, with_profiles=[pr,], od_list=od_list_test_new)
            
            expert_trajs_train_pr = expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=n_samples_per_od, with_profiles=[pr,], od_list=self.od_list_train)
            expert_trajs_test_pr = expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=n_samples_per_od, with_profiles=[pr,], od_list=od_list_test_new)
            

                

            avg_cost_learned_trajs_train, std_learned_trajs_train , allcosts_lttr = calculate_expected_cost_and_std(self.env.cost_model, profile_for_cost_model=pr, trajectories=value_learned_trajs_train, preprocessing = cost_preprocessing,
                                                                                                    bins=bins, file_name=name_method + f'value_learning_with_profile_{pr}', plot_histogram=plot_histograms)
            
            avg_cost_learned_trajs_test, std_learned_trajs_test , allcosts_ltte = calculate_expected_cost_and_std(self.env.cost_model, profile_for_cost_model=pr, trajectories=value_learned_trajs_test, preprocessing = cost_preprocessing,
                                                                                                  bins=bins, file_name=name_method + f'value_learning_with_profile_{pr}_test', plot_histogram=plot_histograms)
            
            avg_cost_expert_trajs_train, std_expert_trajs_train , allcosts_ettr = calculate_expected_cost_and_std(self.env.cost_model, profile_for_cost_model=pr, trajectories=expert_trajs_train_pr, preprocessing = cost_preprocessing,
                                                                                                  bins=bins, file_name=name_method + f'optimal_expert_with_profile_{pr}', plot_histogram=plot_histograms)
            avg_cost_expert_trajs_test, std_expert_trajs_test , allcosts_ette = calculate_expected_cost_and_std(self.env.cost_model, profile_for_cost_model=pr, trajectories=expert_trajs_test_pr, preprocessing = cost_preprocessing,
                                                                                                bins=bins, file_name=name_method + f'optimal_expert_with_profile_{pr}_test', plot_histogram=plot_histograms)
            
            train_rows['expert'][pr] = (avg_cost_expert_trajs_train, std_expert_trajs_train)
            box_plots_train['expert'][pr] = allcosts_ettr

            train_rows['learned'][pr] = (avg_cost_learned_trajs_train, std_learned_trajs_train)
            box_plots_train['learned'][pr] = allcosts_lttr
            test_rows['expert'][pr] = (avg_cost_expert_trajs_test, std_expert_trajs_test)
            box_plots_test['expert'][pr] = allcosts_ette
            test_rows['learned'][pr] = (avg_cost_learned_trajs_test, std_learned_trajs_test)
            box_plots_test['learned'][pr] = allcosts_ltte

        
        df_data = []
        
        def get_key_name(key):
            
            keyname = None
            if 'expert_pr' in key:
                keyname = 'Expert ' + key.split('expert_pr_')[1]
            elif 'society_pr' in key:
                keyname = 'Society ' + key.split('society_pr_')[1]
            elif 'expert' in key:
                keyname = 'Exp + P'

            elif 'learned_pr' in key:
                keyname = 'Learning' + key.split('learned_pr_')[1]
            elif 'learned' in key:
                keyname = 'VL + P'
            else:
                print("UNKNOWN key", key)
                keyname = key
            return keyname
        


        plt.figure(figsize=(30, 10))
        labels = []
            
        for key in box_plots_train.keys():
            labels.append(get_key_name(key))
        
        xlabels = [re.sub("(.{32})", "\\1\n", label, 0, re.DOTALL) for label in labels]

        for i, pr in enumerate(measure_profiles):
            boxes = [box_plots_train[k][pr] for k in box_plots_train.keys()]
            positions = np.array(list((range(len(boxes))))) + i*1.0/len(measure_profiles)
            bp = plt.boxplot(boxes, positions=positions,patch_artist=True,widths=0.7*1.0/len(measure_profiles))
            for patch in bp['boxes']:
                #color2 = 'r' if color == 'red' else 'g' if color == 'green' else 'b' if color == 'blue' else 'w'
                patch.set_facecolor(PROFILE_COLORS.get(pr, 'w'))
                #patch.set_edgecolor(PROFILE_COLORS.get(pr, 'w'))
            if i == len(measure_profiles)-1:
                for p in positions:
                    plt.axvline(x=(p+ p +1.0/len(measure_profiles))/2, color='blue', linestyle='--')  # Highlighting the minimum value
                        
        title = f"Test set: Boxplot (cost distribution calculated with different measure_profiles)"
        plt.title(title)
        plt.xticks(np.asarray(range(len(box_plots_train))) + (len(measure_profiles)-1)/2.0/len(measure_profiles), labels=xlabels, rotation=75)
        plt.tight_layout()
        plt.savefig(f'results/value_system_identification/{name_method}_boxplot_train.pdf', bbox_inches="tight")
        #plt.show()
        plt.close()

        plt.figure(figsize=(30, 10))
        for i, pr in enumerate(measure_profiles):
            boxes = [box_plots_test[k][pr] for k in box_plots_test.keys()]

            bp = plt.boxplot(boxes, positions=np.array(list((range(len(boxes))))) + i*1.0/len(measure_profiles),patch_artist=True, widths=0.7*1.0/len(measure_profiles))
            for patch in bp['boxes']:
                #color2 = 'r' if color == 'red' else 'g' if color == 'green' else 'b' if color == 'blue' else 'w'
                patch.set_facecolor(PROFILE_COLORS.get(pr, 'w'))
                #patch.set_edgecolor(PROFILE_COLORS.get(pr, 'w'))
                if i == len(measure_profiles)-1:
                    for p in positions:
                        plt.axvline(x=(p+ p +1.0/len(measure_profiles))/2, color='blue', linestyle='--')  # Highlighting the minimum value
                
        title = f"Test set: Boxplot (cost distribution calculated with different measure_profiles)"
        plt.title(title)
        plt.xticks(np.asarray(range(len(box_plots_test))) + (len(measure_profiles)-1)/2.0/len(measure_profiles), labels=xlabels, rotation=75)
        plt.tight_layout()
        plt.savefig(f'results/value_system_identification/{name_method}_boxplot_test.pdf', bbox_inches="tight")
        #plt.show()
        plt.close()

        

        for iprk, prkey in enumerate(measure_profiles):
            boxplot_train_of_prkey = []
            boxplot_test_of_prkey = []
            labels = []
            for key in box_plots_train.keys():
                boxplot_train_of_prkey.append(list(box_plots_train[key][prkey]))
                boxplot_test_of_prkey.append(list(box_plots_test[key][prkey]))
                labels.append(get_key_name(key))

            
            xlabels = [re.sub("(.{32})", "\\1\n", label, 0, re.DOTALL) for label in labels]

            plt.figure(figsize=(14,10))
            plt.boxplot(boxplot_test_of_prkey)
            title = f"Test set: boxplot {name_method}_(cost distribution calculated as with {prkey})"
            plt.title(title)
            plt.xticks(list(i + 0.5 for i in range(len(xlabels))), labels=xlabels, rotation=75)
            plt.tight_layout()
            plt.savefig(f'results/value_system_identification/{name_method}_boxplot_test_costs_calculated_with_{prkey}.pdf', bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(14,10))
            plt.boxplot(boxplot_train_of_prkey)
            title = f"Train set: boxplot {name_method}_(cost distribution calculated as with {prkey})"
            plt.title(title)
            plt.xticks(list(i + 0.5 for i in range(len(xlabels))), labels=xlabels, rotation=75)
            plt.tight_layout()
            plt.savefig(f'results/value_system_identification/{name_method}_boxplot_train_costs_calculated_with_{prkey}.pdf', bbox_inches="tight")
            plt.close()
            
        for key in train_rows.keys():
            keyname = get_key_name(key)
            profile_data = {'Policy': keyname}
            profile_data.update({BASIC_PROFILE_NAMES.get(pr, tuple(float(f"{t:0.2f}") for t in pr)): train_rows[key][pr] for pr in measure_profiles})
            #profile_data.update({'Societal cost': train_rows[key]['societal_cost']})
            df_data.append(profile_data)

        df_train = pd.DataFrame(df_data)
        df_data = []
        for key in test_rows.keys():
            
            keyname = get_key_name(key)

            profile_data = {'Policy': keyname}
            profile_data.update({BASIC_PROFILE_NAMES.get(pr, tuple(float(f"{t:0.2f}") for t in pr)): test_rows[key][pr] for pr in measure_profiles})
            
            df_data.append(profile_data)
       
        df_test = pd.DataFrame(df_data)

        for metric in similarity_metrics:
            similarity_data_test[metric] = pd.DataFrame(data=similarity_data_test[metric], columns=similarity_columns)
            similarity_data_train[metric] = pd.DataFrame(data=similarity_data_train[metric], columns=similarity_columns)

        return df_train, df_test, train_rows, test_rows, similarity_data_train, similarity_data_test
    
    
    def feature_differences(self, sampled_trajs, obs_matrix, per_state=False, visitations=None, use_info_real_costs=True, sampled_traj_profile_to_expert_traj_profile_mapping={}):
        fd_per_traj = dict()
        fd_norm_per_traj = dict()

        obs_matrix_np = obs_matrix.detach().cpu().numpy()

        
        profile_mapping = sampled_traj_profile_to_expert_traj_profile_mapping

        for traj in sampled_trajs:
            
            pr = traj.infos[0]['profile']
            odpr_of_sampled_trajs = ((traj.infos[0]['orig'], traj.infos[0]['des']), pr)
            expert_pr = profile_mapping.get(pr, pr)
            corresponding_odpr_of_expert_trajs = ((traj.infos[0]['orig'], traj.infos[0]['des']), expert_pr)
            
            expert_trajs_of_corresponding_odpr = self.expert_trajectories_per_odpr[corresponding_odpr_of_expert_trajs]
            
            if per_state is False:
                fds = []
                fds_norm = []
                if use_info_real_costs is False:
                    traj_obs = np.asarray([obs_matrix_np[o[0],traj.infos[0]['des'],:] for o in traj.obs])
                else:
                    traj_obs = np.asarray([self.env.cost_model(expert_pr, normalization=FeaturePreprocess.NORMALIZATION)(traj.obs[i]) for i in range( len(traj))])
                        
                for expert_traj in expert_trajs_of_corresponding_odpr:
                    if use_info_real_costs is False:
                        expert_traj_obs = np.asarray([obs_matrix_np[o[0],expert_traj.infos[0]['des'],:] for o in expert_traj.obs])
                    else:
                        expert_traj_obs = np.asarray([self.env.cost_model(expert_pr, normalization=FeaturePreprocess.NORMALIZATION)(expert_traj.obs[i]) for i in range( len(expert_traj))])
                    
                    fd_diff = np.sum(traj_obs, axis=0) - np.sum(expert_traj_obs, axis=0)
                    fd_norm_per_expert = np.linalg.norm(fd_diff)
                    fd_per_expert = np.mean(fd_diff)

                    fds_norm.append(fd_norm_per_expert)

                    fds.append(fd_per_expert)
                fd_norm = np.max(fds_norm)
                fd = np.mean(fds)
                
            else:
                assert visitations is not None
                traj_obs = np.multiply(obs_matrix_np[:,traj.infos[0]['des'],:], visitations[odpr_of_sampled_trajs].reshape(-1, 1))
                expected_expert_traj_obs = np.zeros_like(obs_matrix_np[:,expert_trajs_of_corresponding_odpr[0].infos[0]['des'],:])
                for expert_traj in expert_trajs_of_corresponding_odpr:
                    expert_traj_obs = np.multiply(obs_matrix_np[:,expert_traj.infos[0]['des'],:], self.demo_state_om_per_profile[corresponding_odpr_of_expert_trajs].reshape(-1, 1))
                    expected_expert_traj_obs+=expert_traj_obs
                expected_expert_traj_obs/=len(expert_trajs_of_corresponding_odpr)

                fd_norm = np.linalg.norm(traj_obs - expected_expert_traj_obs, axis=-1)
                fd = np.mean(traj_obs - expected_expert_traj_obs, axis=-1)
                assert fd_norm.shape == (obs_matrix_np.shape[0],)
             
            fd_per_traj[corresponding_odpr_of_expert_trajs] = fd
            fd_norm_per_traj[corresponding_odpr_of_expert_trajs] = fd_norm
        return fd_per_traj, fd_norm_per_traj

    def calculate_rewards_for_destinations_and_profile(self, destinations, chosen_profile, obs_mat = None, dones= None):
        if obs_mat is None:
            obs_mat = self.env.observation_matrix
            obs_mat = th.as_tensor(
                obs_mat,
                dtype=self._reward_net.dtype,
                device=self._reward_net.device,
            )

        if dones is None:
            dones = th.as_tensor(self.env.done_matrix,dtype=self._reward_net.dtype)

        
        state_space_is_1d = isinstance(self.env.state_space, gym.spaces.Discrete)
        previous_rew_mode = self._reward_net.mode
        
        with th.no_grad():
            self._reward_net.set_mode(TrainingModes.VALUE_GROUNDING_LEARNING)
            self._reward_net.set_profile(chosen_profile)
            if state_space_is_1d:
                predicted_r = squeeze_r(self._reward_net(None, None, obs_mat, dones[:, destinations[0]]))
                
                assert predicted_r.shape == (obs_mat.shape[0],)
                predicted_r_np = predicted_r.detach().cpu().numpy()
            else:
                predicted_r_np = np.zeros_like(self.env.reward_matrix[list(self.env.reward_matrix.keys())[0]])
                for d in destinations:
                    predicted_r = squeeze_r(self._reward_net(None, None, obs_mat[:,d,:], dones[:, d]))
                    #predicted_r/= -th.max(predicted_r)
                    assert predicted_r.shape == (obs_mat.shape[0],)

                    predicted_r_np[:,d] = predicted_r.detach().cpu().numpy()
            assert np.all(predicted_r_np[:, destinations] < 0.0)

            self._reward_net.set_mode(previous_rew_mode)

        return predicted_r_np
    
    def _train_step_ponderated_profiles(self, obs_mat: th.Tensor, dones: th.Tensor = None, od_train_per_profile: dict = None, profile_ponderation: list = None, loss_weighting=1.0) -> Tuple[np.ndarray, np.ndarray]:
        
        th.autograd.set_detect_anomaly(False)
        
        learned_policy_profile  = self._reward_net.get_learned_profile()
        
        predicted_r_np = self.calculate_rewards_for_destinations_and_profile(self.destinations, learned_policy_profile, obs_mat=obs_mat, dones=dones)
        
        if self.use_mce_om:
            pi = self.mce_partition_fh(predicted_r_np, profile=learned_policy_profile) 
    
            _, visitations = self.mce_occupancy_measures(
                reward=predicted_r_np,
                pi=pi,
                profile=learned_policy_profile
            )
            sampler: SimplePolicy = SimplePolicy.from_policy_matrix(pi, real_env=self.env)
            sampled_trajs= sampler.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[learned_policy_profile,])
            
        else:
            if self.use_dijkstra:
                sampler: SimplePolicy = SimplePolicy.from_environment_expert(self.env, profiles=[learned_policy_profile, ], 
                                                custom_cost=lambda state_des, profile: (-predicted_r_np[state_des[0], state_des[1]]))
                
            else:
                pi = self.mce_partition_fh(predicted_r_np, profile=learned_policy_profile, 
                                           #od_list=self.od_list_train
                                           )
                sampler: SimplePolicy = SimplePolicy.from_policy_matrix(pi, real_env=self.env)
                
            sampled_trajs= sampler.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[learned_policy_profile,])
            
            visitations = get_demo_oms_from_trajectories(sampled_trajs, state_dim=self.env.state_dim)
            
            
        # Forward/back/step (grads are zeroed at the top).
        # weights_th(s) = \pi(s) - D(s)

        self.optimizer.zero_grad()  
        grad_norm = 0
        
        loss_total: th.Tensor = None
        mntensor_train_per_pr = dict()
        mntensor_test_per_pr = dict()
        rewards_per_pr = dict()
        
        for chosen_profile in BASIC_PROFILES:
            if len(od_train_per_profile[chosen_profile]) < 1:
                continue

            if self.training_mode != TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
                learned_policy_profile = chosen_profile
                self._reward_net.set_profile(learned_policy_profile)

            assert chosen_profile in self.expert_trajectories_per_pr.keys()

            sampled_trajs_train = []
            sampled_trajs_test = [] 

            rewards_per_pr[chosen_profile] = predicted_r_np
            for t in sampled_trajs:
                if (t.infos[0]['orig'], t.infos[0]['des']) in od_train_per_profile[chosen_profile]:
                    sampled_trajs_train.append(t)
                    #od_profiles_of_trajs_train.append((t.infos[0]['orig'], t.infos[0]['des']))
                if (t.infos[0]['orig'], t.infos[0]['des']) in self.od_list_test:
                    sampled_trajs_test.append(t)

            
            chosen_profile_expert_trajs_train = [t for t in self.expert_trajectories if tuple(t.infos[0]['profile']) == tuple(chosen_profile) and (t.infos[0]['orig'], t.infos[0]['des']) in od_train_per_profile[chosen_profile]]
            od_profiles_of_trajs_train = [((t.infos[0]['orig'], t.infos[0]['des']), chosen_profile) for t in chosen_profile_expert_trajs_train]
            #chosen_profile_trajs_test = [t for t in chosen_profile_trajs if (t.infos[0]['orig'], t.infos[0]['des']) in self.od_list_test]
            #od_profiles_of_trajs_test = [((t.infos[0]['orig'], t.infos[0]['des']), chosen_profile) for t in chosen_profile_trajs_test]


            visitation_count_diffs = th.as_tensor(
                np.asarray([
                visitations[(odpr[0], learned_policy_profile)] - self.demo_state_om_per_profile[odpr] for odpr in od_profiles_of_trajs_train]),
                    dtype=self._reward_net.dtype,
                    device=self._reward_net.device,
                )
            fd_train, fd_normed_train = self.feature_differences(sampled_trajs_train, obs_mat, per_state=False, use_info_real_costs=True, 
                                                                    sampled_traj_profile_to_expert_traj_profile_mapping=
                                                                    {learned_policy_profile: chosen_profile})
            fd_normed_tensor_train = th.as_tensor(np.asarray([fd_normed_train[(od, chosen_profile)] for od in od_train_per_profile[chosen_profile]]), device=self._reward_net.device, dtype=self._reward_net.dtype)
            #fd_tensor_train = th.as_tensor(np.asarray([fd_train[odpr] for odpr in od_profiles_of_trajs_train]), device=self.reward_net.device, dtype=self.reward_net.dtype)
            mntensor_train_per_pr[chosen_profile] = (th.mean(fd_normed_tensor_train)*profile_ponderation[chosen_profile]).detach().numpy()

            fd_test, fd_normed_test = self.feature_differences(
                sampled_trajs_test, obs_mat, per_state=False, 
                use_info_real_costs=True, 
                sampled_traj_profile_to_expert_traj_profile_mapping={learned_policy_profile: chosen_profile})
            
            fd_normed_tensor_test = th.as_tensor(np.asarray([fd_normed_test[(od, chosen_profile)] for od in self.od_list_test]), device=self._reward_net.device, dtype=self._reward_net.dtype)
            #fd_tensor_test = th.as_tensor(np.asarray([fd_test[odpr] for odpr in od_profiles_of_trajs_test]), device=self.reward_net.device, dtype=self.reward_net.dtype)
            mntensor_test_per_pr[chosen_profile] = (th.mean(fd_normed_tensor_test)*profile_ponderation[chosen_profile]).detach().numpy()

            obs_matrix_all_trajs = th.vstack([obs_mat[:,t.infos[0]['des'],:] for t in chosen_profile_expert_trajs_train])
            done_matrix_all_trajs = th.vstack([dones[:, t.infos[0]['des']] for t in chosen_profile_expert_trajs_train])
            
            rewards =  self._reward_net(None, None, obs_matrix_all_trajs, done_matrix_all_trajs).reshape_as(visitation_count_diffs)
            
            assert visitation_count_diffs.shape == rewards.shape, f"D: {visitation_count_diffs.shape}, R: {rewards.shape}"
            #assert feature_expectation_diffs.shape == rewards.shape, f"D: {feature_expectation_diffs.shape}, R: {rewards.shape}"
            assert th.all(rewards < 0.0)
            #print(mntensor*rewards, (mntensor*rewards).shape)
            
            if self.fd_lambda > 0.0:
                visited_states_for_each_od = (
                        th.as_tensor(
                            np.asarray([visitations[(odpr[0], learned_policy_profile)] for odpr in od_profiles_of_trajs_train]),
                            device=self._reward_net.device, dtype=self._reward_net.dtype) > 0)
            
                fd_on_visited_states = th.mul(
                        visited_states_for_each_od, 
                        fd_normed_tensor_train.reshape(-1,1)
                        )
                losses = th.mul(
                        visitation_count_diffs + self.fd_lambda/(2*len(chosen_profile_expert_trajs_train))*fd_on_visited_states,
                    #squeeze_r(self.reward_net(obs_mat, None, None, dones[:, self.destinations[0]])) if state_space_is_1d else
                    rewards) 
            else:
                losses = th.mul(
                        visitation_count_diffs,
                    #squeeze_r(self.reward_net(obs_mat, None, None, dones[:, self.destinations[0]])) if state_space_is_1d else
                    rewards)
            #sum_of_all_lengths = np.sum([len(t) for t in chosen_profile_trajs_train])
            #weights = th.tensor([len(t)/sum_of_all_lengths for t in self.expert_trajectories], dtype=losses.dtype, requires_grad=False)
            
            #losses = th.matmul(fd_tensor, losses)
            # This alost works but vanishes? loss = th.sum(losses)/len(chosen_profile_trajs_train)

            

            with th.no_grad():
                wheres = th.where(losses != 0.0)
            if wheres[0].shape[0] <= 1:
                loss = th.sum(losses)
            else:
                #loss = th.sum(losses)/len(chosen_profile_expert_trajs_train)
                loss = th.sum(losses[wheres])/len(set(wheres[0].tolist()))#/len([odpr for odpr in od_profiles_of_trajs_train if losses[:,odpr[0][1]]]) #/len([t for t in self.expert_trajectories if tuple(t.infos[0]['profile']) == tuple(chosen_profile)])
            if loss_total is None:
                loss_total = loss*profile_ponderation[chosen_profile]
            else:
                loss_total += loss*profile_ponderation[chosen_profile]

        loss_total.backward(retain_graph=False)
        self.optimizer.step()
            
        grads = []
        for p in self._reward_net.parameters():
            assert p.grad is not None  # for type checker
            grads.append(p.grad)
            
            print("GRAD: ", p.names, p.grad)
        print("REAL LOSS:", loss)
        print("REWARD NORM:", rewards.norm())
        # TODO: Loss should be 0 when the profile has converged (?) ... or at least the gradient. Still it converges without problem though shakingly
        grad_norm += util.tensor_iter_norm(grads).item()
        
        grad_norm_per_pr = collections.defaultdict(lambda: grad_norm)

        return rewards_per_pr, visitations, grad_norm_per_pr, mntensor_train_per_pr, mntensor_test_per_pr, learned_policy_profile


    def _train_step_sampling_trajectories(self, obs_mat: th.Tensor, dones: th.Tensor = None, society_profiles: list = [], society_profile_distribution: np.ndarray = np.array([]), batch_size = 100) -> Tuple[np.ndarray, np.ndarray]:
        
        th.autograd.set_detect_anomaly(False)
        with th.no_grad():
            learned_policy_profile  = self._reward_net.get_learned_profile()
            
            predicted_r_np = self.calculate_rewards_for_destinations_and_profile(self.destinations, learned_policy_profile, obs_mat=obs_mat, dones=dones)
            pure_rewards_per_pure_pr = dict()
            for pr in BASIC_PROFILES:
                pure_rewards_per_pure_pr[pr] = self.calculate_rewards_for_destinations_and_profile(self.destinations, pr, obs_mat=obs_mat, dones=dones)
            
            if self.use_mce_om:
                pi = self.mce_partition_fh(predicted_r_np, profile=learned_policy_profile) 
        
                _, visitations = self.mce_occupancy_measures(
                    reward=predicted_r_np,
                    pi=pi,
                    profile=learned_policy_profile
                )
                sampler: SimplePolicy = SimplePolicy.from_policy_matrix(pi, real_env=self.env)
                #sampled_trajs= sampler.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[learned_policy_profile,])
                
            else:
                if self.use_dijkstra:
                    sampler: SimplePolicy = SimplePolicy.from_environment_expert(self.env, profiles=[learned_policy_profile, ], 
                                                    custom_cost=lambda state_des, profile: (-predicted_r_np[state_des[0], state_des[1]]))
                    
                else:
                    pi = self.mce_partition_fh(predicted_r_np, profile=learned_policy_profile, 
                                            #od_list=self.od_list_train
                                            )
                    sampler: SimplePolicy = SimplePolicy.from_policy_matrix(pi, real_env=self.env)
                    
                #sampled_trajs= sampler.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[learned_policy_profile,])
                
                #visitations = get_demo_oms_from_trajectories(sampled_trajs, state_dim=self.env.state_dim)
                
                
        # Forward/back/step (grads are zeroed at the top).
        # weights_th(s) = \pi(s) - D(s)

        
            mntensor_train_per_pr = dict()
            mntensor_test_per_pr = dict()
            rewards_per_pr = dict()
            
            od_list_array = np.array(self.od_list_train)
            od_list_array_test = np.array(self.od_list_test)
            society_array = np.asarray(society_profiles)

                
            ods_index = np.random.choice(len(self.od_list_train), size=batch_size, replace=True)
            ods = [(od[0], od[1]) for od in od_list_array[ods_index].tolist()]
            profiles_index = np.random.choice(len(society_profiles), size=batch_size, p=society_profile_distribution, replace=True)
            
            profiles = [tuple(pr) for pr in society_array[profiles_index].tolist()]

            ods_index_test = np.random.choice(len(self.od_list_test), size=batch_size, replace=True)
            ods_test = [(od[0], od[1]) for od in od_list_array_test[ods_index_test].tolist()]
            profiles_index_test = np.random.choice(len(society_profiles), size=batch_size, p=society_profile_distribution, replace=True)
            
            profiles_test = [tuple(pr) for pr in society_array[profiles_index_test].tolist()]

            assert len(profiles) == len(ods)
            learned_policy_trajs = []
            expert_policy_trajs = []

            learned_policy_trajs_test = []
            expert_policy_trajs_test = []

            visitations = dict()
            expert_visitations = dict()
            for od, pr in zip(ods, profiles):
                learned_policy_trajs.extend(sampler.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[learned_policy_profile,], od_list=[od,]))
                
                expert_policy_trajs.extend(self.expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[pr,], od_list=[od,]))
            
            for od, pr in zip(ods_test, profiles_test):
                learned_policy_trajs_test.extend(sampler.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[learned_policy_profile,], od_list=[od,]))
                
                expert_policy_trajs_test.extend(self.expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[pr,], od_list=[od,]))
             
                 
            sampled_trajs_train = sampler.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[learned_policy_profile,], od_list=self.od_list_train)
            sampled_trajs_test = sampler.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[learned_policy_profile,], od_list=self.od_list_test)

            all_trajs = deepcopy(sampled_trajs_train)
            all_trajs.extend(sampled_trajs_test)

            visitations = get_demo_oms_from_trajectories(learned_policy_trajs, state_dim=self.env.state_dim)
            all_visitations = get_demo_oms_from_trajectories(all_trajs, state_dim= self.env.state_dim)
            expert_visitations = get_demo_oms_from_trajectories(expert_policy_trajs, state_dim=self.env.state_dim)

            
            costs_train, _, ods_agou= calculate_expected_similarities_and_std(cost_model_from_reward_net(self._reward_net, env=self.env, precalculated_rewards_per_pure_pr=pure_rewards_per_pure_pr), sample_trajs=learned_policy_trajs, expert_trajs=expert_policy_trajs, profile_for_cost_model=(1.0,1.0,1.0), profile_criteria=BASIC_PROFILES,preprocessing=FeaturePreprocess.NORMALIZATION,return_ods=True)    
            costs_test, _, ods_agou_test= calculate_expected_similarities_and_std(cost_model_from_reward_net(self._reward_net, env=self.env,precalculated_rewards_per_pure_pr=pure_rewards_per_pure_pr), sample_trajs=learned_policy_trajs_test, expert_trajs=expert_policy_trajs_test, profile_for_cost_model=(1.0,1.0,1.0), profile_criteria=BASIC_PROFILES,preprocessing=FeaturePreprocess.NORMALIZATION,return_ods=True)    
            
            for chosen_profile in set(bpr for i, bpr in enumerate(BASIC_PROFILES) if society_profile_distribution[i] > 0.0):
                mntensor_train_per_pr[chosen_profile] = profiled_aggregated_similarity_Agourogiannis_et_al_2023(costs_train['sample'], costs_train['expert'], preference_weights=chosen_profile)
                mntensor_test_per_pr[chosen_profile] = profiled_aggregated_similarity_Agourogiannis_et_al_2023(costs_test['sample'], costs_test['expert'], preference_weights=chosen_profile)

                rewards_per_pr[chosen_profile] = predicted_r_np
        
        self.optimizer.zero_grad()  
        grad_norm = 0
        ctrainsamp = th.as_tensor(costs_train['sample'],device=self._reward_net.device, dtype=self._reward_net.dtype)
        ctrainexp = th.as_tensor(costs_train['expert'],device=self._reward_net.device, dtype=self._reward_net.dtype)
        ctestsamp = th.as_tensor(costs_test['sample'],device=self._reward_net.device, dtype=self._reward_net.dtype)
        ctestexp = th.as_tensor(costs_test['expert'],device=self._reward_net.device, dtype=self._reward_net.dtype)
        
        agou_tr = 1.0- th.abs(1.0- 1.0/(-self._reward_net.trained_profile_net(- th.maximum(ctrainsamp, ctrainexp)/th.minimum(ctrainsamp, ctrainexp)))) #profiled_aggregated_similarity_Agourogiannis_et_al_2023(, )
        assert th.all(agou_tr >= 0.0)
        with th.no_grad():
            agou_test = 1.0- th.abs(1.0 - 1/(- self._reward_net.trained_profile_net(- th.maximum(ctestsamp, ctestexp)/th.minimum(ctestsamp, ctestexp)))) #profiled_aggregated_similarity_Agourogiannis_et_al_2023(, )
            assert th.all(agou_test >= 0.0)
            mntensor_train_per_pr['learned'] = agou_tr.detach().numpy()
            mntensor_test_per_pr['learned'] = agou_test.detach().numpy()
            ods_weights= ods_agou
            jaccard_per_pr_expert_train, jaccard_expert_train_normal, jaccard_tr, ods_jaccard_tr = jaccard_similarity(cost_model_from_reward_net(self._reward_net, env=self.env, precalculated_rewards_per_pure_pr=pure_rewards_per_pure_pr), profile_for_cost_model=learned_policy_profile, sample_trajs=learned_policy_trajs, expert_trajs=expert_policy_trajs, profile_criteria=set(profiles), preprocessing =FeaturePreprocess.NORMALIZATION,return_per_od=True)
            jaccard_per_pr_expert_train, jaccard_expert_train_normal, jaccard_tst, ods_jaccard_tr = jaccard_similarity(cost_model_from_reward_net(self._reward_net, env=self.env, precalculated_rewards_per_pure_pr=pure_rewards_per_pure_pr), profile_for_cost_model=learned_policy_profile, sample_trajs=learned_policy_trajs_test, expert_trajs=expert_policy_trajs_test, profile_criteria=set(profiles), preprocessing =FeaturePreprocess.NORMALIZATION,return_per_od=True)
        


            jaccard_tr = th.minimum(th.as_tensor(jaccard_tr), th.tensor(1.0))
            jaccard_tst = th.minimum(th.as_tensor(jaccard_tst), th.tensor(1.0))
        
        
        if self.fd_lambda != 0.0:
            agou_tr = th.minimum(agou_tr,  th.tensor(1.0, requires_grad=True))
            
            COST_PREFERENCE = 3.0
            #OVERLAP_REFERENCE = 1.0
            #OVERLAP_USE = 0.0
            # TODO: Investigate using similarities to ponder bigger or shorter updates
            weights = th.float_power((1.0-agou_tr), 1.0/COST_PREFERENCE) #* OVERLAP_USE*th.float_power((1.0-jaccard_tr), 1.0/OVERLAP_REFERENCE)
        else:
            weights = th.ones_like(agou_tr)
        assert th.all(weights >= 0.0)
        
        visitation_count_diffs = th.vstack([
            weights[ods_weights.index(od)]*th.as_tensor(visitations[(od, learned_policy_profile)] - expert_visitations[(od,pr)], dtype=self._reward_net.dtype,
                device=self._reward_net.device) for od, pr in zip(ods, profiles)],
            )
        obs_matrix_all_trajs = th.vstack([obs_mat[:,od[1],:] for od in ods])
        done_matrix_all_trajs = th.vstack([dones[:,od[1]] for od in ods])
        
        rewards =  self._reward_net(None, None, obs_matrix_all_trajs, done_matrix_all_trajs).reshape_as(visitation_count_diffs)
        
    
        assert visitation_count_diffs.shape == rewards.shape, f"D: {visitation_count_diffs.shape}, R: {rewards.shape}"
        #assert feature_expectation_diffs.shape == rewards.shape, f"D: {feature_expectation_diffs.shape}, R: {rewards.shape}"
        assert th.all(rewards < 0.0)
        #print(mntensor*rewards, (mntensor*rewards).shape)
        losses = th.mul(
                    visitation_count_diffs,
                #squeeze_r(self.reward_net(obs_mat, None, None, dones[:, self.destinations[0]])) if state_space_is_1d else
                rewards)

        

        loss = th.sum(losses)/batch_size

        loss.backward(retain_graph=False)
        self.optimizer.step()
        
        with th.no_grad():
            grads = []
            for p in self._reward_net.parameters():
                assert p.grad is not None  # for type checker
                grads.append(p.grad)
                
                print("GRAD: ", p.names, p.grad)
            print("REAL LOSS:", loss)
            print("REWARD NORM:", rewards.norm())
            # TODO: Loss should be 0 when the profile has converged (?) ... or at least the gradient. Still it converges without problem though shakingly in learning from societies
            grad_norm += util.tensor_iter_norm(grads).item()
            
            grad_norm_per_pr = collections.defaultdict(lambda: grad_norm)

            visitations_test = get_demo_oms_from_trajectories(learned_policy_trajs_test, state_dim=self.env.state_dim)
            expert_visitations_test = get_demo_oms_from_trajectories(expert_policy_trajs_test, state_dim=self.env.state_dim)
            
            visitation_count_diffs_abs_train = np.sum([np.abs(visitations[(od, learned_policy_profile)] - expert_visitations[(od,pr)]) for od, pr in zip(ods, profiles)], axis=1)
            visitation_count_diffs_abs_test = np.sum([np.abs(visitations_test[(od, learned_policy_profile)] - expert_visitations_test[(od,pr)]) for od, pr in zip(ods_test, profiles_test)], axis=1)
            
            
        return rewards_per_pr, all_visitations, grad_norm_per_pr, mntensor_train_per_pr, mntensor_test_per_pr, learned_policy_profile, jaccard_tr.detach().numpy(), jaccard_tst.detach().numpy(), visitation_count_diffs_abs_train, visitation_count_diffs_abs_test
    
    def _train_step_value_learning(self, obs_mat: th.Tensor, dones: th.Tensor = None, chosen_profile=(1.0,0.0,0.0), loss_weighting=1.0,  batch_size=None) -> Tuple[np.ndarray, np.ndarray]:
        
        th.autograd.set_detect_anomaly(False)
        learned_policy_profile  = self._reward_net.get_learned_profile() if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION else chosen_profile
        pure_rewards_per_pure_pr = dict()
        for pr in BASIC_PROFILES:
            pure_rewards_per_pure_pr[pr] = self.calculate_rewards_for_destinations_and_profile(self.destinations, pr, obs_mat=obs_mat, dones=dones)
        
        predicted_r_np = self.calculate_rewards_for_destinations_and_profile(self.destinations, learned_policy_profile, obs_mat=obs_mat, dones=dones)
        
        if batch_size is None:
            batch_od_train = self.od_list_train
        else:
            ods_index = np.random.choice(len(self.od_list_train), size=batch_size, replace=True)
            batch_od_train = [(od[0], od[1]) for od in np.array(self.od_list_train)[ods_index].tolist()]

        if self.use_mce_om:
            pi = self.mce_partition_fh(predicted_r_np, profile=learned_policy_profile) 
    
            _, visitations = self.mce_occupancy_measures(
                reward=predicted_r_np,
                pi=pi,
                profile=learned_policy_profile
            )
            sampler: SimplePolicy = SimplePolicy.from_policy_matrix(pi, real_env=self.env)
            sampled_trajs= sampler.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[learned_policy_profile,])
            
        else:
            if self.use_dijkstra:
                sampler: SimplePolicy = SimplePolicy.from_environment_expert(self.env, profiles=[learned_policy_profile, ], 
                                                custom_cost=lambda state_des, profile: (-predicted_r_np[state_des[0], state_des[1]]))
                
            else:
                pi = self.mce_partition_fh(predicted_r_np, profile=learned_policy_profile, 
                                           #od_list=self.od_list_train
                                           )
                sampler: SimplePolicy = SimplePolicy.from_policy_matrix(pi, real_env=self.env)
                
            sampled_trajs= sampler.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=[learned_policy_profile,])
            
            visitations = get_demo_oms_from_trajectories(sampled_trajs, state_dim=self.env.state_dim)
            
            
        # Forward/back/step (grads are zeroed at the top).
        # weights_th(s) = \pi(s) - D(s)

        self.optimizer.zero_grad()  
        grad_norm = 0
        """if False:
            loss = th.mean(th.vstack([
                th.dot(th.as_tensor(
                    np.mean([visitations[(sod, chosen_profile)] - self.demo_state_om_per_profile[(sod, chosen_profile)] for sod in self.env.od_list_int if sod[1] == d], axis=0),
                    dtype=self._reward_net.dtype,
                    device=self._reward_net.device,
                ), 
                squeeze_r(self.reward_net(None, None, obs_mat, dones[:, self.destinations_train[0]])) if state_space_is_1d else
                squeeze_r(self.reward_net(None, None, obs_mat[:,d,:], dones[:, d]))
                ) 
                for d in  self.destinations_train]).reshape((len(self.destinations_train),)))
                #print(predicted_r[0:10])
                    # This gives the required gradient:
                    #   E_\pi[\nabla r_\theta(S)] - E_D[\nabla r_\theta(S)].
        else:
        """
        sampled_trajs_train = []
        sampled_trajs_test = [] 

        for t in sampled_trajs:
            if (t.infos[0]['orig'], t.infos[0]['des']) in self.od_list_train:
                sampled_trajs_train.append(t)
                #od_profiles_of_trajs_train.append((t.infos[0]['orig'], t.infos[0]['des']))
            if (t.infos[0]['orig'], t.infos[0]['des']) in self.od_list_test:
                sampled_trajs_test.append(t)

        #chosen_profile_expert_trajs_train = [t for t in self.expert_trajectories if tuple(t.infos[0]['profile']) == tuple(chosen_profile) and (t.infos[0]['orig'], t.infos[0]['des']) in self.od_list_train]
        chosen_profile_expert_trajs_train = [t for od in  batch_od_train for t in self.expert_trajectories_per_odpr[(od, chosen_profile)]]
        od_profiles_of_trajs_train = [((t.infos[0]['orig'], t.infos[0]['des']), chosen_profile) for t in chosen_profile_expert_trajs_train]
        #chosen_profile_trajs_test = [t for t in chosen_profile_trajs if (t.infos[0]['orig'], t.infos[0]['des']) in self.od_list_test]
        #od_profiles_of_trajs_test = [((t.infos[0]['orig'], t.infos[0]['des']), chosen_profile) for t in chosen_profile_trajs_test]


        visitation_count_diffs = th.as_tensor(
            np.asarray([
            visitations[(odpr[0], learned_policy_profile)] - self.demo_state_om_per_profile[odpr] for odpr in od_profiles_of_trajs_train]),
                dtype=self._reward_net.dtype,
                device=self._reward_net.device,
            )
        
        costs_train, agou , ods_agou= calculate_expected_similarities_and_std(cost_model_from_reward_net(self._reward_net, env=self.env, precalculated_rewards_per_pure_pr=pure_rewards_per_pure_pr), sample_trajs=sampled_trajs_train, expert_trajs=chosen_profile_expert_trajs_train, profile_for_cost_model=(1.0,1.0,1.0), profile_criteria=BASIC_PROFILES,preprocessing=FeaturePreprocess.NORMALIZATION,return_ods=True)    
        #jaccard_per_pr_expert_train, jaccard_expert_train_normal, jaccard_expert_train_costs, ods_jaccard = jaccard_similarity(cost_model_from_reward_net(self._reward_net,env=self.env,precalculated_rewards_per_pure_pr=pure_rewards_per_pure_pr), profile_for_cost_model=(1.0,1.0,1.0), sample_trajs=sampled_trajs_train, expert_trajs=expert_trajs_train, profile_criteria=BASIC_PROFILES, preprocessing =FeaturePreprocess.NORMALIZATION,return_per_od=True)
        
        costs_test, agou_test, ods_agou_test= calculate_expected_similarities_and_std(cost_model_from_reward_net(self._reward_net, env=self.env,precalculated_rewards_per_pure_pr=pure_rewards_per_pure_pr), sample_trajs=sampled_trajs_train, expert_trajs=chosen_profile_expert_trajs_train, profile_for_cost_model=(1.0,1.0,1.0), profile_criteria=BASIC_PROFILES,preprocessing=FeaturePreprocess.NORMALIZATION,return_ods=True)    
        
        mntensor_train =np.mean(profiled_aggregated_similarity_Agourogiannis_et_al_2023(costs_train['sample'], costs_train['expert'], preference_weights=chosen_profile))
        mntensor_test = np.mean(profiled_aggregated_similarity_Agourogiannis_et_al_2023(costs_test['sample'], costs_test['expert'], preference_weights=chosen_profile))
            
        """fd_train, fd_normed_train = self.feature_differences(sampled_trajs_train, obs_mat, per_state=False, use_info_real_costs=True, 
                                                                sampled_traj_profile_to_expert_traj_profile_mapping=
                                                                {learned_policy_profile: chosen_profile})
        fd_normed_tensor_train = th.as_tensor(np.asarray([fd_normed_train[(od, chosen_profile)] for od in self.od_list_train]), device=self._reward_net.device, dtype=self._reward_net.dtype)
        #fd_tensor_train = th.as_tensor(np.asarray([fd_train[odpr] for odpr in od_profiles_of_trajs_train]), device=self.reward_net.device, dtype=self.reward_net.dtype)
        mntensor_train = th.mean(fd_normed_tensor_train)

        fd_test, fd_normed_test = self.feature_differences(
            sampled_trajs_test, obs_mat, per_state=False, 
            use_info_real_costs=True, 
            sampled_traj_profile_to_expert_traj_profile_mapping={learned_policy_profile: chosen_profile})
        
        fd_normed_tensor_test = th.as_tensor(np.asarray([fd_normed_test[(od, chosen_profile)] for od in self.od_list_test]), device=self._reward_net.device, dtype=self._reward_net.dtype)
        #fd_tensor_test = th.as_tensor(np.asarray([fd_test[odpr] for odpr in od_profiles_of_trajs_test]), device=self.reward_net.device, dtype=self.reward_net.dtype)
        mntensor_test = th.mean(fd_normed_tensor_test)"""

        obs_matrix_all_trajs = th.vstack([obs_mat[:,t.infos[0]['des'],:] for t in chosen_profile_expert_trajs_train])
        done_matrix_all_trajs = th.vstack([dones[:, t.infos[0]['des']] for t in chosen_profile_expert_trajs_train])
        
        rewards =  self._reward_net(None, None, obs_matrix_all_trajs, done_matrix_all_trajs).reshape_as(visitation_count_diffs)
        
        assert visitation_count_diffs.shape == rewards.shape, f"D: {visitation_count_diffs.shape}, R: {rewards.shape}"
        #assert feature_expectation_diffs.shape == rewards.shape, f"D: {feature_expectation_diffs.shape}, R: {rewards.shape}"
        assert th.all(rewards < 0.0)
        
        losses = th.mul(
                    visitation_count_diffs,
                #squeeze_r(self.reward_net(obs_mat, None, None, dones[:, self.destinations[0]])) if state_space_is_1d else
                rewards)
        #sum_of_all_lengths = np.sum([len(t) for t in chosen_profile_trajs_train])
        #weights = th.tensor([len(t)/sum_of_all_lengths for t in self.expert_trajectories], dtype=losses.dtype, requires_grad=False)
        
        #losses = th.matmul(fd_tensor, losses)
        # This alost works but vanishes? loss = th.sum(losses)/len(chosen_profile_trajs_train)

        

        with th.no_grad():
            wheres = th.where(losses != 0.0)
        if wheres[0].shape[0] <= 1:
            loss = th.sum(losses)
        else:
            loss = th.sum(losses[wheres])/len(set(wheres[0].tolist()))#/len([odpr for odpr in od_profiles_of_trajs_train if losses[:,odpr[0][1]]]) #/len([t for t in self.expert_trajectories if tuple(t.infos[0]['profile']) == tuple(chosen_profile)])
        loss *= loss_weighting

        loss.backward(retain_graph=False)
        self.optimizer.step()
            
        grads = []
        for p in self._reward_net.parameters():
            assert p.grad is not None  # for type checker
            grads.append(p.grad)
            
            print("GRAD: ", p.names, p.grad)
        print("REAL LOSS:", loss)
        print("REWARD NORM:", rewards.norm())
        grad_norm += util.tensor_iter_norm(grads).item()

        predicted_r_np_all_states = predicted_r_np

        return predicted_r_np_all_states, visitations, grad_norm, mntensor_train, mntensor_test, learned_policy_profile

    def divide_od_list_as_per_profile(self, od_list, pr):
        total_elements = len(od_list)
        
        groups = dict()
        prev_pos = 0
        shuffled_od = np.random.permutation(od_list)
        remaining_elements = total_elements
        for i, pure_profile in enumerate(BASIC_PROFILES):
            if np.sum(pr[i:]) == 0.0:
                proportion = 0
            else:
                 proportion = pr[i]/np.sum(pr[i:])
            
            to_put =round(proportion*remaining_elements)
            group = shuffled_od[prev_pos:prev_pos+to_put].tolist()
            remaining_elements -= len(group)
            prev_pos = prev_pos + len(group)

            groups[pure_profile] = [tuple(od) for od in group]

        
        return groups
    
    def train(self, max_iter: int = 1000,render_partial_plots=True, training_mode = None, training_set_mode = None, batch_size=None, wait_until_end_of_iterations=False) -> np.ndarray:
        
        if training_mode is not None:
            self.training_mode = training_mode
        if training_set_mode is not None:
            self.training_set_mode = training_set_mode

        self._reward_net.set_mode(self.training_mode)

        # use the same device and dtype as the rmodel parameters
        obs_mat = self.env.observation_matrix
        
        torch_obs_mat = th.as_tensor(
            obs_mat,
            dtype=self._reward_net.dtype,
            device=self._reward_net.device,
        )
        
        dones = th.as_tensor(self.env.done_matrix,dtype=self._reward_net.dtype)
        assert self.demo_state_om is not None or self.demo_state_om_per_profile is not None
        assert self.demo_state_om_per_profile[tuple(list(self.demo_state_om_per_profile.keys())[0])].shape == (len(obs_mat),)
        
        self.eval_profiles = self.training_profiles if self.training_set_mode != TrainingSetModes.PROFILED_SOCIETY else BASIC_PROFILES
        

        mean_absolute_difference_in_visitation_counts_per_profile_test = {pr: [] for pr in self.eval_profiles}
        mean_absolute_difference_in_visitation_counts_per_profile_train = {pr: [] for pr in self.eval_profiles}
        if self.training_set_mode == TrainingSetModes.PROFILED_SOCIETY:
            mean_absolute_difference_in_visitation_counts_per_profile_train['learned'] = []
            mean_absolute_difference_in_visitation_counts_per_profile_test['learned'] = []

        grad_norms_per_iteration_per_profile = {pr: [] for pr in self.eval_profiles}

        overlapping_proportions_per_iteration_per_profile_test = {pr: [] for pr in self.eval_profiles}
        overlapping_proportions_per_iteration_per_profile_train = {pr: [] for pr in self.eval_profiles}

        jaccard_pure_per_iteration_per_profile_test = {pr: [] for pr in self.eval_profiles}
        jaccard_pure_per_iteration_per_profile_train = {pr: [] for pr in self.eval_profiles}

        sorted_overlap_indices_test = {pr: [] for pr in self.eval_profiles}
        sorted_overlap_indices_train = {pr: [] for pr in self.eval_profiles}

        pi_per_profile  = {pr: None for pr in self.eval_profiles}
        feature_expectation_differences_per_iteration_per_profile_test = {pr: [] for pr in self.eval_profiles}
        feature_expectation_differences_per_iteration_per_profile_train = {pr: [] for pr in self.eval_profiles}

        historic_avg_agou_cost, historic_avg_jaccard, historic_std_agou_cost, historic_std_jaccard, historic_avg_vc, historic_std_vc= [], [], [],[],[],[]
        historic_avg_agou_cost_test, historic_avg_jaccard_test, historic_std_agou_cost_test, historic_std_jaccard_test, historic_avg_vc_test, historic_std_vc_test= [], [], [],[],[],[]

        with networks.training(self._reward_net):
            # switch to training mode (affects dropout, normalization)
            #current_batch_of_destinations = self.rng.permutation(self.destinations)
            for t in range(max_iter):
                skip_pr = collections.defaultdict(lambda: False) # skip evaluating with a profile not being trained on (PROFILED_SOCIETY)
                if t%5 == 0 and self.stochastic_expert:
                    sampled_expert_trajectories = self.expert_policy.sample_trajectories(stochastic=self.stochastic_expert, repeat_per_od=self.n_repeat_per_od_monte_carlo, with_profiles=BASIC_PROFILES if self.training_set_mode == TrainingSetModes.PROFILED_SOCIETY else self.training_profiles, od_list=self.od_list_train)
                    self._set_demo_oms_from_trajectories(sampled_expert_trajectories)

                predicted_rewards_per_profile = dict()

                original_od_train = deepcopy(self.od_list_train)

                if self.training_set_mode == TrainingSetModes.PROFILED_SOCIETY:
                    
                    od_train_per_basic_profile = self.divide_od_list_as_per_profile(self.od_list_train, self.training_profiles[t%len(self.training_profiles)])

                    profile_proportions = np.asarray(self.training_profiles[t%len(self.training_profiles)]) / np.sum(self.training_profiles[t%len(self.training_profiles)])
                    for pr, ods in od_train_per_basic_profile.items():
                        skip_pr[pr] = len(ods) < 1 # If no examples for a certain profile, we do not eval and train  with it.
                else:
                    od_train_per_basic_profile = {pr: self.od_list_train for pr in self.training_profiles}
                
                grad_norm_per_pr = dict()
                fd_train_per_pr = dict()
                fd_test_per_pr = dict()
                visitations = dict()
                learned_profiles_per_pr = {pr: pr for pr in self.eval_profiles}
                predicted_rewards_per_profile = dict()

                avg_soc_cost, std_soc_cost = None, None

                if self.training_set_mode == TrainingSetModes.PROFILED_SOCIETY:
                    """predicted_rewards_per_profile, visitations, grad_norm_per_pr, fd_train_per_pr, fd_test_per_pr, learned_policy_profile = self._train_step_ponderated_profiles(
                        torch_obs_mat, dones, od_train_per_profile=od_train_per_basic_profile, profile_ponderation={pr: profile_proportions[BASIC_PROFILES.index(pr)] for pr in BASIC_PROFILES})"""
                    predicted_rewards_per_profile, visitations, grad_norm_per_pr, fd_train_per_pr, fd_test_per_pr, learned_policy_profile, jaccard_sim_train, jaccard_sim_test, vc_diffs_abs_train, vc_diffs_abs_test = self._train_step_sampling_trajectories(
                        torch_obs_mat, dones, society_profiles=self.eval_profiles, society_profile_distribution=[profile_proportions[BASIC_PROFILES.index(pr)] for pr in self.eval_profiles], batch_size=batch_size)
                    learned_profiles_per_pr = {pr: learned_policy_profile for pr in self.eval_profiles}
                    agou = fd_train_per_pr['learned']
                    historic_avg_agou_cost.append(np.mean(agou))
                    historic_std_agou_cost.append(np.std(agou))
                    historic_avg_jaccard.append(np.mean(jaccard_sim_train))
                    historic_std_jaccard.append(np.std(jaccard_sim_train))
                    historic_avg_vc.append(np.mean(vc_diffs_abs_train))
                    historic_std_vc.append(np.std(vc_diffs_abs_train))
                    
                    agou_test = fd_test_per_pr['learned']
                    historic_avg_agou_cost_test.append(np.mean(agou_test))
                    historic_std_agou_cost_test.append(np.std(agou_test))
                    historic_avg_jaccard_test.append(np.mean(jaccard_sim_test))
                    historic_std_jaccard_test.append(np.std(jaccard_sim_test))
                    historic_avg_vc_test.append(np.mean(vc_diffs_abs_test))
                    historic_std_vc_test.append(np.std(vc_diffs_abs_test))

                else:
                    
                    for pr in self.rng.permutation(self.eval_profiles , axis=0):
                        pr = tuple(pr)
                        if skip_pr[pr]:
                            print("NO DATA FOR PROFILE: ", pr, " skipping...")
                            continue
                        print("Train with ", pr)
                        
                        self.od_list_train = od_train_per_basic_profile[pr]
                        predicted_r_np_all_od, visitations_pr, grad_norm, fd_train, fd_test, learned_policy_profile = self._train_step_value_learning(
                                        torch_obs_mat,
                                        dones=dones, chosen_profile=pr, loss_weighting=1.0 if self.training_set_mode == TrainingSetModes.PROFILED_EXPERT else profile_proportions[BASIC_PROFILES.index(pr)],  batch_size=batch_size)
                        self.od_list_train = deepcopy(original_od_train)
                        grad_norm_per_pr[pr] = grad_norm
                        fd_train_per_pr[pr] = fd_train
                        fd_test_per_pr[pr] = fd_test
                        visitations.update(visitations_pr)
                        learned_profiles_per_pr[pr] = learned_policy_profile
                        predicted_rewards_per_profile[pr] = predicted_r_np_all_od
                            
                for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]:
                    #di+=1
                    # these are just for termination conditions & debug logging
                    # TRAIN
                    #print(self.training_set_mode, self.training_mode)
                    #print(self.eval_profiles, skip_pr)
                    #print(fd_train_per_pr.keys())
                    feature_expectation_differences_per_iteration_per_profile_train[pr].append(np.mean(fd_train_per_pr[pr]))
                    
                    difference_in_visitation_counts_train = np.array([self.demo_state_om_per_profile[(od, pr)] - visitations[(od,learned_profiles_per_pr[pr])] for od in self.od_list_train])
                    absolute_difference_in_visitation_counts_train = np.abs(difference_in_visitation_counts_train)

                    absolute_difference_in_visitation_counts_per_od_train = np.sum(absolute_difference_in_visitation_counts_train, axis=1)
                    #max_delta_per_od = np.max(absolute_difference_in_visitation_counts, axis=-1)

                    mean_absolute_difference_in_visitation_counts_train = np.mean(absolute_difference_in_visitation_counts_per_od_train)

                    grad_norms_per_iteration_per_profile[pr].append(grad_norm_per_pr[pr])
                    mean_absolute_difference_in_visitation_counts_per_profile_train[pr].append(mean_absolute_difference_in_visitation_counts_train)
                    
                    # TEST 
                    feature_expectation_differences_per_iteration_per_profile_test[pr].append(np.mean(fd_test_per_pr[pr]))
                    
                    difference_in_visitation_counts_test = np.array([self.demo_state_om_per_profile[(od, pr)] - visitations[(od,learned_profiles_per_pr[pr])] for od in self.od_list_test])
                    absolute_difference_in_visitation_counts_test = np.abs(difference_in_visitation_counts_test)

                    absolute_difference_in_visitation_counts_per_od_test = np.sum(absolute_difference_in_visitation_counts_test, axis=1)#max_delta_per_od = np.max(absolute_difference_in_visitation_counts, axis=-1)

                    mean_absolute_difference_in_visitation_counts_test = np.mean(absolute_difference_in_visitation_counts_per_od_test)

                    mean_absolute_difference_in_visitation_counts_per_profile_test[pr].append(mean_absolute_difference_in_visitation_counts_test)
                
                if self.training_set_mode == TrainingSetModes.PROFILED_SOCIETY:
                    

                    mean_absolute_difference_in_visitation_counts_per_profile_train['learned'].append(historic_avg_vc[-1])
                    
                    mean_absolute_difference_in_visitation_counts_per_profile_test['learned'].append(historic_avg_vc_test[-1])
                    
                if self.use_dijkstra:
                    
                    sampler: SimplePolicy = SimplePolicy.from_environment_expert(self.env, [prs for prs in self.eval_profiles if not skip_pr[prs]], custom_cost=lambda state_des, 
                                                                             pr: (-predicted_rewards_per_profile[pr][state_des[0], state_des[1]]))
                else:
                    
                    for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]:
                        pi_per_profile[pr] = self.mce_partition_fh(
                                                reward_matrix=predicted_r_np_all_od,
                                                profile=pr,
                                                #od_list=self.env.od_list_int
                                        
                                            )
                    sampler: SimplePolicy = SimplePolicy.from_policy_matrix(pi_per_profile, real_env=self.env)
                
                
                sampled_trajs = sampler.sample_trajectories(stochastic=False, repeat_per_od=1, with_profiles=[prs for prs in self.eval_profiles if not skip_pr[prs]], od_list=self.env.od_list_int)
                #sampled_trajs_test = sampler.sample_trajectories(stochastic=False, repeat_per_od=1, profile=self.sampling_profiles, od_list=self.od_list_train)
                

                
                overlapping_proportion_test = {pr: 0 for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]}
                overlap_sorted_test = {pr: [] for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]}

                overlapping_proportion_train = {pr: 0 for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]}
                overlap_sorted_train = {pr: [] for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]}
                
                jaccards_per_pr_train = {pr: [] for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]}
                jaccards_per_pr_test = {pr: [] for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]}
                avg_jaccard_per_pr_test =  {pr: 0 for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]}
                avg_jaccard_per_pr_train =  {pr: 0 for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]}

                for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]:
                    overlap_of_this_pr_test = []
                    total_possible_overlaps_of_this_pr_test = 0

                    sampled_trajs_of_pr = [t for t in sampled_trajs if t.infos[0]['profile'] == pr]

                    overlap_of_this_pr_train = []
                    total_possible_overlaps_of_this_pr_train = 0

                    for odpr, expert_trajs in self.expert_trajectories_per_odpr.items():
                        if odpr[1] == pr:
                            for traj in [t for t in sampled_trajs_of_pr if (t.infos[0]['orig'], t.infos[0]['des']) == odpr[0]]:

                                if odpr[0] in self.od_list_test:
                                    overlapping_nodes = 0

                                    best_traj = None
                                    best_overlapping_proportion = 0
                                    for expert_traj in expert_trajs:
                                        if (expert_traj.infos[0]['orig'], expert_traj.infos[0]['des']) in self.od_list_test:
                                            

                                            aux_overlap_nodes = 0
                                            expert_traj = expert_trajs[0]
                                            lexpert = len(expert_traj)
                                            for i in range(lexpert):
                                                if i < len(traj):
                                                    do_overlap = (1 if expert_traj.obs[i] == traj.obs[i] else 0)
                                                    aux_overlap_nodes+=do_overlap
                                            oprop = aux_overlap_nodes/lexpert
                                            if oprop >= best_overlapping_proportion:
                                                best_traj = expert_traj
                                                best_overlapping_proportion = oprop

                                    overlapping_nodes = 0

                                    expert_traj = best_traj
                                    for i in range(len(expert_traj)):
                                        if i < len(traj):
                                            do_overlap = (1 if expert_traj.obs[i] == traj.obs[i] else 0)
                                            overlapping_nodes+=do_overlap
                                            overlapping_proportion_test[pr]+=do_overlap
                                        total_possible_overlaps_of_this_pr_test+=1
                                    #edit_dist = editdistance.eval(traj.obs, expert_traj.obs)
                                    _,_,jreal, jaccard_dist = jaccard(traj,expert_traj,cost_model=self.env.cost_model, profile=pr,preprocessing=FeaturePreprocess.NORMALIZATION)
                                    overlap_of_this_pr_test.append(overlapping_nodes/len(expert_traj))

                                    jaccards_per_pr_test[pr].append(jaccard_dist)
                                    avg_jaccard_per_pr_test[pr]+=jaccard_dist/len(self.od_list_test)

                                if odpr[0] in self.od_list_train:
                                    overlapping_nodes = 0

                                    best_traj = None
                                    best_overlapping_proportion = 0
                                    for expert_traj in expert_trajs:
                                        if (expert_traj.infos[0]['orig'], expert_traj.infos[0]['des']) in self.od_list_train:
                                            aux_overlap_nodes = 0
                                            expert_traj = expert_trajs[0]
                                            lexpert = len(expert_traj)
                                            for i in range(lexpert):
                                                if i < len(traj):
                                                    do_overlap = (1 if expert_traj.obs[i] == traj.obs[i] else 0)
                                                    aux_overlap_nodes+=do_overlap
                                            oprop = aux_overlap_nodes/lexpert
                                            if oprop >= best_overlapping_proportion:
                                                best_traj = expert_traj
                                                best_overlapping_proportion = oprop

                                    overlapping_nodes = 0

                                    expert_traj = best_traj
                                    for i in range(len(expert_traj)):
                                        if i < len(traj):
                                            do_overlap = (1 if expert_traj.obs[i] == traj.obs[i] else 0)
                                            overlapping_nodes+=do_overlap
                                            overlapping_proportion_train[pr]+=do_overlap
                                        total_possible_overlaps_of_this_pr_train+=1
                                    #edit_dist = editdistance.eval(traj.obs, expert_traj.obs)
                                    _,_, jreal, jaccard_dist = jaccard(traj,expert_traj,cost_model=self.env.cost_model, profile=pr,preprocessing=FeaturePreprocess.NORMALIZATION)
                                    overlap_of_this_pr_train.append(overlapping_nodes/len(expert_traj))

                                    #overlap_of_this_pr_train.append(overlapping_nodes/len(expert_traj))
                                    jaccards_per_pr_train[pr].append(jaccard_dist)
                                    avg_jaccard_per_pr_train[pr]+=jaccard_dist/len(self.od_list_train)
                    
                    overlapping_proportion_test[pr]/=total_possible_overlaps_of_this_pr_test
                    
                    npoverlap_test = np.array(overlap_of_this_pr_test)
                    sorted_overlap_indices_test[pr] = np.argsort(npoverlap_test)
                    #print(sorted_overlap_indices)
                    overlap_sorted_test[pr] = npoverlap_test[sorted_overlap_indices_test[pr]]
                    
                    overlapping_proportions_per_iteration_per_profile_test[pr].append(overlapping_proportion_test[pr])
                    jaccard_pure_per_iteration_per_profile_test[pr].append(avg_jaccard_per_pr_test[pr])

                    overlapping_proportion_train[pr]/=total_possible_overlaps_of_this_pr_train

                    npoverlap_train = np.array(overlap_of_this_pr_train)
                    sorted_overlap_indices_train[pr] = np.argsort(npoverlap_train)
                    #print(sorted_overlap_indices)
                    overlap_sorted_train[pr] = npoverlap_train[sorted_overlap_indices_train[pr]]

                    overlapping_proportions_per_iteration_per_profile_train[pr].append(overlapping_proportion_train[pr])

                    jaccard_pure_per_iteration_per_profile_train[pr].append(avg_jaccard_per_pr_train[pr])
                    
                termination_condition_met = False
                if wait_until_end_of_iterations is False:
                    if self.training_set_mode != TrainingSetModes.PROFILED_SOCIETY:
                        termination_condition_met = (max(mean_absolute_difference_in_visitation_counts_per_profile_train[pr][-1] for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]) <= self.mean_vc_diff_eps or max(grad_norms_per_iteration_per_profile[pr][-1] for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]) <= self.grad_l2_eps) and self.overlaping_percentage <= min(overlapping_proportion_train[pr] for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]])
                    else:
                        termination_condition_met = max(grad_norms_per_iteration_per_profile[pr][-1] for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]) <= self.grad_l2_eps and historic_avg_jaccard[-1] >= self.overlaping_percentage
                if (self.log_interval is not None and 0 == (t % self.log_interval)) or termination_condition_met or (t + 1 == max_iter):
                    print("Logging")
                    params = self._reward_net.parameters()
                    weight_norm = util.tensor_iter_norm(params).item()

                    
                    self.logger.record("iteration", t)
                    self.logger.record("max mean_absolute_difference_in_visitation_counts", max(mean_absolute_difference_in_visitation_counts_per_profile_train[pr][-1] for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]))
                    self.logger.record("weight_norm", weight_norm)
                    self.logger.record("max grad_norm", max(grad_norms_per_iteration_per_profile[pr][-1] for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]))
                    #self.logger.record("Linf_delta found in OD, state: ", str((np.asarray(self.env.od_list_int)[mean_absolute_difference_in_visitation_counts_index[pr][0]], mean_absolute_difference_in_visitation_counts_index[pr][1])))
                    
                    print("PARAMS:", list(self._reward_net.parameters()))
                    print("Value matrix: ", self._reward_net.value_matrix())
                    if self.training_mode==TrainingModes.VALUE_SYSTEM_IDENTIFICATION or self.training_mode==TrainingModes.SIMULTANEOUS:
                        print("Previous Profile: ", learned_policy_profile)
                        print("New Learned Profile: ", self._reward_net.get_learned_profile())
                        self.logger.record("Previous Profile: ", [f"{p:0.2f}" for p in learned_policy_profile])
                        self.logger.record("New Learned Profile: ", [f"{p:0.2f}" for p in self._reward_net.get_learned_profile()])
                    self.logger.dump(t)
                    
                    #losses.append(loss)# Plot the bar plot
                    mean_absolute_difference_in_visitation_counts_per_profile_colors = PROFILE_COLORS
                    overlapping_colors = PROFILE_COLORS
                    grad_norm_colors = PROFILE_COLORS

                    if len(historic_avg_agou_cost) > 0:
                        plt.figure(figsize=(6,6))
                        plt.title(f"Train: Similarity metrics")
                        plt.xlabel("Training Iteration")
                        plt.ylabel("Average similarity over OD pairs")
                        agou = np.asarray(historic_avg_agou_cost)
                        jaccard_metric =np.asarray(historic_avg_jaccard)
                        std_agou =np.asarray(historic_std_agou_cost)
                        std_jaccard = np.asarray(historic_std_jaccard)
                        
                        plt.plot(agou,
                                        color='orange', 
                                        label=f'PS: {float(agou[-1]):0.2f}'
                                        )
                        plt.plot(jaccard_metric,
                                        color='black', 
                                        label=f'JACC: {float(jaccard_metric[-1]):0.2f}'
                                        ) 
                        
                        plt.fill_between([i for i in range(len(agou))], agou-std_agou, agou+std_agou,edgecolor='orange',alpha=0.1, facecolor='orange')
                        plt.fill_between([i for i in range(len(agou))], jaccard_metric-std_jaccard, jaccard_metric+std_jaccard,edgecolor='black',alpha=0.1, facecolor='black')
                        
                        plt.legend()
                        plt.grid()
                        plt.savefig(f'plots/Maxent_{self.name_method}_similarities_train_{self.training_mode.value}_via_{self.training_set_mode.value}.pdf')

                        plt.figure(figsize=(6,6))
                        plt.title(f"Test: Similarity metrics (assuming learned profile)")
                        plt.xlabel("Training Iteration")
                        plt.ylabel("Average similarity over OD pairs")
                        agou_test = np.asarray(historic_avg_agou_cost_test)
                        jaccard_metric_test =np.asarray(historic_avg_jaccard_test)
                        std_agou_test =np.asarray(historic_std_agou_cost_test)
                        std_jaccard_test = np.asarray(historic_std_jaccard_test)
                        
                        plt.plot(agou_test,
                                        color='orange', 
                                        label=f'PS: {float(agou[-1]):0.2f}'
                                        )
                        plt.plot(jaccard_metric_test,
                                        color='black', 
                                        label=f'JACC: {float(jaccard_metric[-1]):0.2f}'
                                        ) 
                        
                        plt.fill_between([i for i in range(len(agou_test))], agou_test-std_agou_test, agou_test+std_agou_test,edgecolor='orange',alpha=0.1, facecolor='orange')
                        plt.fill_between([i for i in range(len(jaccard_metric_test))], jaccard_metric_test-std_jaccard_test, jaccard_metric_test+std_jaccard_test,edgecolor='black',alpha=0.1, facecolor='black')
                        
                        plt.legend()
                        plt.grid()
                        plt.savefig(f'plots/Maxent_{self.name_method}_similarities_test_{self.training_mode.value}_via_{self.training_set_mode.value}.pdf')

                        plt.figure(figsize=(6,6))
                        plt.title(f"Train: Total absolute difference in visitation counts")
                        plt.xlabel("Training Iteration")
                        plt.ylabel("Average TVC over OD pairs")
                        tvc =np.asarray(historic_avg_vc)
                        std_tvc =np.asarray(historic_std_vc)
                        
                        
                        plt.plot(tvc,
                                        color='magenta', 
                                        label=f'TVC: {float(tvc[-1]):0.2f}'
                                        )
                        plt.fill_between([i for i in range(len(agou))], tvc-std_tvc, tvc+std_tvc,edgecolor='magenta',alpha=0.1, facecolor='magenta')
                        
                        plt.legend()
                        plt.grid()
                        plt.savefig(f'plots/Maxent_{self.name_method}_differences_train_{self.training_mode.value}_via_{self.training_set_mode.value}.pdf')

                        plt.figure(figsize=(6,6))
                        plt.title(f"Test: Total absolute difference in visitation counts")
                        plt.xlabel("Training Iteration")
                        plt.ylabel("Average TVC over OD pairs")
                        tvc_test =np.asarray(historic_avg_vc_test)
                        std_tvc_test =np.asarray(historic_std_vc_test)
                        
                        plt.plot(tvc_test,
                                        color='magenta', 
                                        label=f'TVC: {float(tvc_test[-1]):0.2f}'
                                        )
                        plt.fill_between([i for i in range(len(tvc_test))], tvc_test-std_tvc_test, tvc_test+std_tvc_test,edgecolor='magenta',alpha=0.1, facecolor='magenta')
                        
                        plt.legend()
                        plt.grid()
                        plt.savefig(f'plots/Maxent_{self.name_method}_differences_test_{self.training_mode.value}_via_{self.training_set_mode.value}.pdf')
                    
                    
                    print("-----------------------------------------------------------------------")
                    print("FIGURE TEST: ")
                    plt.figure(figsize=[18, 12])
                    #print(mean_absolute_difference_in_visitation_counts_per_profile)
                    plt.subplot(2, max(3,len([prs for prs in self.eval_profiles if not skip_pr[prs]])), 1)
                    """plt.title(f"Average trajectory overlap with profiles")
                        plt.xlabel("Training Iteration")
                        plt.ylabel("Average trajectory overlap proportion")
                        for _, pr in enumerate([prs for prs in self.eval_profiles if not skip_pr[prs]]):
                            #plt.plot(mean_absolute_difference_in_visitation_counts_per_profile[pr], color=mean_absolute_difference_in_visitation_counts_per_profile_colors[pi], label='Maximum difference in visitation counts')
                            #print(overlapping_proportions_per_iteration_per_profile_test[pr])
                            plt.plot(overlapping_proportions_per_iteration_per_profile_test[pr],
                                    color=overlapping_colors.get(pr,'black'), 
                                    label=f'P: \'{BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))}\'\nLast: {float(overlapping_proportions_per_iteration_per_profile_test[pr][-1]):0.2f}'
                                    )
                        plt.legend()
                        plt.grid()"""
                    
                    plt.title(f"Average trajectory jaccard similarity with profiles")
                    plt.xlabel("Training Iteration")
                    plt.ylabel("Average trajectory jaccard similarity with profiles")
                    for _, pr in enumerate([prs for prs in self.eval_profiles if not skip_pr[prs]]):
                        #plt.plot(mean_absolute_difference_in_visitation_counts_per_profile[pr], color=mean_absolute_difference_in_visitation_counts_per_profile_colors[pi], label='Maximum difference in visitation counts')
                        #print(overlapping_proportions_per_iteration_per_profile_test[pr])
                        plt.plot(jaccard_pure_per_iteration_per_profile_test[pr],
                                color=overlapping_colors.get(pr,'black'), 
                                label=f'P: \'{BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))}\'\nLast: {float(jaccard_pure_per_iteration_per_profile_test[pr][-1]):0.2f}'
                                )
                    if self.training_set_mode == TrainingSetModes.PROFILED_SOCIETY and len(historic_avg_jaccard_test) > 0:
                        
                        plt.plot(historic_avg_jaccard_test,
                                        color='black', 
                                        label=f'P: \'{BASIC_PROFILE_NAMES.get(learned_policy_profile, str(tuple([float(f"{l:0.2f}") for l in learned_policy_profile])))}\'\nLearned Prof. Sim: {float(historic_avg_jaccard_test[-1]):0.2f}'
                                        ) 
                    plt.legend()
                    plt.grid()

                    plt.subplot(2, max(3,len([prs for prs in self.eval_profiles if not skip_pr[prs]])), 2)
                    plt.title(f"Mean absolute difference in visitation counts by profile")
                    plt.xlabel("Training Iteration")
                    plt.ylabel("Mean absolute difference in visitation counts")
                    for _, pr in enumerate([prs for prs in self.eval_profiles if not skip_pr[prs]]):
                        #print(mean_absolute_difference_in_visitation_counts_per_profile_test[pr])
                        plt.plot(mean_absolute_difference_in_visitation_counts_per_profile_test[pr], 
                                    color=mean_absolute_difference_in_visitation_counts_per_profile_colors.get(pr,'black'), 
                                    label=f'P: {BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))}\nLast: {float(mean_absolute_difference_in_visitation_counts_per_profile_test[pr][-1]):0.2f}).')
                    if 'learned' in mean_absolute_difference_in_visitation_counts_per_profile_test.keys():
                        plt.plot(mean_absolute_difference_in_visitation_counts_per_profile_test['learned'], 
                                    color='magenta', 
                                    label=f'P: {BASIC_PROFILE_NAMES.get(learned_policy_profile, str(tuple([float(f"{l:0.2f}") for l in learned_policy_profile])))}\nLast iter: {float(mean_absolute_difference_in_visitation_counts_per_profile_test["learned"][-1]):0.2f}).')
                    
                    plt.legend()
                    plt.grid()

                    plt.subplot(2, max(3,len([prs for prs in self.eval_profiles if not skip_pr[prs]])), 3)
                    plt.title(f"Expected trajectory profiled cost similarity per iteration")
                    plt.xlabel("Training Iteration")
                    plt.ylabel("Expected trajectory profiled cost similarity")
                    for _, pr in enumerate([prs for prs in self.eval_profiles if not skip_pr[prs]]):
                        #print(feature_expectation_differences_per_iteration_per_profile_test[pr])

                        plt.plot(feature_expectation_differences_per_iteration_per_profile_test[pr], 
                                 color=grad_norm_colors.get(pr,'black'),
                                 label=f'P: {BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))}\nLast: {float(feature_expectation_differences_per_iteration_per_profile_test[pr][-1]):0.2f}).')
                        #plt.axvline(x=0, color=grad_norm_colors.get(pr,'black'), linestyle='--', label=f'FED ({BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))}): {float(feature_expectation_differences_per_iteration_per_profile[pr][-1]):0.2f}).')  # Highlighting the minimum value
                    if self.training_set_mode == TrainingSetModes.PROFILED_SOCIETY and len(historic_avg_agou_cost_test) > 0:
                        
                        plt.plot(historic_avg_agou_cost_test,
                                        color='orange', 
                                        label=f'P: \'{BASIC_PROFILE_NAMES.get(learned_policy_profile, str(tuple([float(f"{l:0.2f}") for l in learned_policy_profile])))}\'\nLearned Prof. Sim: {float(historic_avg_agou_cost_test[-1]):0.2f}'
                                        ) 
                    plt.legend()
                    plt.grid()
                    
                    for pindex, pr in enumerate([prs for prs in self.eval_profiles if not skip_pr[prs]]):
                        plt.subplot(2, max(3,len([prs for prs in self.eval_profiles if not skip_pr[prs]])), pindex+4)
                        plt.bar(range(len(overlap_sorted_test[pr])), overlap_sorted_test[pr], color=overlapping_colors.get(pr,'black'))
                        plt.xlabel('OD pairs')
                        plt.ylabel('Best overlap proportion with an expert trajectory')
                        plt.title(f'Best overlap proportion with {BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))} agent per OD')
                        plt.xticks(range(len(overlap_sorted_test[pr])), labels=[str(od) for od in np.asarray(self.od_list_test)[sorted_overlap_indices_test[pr]]], rotation=90, fontsize=2)  # Setting x-axis ticks
                        plt.axvline(x=0, color='blue', linestyle='--', label=f'Worst Overlap: {float(np.min(overlap_sorted_test[pr])):0.2f}\n(at {np.asarray(self.od_list_test)[sorted_overlap_indices_test[pr][0]]}).\nOverlap Proportion: {overlapping_proportion_test[pr]:0.2f}')  # Highlighting the minimum value
                        plt.legend()
                        #plt.grid()
                    

                    # plt.show()
                    if self.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING:
                        test_file = f'results/value_grounding_learning/Maxent_{self.name_method}_learning_curves_test_{self.training_mode.value}_via_{self.training_set_mode.value}.pdf'
                    else:
                        test_file = f'plots/Maxent_{self.name_method}_learning_curves_test_{self.training_mode.value}_via_{self.training_set_mode.value}.pdf'
                    
                    plt.savefig(test_file)
                    plt.close()
                    print("Fig saved into file: ", test_file)
                    print("-----------------------------------------------------------------------")
                    

                    print("-----------------------------------------------------------------------")
                    print("FIGURE TRAIN: ")
                    plt.figure(figsize=[18, 12])
                    #print(mean_absolute_difference_in_visitation_counts_per_profile)
                    
                    """plt.title(f"Average trajectory overlap with profiles")
                    plt.xlabel("Training Iteration")
                    plt.ylabel("Average trajectory overlap proportion")
                    for _, pr in enumerate([prs for prs in self.eval_profiles if not skip_pr[prs]]):
                        #plt.plot(mean_absolute_difference_in_visitation_counts_per_profile[pr], color=mean_absolute_difference_in_visitation_counts_per_profile_colors[pi], label='Maximum difference in visitation counts')
                        #print(overlapping_proportions_per_iteration_per_profile_test[pr])
                        plt.plot(overlapping_proportions_per_iteration_per_profile_train[pr],
                                color=overlapping_colors.get(pr,'black'), 
                                label=f'P: \'{BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))}\'\nLast: {float(overlapping_proportions_per_iteration_per_profile_train[pr][-1]):0.2f}'
                                )
                    plt.legend()
                    plt.grid()"""
                    plt.subplot(2, max(3,len([prs for prs in self.eval_profiles if not skip_pr[prs]])), 1)
                    plt.title(f"Average trajectory jaccard similarity with profiles")
                    plt.xlabel("Training Iteration")
                    plt.ylabel("Average trajectory jaccard similarity")
                    for _, pr in enumerate([prs for prs in self.eval_profiles if not skip_pr[prs]]):
                        #plt.plot(mean_absolute_difference_in_visitation_counts_per_profile[pr], color=mean_absolute_difference_in_visitation_counts_per_profile_colors[pi], label='Maximum difference in visitation counts')
                        #print(overlapping_proportions_per_iteration_per_profile_test[pr])
                        plt.plot(jaccard_pure_per_iteration_per_profile_train[pr],
                                color=overlapping_colors.get(pr,'black'), 
                                label=f'P: \'{BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))}\'\nLast: {float(jaccard_pure_per_iteration_per_profile_train[pr][-1]):0.2f}'
                                )
                    if self.training_set_mode == TrainingSetModes.PROFILED_SOCIETY and len(historic_avg_jaccard) > 0:
                        
                        plt.plot(historic_avg_jaccard,
                                        color='black', 
                                        label=f'P: \'{BASIC_PROFILE_NAMES.get(learned_policy_profile, str(tuple([float(f"{l:0.2f}") for l in learned_policy_profile])))}\'\nLearned Prof. Sim: {float(historic_avg_jaccard[-1]):0.2f}'
                                        )  
                    plt.legend()
                    plt.grid()

                    plt.subplot(2, max(3,len([prs for prs in self.eval_profiles if not skip_pr[prs]])), 2)
                    plt.title(f"Expected visitation count difference per profile")
                    plt.xlabel("Training Iteration")
                    plt.ylabel("Expected absolute difference in visitation counts")
                    for _, pr in enumerate([prs for prs in self.eval_profiles if not skip_pr[prs]]):
                        #print(mean_absolute_difference_in_visitation_counts_per_profile_train[pr])
                        plt.plot(mean_absolute_difference_in_visitation_counts_per_profile_train[pr], 
                                    color=mean_absolute_difference_in_visitation_counts_per_profile_colors.get(pr,'black'), 
                                    label=f'P: {BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))}\nLast iter: {float(mean_absolute_difference_in_visitation_counts_per_profile_train[pr][-1]):0.2f}).')
                    if 'learned' in mean_absolute_difference_in_visitation_counts_per_profile_train.keys():
                        plt.plot(mean_absolute_difference_in_visitation_counts_per_profile_train['learned'], 
                                    color='magenta',  
                                    label=f'P: {BASIC_PROFILE_NAMES.get(learned_policy_profile, str(tuple([float(f"{l:0.2f}") for l in learned_policy_profile])))}\nLast iter: {float(mean_absolute_difference_in_visitation_counts_per_profile_train["learned"][-1]):0.2f}).')
                    plt.legend()
                    plt.grid()

                    plt.subplot(2, max(3,len([prs for prs in self.eval_profiles if not skip_pr[prs]])), 3)
                    plt.title(f"Expected profiled cost similarity per profile")
                    plt.xlabel("Training Iteration")
                    plt.ylabel("Expected trajectory profiled cost similarity")
                    for _, pr in enumerate([prs for prs in self.eval_profiles if not skip_pr[prs]]):
                        #print(feature_expectation_differences_per_iteration_per_profile_train[pr])

                        plt.plot(feature_expectation_differences_per_iteration_per_profile_train[pr], 
                                 color=grad_norm_colors.get(pr,'black'),
                                 label=f'P: {BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))}\nLast iter: {float(feature_expectation_differences_per_iteration_per_profile_train[pr][-1]):0.2f}).')
                        #plt.axvline(x=0, color=grad_norm_colors.get(pr,'black'), linestyle='--', label=f'FED ({BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))}): {float(feature_expectation_differences_per_iteration_per_profile[pr][-1]):0.2f}).')  # Highlighting the minimum value
                    if self.training_set_mode == TrainingSetModes.PROFILED_SOCIETY and len(historic_avg_agou_cost) > 0:
                        
                        plt.plot(historic_avg_agou_cost,
                                        color='orange', 
                                        label=f'P: \'{BASIC_PROFILE_NAMES.get(learned_policy_profile, str(tuple([float(f"{l:0.2f}") for l in learned_policy_profile])))}\'\nLearned Prof. Sim: {float(historic_avg_agou_cost[-1]):0.2f}'
                                        ) 
                    plt.legend()
                    plt.grid()
                    
                    for pindex, pr in enumerate([prs for prs in self.eval_profiles if not skip_pr[prs]]):
                        plt.subplot(2, max(3,len([prs for prs in self.eval_profiles if not skip_pr[prs]])), pindex+4)
                        plt.bar(range(len(overlap_sorted_train[pr])), overlap_sorted_train[pr], color=overlapping_colors.get(pr,'black'))
                        plt.xlabel('OD pairs')
                        plt.ylim(top=1.1)
                        plt.ylabel('Best overlap proportion with an expert trajectory')
                        plt.title(f'Best overlap proportion with {BASIC_PROFILE_NAMES.get(pr, str(tuple([float(f"{l:0.2f}") for l in pr])))} agent per OD')
                        plt.xticks(range(len(overlap_sorted_train[pr])), labels=[str(od) for od in np.asarray(self.od_list_train)[sorted_overlap_indices_train[pr]]], rotation=90, fontsize=2)  # Setting x-axis ticks
                        plt.axvline(x=0, color='blue', linestyle='--', label=f'Worst Overlap: {float(np.min(overlap_sorted_train[pr])):0.2f}\n(at {np.asarray(self.od_list_train)[sorted_overlap_indices_train[pr][0]]}).\nOverlap Proportion: {overlapping_proportion_train[pr]:0.2f}')  # Highlighting the minimum value
                        plt.legend()
                        #plt.grid()
                    
                    if self.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING:
                        train_file = f'results/value_grounding_learning/Maxent_{self.name_method}_learning_curves_train_({self.training_mode.value}_via_{self.training_set_mode.value}).pdf'
                    else:
                        train_file = f'plots/Maxent_{self.name_method}_learning_curves_train_({self.training_mode.value}_via_{self.training_set_mode.value}).pdf'
                    
                    plt.savefig(train_file)
                    plt.close()
                    print("Fig saved into file: ", train_file)
                    print("-----------------------------------------------------------------------")
                    

                    self._policy.set_pi_or_policy(sampler)

                    for pr in [prs for prs in self.eval_profiles if not skip_pr[prs]]:
                        self._policy.set_profile(pr)

                        if render_partial_plots: # RENDER WORST OVERLAP
                            
                            
                            orig, des  = tuple(np.asarray(self.od_list_train
                                                          )[sorted_overlap_indices_train[pr][0]])

                            sampler: SimplePolicy = SimplePolicy.from_sb3_policy(self.policy, real_env = self.env)
                            path, edge_path = sampler.sample_path(start=orig, des = des, stochastic=False, profile=pr,t_max=1000)
                            expert_sampler: SimplePolicy = SimplePolicy.from_trajectories(self.expert_trajectories, real_env = self.env)
                            expert_path, expert_edge_path = expert_sampler.sample_path(start=orig, des = des, stochastic=False, profile=pr,t_max=1000)
                            print(f"Learned path from {orig} to {des}", edge_path)
                            print(f"Expert path from {orig} to {des}", expert_edge_path)
                            #print("PRNP", predicted_r_np)
                            self.env.render(caminos_by_value={'sus': [path,], 'eff': [expert_path]}, file=f"test_me_partial_{t}_{pr}_train.png", show=False,show_edge_weights=False)
                            

                if termination_condition_met:
                    break
        
        self._policy.set_pi_or_policy(sampler)
        
        return (predicted_rewards_per_profile, 
    {"op": overlapping_proportions_per_iteration_per_profile_test,
     "ps": feature_expectation_differences_per_iteration_per_profile_test, 
     "vc": mean_absolute_difference_in_visitation_counts_per_profile_test,
     "jacc": jaccard_pure_per_iteration_per_profile_test
    }, 
     {"op": overlapping_proportions_per_iteration_per_profile_train,
      "ps": feature_expectation_differences_per_iteration_per_profile_train, 
      "vc": mean_absolute_difference_in_visitation_counts_per_profile_train,
      "jacc": jaccard_pure_per_iteration_per_profile_train})
    

    @property
    def policy(self) -> TabularPolicyPerProfile:
        return self._policy
