from abc import abstractmethod
import signal
from typing import Any, Callable, Dict, Iterable, List, Mapping, NoReturn, Optional, Tuple, Union
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import sys

import torch

from src.network_env import FeaturePreprocess, RoadWorldGym
from src.values_and_costs import BASIC_PROFILES

from imitation.data.types import Trajectory


from scipy.special import softmax

from src.vsl_policies import ValueSystemLearningPolicy

if "../" not in sys.path:
  sys.path.append("../") 


from collections import defaultdict

from stable_baselines3.common.policies import BasePolicy

from scipy.spatial import distance

def profiled_similarity_Agourogiannis_et_al_2023(traj1, traj2, cost_model, preference_weights=(1,1,1), profiles=BASIC_PROFILES, preprocessing=None):
    traj1_cost_per_profile = np.asarray([np.sum([cost_model(pr, normalization=preprocessing)(traj1.obs[i]) for i in range(len(traj1))]) for pr in profiles])
    traj2_cost_per_profile = np.asarray([np.sum([cost_model(pr, normalization=preprocessing)(traj2.obs[i]) for i in range(len(traj2))]) for pr in profiles])
    np_prefs = np.asarray(preference_weights)
    assert np.allclose(np.asarray([1.0,]), np.array([np.sum(np_prefs),]))
    return 1 - np.abs(1 - len(profiles)/np.dot(traj1_cost_per_profile/traj2_cost_per_profile, np_prefs))

def jensen_shannon_distance_at_weighted_criteria(sample_costs, expert_costs, preference_weights, bins=100):

    range_bins = (min([np.min(sample_costs), np.min(expert_costs)]), max([np.max(sample_costs), np.max(expert_costs)]))

    jsd_per_criterion = np.asarray([0]*len(preference_weights))
    jsd_final = 0.0
    assert np.allclose(np.sum(preference_weights) , 1.0)

    for j, prw in enumerate(preference_weights):
        cost_distribution_sample = np.histogram(bins=len(sample_costs[:,j]), a = sample_costs[:,j],density=True)[0]
        cost_distribution_expert = np.histogram(bins=len(expert_costs[:,j]), a = expert_costs[:,j],density=True)[0]

        jsd_per_criterion[j] = distance.jensenshannon(cost_distribution_sample, cost_distribution_expert)
        
        jsd_final += jsd_per_criterion[j]*prw # TODO Jensen shannon does not work for some reason...

    return jsd_final, jsd_per_criterion


def profiled_aggregated_similarity_Agourogiannis_et_al_2023(sample_costs, expert_costs, preference_weights=(1,1,1)):

    if isinstance(preference_weights, tuple):
        np_prefs = np.array(preference_weights, dtype=np.float64)/np.sum(preference_weights)
        assert np.allclose(np.asarray([1.0,]), np.array([np.sum(np_prefs),]))
        
        res =  1.0 - np.abs(1.0- 1.0/np.dot(np.maximum(sample_costs,expert_costs)/np.minimum(expert_costs, sample_costs), np_prefs))
        assert np.all(res >= 0.0)
        return res
    if isinstance(preference_weights, torch.Tensor):
        assert torch.allclose(torch.as_tensor([1.0,]), torch.tensor([torch.sum(preference_weights),]))
        
        res = 1.0 - torch.abs(1.0- 1.0/torch.dot(torch.maximum(sample_costs,expert_costs)/torch.minimum(sample_costs, expert_costs), preference_weights))
        assert torch.all(res >= 0.0)
        return res
def jaccard(traj, traj2, cost_model, profile, preprocessing):
    t1, t2 = set(traj.obs), set(traj2.obs)
    intersection = t1.intersection(t2)
    union = t1.union(t2)
    intersection_cost = np.sum([cost_model(profile, normalization=preprocessing)(obs) for obs in intersection])
    union_cost = np.sum([cost_model(profile, normalization=preprocessing)(obs) for obs in union])

    return intersection, union, len(intersection)/len(union), intersection_cost/union_cost

def jaccard_similarity(cost_model, profile_for_cost_model, sample_trajs, expert_trajs, profile_criteria, preprocessing= FeaturePreprocess.NORMALIZATION, return_per_od=False):
    trajs_per_od = dict()
    n_trajs_per_od = dict()
    
    jaccard_per_pr = dict()

    jaccard_normal = 0.0

    jaccard_cost = 0.0

    jacc_per_od_normal = dict()
    jacc_per_od_cost = dict()
    jacc_per_od_pr = dict()

    jaccs_normal = []
    jaccs_cost = []
    jaccs_per_pr = {pr: [] for pr in profile_criteria}

    for sampled_traj in sample_trajs:
        od_sampled_traj = (sampled_traj.infos[0]['orig'], sampled_traj.infos[0]['des'])
        if od_sampled_traj not in trajs_per_od.keys():
            trajs_per_od[od_sampled_traj] = {'sample': [], 'expert': []}
        trajs_per_od[od_sampled_traj]['sample'].append(sampled_traj)
    
    for expert_traj in expert_trajs:
        od_expert_traj = (expert_traj.infos[0]['orig'], expert_traj.infos[0]['des'])
        if od_expert_traj not in trajs_per_od.keys():
            continue
        trajs_per_od[od_expert_traj]['expert'].append(expert_traj)

    ods= []
    for od, sampled_and_experts in trajs_per_od.items():
        ods.append(od)
        jacc_per_od_normal[od] = []
        jacc_per_od_cost[od] = []
        jacc_per_od_pr[od] = {pr: [] for pr in profile_criteria}
        for t in sampled_and_experts['sample']:
            t_to_experts_normal = []
            t_to_experts_cost = []
            t_to_experts_per_pr = {pr: [] for pr in profile_criteria}
            for e in sampled_and_experts['expert']:
                intersection, union, normal_jaccard, cost_jaccard = jaccard(t,e,cost_model=cost_model, profile=profile_for_cost_model, preprocessing=preprocessing)
                for pr in profile_criteria:
                    cost_jaccard_pr = np.sum([cost_model(pr, normalization=preprocessing)(obs) for obs in intersection])/ np.sum([cost_model(pr, normalization=preprocessing)(obs) for obs in union])
                    t_to_experts_per_pr[pr].append(cost_jaccard_pr)
                t_to_experts_cost.append(cost_jaccard)
                t_to_experts_normal.append(normal_jaccard)
            jacc_per_od_normal[od].append(np.mean(t_to_experts_normal))
            jacc_per_od_cost[od].append(np.mean(t_to_experts_cost))
            for pr in profile_criteria:
                jacc_per_od_pr[od][pr].append(np.mean(t_to_experts_per_pr[pr]))
        jaccs_normal.append(np.mean(jacc_per_od_normal[od]))
        jaccs_cost.append(np.mean(jacc_per_od_cost[od]))
        for pr in profile_criteria:
            jaccs_per_pr[pr].append(np.mean(jacc_per_od_pr[od][pr]))

    for pr in profile_criteria:
        jaccard_per_pr[pr] = np.mean(jaccs_per_pr[pr])
    jaccard_normal = np.mean(jaccs_normal)
    jaccard_cost = np.mean(jaccs_cost)
    if return_per_od:
        return jaccs_per_pr, jaccs_normal, jaccs_cost, ods
    return jaccard_per_pr, jaccard_normal, jaccard_cost
def calculate_expected_similarities_and_std(cost_model, profile_for_cost_model, sample_trajs, expert_trajs, profile_criteria, preprocessing= FeaturePreprocess.NORMALIZATION, standarize_cumulative_costs=False, return_ods=False):
    
    jensen_distances_per_od = dict()
    agourogiannis_similarity_per_od = dict()

    trajs_costs_per_od = dict()
    
    n_trajs_per_od = dict()
    for sampled_traj in sample_trajs:
        od_sampled_traj = (sampled_traj.infos[0]['orig'], sampled_traj.infos[0]['des'])
        if od_sampled_traj not in trajs_costs_per_od.keys():
            trajs_costs_per_od[od_sampled_traj] = {'sample': np.asarray([0.0 for pr in profile_criteria]), 'expert': np.asarray([0.0 for pr in profile_criteria])}
            n_trajs_per_od[od_sampled_traj] = {'sample': 0, 'expert': 0}
            

        trajs_costs_per_od[od_sampled_traj]['sample'] += np.asarray([np.sum(
            np.asarray([cost_model(pr, normalization=preprocessing)(
                sampled_traj.obs[i]) for i in range(len(sampled_traj))], dtype=np.float64)) for pr in profile_criteria])
        n_trajs_per_od[od_sampled_traj]['sample'] += 1
    
    for expert_traj in expert_trajs:
        od_expert_traj = (expert_traj.infos[0]['orig'], expert_traj.infos[0]['des'])
        if od_expert_traj not in trajs_costs_per_od.keys():
            continue
        trajs_costs_per_od[od_expert_traj]['expert'] += np.asarray([np.sum(
            np.asarray([cost_model(pr, normalization=preprocessing)(
                expert_traj.obs[i]) for i in range(len(expert_traj))], dtype=np.float64)) for pr in profile_criteria])
        n_trajs_per_od[od_expert_traj]['expert'] += 1

    """if society_trajs is not None:
        for soc_traj in society_trajs:
            od_soc_traj = (society_trajs.infos[0]['orig'], soc_traj.infos[0]['des'])
            if od_soc_traj not in trajs_costs_per_od.keys():
                continue
            trajs_costs_per_od[od_soc_traj]['society'] += np.asarray([np.sum(
                np.asarray([cost_model(pr, normalization=preprocessing)(
                    soc_traj.obs[i]) for i in range(len(soc_traj))], dtype=np.float64)) for pr in profile_criteria])
            n_trajs_per_od[od_soc_traj]['society'] += 1"""
    
    all_costs = {'sample': [], 'expert': []}
    ods = []
    for od, sample_and_expert_trajs in trajs_costs_per_od.items():
        ods.append(od)
        if n_trajs_per_od[od]['expert'] == 0 or n_trajs_per_od[od]['sample'] == 0:
            trajs_costs_per_od.pop(od)
            
            continue
        all_costs['sample'].append(np.asarray(sample_and_expert_trajs['sample'])/n_trajs_per_od[od]['sample'])
        all_costs['expert'].append(np.asarray(sample_and_expert_trajs['expert'])/n_trajs_per_od[od]['expert']) # coste medio de ecologia de todas las rutas.
    
    all_costs['sample'] = np.asarray(all_costs['sample'])
    all_costs['expert'] = np.asarray(all_costs['expert'])
    
    agourogiannis_similarity = profiled_aggregated_similarity_Agourogiannis_et_al_2023(all_costs['sample'] , all_costs['expert'] , preference_weights=profile_for_cost_model)
    
    #jensen_distance, jsd_per_criterion = jensen_shannon_distance_at_weighted_criteria(all_costs['sample'] , all_costs['expert'] , preference_weights=profile_for_cost_model)
    
    if standarize_cumulative_costs:
        for pr in range(len(profile_criteria)):
            all_costs['sample'][:,pr] = (all_costs['sample'][:,pr]-np.min(all_costs['sample'][:,pr]))/(np.max(all_costs['sample'][:,pr])-np.min(all_costs['sample'][:,pr]))
            all_costs['expert'][:,pr] = (all_costs['expert'][:,pr]-np.min(all_costs['expert'][:,pr]))/(np.max(all_costs['expert'][:,pr])-np.min(all_costs['expert'][:,pr]))
    
    if return_ods:
        return all_costs, agourogiannis_similarity, ods
    return all_costs, agourogiannis_similarity

def calculate_expected_cost_and_std(cost_model, profile_for_cost_model, trajectories, preprocessing = None, mode='cost_model', bins=30, file_name='expected_costs', plot_histogram=False, standarize_cumulative_costs=False):
    
    if mode == 'cost_model':
        total_costs = []
        total_costs_by_od = dict()
        n_trajs_per_od = dict()


        for traj in trajectories:
            cost_vector = np.asarray([cost_model(profile_for_cost_model, 
                                                normalization=preprocessing)(traj.obs[i]) for i in range(len(traj))])
            total_costs.append(np.sum(cost_vector))
            od = (traj.infos[0]['orig'], traj.infos[0]['des'])
            if od not in total_costs_by_od.keys():
                total_costs_by_od[od] = 0
                n_trajs_per_od[od] = 0 
            total_costs_by_od[od] += np.sum(cost_vector)
            n_trajs_per_od[od] += 1

        allcosts = []
        for od in total_costs_by_od.keys():
            total_costs_by_od[od] /= n_trajs_per_od[od]
            allcosts.append(total_costs_by_od[od])
        allcosts = np.asarray(allcosts)
    elif mode == 'societal_cost':


        total_costs = []
        total_costs_by_odpr = dict()
        n_trajs_per_odpr = dict()


        for traj in trajectories:
            cost_vector = np.asarray([cost_model(tuple(traj.infos[0]['profile']), 
                                                normalization=preprocessing)(traj.obs[i]) for i in range(len(traj))])
            total_costs.append(np.sum(cost_vector))
            odpr = ((traj.infos[0]['orig'], traj.infos[0]['des']), tuple(traj.infos[0]['profile'])) 
            if odpr not in total_costs_by_odpr.keys():
                total_costs_by_odpr[odpr] = 0
                n_trajs_per_odpr[odpr] = 0 
            total_costs_by_odpr[odpr] += np.sum(cost_vector)
            n_trajs_per_odpr[odpr] += 1

        total_costs_group_by_od = dict()
        for odpr in total_costs_by_odpr.keys():
            #proportion = float(np.asarray(profile_for_cost_model)[np.where(np.asarray(odpr[1], dtype=np.float64)==1.0)])
            
            total_costs_by_odpr[odpr] /= n_trajs_per_odpr[odpr]
            od = odpr[0]
            if od not in total_costs_group_by_od.keys():
                total_costs_group_by_od[od] = []
            total_costs_group_by_od[od].append(total_costs_by_odpr[odpr])
        allcosts = []
        for od in total_costs_group_by_od.keys():
            allcosts.append(sum(total_costs_group_by_od[od])/len(total_costs_group_by_od[od]))

        allcosts = np.asarray(allcosts)

    #avg_cost = np.mean(allcosts)
    #std_dev = np.std(allcosts)

    if standarize_cumulative_costs:
        allcosts = (allcosts)/(np.max(allcosts))
    
    avg_cost = np.mean(allcosts)
    std_dev = np.std(allcosts)

    if plot_histogram:
        plt.hist(allcosts, bins=bins)
        plt.savefig('results/value_system_identification/cost_distribution_plots/'+file_name+f'_cost_calculated_assuming_profile_{profile_for_cost_model}.png')
        plt.close()

    return avg_cost, std_dev, allcosts


class TabularPolicyPerProfile(BasePolicy):
    """A tabular policy. Cannot be trained -- prediction only."""

    pi: np.ndarray
    rng: np.random.Generator

    def __init__(
        self,
        state_space: gym.Space,
        action_space: gym.Space,
        pi: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Builds TabularPolicy.

        Args:
            state_space: The state space of the environment.
            action_space: The action space of the environment.
            pi: A tabular policy. Three-dimensional array, where pi[t,s,a]
                is the probability of taking action a at state s at timestep t.
            rng: Random state, used for sampling when `predict` is called with
                `deterministic=False`.
        """
        assert isinstance(state_space, gym.spaces.Discrete) or isinstance(state_space, gym.spaces.MultiDiscrete), "state not tabular"
        assert isinstance(action_space, gym.spaces.Discrete), "action not tabular"
        # What we call state space here is observation space in SB3 nomenclature.
        
        super().__init__(observation_space=
                         gym.spaces.MultiDiscrete((state_space.n,state_space.n)) 
                         if isinstance(state_space, gym.spaces.Discrete) 
                         else gym.spaces.MultiDiscrete((state_space.nvec[0],state_space.nvec[1])), action_space=action_space)
        self.rng = rng
        self.pi_by_profile = dict()
        self.profile = None
        self.use_sampler = False
        self.sampler = None
        
        self.set_pi_or_policy(pi)

    def set_pi_or_policy(self, pi: Union[np.ndarray, Dict]) -> None:
        """Sets tabular policy to `pi`."""

        if isinstance(pi, dict):
            self.use_sampler =False
            for pr, pi_pr in pi.items():
                assert pi_pr.ndim == 3, "expected three-dimensional policy"
                assert np.allclose(pi_pr.sum(axis=-1), 1), "policy not normalized"
                assert np.all(pi_pr >= 0), "policy has negative probabilities"
                self.pi_by_profile[pr] = pi_pr
        elif isinstance(pi, SimplePolicy) or isinstance(pi, ValueIterationPolicyRoadWorld):
            self.use_sampler = True
            self.sampler = pi
        else:
            assert pi.ndim == 3, "expected three-dimensional policy"
            assert np.allclose(pi.sum(axis=-1), 1), "policy not normalized"
            assert np.all(pi >= 0), "policy has negative probabilities"
            self.use_sampler = False
            self.pi = pi

    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        raise NotImplementedError("Should never be called as predict overridden.")
    def set_profile(self, profile):
        self.profile = profile
        if self.use_sampler:
            return
        else:
            self.pi = self.pi_by_profile[profile]

    def forward(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> NoReturn:
        raise NotImplementedError("Should never be called.")  # pragma: no cover

    def predict(
        self,
        observation: Union[np.ndarray, Mapping[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Based on https://imitation.readthedocs.io/en/latest/algorithms/mce_irl.html
        """
        if state is None:
            timesteps = np.zeros(len(observation), dtype=int)
        else:
            assert len(state) == 1
            timesteps = state[0]
        assert len(timesteps) == len(observation), "timestep and obs batch size differ"

        if episode_start is not None:
            timesteps[episode_start] = 0

        actions: List[int] = []
        for obs, t in zip(observation, timesteps):
        
            assert self.observation_space.contains(obs), "illegal state"
            if self.use_sampler:
                act, _ = self.sampler.get_action(obs, exploration=0.0, training=False, 
                                                     stochastic=not deterministic, profile=self.profile)
                actions.append(act)   
            else:
                dist = self.pi[obs[1], obs[0], :]
                if deterministic:
                    actions.append(int(dist.argmax()))
                else:
                    actions.append(self.rng.choice(len(dist), p=dist))

        timesteps += 1  # increment timestep
        state = (timesteps,)
        return np.array(actions), state


class SimplePolicy(ValueSystemLearningPolicy):

    def fit_to_profile(self, profile, *args, **kwargs):
        return self.train(profile=profile, *args, **kwargs)
    
    def from_environment_expert(environment: RoadWorldGym, profiles = BASIC_PROFILES, custom_cost= None, on_od_list=None):
        pol = SimplePolicy(environment)
        _state_to_action = dict()
        
        def _sample_actions(state, exploration=0, training=False, with_probabilities=False, with_logits=False, profile=None,stochastic=False):
            
            assert with_logits is False and with_probabilities is False and stochastic is False
            #print(len(_state_to_action.keys()))
            edge_state = environment.get_edge_to_edge_state(state, to_tuple=True)
            
            av_actions = np.asarray(environment.get_available_actions_from_state(edge_state))
            
            if np.random.rand() >= exploration:
                actions = _state_to_action.get((edge_state, profile), None)
                if actions is None:
                    if edge_state[0] == edge_state[1]:
                        actions =  environment.get_available_actions_from_state((edge_state[1], edge_state[1]))
                        _state_to_action[((edge_state[1], edge_state[1]),profile)] = actions
                        
                    else:
                        actions = []
                        path = environment.shortest_path_edges(profile=profile, to_state=edge_state[1], from_state=edge_state[0], with_length=False, all_alternatives=False, 
                                                                        custom_cost=custom_cost, flattened=False)
                        for iedge in range(1,len(path)):
                            prev_edge = path[iedge-1]
                            edge = path[iedge]
                            prev_state = (prev_edge, edge_state[1])
                            av_actions_prev = np.asarray(environment.get_available_actions_from_state(prev_state))
                            av_states = [environment.get_state_des_transition(prev_state, av_c) for av_c in av_actions_prev]
                            estate = (edge, edge_state[1])
                            istate_or_act = av_states.index(estate)
                            optimal_action = av_actions_prev[istate_or_act]

                            if iedge == 1:
                                actions.append(optimal_action)

                            _state_to_action[(prev_state, profile)] = [optimal_action,]
                if actions is not None or len(actions) > 0:
                    probs = np.zeros_like(np.asarray(av_actions), dtype=np.float64)
                    for a in actions:
                        probs[np.where(av_actions==a)[0][0]] = 1.0/float(len(actions))
                else: 
                    actions = [np.random.choice(av_actions),]
                    probs = np.ones_like(av_actions)/av_actions.shape[0]

            else: 
                actions = [np.random.choice(av_actions),]
                probs = np.ones_like(av_actions)/av_actions.shape[0]

            return actions, av_actions, probs 
        pol.sample_actions = _sample_actions
        return pol
        
    def from_environment_expert_old(environment: RoadWorldGym, profiles = BASIC_PROFILES, custom_cost= None, on_od_list=None):
        expert_trajs = list()
        for od in (environment.od_list_int if on_od_list is None else on_od_list):
            #print("new od", od)
            for pr in profiles:
                #print(pr)
                paths = environment.shortest_path_edges(profile=pr, to_state=od[1], from_state=od[0], with_length=False, all_alternatives=True, 
                                                                    custom_cost=custom_cost, flattened=False)
                #print("done", len(paths))
                for path in paths:
                    len_traj = len(path)
                    actions = [0]*(len_traj-1)
                    obs = [0]*len_traj
                    infos = [{}]*(len_traj-1)
                    terminal = [False]*len_traj
                    terminal[-1] = True
                    for ie, edge in enumerate(path[0:len_traj-1]):
                        state_des = (edge, od[1])
                        obs[ie] = state_des
                        av_acts = environment.get_available_actions_from_state(state_des)
                        for a in av_acts:
                            new_st = environment.get_state_des_transition(state_des, a)
                            #print(new_st, paths[ie+1])
                            if new_st[0] == path[ie+1]:
                                actions[ie] = a
                                break
                        infos[ie] = dict()
                        infos[ie]['profile'] = pr
                        infos[ie]['orig'] = od[0]
                        infos[ie]['des'] = od[1]
                    obs[-1] = (path[-1], od[1])
                    expert_traj = Trajectory(
                        obs=obs,
                        acts=actions,
                        infos=infos,
                        terminal=terminal
                    )
                    expert_trajs.append(expert_traj)

        return SimplePolicy.from_trajectories(expert_trajs, environment)
        

    def from_trajectories(expert_trajectories: Iterable[Trajectory],  real_env: RoadWorldGym):
        pol = SimplePolicy(real_env)
        _state_to_action = dict()

        def _sample_actions(state, exploration=0, training=False, with_probabilities=False, with_logits=False, profile=None,stochastic=False):
            print("FT???")
            assert with_logits is False and with_probabilities is False and stochastic is False
            edge_state = real_env.get_edge_to_edge_state(state, to_tuple=True)
            
            av_actions = np.asarray(real_env.get_available_actions_from_state(edge_state))
            
            if np.random.rand() >= exploration:
                actions = _state_to_action.get((edge_state, profile), None)
                if actions is None:
                    actions = []
                    for traj in expert_trajectories:
                        if traj.infos[0]['profile'] != profile:
                            continue
                        for obs, act in zip(traj.obs[:-1], traj.acts):
                            estate = real_env.get_edge_to_edge_state(obs, to_tuple=True)
                            if estate == edge_state:
                                actions.append(act)
                                break
                    if len(actions) == 0:
                        print(state)
                        print(profile)
                        print(set(traj.infos[0]['profile'] for traj in expert_trajectories))
                        print(profile in set(traj.infos[0]['profile'] for traj in expert_trajectories))
                        
                        m = min(len(traj) for traj in expert_trajectories)
                        for traj in expert_trajectories:
                            if m == len(traj):
                                print(traj)
                                print(traj.infos[0]['orig'], traj.infos[0]['des'])

                    _state_to_action[(edge_state, profile)] = actions
                if actions is not None or len(actions) > 0:
                    probs = np.zeros_like(np.asarray(av_actions), dtype=np.float64)
                    for a in actions:
                        probs[np.where(av_actions==a)[0][0]] = 1.0/float(len(actions))
                else: 
                    actions = [np.random.choice(av_actions),]
                    probs = np.ones_like(av_actions)/av_actions.shape[0]

            else: 
                actions = [np.random.choice(av_actions),]
                probs = np.ones_like(av_actions)/av_actions.shape[0]

            return actions, av_actions, probs 
        pol.sample_actions = _sample_actions
        return pol
    
    def from_policy_matrix(pi: np.ndarray, real_env: RoadWorldGym):
         
        pol = SimplePolicy(real_env)
        
    
        def _sample_actions(state, exploration=0, training=False, with_probabilities=False, with_logits=False, profile=None,stochastic=False):
            
            edge_state = real_env.get_edge_to_edge_state(state, to_tuple=True)
            
            av_actions = np.asarray(real_env.get_available_actions_from_state(edge_state))
            
            if np.random.rand() >= exploration:
                if isinstance(pi, np.ndarray):
                    probs = pi[edge_state[1], edge_state[0],:]
                    av_action_probs = pi[edge_state[1], edge_state[0], av_actions]
                else:
                    probs = pi[profile][edge_state[1], edge_state[0],:]
                    av_action_probs = pi[profile][edge_state[1], edge_state[0], av_actions]
                #print(av_action_probs)
                assert len(av_action_probs) > 0
                av_action_probs/=np.sum(av_action_probs)
                if stochastic:
                    action = np.random.choice(av_actions, p=av_action_probs)
                    actions = [action,]
                else:
                    #av_action_probs = pi[edge_state[1], edge_state[0], av_actions]
                    
                    max_prob = np.max(av_action_probs)
                    max_q = np.where(av_action_probs == max_prob)[0]
                    actions = av_actions[max_q]
                    #print(actions)
                    assert len(actions) > 0
                    if tuple(edge_state) == (226, 591) and False:
                        print(av_action_probs)
                        print(av_actions)
                        print("REWARDS", [real_env.get_reward(edge_state[0],a,edge_state[1],tuple(profile)) for a in av_actions])
                        print("FINAL ACTIONS", actions)
                        exit(0)

                    #probs = np.zeros_like(np.asarray(av_actions), dtype=np.float32)
                    #probs[av_actions==action] = 1.0

            else: 
                actions = [np.random.choice(av_actions),]
                probs = np.ones_like(av_actions)/av_actions.shape[0]

            return actions, av_actions, probs 
        pol.sample_actions = _sample_actions
        
        return pol

    def from_sb3_policy(policy: TabularPolicyPerProfile, real_env: RoadWorldGym):

        if policy.use_sampler: 
            return policy.sampler
        pol = SimplePolicy(real_env)
        def _sample_actions(state, exploration=0, training=False, with_probabilities=False, with_logits=False, profile=None,stochastic=False):
            assert with_logits is False and with_probabilities is False
            edge_state = real_env.get_edge_to_edge_state(state, to_tuple=True)
            
            av_actions = np.asarray(real_env.get_available_actions_from_state(edge_state))
            
            if np.random.rand() >= exploration:
                
                actions, _ = policy.predict(observation=[edge_state,], deterministic=not stochastic)
                action = actions[0]

                probs = np.zeros_like(np.asarray(av_actions), dtype=np.float64)
                probs[av_actions==action] = 1.0

            else: 
                action = np.random.choice(av_actions) 
                probs = np.ones_like(av_actions)/av_actions.shape[0]

            return [action,], av_actions, probs 
        pol.sample_actions = _sample_actions
        return pol

    def __init__(self, env: RoadWorldGym, use_checkpoints=False):
        super().__init__(env=env, use_checkpoints=use_checkpoints)
        self.environ = env
        


    def set_training_data(self, pre_reset):
        if pre_reset is not None:
            self.environ.od_list = pre_reset[0]
            self.environ.od_dist = pre_reset[1]
    
    def _save_checkpoint(self, save_last=True):
        pass
    
    def fit(self, *args, **kwargs):
        return self.train(*args, **kwargs)
    
    def train(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def act(self, *args, **kwargs):
        return self.get_action(*args,**kwargs)[0]

    def sample_actions(self, state, exploration = 0, training=False, with_probabilities=True, with_logits=False, stochastic=False, profile=None)-> tuple[List[int], List[int], List[float]]:
        
        return [], [], [] #sampled, available ones, q-value function for the available ones as probabilties in [0,1], act estimated values

    def get_action(self, state, exploration=0, training=False, stochastic=False, profile=None):
        sampled, available_acts, act_probs = self.sample_actions(state, exploration=exploration, training=training, with_probabilities=stochastic, with_logits=False, profile=profile, stochastic=stochastic)  # Exploit learned values
        action = np.random.choice(sampled)
        index = list(available_acts).index(action)
        return action, index
    
    
    def get_reward_model_from_profile(self, profile):
        return lambda s,a,d: self.environ.get_reward(s,a,d,profile=profile) # if None, environment resorts to its default reward
    
    
    
    def sample_trajectory(self, start, des, stochastic=False, profile=None, t_max=1000, only_states=True, with_policy_observations=False):
        terminated = False
        truncated = False
        profile = profile if profile is not None else self.environ.last_profile
        policy_state, info = self.environ.reset(st=start,des= des, profile=profile)

        #reward_function = lambda s,a,d: 1 # for fast computation

        if only_states:
            edge_path = []
            path = []
            edge_path.append(self.environ.cur_state)
            t = 0
            while not (terminated or truncated) and t < t_max:
                
                action, index = self.get_action(policy_state, exploration=0, training=False, stochastic=stochastic, profile=profile)
                #print(self.environ.cur_state, action, index)
                policy_state, rew, terminated, truncated, info = self.environ.step(action)
                
                edge_path.append(self.environ.cur_state)
            
                t+=1
            return edge_path
        else:
            obs = [policy_state if with_policy_observations else self.environ.cur_state]
            acts = []
            infos = []
            terminal = []
            #edge_path.append(self.environ.cur_state)
            t = 0
            while not (terminated or truncated) and t < t_max:
                
                action, index = self.get_action(policy_state, exploration=0, training=False, stochastic=stochastic, profile=profile)
                
                policy_state, rew, terminated, truncated, info = self.environ.step(action)
                obs.append(policy_state if with_policy_observations else self.environ.cur_state)
                terminal.append(terminated)
                #state_des = self.environ.get_edge_to_edge_state(obs)

                acts.append(action)
                info['orig'] = start
                info['des'] = des
                info['profile'] = profile
                
                infos.append(info)
                t+=1
            
            return Trajectory(obs=obs, acts=acts, infos = infos, terminal=terminal)
    def sample_trajectories(self, stochastic=False, repeat_per_od = 1, with_profiles=[(1.0,0.0,0.0),], t_max=100, od_list=None):
        trajs = []
        od_list = od_list if od_list is not None else self.environ.od_list_int
        for r in range(repeat_per_od):
            for pr in with_profiles:
                pr = tuple(pr)
                trajs.extend([self.sample_trajectory(start=s, des = d, t_max=t_max, stochastic=stochastic, profile=pr, only_states=False, with_policy_observations=True) for (s,d) in od_list])
        return trajs

    def sample_path(self, start, des, stochastic=False, profile=None, t_max=1000):
        edge_path = self.sample_trajectory(start, des, stochastic, profile, t_max, with_policy_observations=True, only_states=True)
        path = self.environ.edge_path_to_node_path(edge_path)
        
        return path, edge_path
    


RESET_VAL = -1.0

class ValueIterationPolicyRoadWorld(SimplePolicy):
    def __init__(self, env: RoadWorldGym):
        super().__init__(env)
        self.values = defaultdict(lambda: np.zeros(1))
        self.all_states = []
        self.policy = defaultdict(lambda: np.zeros(1))
        self.n_states = 0
        n_edges = max(self.environ.valid_edges)+1
        
        for state in env.valid_edges:
                for des in env.valid_edges:
                    n_actions = len(self.environ.get_available_actions_from_state((state,des)))
                    self.all_states.append((state,des))
                    self.policy[(state,des)] = np.ones(n_actions)/n_actions
                    self.n_states+=1
        self.values = np.full((n_edges, n_edges), fill_value=RESET_VAL, dtype=np.float64)
        np.fill_diagonal(self.values, 0.0)

        self.q_vals = np.zeros((n_edges, n_edges, n_edges), dtype=np.float64)

        od_list_int = self.environ.od_list_int
        #print("TRAIN WITH: ", start_0, end_0)

        goal_edges_with_rep = [od[1] for od in od_list_int]
        self.goal_edges = np.array(list(set(goal_edges_with_rep)))
        needed_states_set = list(set(s for s in self.all_states if s[1] in self.goal_edges))
        needed_states = np.array(needed_states_set)
        self.needed_edges = np.array(list(set(e for e,d in needed_states)))
        self.cached_states_actions = dict()
        self.cached_rewards = dict()

        self.cached_states_actions = dict()

        self.min_abs_reward = 1.0

        self.pi_from_d = np.zeros((self.environ.n_states, self.environ.n_states, self.environ.n_actions), dtype=np.float64)
        
        for des in self.environ.states:
            self.pi_from_d[des,self.environ.policy_mask] = 1.0
            self.pi_from_d[des, :,:]/=self.pi_from_d[des,:,:].sum(axis=-1, keepdims=True)        
    
    def train(self, error=1.0, verbose=True, profile=[1, 0, 0], custom_reward_function=None, reference_experts=None):
        
        #policy.set_training_data(([od_list[0],], [od_dist[0],]))
        #policy.get_reward_model_from_profile = lambda profile: (lambda s,a,d: env.get_reward(s,a,d))
        self.value_iteration(
            error=error, 
            profile=profile, 
            verbose=verbose, 
            custom_reward=custom_reward_function, 
            expert_paths_per_od_profile=reference_experts)
    

    def _retrieve_states_actions_and_rewards_from_state_des(self, state_des, reward_function):
        
        actions_from_s, states_from_s = self.cached_states_actions.get(state_des, (None,None))
        if actions_from_s is None:
            actions_from_s = np.asarray(self.environ.get_available_actions_from_state(state_des), dtype=np.int16)
            states_from_s = np.asarray([self.environ.get_state_des_transition(state_des, a)[0] for a in actions_from_s], dtype=np.int16)
            self.cached_states_actions[state_des] = (actions_from_s, states_from_s)
        if reward_function is not None:
            rewards = self.cached_rewards.get(state_des, None)
            if rewards is None:
                if state_des[0] != state_des[1]:
                    rewards = np.asarray([reward_function(state_des[0], a, state_des[1]) for a in actions_from_s], dtype=np.float64)
                else:
                    rewards = np.zeros_like(actions_from_s, dtype=np.float64)
                self.cached_rewards[state_des] = rewards
        else:
            rewards = []


        return actions_from_s, states_from_s, rewards
        
    def value_iteration(self, error=0.05, profile=None, verbose=False, custom_reward=None, reset_vi=True, goal_subset=None, expert_paths_per_od_profile=None, full_match=True, od_list = None, reset_with_values_and_qs=None):
        od_list_used = self.environ.od_list_int if od_list is None else od_list

        needed_edges = self.needed_edges
        goals = self.goal_edges if goal_subset is None else list(set(goal_subset))
        
        if od_list is not None:
            goals = list(set([od[1] for od in od_list_used]))


        reward_function = self.get_reward_model_from_profile(profile=profile) if custom_reward is None else custom_reward
        actual_expert_paths_per_od_profile = dict()
        
        if expert_paths_per_od_profile is not None:
            if isinstance(expert_paths_per_od_profile[list(expert_paths_per_od_profile.keys())[0]][0], Trajectory):
                actual_expert_paths_per_od_profile = dict()
                for k,trajs in expert_paths_per_od_profile.items():
                    actual_expert_paths_per_od_profile[k] = [[o[0] for o in t.obs] for t in trajs]
            
        else:
            actual_expert_paths_per_od_profile = expert_paths_per_od_profile
        
        if reset_vi:
            print("RESETTED VI")
            self.values[needed_edges][:, goals] = RESET_VAL
            self.q_vals.fill(0.0)
        np.fill_diagonal(self.values, 0.0)
        #needed_states = zip(needed_edges, goals)

        self.cached_rewards = dict() # New rewards needed to be recalculated.
        iteration = 0
        diff= 10000
        self.min_abs_reward = error
        ds = 1000

        if reset_with_values_and_qs is not None:
            for s in needed_edges:
                for d in goals:
                    self.values[s,d] = reset_with_values_and_qs[0][s,d]
                    self.q_vals[s, :, d] = reset_with_values_and_qs[1][s,:,d]
        while True:
            np.fill_diagonal(self.values, 0.0) # Alg 2, line 3 in Wulfmeier 2015
            v = self.values.copy()
                          
            for s in needed_edges:
                for d in goals:
                    actions_from_s, states_from_s, rewards = self._retrieve_states_actions_and_rewards_from_state_des((s,d), reward_function)
                    assert np.all(rewards <= 0.0) 
                    
                    if s == d:
                        assert np.allclose(rewards, np.zeros_like(rewards))
                        assert all(x == states_from_s[0] for x in states_from_s)
                    else:
                        if iteration == 0:
                            if np.any(rewards != 0.0):
                                abs_rews = np.abs(rewards)
                                self.min_abs_reward = min(min(abs_rews[abs_rews>0]), self.min_abs_reward)
                    
                    self.q_vals[s,actions_from_s, d] = rewards + 0.9999*self.values[states_from_s, d]
                    
                    if s!= d:
                        self.values[s, d] = np.max(self.q_vals[s,actions_from_s,d])
                    else:
                        self.values[s,d] = 0.0
                    if (iteration + 1)% 10 == 0:
                        q_diff = self.q_vals[s,actions_from_s,d]-self.values[s,d]
                        self.policy[(s,d)] = softmax(q_diff)
                        self.pi_from_d[d,s,:] = 0.0
                        self.pi_from_d[d, s, actions_from_s] = self.policy[(s,d)]
            last_diff = diff
            ls = ds
            diff = np.max(np.abs(self.values - v))
            ds = np.sum(np.abs(self.values - v))
            iteration += 1

            desired_err = min(error, self.min_abs_reward)
            if verbose:
                print("DIFF - DESIRED DIFF", diff, desired_err, np.where(np.abs(self.values - v) == diff)[0])
                print("ITERATION: ", iteration, goals)
                print("DELTA_DIFF", abs(ds - ls))
                
            assert np.allclose(self.values[goals, goals], 0.0)


            if abs(ls - ds) <= 0.00000001 and diff < desired_err:
                print("Final Diff VI: ", diff, "at it: ", iteration)
                assert diff == 0.0

                if actual_expert_paths_per_od_profile is not None:
                        good_path = False
                        n_ok = 0
                        n_matched = 0
                        for od in od_list_used:
                            path, edge_path = self.sample_path(od[0], od[1], stochastic=False, profile=profile, t_max=200)
                            
                            expert_ones = actual_expert_paths_per_od_profile[(od, profile)]
                            
                            
                            assert edge_path[0] == od[0]
                            for e in expert_ones:
                                assert e[0] == od[0] and e[-1] == od[1]
                            
                            if full_match:
                                
                                good_path = edge_path in expert_ones
                            else:
                                good_path = edge_path[0] == od[0] and edge_path[-1] == od[1]
                                
                            n_ok += 1 if good_path else 0
                            n_matched += 1 if (edge_path[0] == od[0] and edge_path[-1] == od[1]) else 0
                        
                        print("---------------------------")
                        print("NOK: ", n_ok, "MATCHED OD: ", n_matched)
                        print("---------------------------")
                        
                        if n_ok == len(od_list_used):
                            if verbose:
                                print("All paths achieve end")
                            return self.pi_from_d
                        
                break
            
        
        return self.pi_from_d
    
    def sample_actions(self, state, exploration=0, training=False, with_probabilities=False, with_logits=False, stochastic=False, profile=None) -> tuple[List[int], List[int], List[float]]:
        state = self.environ.get_edge_to_edge_state(state)
        action_probs = self.policy[state] 
        #print(action_probs)
        av_Actions = np.asarray(self.environ.get_available_actions_from_state(state))

        if stochastic:
            sampled_actions = np.array([np.random.choice(av_Actions, size=1, p=action_probs),])
        else:
            nmax = np.max(action_probs)
            max_q = np.where(action_probs == nmax)[0]
            sampled_actions = av_Actions[max_q]
        #print(sampled_actions)
        if with_logits:
            logits = np.log(action_probs/(1-action_probs))
            return sampled_actions, av_Actions, action_probs, logits
        else:
            return sampled_actions, av_Actions, action_probs




def check_policy_gives_optimal_paths(env_single, policy, profiles = [(1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)]):
    for od in env_single.od_list_int:
        for p in profiles:
            path_p, edge_path_p = policy.sample_path(start=od[0], des=od[1], t_max=1000, profile=p, stochastic=False)

            edge_path_real = env_single.shortest_path_edges(profile=p, to_state=od[1], from_state=od[0])
            print(od)
            print(edge_path_real, edge_path_p)
            assert np.allclose(np.asarray(edge_path_p), np.asarray(edge_path_real)), od
        