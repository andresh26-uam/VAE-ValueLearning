from collections import defaultdict, namedtuple
from copy import deepcopy
import enum
from typing import Union
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd

from torch.functional import F
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from gymnasium import spaces

from use_cases.roadworld_env_use_case.values_and_costs import BASIC_PROFILE_NAMES, BASIC_PROFILES, PROFILE_COLORS, PROFILE_NAMES_TO_TUPLE, VALUE_COSTS, VALUE_COSTS_PRE_COMPUTED_FEATURES
from use_cases.roadworld_env_use_case.network_utils import angle_between

from use_cases.roadworld_env_use_case.utils.load_data import load_link_feature, load_path_feature, minmax_normalization, minmax_normalization01

Step = namedtuple('Step', ['cur_state', 'action', 'next_state', 'reward', 'done'])
INVALID_EDGES = {604, 515}
NETWORK_PLOTS_DIR = 'use_cases/roadworld_env_use_case/plots/network_plots/'
PATH_FEATURES_PATH = 'use_cases/roadworld_env_use_case/data/path_features.npy'
MAX_PATH_FEATURES_PATH = 'use_cases/roadworld_env_use_case/data/max_path_features.npy'

class FeatureSelection(enum.Enum):

    ONE_HOT_ORIGIN_ONLY = 'origin_only'
    ONE_HOT_ALL = 'all'
    ONE_HOT_ORIGIN_AND_DEST = 'origin_and_dest'
    ONLY_COSTS = 'only_costs'
    DEFAULT = None



class FeaturePreprocess(enum.Enum):
    NORMALIZATION = 'norm'
    STANDARIZATION = 'std'
    NO_PREPROCESSING = 'none'

class RoadWorld(object):
    """
    Environment
    """

    def __init__(self, network_path, edge_path, pre_reset=None, origins=None,
                 destinations=None, k=8, max_iter=200):
        self.last_profile = None
        self.network_path = network_path
        self.netin = origins
        self.netout = destinations
        self.k = k
        self.max_route_length = 0
        self.max_iter = max_iter

        # define transition matrix
        netconfig = np.load(self.network_path)
        netconfig = pd.DataFrame(netconfig, columns=["from", "con", "to"])
        netconfig_dict = {}
        states_from_state = {}
        state_list_from_state = {}
        pre_state = {}
        pre_state_list = {}
        action_list_from_state = {}
        for i in range(len(netconfig)):
            fromid, con, toid = netconfig.loc[i]

            if fromid in netconfig_dict.keys():
                netconfig_dict[fromid][con] = toid
                states_from_state[fromid][toid] = toid
                state_list_from_state[fromid].append(toid)
                action_list_from_state[fromid].append(con)
            else:
                states_from_state[fromid] = {}
                state_list_from_state[fromid] = []
                netconfig_dict[fromid] = {}
                action_list_from_state[fromid] = []
                netconfig_dict[fromid][con] = toid
                states_from_state[fromid][toid] = toid
                state_list_from_state[fromid].append(toid)
                action_list_from_state[fromid].append(con)
            if toid in pre_state.keys():
                pre_state[toid][con] = fromid
                pre_state_list[toid].append(fromid)
            else:
                pre_state[toid] = {}
                pre_state_list[toid] = []
                pre_state[toid][con] = fromid
                pre_state_list[toid].append(fromid)

        self.pre_state = pre_state
        self.pre_state_list = pre_state_list
        
        self.netconfig = netconfig_dict
        self.states_from_state = states_from_state
        self.state_list_from_state = state_list_from_state
        self.action_list_from_state = action_list_from_state
        # define states and actions
        edge_df = pd.read_csv(edge_path, header=0, usecols=['n_id'])
        
        # self.terminal = len(self.states)  # add a terminal state for destination
        self.states = edge_df['n_id'].tolist()
        
        self.pad_idx = len(self.states)
        #self.states.append(self.pad_idx)
        self.actions = range(k)  # k represents action to terminal
        #self.actions = edge_df['n_id'].tolist()
        #self.actions.append(self.pad_idx)

        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        print('n_states', self.n_states)
        print('n_actions', self.n_actions)

        self.rewards = [0 for _ in range(self.n_states)]

        #self.state_action_pair = [[(s, a) for a in self.get_action_list(s)] for s in self.states]
        #print(self.state_action_pair)
        #self.link_to_link_pairs = np.asarray([[a for a in self.get_action_list(s)] for s in self.states])
        #self.num_sapair = len(self.state_action_pair)
        #print('n_sapair', self.num_sapair)

        #self.sapair_idxs = self.state_action_pair  # I think in our case this two should be the same
        self.policy_mask = np.full([self.n_states, self.n_actions], dtype=np.bool_, fill_value=False)
        self.policy_unavailable_mask = np.full([self.n_states, self.n_actions], dtype=np.bool_, fill_value=True)
        self.state_action = np.zeros([self.n_states, self.n_actions], dtype=np.int32)
        print('policy mask', self.policy_mask.shape)
        for s in self.states:
            #self.state_action[s, :] = [-1]*self.state_action.shape[1]
            for a in self.get_action_list(s):
                self.policy_mask[s, a] = True
                self.policy_unavailable_mask[s,a] = False
                self.state_action[s, a] = self.netconfig[s][a]
                
        self.cur_state = None
        self.cur_des = None

        self.valid_edges = np.asarray(list(set(self.netconfig.keys()).difference(INVALID_EDGES)))
        
        self.pre_acts_and_pre_states = dict()
        for s in self.states:
            pre_s = self.pre_state[s]
            matrix = np.asarray(list(pre_s.items()))
            self.pre_acts_and_pre_states[s] = list(matrix[:,0]), list(matrix[:,1])
            

        if pre_reset is not None:
            self.od_list = pre_reset[0]
            self.od_dist = pre_reset[1]
            self.od_list_int = [(int(od.split('_')[0]), int(od.split('_')[1])) for od in self.od_list]
        self.destination_list = [d for o,d in self.od_list_int]
        self.all_rewards_per_profile = dict()
        self.all_rewards_per_profile[self.last_profile] = dict()
        self._get_reward = None
    
    def get_unavailable_action_mask(self, states):
        return self.policy_unavailable_mask[states[:,0]]
    def get_action_mask(self, states):
        #mask = np.zeros(shape=(states.shape[0], self.n_actions))
        return self.policy_mask[states[:,0]]
    
    def _get_random_initial_state(self):
        ori = np.random.choice(np.asarray(list(set(self.valid_edges))))
        des = np.random.choice(np.asarray(list(set(self.valid_edges).difference(set([ori,])))))
        return (ori, des)

    def reset(self, seed=None, options={}, st=None, des=None, full_random=False, profile=None):
        self.iterations = 0
        if seed is not None:
            np.random.seed(seed)

        if profile is not None:
            self.last_profile = profile
        
        self.cur_state = None
        while self.cur_state == self.cur_des or self.cur_state is None:
            if st is not None and des is not None:
                self.cur_state, self.cur_des = int(st), int(des)
            else:
                if des is None:
                    if not full_random:
                        od_idx = np.random.choice(self.od_list, 1)
                        ori, des = od_idx[0].split('_')
                    else:
                        ori = np.random.choice(np.asarray(list(set(self.valid_edges))))
                        des = np.random.choice(np.asarray(list(set(self.valid_edges).difference(set([ori,])))))
                else:
                    ori = np.random.choice(np.asarray(list(set(self.valid_edges).difference(set([des,])))))      
                self.cur_state, self.cur_des = int(ori), int(des)
        self.state = (self.cur_state, self.cur_des)

        return self.cur_state, self.cur_des
    
    
    def get_reward(self, state, action, des, profile=None):
        #print("GET_REWARD AT ", state, self.netconfig[state])
        return self._get_reward(state, self.get_state_des_transition((int(state), des), action)[0], des, profile)
    
    
    def step(self, action, reward_function=None):
        """
        Step function for the agent to interact with gridworld
        inputs:
          action        action taken by the agent
        returns
          current_state current state
          action        input action
          next_state    next_state
          reward        reward on the next state
          is_done       True/False - if the agent is already on the terminal states
        """
        prev_state = self.cur_state
        try:
            next_state, self.cur_des = self.get_state_des_transition((self.cur_state,self.cur_des), action)
        except:
            tmp_dict = self.netconfig.get(self.cur_state, None)
            if tmp_dict is not None:
                next_state = tmp_dict.get(action, self.cur_state)
            else:
                next_state = self.cur_state
        
        if reward_function is not None:
            reward = reward_function(self.cur_state, action, self.cur_des)
        else:
            reward =  self.get_reward(self.cur_state, action, self.cur_des, profile=self.last_profile)
        self.cur_state = next_state
        done = (self.cur_state == self.cur_des) #or (self.cur_state == self.pad_idx)
        self.iterations += 1
        self.state = (self.cur_state, self.cur_des)
        #info = {"moved_to": self.cur_state, "reward": reward}
        
        trunc = self.iterations >= self.max_iter or action not in self.action_list_from_state[prev_state]
        done = done or action not in self.action_list_from_state[prev_state]
        return self.state, reward, done, trunc, {}

    def get_state_transition(self, state, action):
        #return self.netconfig[state][action]
        return self.netconfig[state][action]
    

    def get_state_des_transition(self, state_des, action):
        if state_des[0] == state_des[1]:
            return state_des
        else:
            return self.netconfig[state_des[0]][action], state_des[1]
        
    def get_action_list(self, state):
        if state in self.netconfig.keys():
            return list(self.netconfig[state].keys())
        else:
            return list()

    def import_demonstrations(self, demopath, od=None, n_rows=None):
        demo = pd.read_csv(demopath, header=0, nrows=n_rows)
        expert_st, expert_des, expert_ac, expert_st_next = [], [], [], []
        for demo_str, demo_des in zip(demo['path'].tolist(), demo['des'].tolist()):
            cur_demo = [int(r) for r in demo_str.split('_')]
            len_demo = len(cur_demo)
            for i0 in range(1, len_demo):
                cur_state = cur_demo[i0 - 1]
                next_state = cur_demo[i0]
                action_list = self.get_action_list(cur_state)
                j = [self.get_state_transition(cur_state, a0) for a0 in action_list].index(next_state)
                action = action_list[j]
                expert_st.append(cur_state)
                expert_des.append(demo_des)
                expert_ac.append(action)
                expert_st_next.append(next_state)
        return torch.LongTensor(expert_st), torch.LongTensor(expert_des), torch.LongTensor(expert_ac), torch.LongTensor(
            expert_st_next)

    def import_demonstrations_step(self, demopath, n_rows=None):
        demo = pd.read_csv(demopath, header=0, nrows=n_rows)
        trajs = []
        for demo_str, demo_des in zip(demo['path'].tolist(), demo['des'].tolist()):
            cur_demo = [int(r) for r in demo_str.split('_')]
            len_demo = len(cur_demo)
            episode = []
            for i0 in range(1, len_demo):
                cur_state = cur_demo[i0 - 1]
                next_state = cur_demo[i0]

                action_list = self.get_action_list(cur_state)
                j = [self.get_state_transition(cur_state, a0) for a0 in action_list].index(next_state)
                action = action_list[j]

                reward = self.get_reward(cur_state)
                is_done = next_state == demo_des

                episode.append(
                    Step(cur_state=cur_state, action=action, next_state=next_state, reward=reward, done=is_done))
            trajs.append(episode)
            self.max_route_length = len(episode) if self.max_route_length < len(episode) else self.max_route_length
        print('max_route_length', self.max_route_length)
        print('n_traj', len(trajs))
        return trajs
    
import gymnasium as gym
import networkx as nx



DATA_FOLDER = 'use_cases/roadworld_env_use_case/data/'
TRAINED_MODELS = 'use_cases/roadworld_env_use_case/trained_models/'
"""
def ini_od_dist(train_path):
    # find the most visited destination in train data
    df = pd.read_csv(train_path)
    num_trips = len(df)
    df['od'] = df.apply(lambda row: '%d_%d' % (row['ori'], row['des']), axis=1)
    df = df[['od', 'path']]
    df = df.groupby('od').count()
    df['path'] = df['path'] / num_trips
    print(df['path'].sum())
    return df.index.tolist(), df['path'].tolist()"""

import networkx as nx

def get_color_gradient(c1, c2, mix):
    """
    Given two hex colors, returns a color gradient corresponding to a given [0,1] value
    """
    c1_rgb = np.array(c1)
    c2_rgb = np.array(c2)
    
    return ((1-mix)*c1_rgb + (mix*c2_rgb))

def visualize_graph(graph: nx.Graph,posiciones, show = False, save_to="vgraph_demo.png", show_edge_weights = True, caminos_by_value={'sus': [], 'sec': [], 'eff': []}, custom_weights: dict =None, custom_weights_dest: int = None, plot_by_value=False):
    
    #posiciones = {node: node for node in self.graph.nodes()}
    

    fig, ax = plt.subplots(figsize=(30,20))
    
    nodelist = [n for n  in graph.nodes() if n not in (2389906350, 856926212)]
    edgelist = [edge for edge in graph.edges() if edge != (2389906350, 856926212) and edge != (856926212,2389906350)]
    
    #posiciones = dict([(n, posiciones[n]) for n in nodelist])
    #node_size=dict([(n, 100) if n in (END, START) else (n, 5) for n in nodelist])
    node_size = 10
    
    node_color = "tab:blue"
    #edge_color = "tab:gray"

    if custom_weights is None:
        
        max_eco = np.max([graph.get_edge_data(edge[0], edge[1])['sus'] for edge in edgelist])
        eco_vals = [graph.get_edge_data(edge[0], edge[1])['sus'] for edge in edgelist]

        max_sec = np.max([graph.get_edge_data(edge[0], edge[1])['sec'] for edge in edgelist])
        sec_vals = [graph.get_edge_data(edge[0], edge[1])['sec'] for edge in edgelist]

        max_eff = np.max([graph.get_edge_data(edge[0], edge[1])['eff'] for edge in edgelist])
        eff_vals = [graph.get_edge_data(edge[0], edge[1])['eff'] for edge in edgelist]
    else:
        max_eco = np.max([custom_weights[PROFILE_NAMES_TO_TUPLE['sus']][graph.get_edge_data(edge[0], edge[1])['id'], custom_weights_dest] for edge in edgelist])
        eco_vals = [custom_weights[PROFILE_NAMES_TO_TUPLE['sus']][graph.get_edge_data(edge[0], edge[1])['id'], custom_weights_dest] for edge in edgelist]
        
        max_sec = np.max([custom_weights[PROFILE_NAMES_TO_TUPLE['sec']][graph.get_edge_data(edge[0], edge[1])['id'], custom_weights_dest] for edge in edgelist])
        sec_vals = [custom_weights[PROFILE_NAMES_TO_TUPLE['sec']][graph.get_edge_data(edge[0], edge[1])['id'], custom_weights_dest] for edge in edgelist]
        
        max_eff = np.max([custom_weights[PROFILE_NAMES_TO_TUPLE['eff']][graph.get_edge_data(edge[0], edge[1])['id'], custom_weights_dest] for edge in edgelist])
        eff_vals = [custom_weights[PROFILE_NAMES_TO_TUPLE['eff']][graph.get_edge_data(edge[0], edge[1])['id'], custom_weights_dest] for edge in edgelist]

    edge_color_eco = [float(eco_val/max_eco) for eco_val in eco_vals]
    edge_color_eff = [float(eff_val/max_eff) for eff_val in eff_vals]
    edge_color_sec = [float(sec_val/max_sec) for sec_val in sec_vals]
    
    edge_size = [2]*len(edgelist)
    if plot_by_value:
        
        nx.draw_networkx(graph, pos=posiciones, nodelist=nodelist,edgelist=edgelist, ax=ax, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_color,  width=edge_size, edge_color=edge_color_eco, edge_cmap = plt.cm.Greens_r, edge_vmin = 0.3, edge_vmax = 1, connectionstyle=f'arc3, rad = {0.3}')
        nx.draw_networkx(graph, pos=posiciones, nodelist=nodelist,edgelist=edgelist, ax=ax, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_color,  width=edge_size, edge_color=edge_color_sec, edge_cmap = plt.cm.Blues_r, edge_vmin = 0.3, edge_vmax = 1, connectionstyle=f'arc3, rad = {0.15}')
        nx.draw_networkx(graph, pos=posiciones, nodelist=nodelist,edgelist=edgelist, ax=ax, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_color,  width=edge_size, edge_color=edge_color_eff, edge_cmap = plt.cm.Reds_r, edge_vmin = 0.3, edge_vmax = 1, connectionstyle=f'arc3, rad = {0}')

    else:

        edge_color = 'gray'
        nx.draw_networkx(graph, pos=posiciones, nodelist=nodelist,edgelist=edgelist, ax=ax, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_color,  width=edge_size, edge_color=edge_color, edge_cmap = plt.cm.Reds_r, edge_vmin = 0.3, edge_vmax = 1)


    
    
    min_eco = 0
    
    #nx.draw_networkx_edges(graph, pos=posiciones, nodelist=nodelist,edgelist=edgelist, ax=ax, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_color,  width=edge_size, edge_color=[float(eco_val/max_eco) for eco_val in eco_vals], edge_cmap = plt.cm.Reds)
    if show_edge_weights:
        edge_labels = {edge: f"Sus: {eco_val:.2f}\nSec: {sec_val:.2f}\nEff: {eff_val:.2f}" for edge, eco_val, sec_val, eff_val in zip(graph.edges(), eco_vals, sec_vals, eff_vals)}
        nx.draw_networkx_edge_labels(graph, pos=posiciones, edge_labels=edge_labels, font_size=8, font_color='black', horizontalalignment='left')
        
    else:
        edge_labels = None
    
    edge_color_by_value = {}
    edge_size_by_value = {}
    edge_list_by_value = {}
    
    for value, caminos in caminos_by_value.items():
        edge_color_by_value[value] = []
        edge_size_by_value[value] = []
        edge_list_by_value[value] = []
        arc_rad = 0.0 if value == 'sus' else 0.15 if value == 'eff' else 0.0 if value == 'sec' else 0.3
        for camino in caminos:
            if camino is not None:
                
                node_color=["tab:red" if n == camino[-1] else "tab:green" if n == camino[0] else "tab:orange" if n in camino else "tab:blue" for n in nodelist]
                node_size=[150 if n == camino[-1] else 150 if n == camino[0] else 100 if n in camino else 10 for n in nodelist]
                
                for edge in edgelist:
                        valid = False
                        if edge[0] in camino and edge[1] in camino:
                            for i in range(len(camino) - 1):
                                if camino[i] == edge[0] and camino[i + 1] == edge[1]:
                                    valid = True
                                    break
                        if valid:
                            edge_list_by_value[value].append(edge)
                            edge_color_by_value[value].append(PROFILE_COLORS.get(PROFILE_NAMES_TO_TUPLE.get(value, 'unk'),'tab:black'))
                            edge_size_by_value[value].append(5)
       
        nx.draw_networkx_nodes(graph, posiciones, nodelist=nodelist, node_size=node_size, node_color=node_color)
        nx.draw_networkx_edges(graph, posiciones, edge_list_by_value[value], connectionstyle=f'arc3, rad = {arc_rad}', width=edge_size_by_value[value], edge_color=edge_color_by_value[value], label='Shortest path for ' + str(value))
    
    #nx.draw_networkx(self.graph, pos={p: p for p in posiciones if p in camino}, ax=ax, with_labels=False, font_weight='bold', nodelist=[n for n in camino], node_size=[100 if self.points_to_data[tuple(n)]["id"] in (END, START) else 50 for n in camino], node_color=[n for n in node_color if n != 'tab:blue'], edgelist=[ed for ed in self.graph.edges() if ed[0] in camino and ed[1] in camino], edge_color=[ed for ed in edge_color if ed != 'tab:blue'], width=[ed for ed in edge_size if ed > 1.0])#nx.draw_networkx_nodes(G, pos, nodelist=[0, 1, 2, 3], node_color="tab:red", **options)
    #nx.draw_networkx_nodes(G, pos, nodelist=[4, 5, 6, 7], node_color="tab:blue", **options)
    #ax.plot(points[:,0], points[:,1], '.')
    #ax.plot(vertices[:,0], vertices[:,1], 'o')
    #ax.set_xlim([-0.05, 1.05])
    #ax.set_ylim([-0.05, 1.05])
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    fig.legend()
    fig.savefig(save_to)

    if show:
        plt.show()
    plt.close()
    




class RoadWorldGym(RoadWorld,gym.Env):
    def get_edge_to_edge_state(self, observation, to_tuple=True):
        return tuple(observation) if to_tuple else observation
    
    def node_path_to_edge_path(self, path, format_list=False):
        edge_path = []
        
        for i in range(len(path)-1):
            node_from = path[i]
            node_to = path[i+1]
            data = self.graph.get_edge_data(node_from, node_to)
            edge_path.append(str(data['id']) if not format_list else int(data['id']))
        return edge_path if format_list  else '_'.join(edge_path)
    
    def edge_path_to_node_path(self, path):
        node_path = []
        
        for i in range(len(path)):
            edge = path[i]

            node_path.append(self.edge_to_source_dest[edge]['source'])
        node_path.append(self.edge_to_source_dest[edge]['dest'])
        return node_path
    

    
    def get_available_actions_from_state(self, state_des):
        return self.action_list_from_state[state_des[0]]
    
    def get_separate_features(self, state, des, normalized=None):

        if normalized == FeaturePreprocess.NORMALIZATION:
            path_feature = self.path_feature[state, des, :]
            edge_feature = self.link_feature[state, :]
            assert torch.all(edge_feature > 0)
        elif normalized == FeaturePreprocess.STANDARIZATION:
            path_feature = self.path_feature_standarized[state, des, :] 
            #path_feature = torch.zeros_like(self.path_feature[state,des,:])
            edge_feature = self.link_feature_standarized[state, :]
        else:
            path_feature = self.path_feature_no_norm[state, des, :]
            edge_feature = self.link_feature_no_norm[state, :]

        return edge_feature, path_feature
    
    def get_real_reward_for_state_des_per_basic_profile(self, state_des):
        
        return torch.tensor([[self._get_reward_state(int(s[0]), int(s[1]),profile=pr) for pr in BASIC_PROFILES] for s in state_des], dtype=torch.float16)

    def process_features(self, state_des: torch.Tensor, 
                         feature_selection: Union[FeatureSelection, tuple]=FeatureSelection.DEFAULT, 
                         use_real_env_rewards_as_feature: bool = False, 
                         feature_preprocessing: FeaturePreprocess =FeaturePreprocess.NORMALIZATION):
        # print('state', state.shape, 'des', des.shape)
        #print(state_des[:,0], state_des[:,1])
        
        if feature_selection != FeatureSelection.DEFAULT:
            if feature_selection == FeatureSelection.ONLY_COSTS:
                edge_feature, path_feature = self.get_separate_features(state_des[:,0], state_des[:,1], normalized=feature_preprocessing)
                
                return edge_feature[:,-3:]
            if isinstance(feature_selection, tuple):
                edge_feature, path_feature = self.get_separate_features(state_des[:,0], state_des[:,1], normalized=feature_preprocessing)
                
                return edge_feature[:,-3:][:,feature_selection]
            if feature_selection == FeatureSelection.ONE_HOT_ORIGIN_ONLY:
                positions = state_des[:,0]

                features = torch.eye(self.n_states)[positions]
            elif feature_selection == FeatureSelection.ONE_HOT_ORIGIN_AND_DEST:
                #features = self.one_hot_encoding_st_des(state_des)
                features = torch.cat([torch.eye(self.n_states)[state_des[:,0]], torch.eye(self.n_states)[state_des[:,1]]], -1)
            elif feature_selection == FeatureSelection.ONE_HOT_ALL:
                edge_feature, path_feature = self.get_separate_features(state_des[:,0], state_des[:,1], normalized=feature_preprocessing)
                #print(path_feature)
                path_feature = path_feature[:, 1:(path_feature.shape[1]+1)]
                feature = torch.cat([path_feature, edge_feature], -1)  # [batch_size, n_path_feature + n_edge_feature]
                
                features = torch.cat([feature, torch.eye(self.n_states)[state_des[:,0]], torch.eye(self.n_states)[state_des[:,1]]], -1)
            else:
                raise ValueError("Unknown Feature Selection: " + str(feature_selection))
        else:
            edge_feature, path_feature = self.get_separate_features(state_des[:,0], state_des[:,1], normalized=feature_preprocessing)
            features = torch.cat([path_feature, edge_feature], -1)  # [batch_size, n_path_feature + n_edge_feature]
        
        if use_real_env_rewards_as_feature:
            rewards = self.get_real_reward_for_state_des_per_basic_profile(state_des)
        #print("REWARDS", rewards, rewards.shape)
            features = torch.cat([features, rewards], -1) 
            
        return features
    

    def cost_model(self, profile, normalization=None):
        return lambda state_des: torch.tensor(profile, dtype=torch.float32).dot(torch.tensor(
                    [VALUE_COSTS_PRE_COMPUTED_FEATURES[v](*self.get_separate_features(state_des[0], state_des[1], normalization)) for v in VALUE_COSTS_PRE_COMPUTED_FEATURES.keys()],
                     dtype=torch.float32))
    def profile_cost_minimization_path(self, des, profile=(1.0,0.0,0.0), reverse=False, custom_cost=None):
           
        def cost(u,v, weight_dict=None):
            
            if reverse:
                data = self.graph.get_edge_data(v, u)

            else:

                data = self.graph.get_edge_data(u, v)
                #print("IS NOT NONE:", data)
            state = int(data['id'])
            state_des = (state, des)
            if state == des:
                return 0.0
            if custom_cost is None:
                return self.cost_model(profile)(state_des)
            else:
                return custom_cost(state_des, profile)
        return cost
    
    def reduce_size(self):
        pass
        """remove_list = [2716408741, 856926212, 601675165, 601675044, 603891496, 603891416,1882296810, 994212301, 1168461360,603891498, 267671967, 601669728]
        for edge_to_from_con_to in self.netconfig.items():
            #edge_id = edge_to_from_con_to[0]
            edge_id = edge_to_from_con_to[0]
            node_from = edge_df.loc[edge_id]['u']
            node_to = edge_df.loc[edge_id]['v']
            if node_from in remove_list:
                print(node_from)
                try:
                    self.graph.remove_node(node_from)
                except:
                    pass
                try:
                    self.graph.remove_edge(node_from, node_to)
                except:
                    pass
                try:
                    self.graph.remove_node(node_to)
                except:
                    pass

        scc = max(nx.strongly_connected_components(self.graph), key=len)
        print(scc, len(scc))

        self.netconfig[]
        exit(0)"""
    def reset(self, seed=None, options={}, st=None, des=None, full_random=False, profile=None):
        ret, info =  super().reset(seed, options, st, des, full_random, profile), {}
        if profile is not None and profile != self.last_profile:
                self.last_profile = tuple(profile)
        return ret, info

    def _init_graph(self, edge_path, node_path):

        self.graph = nx.DiGraph()

        
        edge_df  = pd.read_csv(edge_path, dtype={'highway': str})
        node_df  = pd.read_csv(node_path, dtype={'highway': str})

        node_positions = dict()
        #graph.add_nodes_from(edges_graph)
        self.edge_to_source_dest = dict()
        for edge_to_from_con_to in self.netconfig.items():
            edge_id = edge_to_from_con_to[0]
            node_from = edge_df.loc[edge_id]['u']
            node_to = edge_df.loc[edge_id]['v']
            node_positions[node_from] = (node_df[node_df['osmid'] == node_from]['x'].values[0], node_df[node_df['osmid'] == node_from]['y'].values[0])
            node_positions[node_to] = (node_df[node_df['osmid'] == node_to]['x'].values[0], node_df[node_df['osmid'] == node_to]['y'].values[0])
            self.edge_to_source_dest[edge_id] = {}
            self.edge_to_source_dest[edge_id]['source'] = node_from
            self.edge_to_source_dest[edge_id]['dest'] = node_to

            self.graph.add_node(node_from)
            self.graph.add_node(node_to)
            self.graph.add_edge(node_from, 
                                node_to, 
                                sus=VALUE_COSTS['sus'](self.edge_feature_old_no_norm[edge_id]),
                                sec=VALUE_COSTS['sec'](self.edge_feature_old_no_norm[edge_id]),
                                eff=VALUE_COSTS['eff'](self.edge_feature_old_no_norm[edge_id]), 
                                id=edge_id, )
        
        
        self.node_positions = node_positions
    def _init_edge_features(self, edge_path):
        edge_feature_no_norm, link_max, link_min = load_link_feature(edge_path)
        
        #path_feature = minmax_normalization01(path_feature_no_norm, path_max, path_min)
        edge_feature = minmax_normalization(edge_feature_no_norm, link_max, link_min)
       
        
        #  edge features = [length, fuel, insec, time]
        edge_feature_real_no_norm = np.zeros((edge_feature.shape[0], 4))
        edge_feature_real_no_norm[:,0] = edge_feature_no_norm[:, 0] # length
        edge_feature_real_no_norm[:,1] = np.array([VALUE_COSTS.get('sus')(ef) for ef in edge_feature_no_norm]) # fuel
        edge_feature_real_no_norm[:,2] = np.array([VALUE_COSTS.get('sec')(ef) for ef in edge_feature_no_norm]) # insec
        edge_feature_real_no_norm[:,3] = np.array([VALUE_COSTS.get('eff')(ef) for ef in edge_feature_no_norm]) # time

        edge_feature_max, edge_feature_min = np.max(edge_feature_real_no_norm, axis=0), np.min(edge_feature_real_no_norm, axis=0)
        self.edge_feature_max = edge_feature_max
        edge_feature_real = minmax_normalization01(edge_feature_real_no_norm.copy(), edge_feature_max, edge_feature_min)
        self.link_feature = torch.from_numpy(edge_feature_real).float()
        self.link_feature_no_norm = torch.from_numpy(edge_feature_real_no_norm).float()

        edge_feature_standarized = np.copy(edge_feature_real_no_norm)

        m = np.mean(edge_feature_standarized, axis=0)
        v = np.std(edge_feature_standarized, axis=0)
        self.link_feature_standarized = torch.from_numpy((edge_feature_standarized - m) / v)
        
        self.edge_feature_old_no_norm = torch.from_numpy(edge_feature_no_norm).float()
        self.edge_feature_old = torch.from_numpy(edge_feature).float()
    
    def _init_path_features(self, path_feature_p):

        # DEFAULT PATH FEATURES: NONE
        self.path_feature = torch.zeros((self.state_dim, self.state_dim,len(BASIC_PROFILES)))
        self.path_feature_no_norm = torch.zeros((self.state_dim, self.state_dim,len(BASIC_PROFILES)))
        self.path_feature_standarized = torch.zeros((self.state_dim, self.state_dim,len(BASIC_PROFILES)))
        
        path_feature_no_norm, path_max, path_min = load_path_feature(path_feature_p)
        
        def calculate_profiled_dist(d):
                dists = np.full((path_feature_no_norm.shape[0], len(BASIC_PROFILES)),fill_value=1e6, dtype=np.float32)
                max_dist_per_pf = np.full((len(BASIC_PROFILES),), fill_value=-10000.0, dtype=np.float32)

                for i, bpf in enumerate(BASIC_PROFILES):
                    max_dist_per_pf[i] = float('-inf')
                    paths_to_d = {edge: self.shortest_path_edges(profile=bpf, to_state=d, from_state=edge, with_length=True) for edge in self.valid_edges}
                    for edge, (path, path_len) in paths_to_d.items():
                        dists[edge, i] = path_len 
                        max_dist_per_pf[i] = max(max_dist_per_pf[i], path_len)
                    dists[:, i] = np.minimum(max_dist_per_pf[i], dists[:, i])

                return dists, max_dist_per_pf
            
        try:
            dist_by_pf = np.load(PATH_FEATURES_PATH)
            max_dists = np.load(MAX_PATH_FEATURES_PATH)
        except EOFError as e:
            dist_by_pf = np.zeros((path_feature.shape[0], path_feature.shape[1],len(BASIC_PROFILES)))
            max_dists = np.ones((path_feature.shape[1],len(BASIC_PROFILES)))
            for d in self.valid_edges:
                dist_by_pf[:,d,:], max_dists[d,:] = calculate_profiled_dist(d)
                #print(d)
                
            np.save( PATH_FEATURES_PATH,dist_by_pf)
            np.save( MAX_PATH_FEATURES_PATH,max_dists)
            dists_file = np.load(PATH_FEATURES_PATH)
            max_dists_f = np.load(MAX_PATH_FEATURES_PATH)

            assert np.allclose(dists_file,  dist_by_pf), (dists_file,dist_by_pf)
            assert np.allclose(max_dists,  max_dists_f)
        
        max_per_pf = np.max(max_dists, axis=0)
        #print(max_per_pf)
        assert max_per_pf.shape == (len(BASIC_PROFILES),) 
        
        
        #  edge features = [length, fuel, insec, time]
        for i in range(max_per_pf.shape[0]):
            self.link_feature[:,i+1] = self.link_feature[:,i+1]/(max_per_pf[i]/self.edge_feature_max[i+1])
        

        if self.need_path_features:
            path_feature_no_norm, path_max, path_min = load_path_feature(path_feature_p)
        
            path_feature = minmax_normalization(path_feature_no_norm, path_max, path_min)

            def calculate_angle(e,d):
                pos_source_e = np.array(self.node_positions[self.edge_to_source_dest[e]['source']] )
                pos_dest_e = np.array(self.node_positions[self.edge_to_source_dest[e]['dest']])
                pos_source_d = np.array(self.node_positions[self.edge_to_source_dest[d]['source']])
                goal_orientation = pos_source_d - pos_source_e
                if np.all(goal_orientation < 0.0001):
                    goal_orientation = np.array(self.node_positions[self.edge_to_source_dest[d]['dest']]) - pos_source_d
                return angle_between(pos_dest_e - pos_source_e, goal_orientation)
            
                
            angles_from_e_to_d = np.zeros((path_feature.shape[0], path_feature.shape[1]))
            for i in np.arange(path_feature.shape[0]):
                for j in np.arange(path_feature.shape[1]):
                    angles_from_e_to_d[i,j] = calculate_angle(i,j)

            

            path_feature_real_no_norm  = np.zeros((path_feature.shape[0], path_feature.shape[1], 2+len(BASIC_PROFILES)))
            path_feature_real_no_norm[:,:,0] = path_feature_no_norm[:,:,0]
            path_feature_real_no_norm[:,:,1] = angles_from_e_to_d
        
            path_feature_real = np.zeros_like(path_feature_real_no_norm)
            path_feature_real[:,:,0] = path_feature[:,:,0]
            path_feature_real[:,:,1] = angles_from_e_to_d/np.pi
        
            self.path_feature = torch.from_numpy(path_feature_real).float()
            self.path_feature_no_norm = torch.from_numpy(path_feature_real_no_norm).float()
            
            def calculate_profiled_dist(d):
                dists = np.full((path_feature_no_norm.shape[0], len(BASIC_PROFILES)),fill_value=1e6, dtype=np.float32)
                max_dist_per_pf = np.full((len(BASIC_PROFILES),), fill_value=-10000.0, dtype=np.float32)

                for i, bpf in enumerate(BASIC_PROFILES):
                    max_dist_per_pf[i] = float('-inf')
                    paths_to_d = {edge: self.shortest_path_edges(profile=bpf, to_state=d, from_state=edge, with_length=True) for edge in self.valid_edges}
                    for edge, (path, path_len) in paths_to_d.items():
                        dists[edge, i] = path_len 
                        max_dist_per_pf[i] = max(max_dist_per_pf[i], path_len)
                    dists[:, i] = np.minimum(max_dist_per_pf[i], dists[:, i])

                return dists, max_dist_per_pf
            
            try:
                dist_by_pf = np.load(PATH_FEATURES_PATH)
                max_dists = np.load(MAX_PATH_FEATURES_PATH)
            except EOFError as e:
                dist_by_pf = np.zeros((path_feature.shape[0], path_feature.shape[1],len(BASIC_PROFILES)))
                max_dists = np.ones((path_feature.shape[1],len(BASIC_PROFILES)))
                for d in self.valid_edges:
                    dist_by_pf[:,d,:], max_dists[d,:] = calculate_profiled_dist(d)
                    #print(d)
                    
                np.save( PATH_FEATURES_PATH,dist_by_pf)
                np.save( MAX_PATH_FEATURES_PATH,max_dists)
                dists_file = np.load(PATH_FEATURES_PATH)
                max_dists_f = np.load(MAX_PATH_FEATURES_PATH)

                assert np.allclose(dists_file,  dist_by_pf), (dists_file,dist_by_pf)
                assert np.allclose(max_dists,  max_dists_f)
                
            for i,pf in enumerate(BASIC_PROFILES):
                path_feature_real_no_norm[:,:,i+2] = dist_by_pf[:,:,i]
                

                path_feature_real[:,:,i+2] = path_feature_real_no_norm[:,:,i+2]/max_dists[:,i].reshape(-1,1)
                
            self.path_feature = torch.from_numpy(path_feature_real).float()
        
            self.path_feature_no_norm = torch.from_numpy(path_feature_real_no_norm).float()

            path_feature_standarized = np.copy(path_feature_real_no_norm)
            
            m = np.mean(path_feature_standarized, axis=(0,1))
            v = np.std(path_feature_standarized, axis=(0,1))
            self.path_feature_standarized = torch.from_numpy((path_feature_standarized - m) / v)
        
        
    def __init__(self, network_path, edge_path, node_path, path_feature_path, pre_reset=[[0,714],], origins=None, destinations=None, profile=(1.0,0.0,0.0),feature_selection=FeatureSelection.DEFAULT, use_optimal_reward_per_profile=False, feature_preprocesssing=FeaturePreprocess.NORMALIZATION,visualize_example=True):
        super().__init__(network_path, edge_path, pre_reset, origins, destinations, 8)
        
        
        network_p = f"{DATA_FOLDER}/transit.npy"
        path_feature_p = f"{DATA_FOLDER}/feature_od.npy"

        
        self.action_dim = self.n_actions
        self.state_dim = self.n_states
        self.use_optimal_reward_per_profile = use_optimal_reward_per_profile
        self.feature_preprocessing = feature_preprocesssing
        
        self.last_profile = tuple(profile)

        self.action_space = Discrete(self.n_states)
        #self.observation_space = MultiDiscrete([self.n_states, self.n_states])
        self.state_space = MultiDiscrete([self.n_states, self.n_states])
        
        self.feature_selection = feature_selection
        self.need_path_features = not ((self.feature_selection == FeatureSelection.ONLY_COSTS) or (isinstance(self.feature_selection, tuple)))
        #print(self.feature_selection)
        #print(self.need_path_features)
        # NEEDS TO BE DONE IN THIS ORDER:

        self._init_edge_features(edge_path)

        self._init_graph(edge_path, node_path)

        self._init_path_features(path_feature_p)

        self.reduce_size()
        
        def _get_one_hot_index(i,j):
                return i * self.n_states + j
       
        self.get_one_hot_index = _get_one_hot_index

        def __get_reward(s,given_edge,d,profile=None):
            if profile is None:
                profile = self.last_profile
            profile = tuple(profile)
            reward_per_dest = self.all_rewards_per_profile.get(profile, None)
            if reward_per_dest is None:
                self.all_rewards_per_profile[profile] = dict()

            shortest_paths = self.all_rewards_per_profile[profile].get(d, None)
            if shortest_paths is None:
                paths = defaultdict(lambda: [])
                paths.update(self.shortest_path_edges(profile, to_state=d, all_alternatives=False, flattened=True))
                self.all_rewards_per_profile[profile][d] = paths
            
            return -0.001 if given_edge in self.all_rewards_per_profile[profile][d][self.edge_to_source_dest[s]['dest']] else -1.0
        def __get_reward_state(s,d,profile=None):
            if profile is None:
                profile = self.last_profile
            profile = tuple(profile)
            reward_per_dest = self.all_rewards_per_profile.get(profile, None)
            if reward_per_dest is None:
                self.all_rewards_per_profile[profile] = dict()

            shortest_paths = self.all_rewards_per_profile[profile].get(d, None)
            if shortest_paths is None:
                paths = defaultdict(lambda: [])
                paths.update(self.shortest_path_edges(profile, to_state=d, all_alternatives=False, flattened=True))
                self.all_rewards_per_profile[profile][d] = paths
            
            return -0.001 if s in self.all_rewards_per_profile[profile][d][self.edge_to_source_dest[s]['source']] else -1.0
        
        self._get_reward = __get_reward
        self._get_reward_state = __get_reward_state

        
        self.n_features = self.process_features(torch.tensor([(107,413),]), feature_selection=feature_selection, use_real_env_rewards_as_feature=self.use_optimal_reward_per_profile,feature_preprocessing=self.feature_preprocessing).shape[1]

        self.observation_space = Box(low=0.0, high=1.0, shape=(self.n_features, ), dtype=np.float32)
        
        
        (self.cur_state, self.cur_des) = super().reset(profile=profile)

        if pre_reset is not None:
            
            origin_dest_pairs = [(int(od.split('_')[0]), int(od.split('_')[1])) for od in self.od_list]
        
            if visualize_example:
                """paths_eco = [self.shortest_paths_nodes(profile=(1.0,0.0,0.0), to_state=od[1], from_state=od[0]) for od in origin_dest_pairs]
                paths_sec = [self.shortest_paths_nodes(profile=[0,1,0], to_state=od[1], from_state=od[0]) for od in origin_dest_pairs]
                paths_eff = [self.shortest_paths_nodes(profile=[0,0,1], to_state=od[1], from_state=od[0]) for od in origin_dest_pairs]"""
                visualize_graph(self.graph,self.node_positions, show_edge_weights=False, caminos_by_value={
                    'sus': [self.shortest_paths_nodes(profile=(1.0,0.0,0.0), to_state=origin_dest_pairs[0][1], from_state=origin_dest_pairs[0][0]),], 
                    'sec': [self.shortest_paths_nodes(profile=[0,1,0], to_state=origin_dest_pairs[0][1], from_state=origin_dest_pairs[0][0]),], 
                    'eff': [self.shortest_paths_nodes(profile=[0,0,1], to_state=origin_dest_pairs[0][1], from_state=origin_dest_pairs[0][0]),]})

        
        
    def get_action(self, state, action_index):
        #print(self.state_list_from_state[self.cur_state], action_index)
        return self.state_list_from_state[state][action_index]
    
    def shortest_paths_nodes(self, profile, to_state, from_state=None, all_alternatives=False, custom_cost=None):

        if all_alternatives == False:
            if from_state is None:
                paths = nx.shortest_path(self.graph, target=self.edge_to_source_dest[to_state]['source'], weight=self.profile_cost_minimization_path(to_state, profile,reverse=True, custom_cost=custom_cost))
                for k in paths.keys():
                    paths[k].append(self.edge_to_source_dest[to_state]['dest'])
                return paths
            else:
                path = nx.shortest_path(self.graph, source=self.edge_to_source_dest[from_state]['dest'], target=self.edge_to_source_dest[to_state]['source'], weight=self.profile_cost_minimization_path(to_state, profile, custom_cost=custom_cost))

                path.insert(0, self.edge_to_source_dest[from_state]['source'])
                path.append(self.edge_to_source_dest[to_state]['dest'])
                return path
        else:

            if from_state is None:
                all_paths = dict()
                for state in self.valid_edges:
                    paths_from_state = list(nx.all_shortest_paths(self.graph, source=self.edge_to_source_dest[state]['dest'], target=self.edge_to_source_dest[to_state]['source'], weight=self.profile_cost_minimization_path(to_state, profile, custom_cost=custom_cost)))
                    for i in range(len(paths_from_state)):
                        paths_from_state[i].insert(0, self.edge_to_source_dest[state]['source'])
                        paths_from_state[i].append(self.edge_to_source_dest[to_state]['dest'])
                    all_paths[self.edge_to_source_dest[state]['source']] = paths_from_state
                return all_paths   
                raise NotImplementedError("all alternatives only when both to_state and from_state specified")
            else:
                paths = list(nx.all_shortest_paths(self.graph, source=self.edge_to_source_dest[from_state]['dest'], target=self.edge_to_source_dest[to_state]['source'], weight=self.profile_cost_minimization_path(to_state, profile, custom_cost=custom_cost)))
                for i in range(len(paths)):
                    paths[i].insert(0, self.edge_to_source_dest[from_state]['source'])
                    paths[i].append(self.edge_to_source_dest[to_state]['dest'])
                return paths
    
    def shortest_path_edges(self, profile, to_state, from_state=None, with_length=False, all_alternatives=False, custom_cost= None, flattened=False):
        
        if all_alternatives:
            node_path_or_paths = self.shortest_paths_nodes(profile, to_state=to_state, from_state=from_state, all_alternatives=True, custom_cost=custom_cost)
            if from_state is None:
                good_edges = dict()
                for key, paths in node_path_or_paths.items():
                    # [108, 358, 222, 524, 32, 37, 31, 27, 467, 352, 313, 136, 53, 435, 404, 407, 488, 410, 482, 413] (shortest for sus)
                    if not flattened:
                        good_edges[key] = []
                        for path in paths:
                            edge_path = self.node_path_to_edge_path(path, format_list=True)
                            
                            if with_length == False:
                                good_edges[key].append( edge_path)
                            else:
                                good_edges[key].append((edge_path, 
                                                sum([self.profile_cost_minimization_path(to_state, profile,reverse=False,custom_cost=custom_cost)(self.edge_to_source_dest[st]['source'], self.edge_to_source_dest[st]['dest'],None) for st in edge_path])))
                    else:
                        good_edges[key] = []
                        if with_length:
                            good_edges[key] = [([],0)]
                        for path in paths:
                            edge_path = self.node_path_to_edge_path(path, format_list=True)
                            if with_length == False:
                                good_edges[key].extend(edge_path)
                                
                            else:
                                length = sum([self.profile_cost_minimization_path(to_state, profile,reverse=False,custom_cost=custom_cost)(self.edge_to_source_dest[st]['source'], self.edge_to_source_dest[st]['dest'],None) for st in edge_path])
                                good_edges[key][0].extend(edge_path)
                                good_edges[key][1]+=length
                            
                return good_edges# if not flattened else good_edges, good_lengths
            
                raise NotImplementedError("all alternatives only when both to_state and from_state specified")
            else:
                if not flattened:
                    edge_paths = []

                    for node_path in node_path_or_paths:
                        edge_path = self.node_path_to_edge_path(node_path, format_list=True)
                        if with_length is False:
                            
                            edge_paths.append(edge_path)
                        else:
                            edge_paths.append((edge_path, sum([self.profile_cost_minimization_path(to_state, profile,reverse=False, custom_cost=custom_cost)(self.edge_to_source_dest[st]['source'], self.edge_to_source_dest[st]['dest'],None) for st in edge_path])))
                    return edge_paths
                else:
                    edges = []
                    assert with_length is False
                    for node_path in node_path_or_paths:
                        edge_path = self.node_path_to_edge_path(node_path, format_list=True)
                        edges.extend(edge_path)
                    return edges
        else:
            node_path_or_paths = self.shortest_paths_nodes(profile, to_state=to_state, from_state=from_state, custom_cost=custom_cost, all_alternatives=all_alternatives)

            if from_state is None:
                
                good_edges = dict()
                for key, path in node_path_or_paths.items():
                    edge_path = self.node_path_to_edge_path(path, format_list=True)
                    
                    if with_length is False:
                        good_edges[key] = edge_path 
                    else:
                        good_edges[key] = (edge_path, 
                                        sum([self.profile_cost_minimization_path(to_state, profile,reverse=False,custom_cost=custom_cost)(self.edge_to_source_dest[st]['source'], self.edge_to_source_dest[st]['dest'],None) for st in edge_path]))
                
                return good_edges
            else:
                edge_path = self.node_path_to_edge_path(node_path_or_paths, format_list=True)
                if with_length is False:
                    return edge_path
                else:
                    return (edge_path, sum([self.profile_cost_minimization_path(to_state, profile,reverse=False,custom_cost=custom_cost)(self.edge_to_source_dest[st]['source'], self.edge_to_source_dest[st]['dest'],None) for st in edge_path]))

    def render(self, caminos_by_value={'sus': [], 'sec': [], 'eff': []}, file='dummy.png', show=True, show_edge_weights=False, custom_weights: dict = None, custom_weights_dest: int = None):
        visualize_graph(self.graph,self.node_positions, caminos_by_value=caminos_by_value, save_to=NETWORK_PLOTS_DIR + file, show_edge_weights=show_edge_weights, show=show, custom_weights=custom_weights, custom_weights_dest = custom_weights_dest)
            
    def states_to_observation(states):
        return states
    
    def one_hot_encoding_st_des(self, state_des: torch.Tensor):
        ts = torch.zeros((state_des.shape[0], self.n_states*self.n_states), requires_grad=False)
        for i in range(state_des.shape[0]):
            ts[i, self.get_one_hot_index(state_des[i,0], state_des[i,1])] = 1
        
        return ts
    

class RoadWorldGymObservationState(RoadWorldGym):
    def __init__(self, network_path, edge_path, node_path, path_feature_path, pre_reset=None, origins=None, destinations=None, profile=(1, 0, 0), visualize_example=False):
        super().__init__(network_path, edge_path, node_path, path_feature_path, pre_reset, origins, destinations, profile, visualize_example, feature_selection=FeatureSelection.DEFAULT)

        self.state_space = deepcopy(self.observation_space)
        self.feature_observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_features, ), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "features": self.feature_observation_space,  # Continuous Box space
            "state": spaces.MultiDiscrete((self.n_states, self.n_actions))  # Discrete MultiDiscrete space
        })

        
        
        self.feature_matrix = np.zeros((self.n_states,self.n_states,self.n_features), dtype=np.float32)
        self._calculated_destinations = np.array([self.cur_des,])
        for s in self.states:
            #print(s)
            #self.reward_matrix[s] = self._get_reward(s,s,self.cur_des,self.last_profile)
            #self.state_action[s, :] = [-1]*self.state_action.shape[1]
            self.feature_matrix[s, self.cur_des, :] = self.process_features(torch.tensor([(s,self.cur_des),]), feature_selection=self.feature_selection, use_real_env_rewards_as_feature=self.use_optimal_reward_per_profile, feature_preprocessing=self.feature_preprocessing).numpy()[0]
        #print(self.feature_matrix[s,self.cur_des])
        self.action_space.sample = self._action_sampler
        self.observation_space.sample = lambda: self._observation_sample()

    def _observation_sample(self):
        ori, des = self._get_random_initial_state()
        obs, info = self.reset(st=ori,des=des)
        return obs

    def _action_sampler(self, obs):
        return np.random.choice(self.get_available_actions_from_state(tuple(obs['state'][0])))

    def states_to_observation(self, state_des: np.ndarray , check_destinations=True) -> list[tuple]:
        state_des = np.asarray(state_des)
        if check_destinations and self._calculated_destinations.shape[0] < self.n_states :
            destinations = state_des[:,1]

            res = np.setdiff1d(destinations, self._calculated_destinations, assume_unique=True)
            if res.shape[0] > 0:
                for s in self.states:
                    self.feature_matrix[s, res, :] = self.process_features(torch.tensor([(s,d) for d in res]), feature_selection=self.feature_selection, use_real_env_rewards_as_feature=self.use_optimal_reward_per_profile, feature_preprocessing=self.feature_preprocessing).numpy()
            #print(self.feature_matrix[s,self.cur_des])
            self._calculated_destinations = np.union1d(self._calculated_destinations, destinations)
            

        return [{'features': f, 'state': np.array(s)} for f,s in zip(self.feature_matrix[state_des[:,0], state_des[:,1], :], state_des)]
        #return np.dstack((self.feature_matrix[state_des[:,0], state_des[:,1], :], state_des), axis=-1)
    def single_state_to_observation(self, single_state_des: tuple, check_destinations=True) -> tuple:
        des = single_state_des[1]
        if check_destinations and des not in self._calculated_destinations and self._calculated_destinations.shape[0] < self.n_states:
            self._calculated_destinations = np.union1d(self._calculated_destinations, [des,])
            for s in self.states:

                self.feature_matrix[s, des, :] = self.process_features(torch.tensor([(s,des),]), self.feature_selection, use_real_env_rewards_as_feature=self.use_optimal_reward_per_profile, feature_preprocessing=self.feature_preprocessing).numpy()[0]
                
                #print((s,des), self.feature_matrix[s,des,:])
        
        return {'features': self.feature_matrix[single_state_des[0], des, :], 'state': np.array(single_state_des)}
        

    def reset(self, seed=None,  options=None, st=None, des=None, full_random=False, profile=None):
        self._is_resetted = True
        state, info = super().reset(seed=seed, options=options, st=st, des=des, full_random=full_random, profile=profile)
        obs, info = self.single_state_to_observation(state, check_destinations=True), info
        
        return obs, info
    
    def step(self, action, reward_function=None):
        
        ns, r, d, t, info =  super().step(action, reward_function=reward_function)
        return self.single_state_to_observation(ns, check_destinations=ns[1] != self.cur_des), r, d, t, info
    
    def get_edge_to_edge_state(self, observation, to_tuple=True):
        return tuple(observation['state']) if to_tuple else observation['state']

class RoadWorldGymPOMDP(RoadWorldGym):
    def states_to_observation(self, state_des: np.ndarray , check_destinations=True):
        return state_des

    def __init__(self, network_path, edge_path, node_path, path_feature_path, pre_reset=[[0, 714]], origins=None, destinations=None, profile=[1, 0, 0], visualize_example=False, horizon=50, feature_selection=FeatureSelection.ONE_HOT_ORIGIN_ONLY, use_optimal_reward_per_profile=False, feature_preprocessing=FeaturePreprocess.NORMALIZATION,):
        super().__init__(network_path, edge_path, node_path, path_feature_path, pre_reset, origins, destinations, profile,feature_selection, use_optimal_reward_per_profile, feature_preprocessing, visualize_example)
        
        self.horizon = horizon
        self.reward_matrix = dict()
        
        self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.int16)
        
        print("N_FEATURES", self.n_features)
        self.observation_matrix = np.zeros((self.n_states,self.n_features), dtype=np.float64)
        self.done_matrix = np.eye(self.n_states, dtype=np.bool_)
        
        self._observation_space = self.observation_space # ?
        
        self.state_space = Discrete(self.n_states)
        
        self.last_profile = tuple(profile)

        (self.cur_state, self.cur_des), info = super().reset(seed=None, options=None, profile=profile, des=destinations[0] if destinations is not None else None)

        
        #self.observation_matrix[:,:] = self.process_features(torch.tensor([(s,self.cur_des) for s in self.states]).detach(), feature_selection=self.feature_selection, use_real_env_rewards_as_feature=self.use_optimal_reward_per_profile, feature_preprocessing=self.feature_preprocessing).numpy()
        #assert np.all(self.observation_matrix > 0)

        self._all_state_des_pairs = {}
        self._all_state_des_pairs_th = {}
        self._all_obs_by_dest = {}
        for d in self.valid_edges:
            self._all_state_des_pairs[d] = np.array([(s,d) for s in range(self.n_states) ], dtype=np.long)
            self._all_state_des_pairs_th[d] = torch.tensor(self._all_state_des_pairs[d], dtype=torch.long).detach()
            self._all_obs_by_dest[d] = self.process_features(
                    self._all_state_des_pairs_th[d],
                    feature_selection=self.feature_selection, 
                    use_real_env_rewards_as_feature=self.use_optimal_reward_per_profile, 
                    feature_preprocessing=self.feature_preprocessing
                ).numpy()

        self.observation_matrix = self._all_obs_by_dest[self.cur_des]

        self.state_actions_with_known_reward = np.zeros((self.n_states,self.n_actions), dtype=np.bool_) 
        print("ROAD_WORLD_POMDP initialied to destination: ", self.cur_des, ", and profile: ", self.last_profile)

        for s in range(self.n_states):
            
            """if s in self.valid_edges:
                self.reward_matrix[self.last_profile][s] = self._get_reward_state(s, self.cur_des, profile)
            else:
                self.reward_matrix[self.last_profile][s] = -1.0
            """
            if s not in self.valid_edges or s == self.cur_des:
                self.state_actions_with_known_reward[s] = True
            
            for a in range(self.n_actions):
                if a in self.get_action_list(s):
                    ns = self.get_state_des_transition((s, self.cur_des), a)[0]
                    self.transition_matrix[s, a, ns] = 1.0
                    if ns == self.cur_des:
                        self.state_actions_with_known_reward[s,a] = True
                else:
                    self.state_actions_with_known_reward[s,a] = True
                    self.transition_matrix[s, a, s] = 1.0

        
        #initial_states = np.array([int(od.split('_')[0]) for od in self.od_list])
        #self.transition_matrix = sp.as_coo(self.transition_matrix)
        
        self.initial_state_dist = np.zeros(self.n_states, dtype=np.float32)
        indices = np.asarray(list(self.valid_edges))
        self.initial_state_dist[indices] = 1/len(self.valid_edges)

    def update_destination(self, new_des, prev_des):
        updated_state_actions = deepcopy(self.state_actions_with_known_reward)
       
        updated_state_actions[new_des] = True
        updated_state_actions[prev_des, self.get_action_list(prev_des)] = False
        updated_state_actions[self.pre_acts_and_pre_states[prev_des][1], self.pre_acts_and_pre_states[prev_des][0]] = False
        updated_state_actions[self.pre_acts_and_pre_states[new_des][1], self.pre_acts_and_pre_states[new_des][0]] = True
        updated_state_actions[new_des] = True

        updated_transition_matrix = deepcopy(self.transition_matrix)
        updated_transition_matrix[new_des, self.get_action_list(new_des), [self.get_state_des_transition((new_des, prev_des), action= a)[0] for a in self.get_action_list(new_des)]] = 0.0
        updated_transition_matrix[new_des, self.get_action_list(new_des), new_des] = 1.0
        #updated_transition_matrix[new_des, self.get_action_list(new_des), [self.get_state_des_transition((new_des, prev_des), action= a)[0] for a in self.get_action_list(new_des)]] = 1.0
        updated_transition_matrix[prev_des, self.get_action_list(prev_des), prev_des] = 0.0
        updated_transition_matrix[prev_des, self.get_action_list(prev_des), [self.get_state_des_transition((prev_des, new_des), action= a)[0] for a in self.get_action_list(prev_des)]] = 1.0
        
        
        self.transition_matrix = updated_transition_matrix
        
        self.state_actions_with_known_reward = updated_state_actions
        self.observation_matrix = self._all_obs_by_dest[new_des]

        self.cur_des = new_des

        
        """old_s_w_k_r = np.zeros_like(self.state_actions_with_known_reward)
        trmat = deepcopy(np.zeros_like(self.transition_matrix))
        for s in range(self.n_states):
            
            if s not in self.valid_edges or s == self.cur_des:
                old_s_w_k_r[s] = True
            
            for a in range(self.n_actions):
                if a in self.get_action_list(s):
                    ns = self.get_state_des_transition((s, self.cur_des), a)[0]
                    trmat[s, a, ns] = 1.0
                    if ns == self.cur_des:
                        old_s_w_k_r[s,a] = True
                else:
                    old_s_w_k_r[s,a] = True
                    trmat[s, a, s] = 1.0
        print(np.where(old_s_w_k_r != updated_state_actions))
        np.testing.assert_allclose(old_s_w_k_r, updated_state_actions)
        np.testing.assert_allclose(trmat, self.transition_matrix)"""
        


    def reset(self,  seed=None,  options=None, st=None, des=None, profile=(1.0, 0.0,0.0), full_random=True):
        prev_des = self.cur_des
        
        ret, info = super().reset(seed, options, st, des, profile=profile, full_random=full_random)
        
        if des is None or self.cur_des != prev_des:
            self.update_destination(self.cur_des, prev_des)

        else:
            assert self.cur_des == prev_des
        self.state = self.cur_state
        return self.state, info
    
    def get_edge_to_edge_state(self, observation, to_tuple=True):
        return (observation, self.cur_des)
    
    def _get_state_des_transition_try(self,state_des, action):
        try:
            return self.get_state_des_transition(state_des, action)[0]
        except:
            return state_des[0]
    
    def step(self, action, reward_function=None, profile=None):
        s, r, d, t, i= super().step(action, reward_function)
        self.state = self.cur_state
        #print("STEP TO ", self.state, r, d, t, i)
        t = t or self.iterations >= self.horizon
        return self.state, r, d, t, i
    
    def get_reward(self, state, action, des, profile=None):

        if des is not None:
            assert des == self.cur_des
        else:
            des = self.valid_edges[0]
            nstate = self._get_state_des_transition_try((int(state), des), action)
            i=0
            while des == state or nstate == des:
                des = self.valid_edges[i]
                nstate = self._get_state_des_transition_try((int(state), des), action)
                i+=1

        nstate = self._get_state_des_transition_try((int(state), des), action)
        if profile is None:
            profile = self.last_profile
        profile = tuple(profile)
        if profile not in self.reward_matrix.keys():
            self.reward_matrix[profile] = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float32)

        
        if state not in self.valid_edges:
            rew = -10000.0
        elif state == des:
            rew = 0.0
        elif nstate == des:
            rew = 1.0 # This is the key... It is hard for RL to converge when 0 (actually learns another thing. To get there need a lot of reward, but not too much to cause overfitting...)
        elif nstate not in self.valid_edges:
            rew =  -10000.0
        elif action not in self.get_action_list(state):
            rew = -10000.0
        else:
            rew =  -self.cost_model(profile=profile, normalization =self.feature_preprocessing)((nstate, des))
            
            if 0.0 != self.reward_matrix[profile][state,action, des]:
                old_res = self.reward_matrix[profile][state,action, des]
                assert rew == old_res
        self.reward_matrix[profile][state,action, des] = rew
            
        return self.reward_matrix[profile][state,action,des]
    

class RoadWorldPOMDPStateAsTuple(RoadWorldGym):
    def __init__(self, network_path, edge_path, node_path, path_feature_path, pre_reset=[[0, 714]], origins=None, destinations=None, profile=[1, 0, 0], visualize_example=False, horizon=50, feature_selection=FeatureSelection.ONE_HOT_ORIGIN_AND_DEST, use_optimal_reward_per_profile = False, feature_preprocessing = FeaturePreprocess.NO_PREPROCESSING):
        super().__init__(network_path, edge_path, node_path, path_feature_path, pre_reset=pre_reset, origins=origins, destinations=destinations, profile=profile, visualize_example=visualize_example, feature_selection=feature_selection, use_optimal_reward_per_profile=use_optimal_reward_per_profile, feature_preprocesssing=feature_preprocessing)
        #self.observation_space = MultiDiscrete(nvec=(self.n_states, self.n_states))
        #self._observation_space = self.observation_space
        self.state_space = MultiDiscrete(nvec=(self.n_states, self.n_states))
        self.reward_matrix = dict()
        
        self.reward_matrix[self.last_profile] = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        self.observation_matrix = np.ones((self.n_states, self.n_states, self.n_features), dtype=np.float64)
        self.done_matrix = np.eye(self.n_states, dtype=np.bool_)
        
        self.horizon = horizon
        
        self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.int16)
        
        for state in self.valid_edges:
            all_states_from_state = []
            for des in self.valid_edges:
                all_states_from_state.append((state,des))
                self.reward_matrix[self.last_profile][state,des] = self._get_reward_state(state,des,self.last_profile)
                # Might need to do this in another moment:
                # self.reward_matrix[self.last_profile][state,des] = 0.0 if state == des else -0.001 if state in self.shortest_path_edges(profile, des, state, all_alternatives=True, flattened=True) else -1.0 
                for a in self.get_action_list(state):
                    
                    #self.reward_matrix[s,a] = self._get_reward(s, self.netconfig[s][a], self.cur_des, profile)
                    self.transition_matrix[state, a, self.get_state_des_transition((state, des), a)[0]] = 1.0
                    
            self.observation_matrix[state, np.asarray(list(self.valid_edges)), :] = self.process_features(torch.tensor(all_states_from_state).detach(),  feature_selection=self.feature_selection, use_real_env_rewards_as_feature=self.use_optimal_reward_per_profile, feature_preprocessing=self.feature_preprocessing).numpy()
            
        (self.cur_state, self.cur_des), info = super().reset(seed=None, options=None, profile=profile, des=None)
        self.initial_state_dist = np.zeros(self.n_states, dtype=np.float32)
        indices = np.asarray(list(self.valid_edges))
        #print(indices)
        self.initial_state_dist[indices] = 1/len(self.valid_edges)
        #for a in self.get_action_list(s):
                #   self.reward_matrix[s,a] = self._get_reward(s, self.netconfig[s][a], self.cur_des, profile)
        assert np.all(self.observation_matrix > 0)
    def get_edge_to_edge_state(self, observation, to_tuple=True):
        return tuple(observation)
    
    def reset(self,  seed=None,  options=None, st=None, des=None, profile=[1, 0, 0], full_random=False):
        
        s, info = super().reset(seed, options, st, des, profile=profile, full_random=full_random)
        self.state = s
        return self.state, info
    
    def step(self, action, reward_function=None, profile=None):
        s, r, d, t, i= super().step(action, reward_function)
        self.state = s
        t = t or self.iterations >= self.horizon
        return self.state, r, d, t, i
    
    def get_reward(self, state, action, des, profile=None):
        nstate = self.get_state_des_transition((int(state), des), action)[0]
        if profile is None:
            profile = self.last_profile
        profile = tuple(profile)
        if profile not in self.reward_matrix.keys():
            self.reward_matrix[profile] = np.zeros((self.n_states, self.n_states), dtype=np.float64) 
        self.reward_matrix[profile][nstate, des] = self._get_reward_state(nstate,des,profile)
        #self.reward_matrix[nstate, des] = self._get_reward_state(nstate,des,profile)
        return self.reward_matrix[profile][nstate, des]
    
    def states_to_observation(self, state_des: np.ndarray , check_destinations=True):
        return state_des
    
        

