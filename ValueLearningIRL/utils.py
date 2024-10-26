import argparse
import ast
import itertools
import json
from math import ceil
import os
from typing import List

import imitation
import numpy as np
import pandas as pd
import torch

from roadworld_env_use_case.values_and_costs import BASIC_PROFILES


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(MODULE_PATH, "checkpoints/")

os.makedirs(CHECKPOINTS, exist_ok=True)

class VectorToIndexConverter:
    def __init__(self, nvec):
        self.lower = np.asarray([0]*len(nvec))
        self.upper = np.asarray(nvec) - 1
        self.rlower = self.lower[::-1]
        self.rupper = self.upper[::-1]
        cm = np.insert(np.array(self.upper[:-1] - self.lower[:-1] + 1),0 ,1)
        self.cfactors = np.cumprod(cm)
        self.factors = np.array(self.upper - self.lower + 1)

    def convert_vec_to_str(self, vec):
        vec = np.asarray(vec)
        
        result = np.sum(np.multiply(vec-self.lower , self.cfactors))
        return str(result)
    def convert_int_to_vec(self, number):
        num_dimensions = len(self.lower)
        result = np.zeros((num_dimensions), dtype=int)

        for i in range(num_dimensions):
            number,r = np.divmod(number, self.factors[i])
            result[i] = r + self.lower[i]
        return result
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
        
def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return {"$array": x.tolist()}  # Make a tagged object
    raise TypeError(x)


def deconvert(x):
    if len(x) == 1:  # Might be a tagged object...
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return np.array(value)  # cast back to array
    return x


class LearningUtils:
    HIDDEN_LEVELS = 5
    GAMMA = 0.999
    BETA = 0.01
    PENALTY = 1
    SEED = 26
    SEED_FILE = None



    def mult_shape(vec):
        m = 1
        for n in vec:
            m = m*n
        return m


    def load_q_table(file, override=False):
        file = os.path.join(CHECKPOINTS, file)
        print("Loading checkpoint Q-table from: ", file)
        if override:
            print("Deleting previous checkpoint file...")
            if os.path.exists(file):
                os.remove(file)

        try:

            f = open(file, "r")
            q_checkpoint = json.load(f, object_hook=deconvert)
            f.close()
        except FileNotFoundError:
            print("Checkpoint file not found, creating...")
            f = open(file, "w")
            q_checkpoint = {}
            f.close()

        return q_checkpoint

    def save_q_table(q_a_table, q_b_table, file, params={}):

        file = os.path.join(CHECKPOINTS, file)
        print("Saving Q-table checkpoint file at: ", file)
        f = open(file, "w")
        save_dict = {}
        save_dict["q_a_table"] = q_a_table
        save_dict["q_b_table"] = q_b_table
        save_dict["params"] = params
        json.dump(save_dict, f, default=convert)

        f.close()




def get_cumulative_rewards(rewards, gamma=0.99):
    G = np.zeros_like(rewards, dtype=float)
    G[-1] = rewards[-1]
    for idx in range(-2, -len(rewards)-1, -1):
        G[idx] = rewards[idx] + gamma * G[idx+1]
    return G


def to_one_hot(y_tensor, ndims):
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot




def create_expert_trajectories(env_creator, from_df_expert_file_or_policy='expert_session_2.csv', repeat_samples=5, profile=(1,0,0), only_edges=False):
    
    from_file = isinstance(from_df_expert_file_or_policy,str)
    

    if from_file == True:
        df_expert = pd.read_csv(from_df_expert_file_or_policy, dtype=object)

        env = env_creator()
        states = []
        actions = []
        infos = []
        rewards = []
        next_states = []
        dones = []
        profiles = []
        for row in df_expert.values:
            for _ in range(repeat_samples):
                t_states = ast.literal_eval(row[0])
                t_actions = ast.literal_eval(row[2])
                #print(t_states)
                
                states.extend( ast.literal_eval(row[0]))
                rewards.extend(ast.literal_eval(row[1]))
                actions.extend(t_actions), 
            
                if len(t_states) < 5:
                    pass#print(t_states, nx_plus_ending)
                if len(t_states) > 1:
                    nx_plus_ending = ast.literal_eval(row[0])[1:]
                    
                else:
                    nx_plus_ending = []
                last_state = env.netconfig[t_states[-1]][t_actions[-1]] if only_edges else (env.netconfig[t_states[-1][0]][t_actions[-1]], t_states[-1][1]) # 413, 413?
                #print(last_state)
                nx_plus_ending = nx_plus_ending + [last_state, ]
                next_states.extend(nx_plus_ending)
                #print(states)
                #print(next_states)
                dones.extend([False]*(len(nx_plus_ending)-1) + [True, ])
                #profiles.extend([(row[-3], row[-2], row[-1]),]*(len(nx_plus_ending)))
                infos.extend([{'profile': (float(row[-3]), float(row[-2]), float(row[-1]))}]*len(nx_plus_ending))
        

        rollouts = imitation.data.types.Transitions(
            obs=np.array(env.states_to_observation(torch.tensor(np.array(states, dtype=np.int32)))), 
            acts=np.array(actions, np.int32), 
            infos=np.array(infos), 
            next_obs=np.array(env.states_to_observation(torch.tensor(np.array(next_states, dtype=np.int32)))), 
            dones=np.array(dones))

    else:
        env = env_creator()

        # Instantiate the PPO agent
        policy = from_df_expert_file_or_policy

        od_list_int = [od.split('_') for od in env.od_list]

        sessions_eco = [policy.generate_session(t_max=env.horizon, od=(int(od_try[0]), int(od_try[1])), profile=profile, with_probs=True, formatted_to_file=False, stochastic=False) for od_try in od_list_int]
    # session = states, rewards, actions, traj_probs.


        df_expert = pd.DataFrame([{'states': session[0],'rewards': session[1],'actions': session[2],'traj_probs': session[3], 'ecopref': profile[0], 'secpref': profile[1], 'effpref': profile[2]} for session in sessions_eco])
        df_expert.to_csv('sample.csv', index=False)

        return create_expert_trajectories(env_creator=lambda: env, from_df_expert_file_or_policy='sample.csv', repeat_samples=repeat_samples, profile=profile, only_edges=only_edges)
        #print(rollouts[0:5])
    return rollouts

def split_od_train_test(od_list, od_dist, split=0.8, to_od_list_int=True):

    if to_od_list_int and isinstance(od_list[0], str):
        od_list = [(int(od.split('_')[0]), int(od.split('_')[1])) for od in od_list  ]
    nod_list= np.asarray(od_list)
    nod_dist = np.asarray(od_dist)

    random_order = np.random.permutation([i for i in range(len(od_list))])
    train_split = [tuple(od) for od in nod_list[random_order[0:ceil(split*len(od_list))]]]
    test_split = [tuple(od) for od in nod_list[random_order[ceil(split*len(od_list)):]]]

    train_split_dist = nod_dist[random_order[0:ceil(split*len(od_dist))]].tolist()
    test_split_dist = nod_dist[random_order[ceil(split*len(od_dist)):]].tolist()

    return train_split, test_split, train_split_dist, test_split_dist

def train_test_split_initial_state_distributions(n_states, split_percentage=0.7):
    n = n_states

    # Step 2: Randomly select 70% of the indexes
    indices = np.arange(n)
    np.random.shuffle(indices)

    split_point = int(split_percentage * n)
    first_indices = indices[:split_point]
    second_indices = indices[split_point:]

    # Step 3: Create uniform distributions
    first_distribution = np.zeros(n)
    second_distribution = np.zeros(n)

    first_distribution[first_indices] = 1 / len(first_indices)
    second_distribution[second_indices] = 1 / len(second_indices)

    # Output the distributions
    return first_distribution, second_distribution

def sample_example_profiles(profile_variety, n_values=3, basic_profiles=BASIC_PROFILES) -> List:
    ratios = np.linspace(0, 1, profile_variety)

    if n_values < 1:
        raise ValueError('Need more values: n_values must be bigger than 0')
    if n_values == 1:
        profile_set = list(ratios)
    if n_values == 2:
        profile_set = [(1.0-ratio, ratio) for ratio in ratios]
    if n_values == 3:
        profile_combinations = [set(itertools.permutations((ratios[i], ratios[j], ratios[-i-j-1]))) for i in range(len(ratios)) for j in range(i, (len(ratios)-i+1)//2)]
    else:
        def recursFind(N, nc=3, i=0, t=0, p=[]):
            if nc==1:
                # No need to explore, last value is N-t
                if N-t>=i:
                    yield p+[N-t]
                else:
                    pass # p+[N-t] is a solution, but it has already been given in another order
            elif i*nc+t>N:
                return # impossible to find nc values>=i (there are some <i. But that would yield to already given solutions)
            else:
                for j in range(i, N):
                    yield from recursFind(N, nc-1, j, t+j, p+[j])

        profile_combinations = [set(itertools.permutations(ratios[i] for i in idx)) for idx in recursFind(len(ratios)-1, n_values)]

    
    if n_values >= 3:
        profile_set = list(set(tuple(float(f"{a_i:0.3f}") for a_i in a) for l in profile_combinations for a in l))
        [profile_set.remove(pr) for pr in basic_profiles]
        for pr in reversed(basic_profiles):
            profile_set.insert(0, pr)

    a = np.array(profile_set, dtype=np.dtype([(f'{i}', float) for i in range(n_values)]))
    sortedprofiles =a[np.argsort(a, axis=-1, order=tuple([f'{i}' for i in range(n_values)]), )]
    profile_set = list(tuple(t) for t in sortedprofiles.tolist())

    profile_set = [tuple(round(num, 2) for num in t) for t in profile_set]
    
    return profile_set



def load_json_config(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def filter_none_args(args):
    """Removes arguments that have a None value."""
    filtered_args = {k: v for k, v in vars(args).items() if v is not None}
    return argparse.Namespace(**filtered_args)