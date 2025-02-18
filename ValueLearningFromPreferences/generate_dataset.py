import argparse
import ast
from collections import defaultdict
import csv
import os
import pprint
import random
from typing import Dict, Sequence
import imitation
import imitation.data
import imitation.data.rollout
from imitation.data.serialize import save, load_with_rewards
import imitation.util
import numpy as np
import torch

from envs.tabularVAenv import TabularVAMDP
from src.algorithms.utils import PolicyApproximators, mce_partition_fh
from envs.firefighters_env import FeatureSelectionFFEnv, FireFightersEnv
from src.data import TrajectoryWithValueSystemRews, load_vs_trajectories, save_vs_trajectories
from src.policies.vsl_policies import VAlignedDictDiscreteStateActionPolicyTabularMDP, VAlignedDictSpaceActionPolicy

from utils import filter_none_args, load_json_config
import gymnasium as gym

DATASETS_PATH = 'datasets/'
TRAJECTORIES_DATASETS_PATH = 'datasets/trajectories/'
COMPARISONS_DATASETS_PATH = 'datasets/comparisons/'
GROUNDINGS_PATH = 'groundings/'

DEFAULT_SEED = 26

def parse_args():
    # IMPORTANT: Default Args are specified depending on the environment in config.json

    parser = argparse.ArgumentParser(
        description="This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected. See the societies.json and algorithm_config.json files")

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument('-dname', '--dataset_name', type=str, default='', required=True, help='Dataset name')
    general_group.add_argument('-gentr', '--gen_trajs', action='store_true', default=False, 
                               help="Generate new trajs for the selected society")
    general_group.add_argument('-genpf', '--gen_preferences', action='store_true', default=False, 
                               help="Generate new preferences among the generated trajectories")
    
    general_group.add_argument('-cf', '--config_file', type=str, default='algorithm_config.json',
                               help='Path to JSON general configuration file (overrides other defaults here, but not the command line arguments)')
    general_group.add_argument('-sf', '--society_file', type=str, default='societies.json',
                               help='Path to JSON society configuration file (overrides other defaults here, but not the command line arguments)')
    general_group.add_argument('-sname', '--society_name', type=str, default='default',
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')
    
    general_group.add_argument('-rg', '--recalculate_groundings', action='store_true', default=True,
                               help='Recalculate custom agent groundings')
    
    general_group.add_argument('-e', '--environment', type=str, default='ff', choices=[
                               'rw', 'ff'], help='environment (roadworld - rw, firefighters - ff, itemgathering - ig)')
    
    general_group.add_argument('-df', '--discount_factor', type=float, default=1.0,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')

    general_group.add_argument('-s', '--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    general_group.add_argument('-varhz', '--end_trajs_when_ended', action='store_true', default=False, 
                               help="Allow trajectories to end when the environment says an episode is done or horizon is reached, whatever happens first, instead of forcing all trajectories to have the length of the horizon")
    
    
    alg_group = parser.add_argument_group('Algorithm-specific Parameters')
    pc_group = alg_group.add_argument_group(
        'Preference Comparisons Parameters')
    pc_group.add_argument('-dfp', '--discount_factor_preferences', type=float,
                          default=1.0, help='Discount factor for preference comparisons')
    

    env_group = parser.add_argument_group('environment-specific Parameters')
    
    env_group.add_argument('-rt', '--retrain', action='store_true',
                           default=False, help='Retrain experts') # TODO: might be needed if we implement non tabular environments...
    env_group.add_argument('-appr', '--approx_expert', action='store_true',
                           default=False, help='Approximate expert')
    env_group.add_argument('-reps', '--reward_epsilon', default=0.0, type=float, 
                           help='Distance between the cummulative rewards of each pair of trajectories for them to be considered as equal in the comparisons')
    
    
    return parser.parse_args()

def calculate_trajectory_save_path(dataset_name, ag, environment_data, society_data):
    return f"{environment_data['name']}/{society_data['name']}/{dataset_name}/trajs_ag_{ag['name']}_{ag['value_system']}_rp_{ag['data']['random_traj_proportion']}_rat_{ag['data']['rationality']}"
def calculate_preferences_save_path(dataset_name, ag, environment_data, society_data, epsilon):
    
    return f"{environment_data['name']}/{society_data['name']}/{dataset_name}/prefs_ag_{ag['name']}_{ag['value_system']}_repsilon_{epsilon}"

def save_trajectories(trajectories: Sequence[TrajectoryWithValueSystemRews], dataset_name, ag, environment_data, society_data):
    path = calculate_trajectory_save_path(dataset_name, ag,environment_data,society_data)
    path = os.path.join(TRAJECTORIES_DATASETS_PATH, path)
    os.makedirs(path, exist_ok=True)
    save_vs_trajectories(path=path, trajectories=trajectories)
def load_trajectories(dataset_name, ag, environment_data, society_data) -> Sequence[TrajectoryWithValueSystemRews]:
    path = calculate_trajectory_save_path(dataset_name, ag,environment_data,society_data)
    return load_vs_trajectories(os.path.join(TRAJECTORIES_DATASETS_PATH, path))

def compare_trajectories(traj_i, traj_j, epsilon=0.0):
    """
    Compare two trajectories based on their discounted sums.
    Args:
        traj_i (float): Discounted sum of the first trajectory.
        traj_j (float): Discounted sum of the second trajectory.
        epsilon (float): Threshold for comparison.
    Returns:
        float: Comparison flag (1.0, 0.5, 0.0).
    """
    comparison = traj_i - traj_j
    if abs(comparison) < epsilon:
        return 0.5
    elif comparison > 0:
        return 1.0
    else:
        return 0.0
def save_preferences(idxs: np.ndarray, discounted_sums: np.ndarray, discounted_sums_per_grounding: np.ndarray, epsilon: float, dataset_name, ag, environment_data, society_data):
    path = calculate_preferences_save_path(dataset_name,ag,environment_data,society_data, epsilon)
    path=os.path.join(COMPARISONS_DATASETS_PATH, path)
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, 'agent_preferences_file.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Traj1', 'Traj2', 'CR1', 'CR2', 'Flag']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(discounted_sums)):
            traj_i = discounted_sums[idxs[i]]
            traj_j = discounted_sums[idxs[(i+1)% len(discounted_sums)]]
            comparison_flag = compare_trajectories(traj_i, traj_j, epsilon=epsilon)
            writer.writerow({'Traj1': idxs[i], 'Traj2': idxs[(i+1)% len(discounted_sums)], 'CR1': traj_i, 'CR2': traj_j, 'Flag': comparison_flag})
    for vi in range(discounted_sums_per_grounding.shape[0]):
        csv_path = os.path.join(path, f'value_{vi}_preferences_file.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Traj1', 'Traj2', 'CR1', 'CR2', 'Flag']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(discounted_sums_per_grounding[vi])):
                traj_i = discounted_sums_per_grounding[vi][idxs[i]]
                traj_j = discounted_sums_per_grounding[vi][idxs[(i+1)% len(discounted_sums_per_grounding[vi])]]
                comparison_flag = compare_trajectories(traj_i, traj_j, epsilon=epsilon)
                writer.writerow({'Traj1': idxs[i], 'Traj2': idxs[(i+1)% len(discounted_sums_per_grounding[vi])], 'CR1': traj_i, 'CR2': traj_j, 'Flag': comparison_flag})
        
def load_preferences(dataset_name, ag, environment_data, society_data, epsilon):
    path = calculate_preferences_save_path(dataset_name, ag,environment_data,society_data,epsilon=epsilon)
    path = os.path.join(COMPARISONS_DATASETS_PATH, path)
    csv_path = os.path.join(path, 'agent_preferences_file.csv')
    idxs = []
    preferences = []

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        
        for row in reader:
            traj1 = int(row['Traj1'])
            traj2 = int(row['Traj2'])
            comparison_flag = float(row['Flag'])
            idxs.append([traj1, traj2])
            preferences.append(comparison_flag)
        
        discounted_sums = np.zeros((reader.line_num,))
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for ir, row in enumerate(reader):
            traj1 = int(row['Traj1'])
            traj2 = int(row['Traj2'])
            cr1 = float(row['CR1'])
            cr2 = float(row['CR2'])
            comparison_flag = float(row['Flag'])

            if discounted_sums[traj1] != 0.0:
                assert discounted_sums[traj1] == cr1
            discounted_sums[traj1] = cr1
            
            if discounted_sums[traj2] != 0.0:
                assert discounted_sums[traj2] == cr2
            assert comparison_flag == preferences[ir]
            discounted_sums[traj2] = cr2
            assert compare_trajectories(cr1,cr2, epsilon) == preferences[ir] 
        
    discounted_sums_per_grounding = np.zeros((len(environment_data['basic_profiles']), discounted_sums.shape[0]))
    preferences_per_grounding = np.zeros((len(environment_data['basic_profiles']), len(preferences)))

    for vi in range(len(environment_data['basic_profiles'])):
        
        csv_path = os.path.join(path, f'value_{vi}_preferences_file.csv')
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for ir, row in enumerate(reader):
                traj1 = int(row['Traj1'])
                traj2 = int(row['Traj2'])
                cr1 = float(row['CR1'])
                cr2 = float(row['CR2'])
                comparison_flag = float(row['Flag'])

                preferences_per_grounding[vi,ir] = comparison_flag
                if discounted_sums_per_grounding[vi,traj1] != 0.0:
                    assert discounted_sums_per_grounding[vi,traj1] == cr1
                discounted_sums_per_grounding[vi,traj1] = cr1
                
                if discounted_sums_per_grounding[vi,traj2] != 0.0:
                    assert discounted_sums_per_grounding[vi,traj2] == cr2
                
                discounted_sums_per_grounding[vi, traj2] = cr2
                assert compare_trajectories(cr1,cr2, epsilon) == preferences_per_grounding[vi,ir] 
    return np.array(idxs), discounted_sums, discounted_sums_per_grounding, np.array(preferences), preferences_per_grounding
if __name__ == "__main__":
    # This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected
    # IMPORTANT: Default Args are specified depending on the environment in config.json
    parser_args = filter_none_args(parse_args())
    # If a config file is specified, load it and override command line args
    config = load_json_config(parser_args.config_file)
    society_config = load_json_config(parser_args.society_file)
    
    pprint.pprint(parser_args)
    np.random.seed(parser_args.seed)
    torch.manual_seed(parser_args.seed)
    random.seed(parser_args.seed)
    

    environment_data = config[parser_args.environment]
    society_data = society_config[parser_args.environment][parser_args.society_name]
    grounding_path = os.path.join('envs', parser_args.environment, GROUNDINGS_PATH)
    dataset_name = parser_args.dataset_name

    
    agent_profiles = [tuple(ag['value_system']) for ag in society_data['agents']]
    agent_groundings = [tuple(ag['grounding']) for ag in society_data['agents']]
    ag_name_to_aggrounding = {ag['name']: tuple(ag['grounding'])  for ag in society_data['agents']}
    grounding_files = society_config[parser_args.environment]['groundings']
    all_agent_groundings_to_save_files = dict({agg: [grounding_files[agg[i]] for i in range(len(agg))] for agg in set(agent_groundings)})
    
    extra_kwargs = {}
    if parser_args.environment == 'ff':
        extra_kwargs = {
            'feature_selection': FeatureSelectionFFEnv(environment_data['feature_selection']),
        }
    environment: TabularVAMDP = gym.make(
            environment_data['name'],
            horizon=environment_data['horizon'], 
            initial_state_distribution=environment_data['initial_state_distribution'], **extra_kwargs)
    environment.reset(seed=parser_args.seed)

    expert_policy_per_grounding_combination: Dict[tuple, VAlignedDictSpaceActionPolicy] = dict()
    
    all_agent_groundings = dict()
    tabular_all_agent_groundings = dict() 

    for agg in all_agent_groundings_to_save_files.keys():
        os.makedirs(grounding_path, exist_ok=True)
        all_agent_groundings[agg], tabular_all_agent_groundings[agg] = environment.calculate_assumed_grounding(variants = agg, 
                                                                     save_folder=grounding_path, 
                                                                     variants_save_files=all_agent_groundings_to_save_files[agg],
                                                                     recalculate=parser_args.recalculate_groundings)
        if parser_args.approx_expert:
            profile_to_assumed_matrix = dict()
            for w in agent_profiles:
                
                _, _, assumed_expert_pi = mce_partition_fh(environment, discount=environment_data['discount'],
                                                        reward=environment.reward_matrix_per_align_func(
                                                            w, custom_grounding = tabular_all_agent_groundings[agg]),
                                                        horizon=environment.horizon,
                                                        approximator_kwargs=society_data['approximator_kwargs'],
                                                        policy_approximator=PolicyApproximators(society_data['policy_approximation_method']),
                                                        deterministic=not society_data['stochastic_expert'])

                profile_to_assumed_matrix[w] = assumed_expert_pi

            expert_policy_per_grounding_combination[agg] = VAlignedDictSpaceActionPolicy(
                policy_per_va_dict=profile_to_assumed_matrix, env=environment, state_encoder=None, expose_state=True)
            
        else:
            raise NotImplementedError("not tested a PPO or other deep learning technique for expert RL generator yet")
    
    # TODO: Generate dataset of trajectories.
    if parser_args.end_trajs_when_ended is True and parser_args.gen_trajs is False:
        raise ValueError("To use varhz flag (variable horizon trajectories), you need to generate trajectories again, i.e. also use the -gentr flag")
    if parser_args.gen_trajs:
        for i, ag in enumerate(society_data['agents']):
            ag_grounding = ag_name_to_aggrounding[ag['name']]
            ag_policy = expert_policy_per_grounding_combination[ag_grounding]
            n_rational_trajs = int(np.ceil(ag['n_agents']*ag['data']['trajectory_pairs']*(1-ag['data']['random_traj_proportion'])))
            n_random_trajs = int(np.floor(ag['n_agents']*ag['data']['trajectory_pairs']*ag['data']['random_traj_proportion']))
            # rationality is epsilon rationality.
            print(f"Agent {i}")
            _, grounding_simple = ag_policy.env.calculate_assumed_grounding(variants = tuple(ag['grounding']), 
                                                                        save_folder=grounding_path, 
                                                                        variants_save_files=all_agent_groundings_to_save_files[tuple(ag['grounding'])],
                                                                        recalculate=False)
            
            print(f"Grounding Done {i}")
            ag_rational_trajs = ag_policy.obtain_trajectories(n_seeds=n_rational_trajs, 
                stochastic=society_data['stochastic_expert'], exploration=1.0-ag['data']['rationality'], 
                align_funcs_in_policy=[tuple(ag['value_system'])], repeat_per_seed=1, with_reward=True, with_grounding=True,
                alignments_in_env=[tuple(ag['value_system'])], end_trajectories_when_ended=parser_args.end_trajs_when_ended)
            print(f"Rational Trajs Done {i} ({len(ag_rational_trajs)})")
            #random policies are equivalent to having rationality 0:
            ag_random_trajs = ag_policy.obtain_trajectories(n_seeds=n_random_trajs,  
                stochastic=society_data['stochastic_expert'], exploration=1.0, align_funcs_in_policy=[tuple(ag['value_system'])], 
                repeat_per_seed=1, with_reward=True, with_grounding=True, alignments_in_env=[tuple(ag['value_system'])], 
                end_trajectories_when_ended=parser_args.end_trajs_when_ended)
            print(f"Random Trajs Done {i} ({len(ag_random_trajs)})")

            # TODO: ADD FOR EACH TRAJECTORY the Value rews.

            all_trajs_ag = []
            all_trajs_ag.extend(ag_random_trajs)
            all_trajs_ag.extend(ag_rational_trajs)
            save_trajectories(all_trajs_ag, dataset_name=dataset_name, ag=ag, society_data=society_data, environment_data=environment_data)
    
        
    if parser_args.gen_preferences:
        for i, ag in enumerate(society_data['agents']):
            agg = tuple(ag['grounding'])
            all_trajs_ag = load_trajectories(dataset_name=dataset_name, ag=ag, society_data=society_data, environment_data=environment_data)
            idxs = np.random.permutation(len(all_trajs_ag)) # Indicates the order of comparison. idxs[0] with idxs[1], then idxs[1] with idxs[2], etc...
            discounted_sums = np.zeros_like(idxs, dtype=np.float32)
            
            ag_grounding = tabular_all_agent_groundings[agg]
            discounted_sums_per_grounding = np.zeros((len(environment_data['basic_profiles']), discounted_sums.shape[0]), dtype=np.float32)
            for i in range((len(all_trajs_ag))):
                discounted_sums[i] = imitation.data.rollout.discounted_sum(all_trajs_ag[i].rews, gamma=parser_args.discount_factor_preferences)
                for vi in range(discounted_sums_per_grounding.shape[0]):
                    grounding_simple = ag_grounding[all_trajs_ag[i].obs[:-1], all_trajs_ag[i].acts][:, vi]
                    list_ = []
                    for o, no, a, info in zip(all_trajs_ag[i].obs[:-1], all_trajs_ag[i].obs[1:], all_trajs_ag[i].acts, all_trajs_ag[i].infos):
                        list_.append(environment.get_reward_per_align_func(align_func=tuple(environment_data['basic_profiles'][vi]), action=a, info=info, obs=o, next_obs=no,custom_grounding=ag_grounding))
                    
                    np.testing.assert_almost_equal(list_ , grounding_simple)
                    np.testing.assert_almost_equal(list_, all_trajs_ag[i].v_rews[vi])
                    assert len(grounding_simple) == len(all_trajs_ag[i].acts)
                    discounted_sums_per_grounding[vi,i] = imitation.data.rollout.discounted_sum(grounding_simple, gamma=parser_args.discount_factor_preferences)
            # We save the comparison of 1 vs 2, 2 vs 3 in the order stablished in discounted_sums.
            
            save_preferences(idxs=idxs, discounted_sums=discounted_sums, discounted_sums_per_grounding=discounted_sums_per_grounding, dataset_name=dataset_name, epsilon=parser_args.reward_epsilon, environment_data=environment_data, society_data=society_data, ag=ag)
    
    #TEST preferences load okey.
    print("TESTING DATA COHERENCE. It is safe to stop this program now...")
    for i, ag in enumerate(society_data['agents']):
        agg = tuple(ag['grounding'])
        # Here idxs is the list of trajectory PAIRS of indices from the trajectory list that are compared.
        idxs, discounted_sums, discounted_sums_per_grounding, preferences, preferences_per_grounding = load_preferences(epsilon=parser_args.reward_epsilon, dataset_name=dataset_name, environment_data=environment_data, society_data=society_data, ag=ag)
        trajs_ag = load_trajectories(dataset_name=dataset_name, ag=ag, environment_data=environment_data,society_data=society_data)

        
        for i in range((len(trajs_ag))):
            assert discounted_sums[i] == imitation.data.rollout.discounted_sum(trajs_ag[i].rews, gamma=parser_args.discount_factor_preferences)
        
        for idx, pr in zip(idxs, preferences):
            assert discounted_sums[idx[0]] == imitation.data.rollout.discounted_sum(trajs_ag[idx[0]].vs_rews, gamma=parser_args.discount_factor_preferences)
            assert discounted_sums[idx[1]] == imitation.data.rollout.discounted_sum(trajs_ag[idx[1]].vs_rews, gamma=parser_args.discount_factor_preferences)
            assert compare_trajectories(discounted_sums[idx[0]], discounted_sums[idx[1]], epsilon=parser_args.reward_epsilon) == pr
        for vi in range(len(environment_data['basic_profiles'])):
            for i in range((len(trajs_ag))):
                assert discounted_sums_per_grounding[vi,i] == imitation.data.rollout.discounted_sum(trajs_ag[i].v_rews[vi], gamma=parser_args.discount_factor_preferences)
        
                assert discounted_sums_per_grounding[vi,i] == imitation.data.rollout.discounted_sum(tabular_all_agent_groundings[agg][trajs_ag[i].obs[:-1], trajs_ag[i].acts][:, vi], gamma=parser_args.discount_factor_preferences)
                
            for idx, pr in zip(idxs, preferences_per_grounding[vi]):
                assert discounted_sums_per_grounding[vi,idx[0]] == imitation.data.rollout.discounted_sum(tabular_all_agent_groundings[agg][trajs_ag[idx[0]].obs[:-1], trajs_ag[idx[0]].acts][:, vi], gamma=parser_args.discount_factor_preferences)
                assert discounted_sums_per_grounding[vi,idx[1]] == imitation.data.rollout.discounted_sum(tabular_all_agent_groundings[agg][trajs_ag[idx[1]].obs[:-1], trajs_ag[idx[1]].acts][:, vi], gamma=parser_args.discount_factor_preferences)
                assert compare_trajectories(discounted_sums_per_grounding[vi,idx[0]], discounted_sums_per_grounding[vi,idx[1]], epsilon=parser_args.reward_epsilon) == pr
        
    print("Dataset generated correctly.")