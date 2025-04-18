import argparse
from collections import defaultdict
import os
import dill
import pprint
import random
from typing import Union
import imitation
import imitation.data
import imitation.data.rollout
import imitation.util
import numpy as np
import torch

from envs.routechoiceApollo import RouteChoiceEnvironmentApollo, RouteChoiceEnvironmentApolloComfort
from src.dataset_processing.trajectories import compare_trajectories
from src.dataset_processing.data import TrajectoryWithValueSystemRews

from src.dataset_processing.datasets import create_dataset
from src.dataset_processing.preferences import load_preferences, save_preferences
from src.dataset_processing.trajectories import load_trajectories, save_trajectories
from src.dataset_processing.utils import DEFAULT_SEED, GROUNDINGS_PATH
from src.utils import filter_none_args, load_json_config
import gymnasium as gym

PICKLED_ENVS = 'datasets/environments/'

def parse_dtype_numpy(choice):
    ndtype = np.float32
    if choice == 'float16':
        ndtype = np.float16
    if choice == 'float32':
        ndtype = np.float32
    if choice == 'float64':
        ndtype = np.float64
    return ndtype


def parse_dtype_torch(choice):
    ndtype = torch.float32
    if choice == 'float16':
        ndtype = torch.float16
    if choice == 'float32':
        ndtype = torch.float32
    if choice == 'float64':
        ndtype = torch.float64
    return ndtype



def parse_args():
    # IMPORTANT: Default Args are specified depending on the environment in config.json

    parser = argparse.ArgumentParser(
        description="This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected. See the societies.json and algorithm_config.json files")

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument(
        '-dname', '--dataset_name', type=str, default='', required=True, help='Dataset name')
    general_group.add_argument('-gentr', '--gen_trajs', action='store_true', default=False,
                               help="Generate new trajs for the selected society")

    general_group.add_argument('-genpf', '--gen_preferences', action='store_true', default=False,
                               help="Generate new preferences among the generated trajectories")

    general_group.add_argument('-dtype', '--dtype', type=parse_dtype_numpy, default=np.float32, choices=[np.float16, np.float32, np.float64],
                               help="Reward data to be saved in this numpy format")

    general_group.add_argument('-a', '--algorithm', default='pc',
                               help="dataset oriented to algorithm")

    general_group.add_argument('-cf', '--config_file', type=str, default='algorithm_config.json',
                               help='Path to JSON general configuration file (overrides other defaults here, but not the command line arguments)')
    """general_group.add_argument('-sf', '--society_file', type=str, default='societies.json',
                               help='Path to JSON society configuration file (overrides other defaults here, but not the command line arguments)')"""
    """general_group.add_argument('-sname', '--society_name', type=str, default='default',
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')
"""
    """general_group.add_argument('-rg', '--recalculate_groundings', action='store_true', default=True,
                               help='Recalculate custom agent groundings')"""

    general_group.add_argument('-e', '--environment', type=str, required=True, choices=[
                               'apollo', ], help='environment (apollo)')

    """general_group.add_argument('-df', '--discount_factor', type=float, default=1.0,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')"""

    general_group.add_argument(
        '-s', '--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    """general_group.add_argument('-varhz', '--end_trajs_when_ended', action='store_true', default=False,
                               help="Allow trajectories to end when the environment says an episode is done or horizon is reached, whatever happens first, instead of forcing all trajectories to have the length of the horizon")
"""
    general_group.add_argument('-tsize', '--test_size', type=float,
                               default=0.2, help='Ratio_of_test_versus_train_preferences')

    alg_group = parser.add_argument_group('Algorithm-specific Parameters')
    pc_group = alg_group.add_argument_group(
        'Preference Comparisons Parameters')
    """pc_group.add_argument('-dfp', '--discount_factor_preferences', type=float,
                          default=1.0, help='Discount factor for preference comparisons')"""

    env_group = parser.add_argument_group('environment-specific Parameters')

    """env_group.add_argument('-rt', '--retrain', action='store_true',
                           # TODO: might be needed if we implement non tabular environments...
                           default=False, help='Retrain experts')"""
    """env_group.add_argument('-appr', '--approx_expert', action='store_true',
                           default=False, help='Approximate expert')"""
    env_group.add_argument('-reps', '--reward_epsilon', default=0.0, type=float,
                           help='Distance between the cummulative rewards of each pair of trajectories for them to be considered as equal in the comparisons')

    return parser.parse_args()


if __name__ == "__main__":
    # This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected
    # IMPORTANT: Default Args are specified depending on the environment in config.json
    parser_args = filter_none_args(parse_args())
    # If a config file is specified, load it and override command line args
    config = load_json_config(parser_args.config_file)
    society_config = load_json_config('societies.json')

    pprint.pprint(parser_args)
    np.random.seed(parser_args.seed)
    torch.manual_seed(parser_args.seed)
    random.seed(parser_args.seed)

    environment_data = config[parser_args.environment]
    # these are real datasets, no different configurations available
    society_data = society_config[parser_args.environment]['default']
    groundings = society_config[parser_args.environment]['groundings']
    alg_config = environment_data['algorithm_config'][parser_args.algorithm]
    grounding_path = os.path.join(
        'envs', parser_args.environment, GROUNDINGS_PATH)
    dataset_name = parser_args.dataset_name

    extra_kwargs = {}
    if parser_args.environment == 'apollo':
        extra_kwargs = {
            'test_size': parser_args.test_size
        }

    environment: Union[RouteChoiceEnvironmentApollo,RouteChoiceEnvironmentApolloComfort] = gym.make(
        environment_data['name'], **extra_kwargs)
    environment.reset(seed=parser_args.seed)

    os.makedirs(os.path.join(
        PICKLED_ENVS, environment_data['name'], dataset_name), exist_ok=True)
    with open(os.path.join(os.path.join(PICKLED_ENVS, environment_data['name'], dataset_name), f"env_kw_{extra_kwargs}.pkl"), 'wb') as f:
        dill.dump(environment, f)

    if parser_args.gen_trajs:
        trajs_by_ag_train = defaultdict(list)
        trajs_by_ag_test = defaultdict(list)
        for traj in environment.routes_train:
            traj: TrajectoryWithValueSystemRews
            trajs_by_ag_train[traj.agent].append(traj)
        for traj in environment.routes_test:
            trajs_by_ag_test[traj.agent].append(traj)

        for ag, trajs in trajs_by_ag_train.items():
            save_trajectories(trajs, dataset_name=dataset_name+'_train', ag={'agent_id': ag, 'name': ag, 'value_system': 'unk', 'data': defaultdict(lambda: 'nd'), 'grounding': list(groundings.keys())},
                              society_data=society_data, environment_data=environment_data, dtype=parser_args.dtype)
        for ag, trajs in trajs_by_ag_test.items():
            save_trajectories(trajs, dataset_name=dataset_name+'_test', ag={'agent_id': ag, 'name': ag, 'value_system': 'unk', 'data': defaultdict(lambda: 'nd'), 'grounding': list(groundings.keys())},
                              society_data=society_data, environment_data=environment_data, dtype=parser_args.dtype)

    if parser_args.gen_preferences:

        for iads, agent_ids_ in enumerate([environment.agent_ids_train, environment.agent_ids_test]):
            suffix = '_train' if iads== 0 else '_test'
            for i, agid in enumerate(agent_ids_):
                ag = {'agent_id': agid, 'name': agid, 'value_system': 'unk', 'data': defaultdict(
                    lambda: 'nd'), 'grounding': list(groundings.keys())}
                all_trajs_ag_train = load_trajectories(
                    dataset_name=dataset_name+suffix, ag=ag, society_data=society_data, environment_data=environment_data, override_dtype=parser_args.dtype)
                idxs = []
                agent_preferences = environment.preferences_per_agent_id[int(agid)]
                preferences_by_idx_pair = {}
                preferences_per_grounding_per_idx_pair = {vi: {} for vi in range(len(environment_data['basic_profiles']))}
                for (t1id, t2id), choice in agent_preferences.items():
                    try:
                        it1, t1 = next((it, t) for it, t in enumerate(
                            all_trajs_ag_train) if t.infos[0]['state'] == t1id)
                        it2, t2 = next((it, t) for it, t in enumerate(
                            all_trajs_ag_train) if t.infos[0]['state'] == t2id)
                    except StopIteration:
                        continue
                    idxs.append(it1)
                    idxs.append(it2)
                    preferences_by_idx_pair[(
                        it1, it2)] = agent_preferences[(t1id, t2id)]
                
                for vi in range(len(environment_data['basic_profiles'])):
                    agent_grounding_preferences = environment.preferences_grounding_per_agent_id[vi][int(agid)]
                
                    for (t1id, t2id), choice in agent_grounding_preferences.items():
                        try:
                            it1, t1 = next((it, t) for it, t in enumerate(
                                all_trajs_ag_train) if t.infos[0]['state'] == t1id)
                            it2, t2 = next((it, t) for it, t in enumerate(
                                all_trajs_ag_train) if t.infos[0]['state'] == t2id)
                        except StopIteration:
                            continue
                        #idxs.append(it1)
                        #idxs.append(it2)
                        preferences_per_grounding_per_idx_pair[vi][(
                            it1, it2)] = agent_grounding_preferences[(t1id, t2id)]
                    
                discounted_sums = np.zeros_like(idxs, dtype=np.float64)

                discounted_sums_per_grounding = np.zeros(
                    (len(environment_data['basic_profiles']), discounted_sums.shape[0]), dtype=np.float64)
                for i in range((len(all_trajs_ag_train))):

                    discounted_sums[i] = imitation.data.rollout.discounted_sum(
                        all_trajs_ag_train[i].vs_rews, gamma=alg_config['discount_factor_preferences'])
                    for vi in range(discounted_sums_per_grounding.shape[0]):
                        list_ = []

                        if environment_data['is_contextual']:
                            environment.contextualize(
                                all_trajs_ag_train[i].infos[0]['context'])
                        for o, no, a, info in zip(all_trajs_ag_train[i].obs[:-1], all_trajs_ag_train[i].obs[1:], all_trajs_ag_train[i].acts, all_trajs_ag_train[i].infos):

                            list_.append(environment.get_reward_per_align_func(align_func=tuple(
                                environment.basic_profiles[vi]), action=a, info=info, obs=o, next_obs=no, custom_grounding=None))

                        
                        np.testing.assert_almost_equal(
                            np.asarray(list_, dtype=parser_args.dtype), all_trajs_ag_train[i].v_rews[vi], decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                        discounted_sums_per_grounding[vi, i] = imitation.data.rollout.discounted_sum(
                            all_trajs_ag_train[i].v_rews[vi], gamma=alg_config['discount_factor_preferences'])

                save_preferences(idxs=idxs, discounted_sums=discounted_sums, discounted_sums_per_grounding=discounted_sums_per_grounding, dataset_name=dataset_name+suffix, epsilon=parser_args.reward_epsilon, environment_data=environment_data, society_data=society_data, ag=ag,
                                real_preference=preferences_by_idx_pair, real_grounding_preference=preferences_per_grounding_per_idx_pair)

    # TEST preferences load okey.
    print("TESTING DATA COHERENCE. It is safe to stop this program now...")
    for i, ag in enumerate(environment.agent_ids_train):
        ag = {'agent_id': ag, 'name': ag, 'value_system': 'unk', 'data': defaultdict(
            lambda: 'nd'), 'grounding': list(groundings.keys())}

        #  Here idxs is the list of trajectory PAIRS of indices from the trajectory list that are compared.
        idxs, discounted_sums, discounted_sums_per_grounding, preferences, preferences_per_grounding = load_preferences(
            epsilon=parser_args.reward_epsilon, dataset_name=dataset_name+'_train', environment_data=environment_data, society_data=society_data, ag=ag, dtype=parser_args.dtype, debug_grounding=False)

        trajs_ag = load_trajectories(dataset_name=dataset_name+'_train', ag=ag,
                                     environment_data=environment_data, society_data=society_data, override_dtype=parser_args.dtype)

        for i in range((len(trajs_ag))):
            np.testing.assert_almost_equal(discounted_sums[i], imitation.data.rollout.discounted_sum(
                trajs_ag[i].vs_rews, gamma=1.0))

        for idx, pr in zip(idxs, preferences):
            assert isinstance(discounted_sums, np.ndarray)
            assert isinstance(idx, np.ndarray)
            idx = [int(ix) for ix in idx]

            np.testing.assert_almost_equal(discounted_sums[idx[0]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                trajs_ag[idx[0]].vs_rews, gamma=1.0)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
            np.testing.assert_almost_equal(discounted_sums[idx[1]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                trajs_ag[idx[1]].vs_rews, gamma=1.0)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
            np.testing.assert_almost_equal(compare_trajectories(
                discounted_sums[idx[0]], discounted_sums[idx[1]], epsilon=parser_args.reward_epsilon), pr, decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
        for vi in range(len(environment_data['basic_profiles'])):
            for i in range((len(trajs_ag))):
                np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, i], parser_args.dtype(imitation.data.rollout.discounted_sum(
                    trajs_ag[i].v_rews[vi], gamma=1.0)), decimal=4 if parser_args.dtype in [np.float32, np.float64] else 3)
                if not environment_data['is_contextual']:
                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, i], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        environment.reward_matrix_per_align_func(environment.basic_profiles[vi])[[trajs_ag[i].infos[0]['state'],], trajs_ag[i].acts], gamma=1.0)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

            for idx, pr in zip(idxs, preferences_per_grounding[:, vi]):
                idx = [int(ix) for ix in idx]

                np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[0]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                    trajs_ag[idx[0]].v_rews[vi], gamma=1.0)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[1]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                    trajs_ag[idx[1]].v_rews[vi], gamma=1.0)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                if not environment_data['is_contextual']:
                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[0]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        environment.reward_matrix_per_align_func(environment.basic_profiles[vi])[[trajs_ag[idx[0]].infos[0]['state'],], trajs_ag[idx[0]].acts], gamma=1.0)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[1]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        environment.reward_matrix_per_align_func(environment.basic_profiles[vi])[[trajs_ag[idx[1]].infos[0]['state'],], trajs_ag[idx[1]].acts], gamma=1.0)), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
                """THIS IS to check the rewards in the grounding are compared as they indicate. It is not useful when ground truth comparisons are given. np.testing.assert_almost_equal(compare_trajectories(
                    discounted_sums_per_grounding[vi, idx[0]], discounted_sums_per_grounding[vi, idx[1]], epsilon=parser_args.reward_epsilon), pr, decimal=4 if parser_args.dtype in [np.float32, np.float64] else 3)
"""
    print("Dataset generated correctly.")
    dataset_train = create_dataset(parser_args, config, society_data, train_or_test='train',
                                   default_groundings=society_config[parser_args.environment]['groundings'])
    print("TRAIN SIZE", len(dataset_train))
    dataset_test = create_dataset(parser_args, config, society_data, train_or_test='test',
                                  default_groundings=society_config[parser_args.environment]['groundings'])
    print("TEST SIZE", len(dataset_test))

