import argparse
from collections import defaultdict
import os
import pprint
import random
from typing import Sequence, Union

import imitation
import numpy as np
import torch

from envs.firefighters_env import FeatureSelectionFFEnv
from envs.tabularVAenv import TabularVAMDP
from generate_dataset import COMPARISONS_DATASETS_PATH, DEFAULT_SEED, GROUNDINGS_PATH, compare_trajectories, load_preferences, load_trajectories, parse_dtype_torch
from src.algorithms.preference_based_vsl import PreferenceBasedClusteringTabularMDPVSL
from src.data import TrajectoryWithValueSystemRewsPair, VSLPreferenceDataset
from src.feature_extractors import ContextualFeatureExtractorFromVAEnv, FeatureExtractorFromVAEnv, OneHotFeatureExtractor
from src.reward_nets.vsl_reward_functions import GroundingEnsemble, LinearVSLRewardFunction, TrainingModes, parse_layer_name
from use_cases.roadworld_env_use_case.network_env import FeaturePreprocess, FeatureSelection
from utils import filter_none_args, load_json_config


import gymnasium as gym


def load_dataset(parser_args, config, society_data={'name': "default", "same_trajectories_for_each_agent_type": False}, train_or_test=None, default_groundings=None):
    environment_data = config[parser_args.environment]

    dataset_name = parser_args.dataset_name
    if train_or_test is not None:
        dataset_name+='_'
        assert train_or_test == 'train' or train_or_test == 'test'
        dataset_name+=train_or_test 

    dataset = VSLPreferenceDataset(n_values=environment_data['n_values'])

    if 'agents' not in society_data.keys():
        agents = []
        # TODO: HERE; THIS PATH IS NOT CORRECT!!
        folder_path = os.path.join(COMPARISONS_DATASETS_PATH, f"{environment_data['name']}/{society_data['name']}/{dataset_name}")
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                if dir_name.startswith("prefs_ag_"):
                    ag_name = dir_name.split("_")[2]
                    agents.append(ag_name)
    else:
        agents = society_data['agents']

    for i, ag in enumerate(agents):
        if 'agents' not in society_data.keys():
            ag = {'agent_id': ag, 'name': ag, 'value_system': 'unk', 'data': defaultdict(lambda: 'nd'), 'n_agents': 1, 'grounding': list(default_groundings.keys())}
        #  Here idxs is the list of trajectory PAIRS of indices from the trajectory list that are compared.
        idxs, discounted_sums, discounted_sums_per_grounding, preferences, preferences_per_grounding = load_preferences(
            epsilon=parser_args.reward_epsilon, dataset_name=dataset_name, environment_data=environment_data, society_data=society_data, ag=ag, dtype=parser_args.dtype)
        trajs_ag = np.asarray(load_trajectories(dataset_name=dataset_name,
                              ag=ag, environment_data=environment_data, society_data=society_data,  override_dtype=parser_args.dtype))
        
        if society_data["same_trajectories_for_each_agent_type"]:
            for t in range(ag['n_agents']-1):
                np.testing.assert_allclose(idxs[0:ag['data']['trajectory_pairs']] % len(trajs_ag), (idxs[(
                    t+1)*ag['data']['trajectory_pairs']:(t+2)*ag['data']['trajectory_pairs']] - ag['data']['trajectory_pairs']*(t+1)) % len(trajs_ag))

                for traj_i in range(ag['data']['trajectory_pairs']):

                    np.testing.assert_allclose(trajs_ag[traj_i + t*ag['data']['trajectory_pairs']].obs, trajs_ag[(
                        t+1)*ag['data']['trajectory_pairs'] + traj_i].obs)

        
        ag_point = 0
        n_pairs_per_agent = len(idxs)//ag['n_agents']
        for id in range(ag['n_agents']):
            agent_id = ag['name']+'_'+str(id)
            ag_idxs = idxs[ag_point:ag_point+n_pairs_per_agent]

            trajectory_pairs: Sequence[TrajectoryWithValueSystemRewsPair] = trajs_ag[ag_idxs]
            dataset.push(trajectory_pairs, preferences[ag_point:ag_point+n_pairs_per_agent], preferences_per_grounding[ag_point:(ag_point+n_pairs_per_agent)], agent_ids=[agent_id]*n_pairs_per_agent, agent_data={agent_id: ag})
            if society_data["same_trajectories_for_each_agent_type"] and ag_point > 0:
                prev_ag_idxs = idxs[ag_point -
                                    n_pairs_per_agent:ag_point+n_pairs_per_agent]
                for j in range(len(ag_idxs)):
                    np.testing.assert_allclose(
                        trajs_ag[ag_idxs][j][0].obs, trajs_ag[prev_ag_idxs][j][0].obs)
                    np.testing.assert_allclose(
                        trajs_ag[ag_idxs][j][1].obs, trajs_ag[prev_ag_idxs][j][1].obs)
                if ag['name'] == las_agent_name:
                    np.testing.assert_allclose(
                        dataset.data_per_agent[last_agent_id].preferences_with_grounding, dataset.data_per_agent[agent_id].preferences_with_grounding)
                    np.testing.assert_allclose([t.obs for t in dataset.data_per_agent[last_agent_id].fragments1], [
                                               t.obs for t in dataset.data_per_agent[agent_id].fragments1])

            ag_point += n_pairs_per_agent
            last_agent_id = agent_id
            las_agent_name = ag['name']

        """for i in range((len(trajs_ag))):
            assert discounted_sums[i] == imitation.data.rollout.discounted_sum(
                trajs_ag[i].rews, gamma=alg_config['discount_factor_preferences'])
        for idx, pr in zip(idxs, preferences):
            assert discounted_sums[idx[0]] == imitation.data.rollout.discounted_sum(
                trajs_ag[idx[0]].rews, gamma=alg_config['discount_factor_preferences'])
            assert discounted_sums[idx[1]] == imitation.data.rollout.discounted_sum(
                trajs_ag[idx[1]].rews, gamma=alg_config['discount_factor_preferences'])
            assert compare_trajectories(
                discounted_sums[idx[0]], discounted_sums[idx[1]], epsilon=parser_args.reward_epsilon) == pr
        for vi in range(len(environment_data['basic_profiles'])):

            for idx, pr in zip(idxs, preferences_per_grounding[vi]):
                assert compare_trajectories(
                    discounted_sums_per_grounding[vi, idx[0]], discounted_sums_per_grounding[vi, idx[1]], epsilon=parser_args.reward_epsilon) == pr"""

    return dataset


def parse_args():
    # IMPORTANT: Default Args are specified depending on the environment in config.json

    parser = argparse.ArgumentParser(
        description="This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected. See the societies.json and algorithm_config.json files")

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument('-dname', '--dataset_name', type=str,
                               default='test_dataset', required=True, help='Dataset name')
    general_group.add_argument('-sname', '--society_name', type=str, default='default',
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')

    general_group.add_argument('-ename', '--experiment_name', type=str,
                               default='test_experiment', required=True, help='Experiment name')
    
    general_group.add_argument('-dtype', '--dtype', type=parse_dtype_torch, default=torch.float32, choices=[torch.float32, torch.float64],
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')


    general_group.add_argument(
        '-s', '--seed', type=int, default=DEFAULT_SEED, required=False, help='Random seed')

    general_group.add_argument('-a', '--algorithm', type=str, choices=[
                               'pc'], default='pc', help='Algorithm to use (preference comparison - pc)')

    general_group.add_argument('-cf', '--config_file', type=str, default='algorithm_config.json',
                               help='Path to JSON general configuration file (overrides other defaults here, but not the command line arguments)')
    general_group.add_argument('-sf', '--society_file', type=str, default='societies.json',
                               help='Path to JSON society configuration file (overrides other defaults here, but not the command line arguments)')

    general_group.add_argument('-sh', '--show', action='store_true', default=False,
                               help='Show plots calculated before saving')

    general_group.add_argument('-e', '--environment', type=str, default='ff', choices=[
                               'rw', 'ff'], help='environment (roadworld - rw, firefighters - ff, itemgathering - ig)')

    general_group.add_argument('-df', '--discount_factor', type=float, default=1.0,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')

    general_group.add_argument('-n', '--n_experiments', type=int,
                               default=1, help='Number of experiment repetitions')

    alg_group = parser.add_argument_group('Algorithm-specific Parameters')

    alg_group.add_argument('-k', '--k_clusters', type=Union[int, list], default=-1,
                           help="Number of clusters per value (overriging configuration file)")

    debug_params = parser.add_argument_group('Debug Parameters')
    debug_params.add_argument('-db', '--check_rewards', action='store_true',
                              default=False, help='Check rewards before learning for debugging')

    env_group = parser.add_argument_group('environment-specific Parameters')

    env_group.add_argument('-rt', '--retrain', action='store_true',
                           default=False, help='Retrain experts (roadworld)')
    env_group.add_argument('-appr', '--approx_expert', action='store_true',
                           default=False, help='Approximate expert (roadworld)')
    env_group.add_argument('-reps', '--reward_epsilon', default=0.0, type=float,
                           help='Distance between the cummulative rewards of each pair of trajectories for them to be considered as equal in the comparisons')

    return parser.parse_args()

def parse_cluster_sizes(k, n_values):
    if isinstance(k, int):
        return [k]*n_values
    elif isinstance(k, list):
        assert len(k) == n_values
        return k
    else:
        raise ValueError(f"Number of clusters not identifiable {k}")


def parse_feature_extractors(environment, environment_data, dtype=torch.float32):
    # Dummy implementation, replace with actual logic
    if environment_data['reward_feature_extractor'] == "FeatureExtractorFromVAEnv":
        reward_net_features_extractor_class = FeatureExtractorFromVAEnv
        reward_net_features_extractor_kwargs = dict(
            env=environment,
            dtype=dtype,
        )
    elif environment_data['reward_feature_extractor'] == "ContextualFeatureExtractorFromVAEnv":
        reward_net_features_extractor_class = ContextualFeatureExtractorFromVAEnv
        reward_net_features_extractor_kwargs = dict(
            env=environment,
            dtype=dtype,
        )
    
    else:
        raise ValueError(
            f"Unknown reward feature extractor {environment_data['reward_feature_extractor']}")
    # "reward_feature_extractor": "FeatureExtractorFromVAEnv",
    # "policy_state_feature_extractor": "OneHotFeatureExtractor",
    if environment_data['policy_state_feature_extractor'] == "OneHotFeatureExtractor":
        policy_features_extractor_class = OneHotFeatureExtractor
        policy_features_extractor_kwargs = dict(
            n_categories=environment.action_dim,
            dtype=dtype)
    else:
        raise ValueError(
            f"Unknown policy feature extractor {environment_data['reward_feature_extractor']}")
    return reward_net_features_extractor_class, policy_features_extractor_class, reward_net_features_extractor_kwargs, policy_features_extractor_kwargs


def parse_optimizer_data(environment_data, alg_config):
    opt_kwargs = environment_data['default_optimizer_kwargs']
    opt_kwargs = opt_kwargs if alg_config['optimizer_kwargs'] == "default" else alg_config['optimizer_kwargs']

    opt_class = environment_data['default_optimizer_class']
    opt_class = opt_class if alg_config['optimizer_class'] == "default" else alg_config['optimizer_class']

    if opt_class == 'Adam':
        opt_class = torch.optim.Adam
    else:
        raise ValueError(f"Unknown optimizer class {opt_class}")
    return opt_kwargs, opt_class


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
    rng_for_algorithms = np.random.default_rng(parser_args.seed)

    environment_data = config[parser_args.environment]
    society_data = society_config[parser_args.environment][parser_args.society_name]
    grounding_path = os.path.join(
        'envs', parser_args.environment, GROUNDINGS_PATH)
    dataset_name = parser_args.dataset_name
    experiment_name = parser_args.experiment_name

    agent_profiles = [tuple(ag['value_system'])
                      for ag in society_data['agents']]
    agent_groundings = [tuple(ag['grounding'])
                        for ag in society_data['agents']]
    ag_name_to_aggrounding = {ag['name']: tuple(
        ag['grounding']) for ag in society_data['agents']}
    grounding_files = society_config[parser_args.environment]['groundings']
    all_agent_groundings_to_save_files = dict(
        {agg: [grounding_files[agg[i]] for i in range(len(agg))] for agg in set(agent_groundings)})

    extra_kwargs = {}
    if parser_args.environment == 'ff':
        extra_kwargs = {
            'feature_selection': FeatureSelectionFFEnv(environment_data['feature_selection']),
            'initial_state_distribution': environment_data['initial_state_distribution']
        }
    if parser_args.environment == 'rw':
        extra_kwargs = {'env_kwargs': {
            'feature_selection': FeatureSelection(environment_data['feature_selection']),
            'feature_preprocessing': FeaturePreprocess(environment_data['feature_preprocessing']),
            
        }}
        if 'Fixed' in environment_data['name'] :
            extra_kwargs['with_destination'] = 64
            
    environment: TabularVAMDP = gym.make(
        environment_data['name'],
        horizon=environment_data['horizon'], **extra_kwargs)
    environment.reset(seed=parser_args.seed)

    
    print("TESTING DATA COHERENCE. It is safe to stop this program now...")

    rewward_net_features_extractor_class, policy_features_extractor_class, features_extractor_kwargs, policy_features_extractor_kwargs = parse_feature_extractors(
        environment, environment_data, dtype=parser_args.dtype)

    alg_config = environment_data['algorithm_config'][parser_args.algorithm]

    data_reward_net = environment_data['default_reward_net']
    data_reward_net.update(alg_config['reward_net'])
    GroundingEnsemble
    reward_net = LinearVSLRewardFunction(
        environment=environment,
        use_state=data_reward_net['use_state'],
        use_action=data_reward_net['use_action'],
        use_next_state=data_reward_net['use_next_state'],
        use_done=data_reward_net['use_done'],
        hid_sizes=data_reward_net['hid_sizes'],
        reward_bias=0,
        basic_layer_classes=[parse_layer_name(
            l) for l in data_reward_net['basic_layer_classes']],
        use_one_hot_state_action=False,
        activations=[parse_layer_name(l)
                     for l in data_reward_net['activations']],
        negative_grounding_layer=data_reward_net['negative_grounding_layer'],
        use_bias=data_reward_net['use_bias'],
        clamp_rewards=data_reward_net['clamp_rewards'],
        mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
        features_extractor_class=rewward_net_features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
        action_features_extractor_class=policy_features_extractor_class,
        action_features_extractor_kwargs=policy_features_extractor_kwargs,
        dtype=parser_args.dtype

    )
    
    opt_kwargs, opt_class = parse_optimizer_data(environment_data, alg_config)

    dataset = load_dataset(parser_args, config, society_data, default_groundings = society_config[parser_args.environment]['groundings'])

    if parser_args.algorithm == 'pc':
        vsl_algo = PreferenceBasedClusteringTabularMDPVSL(
            env=environment,
            reward_net=reward_net,
            optimizer_cls=opt_class,
            optimizer_kwargs=opt_kwargs,
            discount=environment_data['discount'],
            discount_factor_preferences=alg_config['discount_factor_preferences'],
            dataset=dataset,
            training_mode=TrainingModes.SIMULTANEOUS, 
            cluster_sizes=parse_cluster_sizes(
                environment_data['K'] if parser_args.k_clusters == -1 else parser_args.k_clusters, n_values=environment_data['n_values']),
                vs_cluster_sizes=environment_data['L'] if isinstance(environment_data['L'], int) else None,

            learn_stochastic_policy=alg_config['learn_stochastic_policy'],
            use_quantified_preference=alg_config['use_quantified_preference'],
            preference_sampling_temperature=0 if alg_config[
                'use_quantified_preference'] else alg_config['preference_sampling_temperature'],
            log_interval=1,
            reward_trainer_kwargs=alg_config['reward_trainer_kwargs'],
            query_schedule=alg_config['query_schedule'],
            vgl_target_align_funcs=environment_data['basic_profiles'],
            approximator_kwargs=alg_config['approximator_kwargs'],
            policy_approximator=alg_config['policy_approximation_method'],
            rng=rng_for_algorithms,
            # This is only used for testing purposes
            expert_is_stochastic=society_data['stochastic_expert'],
            loss_class=alg_config['loss_class'],
            loss_kwargs=alg_config['loss_kwargs'],
            assume_variable_horizon=environment_data['assume_variable_horizon']

        )
    
    vsl_algo.train(mode=TrainingModes.SIMULTANEOUS,
                   assumed_grounding=None, **alg_config['train_kwargs'])
    # Now we need to train.
