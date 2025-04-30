import argparse
import dill
import pprint
import random
from typing import Union
import numpy as np
import torch
from generate_dataset import parse_dtype_torch
from src.dataset_processing.utils import DATASETS_PATH, DEFAULT_SEED, GROUNDINGS_PATH
from generate_dataset_one_shot_tasks import PICKLED_ENVS
from src.algorithms.clustering_utils import ClusterAssignment
from src.algorithms.preference_based_vsl import PreferenceBasedClusteringTabularMDPVSL
from src.dataset_processing.datasets import calculate_dataset_save_path
from src.reward_nets.vsl_reward_functions import LinearVSLRewardFunction, TrainingModes, parse_layer_name
from train_vsl import load_training_results, parse_cluster_sizes, parse_feature_extractors, parse_optimizer_data, save_training_results
from src.dataset_processing.data import VSLPreferenceDataset
import os
from src.utils import filter_none_args, load_json_config


def parse_args():
    # IMPORTANT: Default Args are specified depending on the environment in config.json

    parser = argparse.ArgumentParser(
        description="This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected. See the societies.json and algorithm_config.json files")

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument('-dname', '--dataset_name', type=str,
                               default='test_dataset', required=True, help='Dataset name')
    """general_group.add_argument('-sname', '--society_name', type=str, default='default',
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')
"""
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
    """ general_group.add_argument('-sf', '--society_file', type=str, default='societies.json',
                               help='Path to JSON society configuration file (overrides other defaults here, but not the command line arguments)')"""

    general_group.add_argument('-sh', '--show', action='store_true', default=False,
                               help='Show plots calculated before saving')

    general_group.add_argument('-e', '--environment', type=str, default='apollo', choices=[
                               'apollo'], help='environment (apollo)')

    general_group.add_argument('-df', '--discount_factor', type=float, default=1.0,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')

    general_group.add_argument('-sp', '--split_ratio', type=float,
                               default=0.0, help='Split ratio for train/test set. 0.2 means 80% train, 20% test')
    alg_group = parser.add_argument_group('Algorithm-specific Parameters')
    alg_group.add_argument('-k', '--k_clusters', type=Union[int, list], default=-1,
                           help="Number of clusters per value (overriging configuration file)")

    debug_params = parser.add_argument_group('Debug Parameters')
    debug_params.add_argument('-db', '--debug_mode', action='store_true',
                              default=False, help='Debug Mode')

    env_group = parser.add_argument_group('environment-specific Parameters')

    env_group.add_argument('-rte', '--retrain_experts', action='store_true',
                           default=False, help='Retrain experts (roadworld)')
    env_group.add_argument('-appr', '--approx_expert', action='store_true',
                           default=False, help='Approximate expert (roadworld)')
    env_group.add_argument('-reps', '--reward_epsilon', default=0.000, type=float,
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
    rng_for_algorithms = np.random.default_rng(parser_args.seed)

    environment_data = config[parser_args.environment]
    society_data = society_config[parser_args.environment]['default']
    parser_args.society_name = 'default'
    grounding_path = os.path.join(
        'envs', parser_args.environment, GROUNDINGS_PATH)
    dataset_name = parser_args.dataset_name
    experiment_name = parser_args.experiment_name

    experiment_name = experiment_name #+ '_' + str(parser_args.split_ratio)

    """agent_profiles = [tuple(ag['value_system'])
                      for ag in society_data['agents']]
    agent_groundings = [tuple(ag['grounding'])
                        for ag in society_data['agents']]
    ag_name_to_aggrounding = {ag['name']: tuple(
        ag['grounding']) for ag in society_data['agents']}
    grounding_files = society_config[parser_args.environment]['groundings']"""
    """all_agent_groundings_to_save_files = dict(
        {agg: [grounding_files[agg[i]] for i in range(len(agg))] for agg in set(agent_groundings)})"""

    extra_kwargs = {}

    if parser_args.environment == 'apollo':
        extra_kwargs = {
            'test_size': parser_args.split_ratio
        }

    try:
        f = open(os.path.join(os.path.join(
            PICKLED_ENVS, environment_data['name'], dataset_name), f"env_kw_{extra_kwargs}.pkl"), 'rb')
        environment = dill.load(f)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print("TESTING DATA COHERENCE. It is safe to stop this program now...")

    rewward_net_features_extractor_class, policy_features_extractor_class, features_extractor_kwargs, policy_features_extractor_kwargs = parse_feature_extractors(
        environment, environment_data, dtype=parser_args.dtype)

    alg_config = environment_data['algorithm_config'][parser_args.algorithm]

    data_reward_net = environment_data['default_reward_net']
    data_reward_net.update(alg_config['reward_net'])

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

    path = os.path.join(
        DATASETS_PATH, calculate_dataset_save_path(dataset_name, environment_data, society_data, epsilon=parser_args.reward_epsilon))
    dataset_train = VSLPreferenceDataset.load(
        os.path.join(path, "dataset_train.pkl"))
    dataset_test = VSLPreferenceDataset.load(
        os.path.join(path, "dataset_test.pkl"))

    if parser_args.algorithm == 'pc':

        # TODO: K FOLD CROSS VALIDATION. AND ALSO TEST SET EVALUATION!!!
        vsl_algo = PreferenceBasedClusteringTabularMDPVSL(
            env=environment,
            reward_net=reward_net,
            optimizer_cls=opt_class,
            optimizer_kwargs=opt_kwargs,
            discount=environment_data['discount'],
            discount_factor_preferences=alg_config['discount_factor_preferences'],
            dataset=dataset_train,
            training_mode=TrainingModes.SIMULTANEOUS,
            cluster_sizes=parse_cluster_sizes(
                environment_data['K'] if parser_args.k_clusters == -1 else parser_args.k_clusters, n_values=environment_data['n_values']),
            vs_cluster_sizes=environment_data['L'] if isinstance(
                environment_data['L'], int) else None,

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
            custom_logger='disable',
            debug_mode=parser_args.debug_mode,
            assume_variable_horizon=environment_data['assume_variable_horizon']

        )
    if parser_args.algorithm == 'pc':
        alg_config['train_kwargs']['experiment_name'] = experiment_name
    target_agent_and_vs_to_learned_ones_s, reward_net_pair_agent_and_vs_s, metrics_s, historic_assignments_s  =vsl_algo.train(mode=TrainingModes.SIMULTANEOUS,
                   assumed_grounding=None, **alg_config['train_kwargs'])
    # Now we need to train.
    save_training_results(experiment_name, target_agent_and_vs_to_learned_ones_s,
                          reward_net_pair_agent_and_vs_s, metrics_s, parser_args={'parser_args': parser_args, 'config': config, 'society_config': society_config})
    print(metrics_s['assignment_memory'])
    target_agent_and_vs_to_learned_ones, reward_net_pair_agent_and_vs, metrics, parser_args, historic_assignments, env_state, n_iterations = load_training_results(
        experiment_name)
    
    assignment: ClusterAssignment = historic_assignments[-1]
    
    assert target_agent_and_vs_to_learned_ones == target_agent_and_vs_to_learned_ones_s, "Mismatch in target_agent_and_vs_to_learned_ones"
    assert reward_net_pair_agent_and_vs.keys() == reward_net_pair_agent_and_vs_s.keys(
    ), "Mismatch in reward_net_pair_agent_and_vs"
    assert metrics.keys() == metrics_s.keys(), "Mismatch in metrics"
