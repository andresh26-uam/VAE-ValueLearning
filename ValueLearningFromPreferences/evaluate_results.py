import argparse
import pprint
import random

import numpy as np
import torch

from generate_dataset import DEFAULT_SEED
from src.algorithms.clustering_utils import ClusterAssignment
from train_vsl import load_training_results
from utils import filter_none_args, load_json_config


def parse_args():
    # IMPORTANT: Default Args are specified depending on the environment in config.json

    parser = argparse.ArgumentParser(
        description="This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected. See the societies.json and algorithm_config.json files")

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument('-sname', '--society_name', type=str, default='default',
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')

    general_group.add_argument('-ename', '--experiment_name', type=str,
                               default='test_experiment', required=True, help='Experiment name')
    
    general_group.add_argument('-sp', '--split_ratio', type=float, default=0.2,
                               help='Test split ratio. If 0.0, no split is done. If 1.0, all data is used for testing.')
    general_group.add_argument('-cf', '--config_file', type=str, default='algorithm_config.json',
                               help='Path to JSON general configuration file (overrides other defaults here, but not the command line arguments)')
    general_group.add_argument('-sf', '--society_file', type=str, default='societies.json',
                               help='Path to JSON society configuration file (overrides other defaults here, but not the command line arguments)')

    general_group.add_argument('-sh', '--show', action='store_true', default=False,
                               help='Show plots calculated before saving')
    general_group.add_argument(
        '-s', '--seed', type=int, default=DEFAULT_SEED, required=False, help='Random seed')

    general_group.add_argument('-e', '--environment', type=str, default='ff', choices=[
                               'rw', 'ff', 'vrw', 'apollo'], help='environment (roadworld - rw, firefighters - ff, variablerw - vrw)')

    return parser.parse_args()

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
    
    
    experiment_name = parser_args.experiment_name
    experiment_name = experiment_name + '_' + str(parser_args.split_ratio)
    
    target_agent_and_vs_to_learned_ones, reward_net_pair_agent_and_vs, metrics, historic_assignments = load_training_results(
        experiment_name)

    
    assignment: ClusterAssignment = historic_assignments[0]
    

    assignment.plot_vs_assignments("demo.png")
    print(assignment)