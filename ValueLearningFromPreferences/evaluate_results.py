import argparse
import os
import pprint
import random

import numpy as np
import torch

from envs.tabularVAenv import TabularVAMDP
from src.dataset_processing.utils import DATASETS_PATH, DEFAULT_SEED
from src.algorithms.clustering_utils import ClusterAssignment, ClusterAssignmentMemory
from src.algorithms.preference_based_vsl import PreferenceBasedClusteringTabularMDPVSL
from src.dataset_processing.data import VSLPreferenceDataset
from src.dataset_processing.datasets import calculate_dataset_save_path
from src.reward_nets.vsl_reward_functions import LinearVSLRewardFunction, TrainingModes
from train_vsl import load_training_results, parse_cluster_sizes, parse_optimizer_data
from src.utils import filter_none_args, load_json_config


import pandas as pd

def generate_assignment_tables(assignment_memory: ClusterAssignmentMemory, experiment_name, output_columns, output_dir="test_results", label='train_set'):
    """
    Generate tables for the first, middle, and last assignments in the assignment memory.

    Args:
        assignment_memory (ClusterAssignmentMemory): The memory containing cluster assignments.
        experiment_name (str): The name of the experiment.
        output_columns (dict): A dictionary specifying which columns to include in the output.
                               Example: {"value_systems": True, "num_clusters": True, ...}
        output_dir (str): The base directory for saving the output files.
    """
    # Ensure output directories exist
    csv_dir = os.path.join(output_dir, experiment_name, label, "csv")
    latex_dir = os.path.join(output_dir, experiment_name, label, "latex")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)

    # Select the first, middle, and last assignments
    assignments = [
        ("first", assignment_memory.memory[0]),
        ("middle", assignment_memory.memory[len(assignment_memory.memory) // 2]),
        ("last", assignment_memory.memory[-1]),
    ]

    for position, assignment in assignments:
        # Prepare data for the table
        data = []
        
        for cluster_idx, agents in enumerate(assignment.assignment_vs):
            if len(agents) == 0:
                continue
            row = {}
            row["Cluster"] = cluster_idx + 1
            if output_columns.get("value_systems", False):
                row["Value System"] = ", ".join(f"{v:.3f}" for v in assignment.get_value_system(cluster_idx))
            if output_columns.get("num_agents", False):
                row["Number of Agents"] = len(agents) 
            if output_columns.get("representativity", False):
                intra_cluster_distances = [d for agent, d in assignment.intra_discordances_vs_per_agent.items() if agent in agents]
                row["Representativeness"] = ClusterAssignment._representativity_cluster(intra_cluster_distances)
            if output_columns.get("conciseness", False):
                if assignment.L == 1:
                    row['Conciseness'] = '-'
                else:
                    inter_cluster_distances = [d for kpair, d in assignment.inter_discordances_vs_per_cluster_pair.items() if kpair[0] == cluster_idx or kpair[1] == cluster_idx]
                    row["Conciseness"] = ClusterAssignment._conciseness(inter_cluster_distances, assignment.L)
            if output_columns.get("combined_score", False):
                # Use '-' if L is 1, otherwise calculate the combined score
                row["Combined Score"] = "-" if assignment.L == 1 else row["Conciseness"]/(1.0-row['Representativeness'])  
            if output_columns.get("grounding_coherence", False):
                # Expand coherence array into separate columns
                coherence_array = assignment.gr_score

                for i, value in enumerate(coherence_array):
                    intra_cluster_distances_gr = [d for agent, d in assignment.intra_discordances_gr_per_agent[i].items() if agent in agents]
                    row[f"Coherence V{i + 1}"] = ClusterAssignment._representativity_cluster(intra_cluster_distances_gr)
            data.append(row)
        # Assingment-level information:
        row = {}
        row["Cluster"] = "Total"
        if output_columns.get("value_systems", False):
            row["Value System"] = ", ".join(f"{v:.3f}" for v in assignment.average_value_system())
        if output_columns.get("num_agents", False):
            row["Number of Agents"] = assignment.n_agents
        if output_columns.get("representativity", False):
            row["Representativeness"] = assignment.representativity_vs()
        if output_columns.get("conciseness", False):
            if assignment.L == 1:
                row['Conciseness'] = '-'
            else:
                row["Conciseness"] = assignment.conciseness_vs()
        if output_columns.get("combined_score", False):
            # Use '-' if L is 1, otherwise calculate the combined score
            row["Combined Score"] = "-" if assignment.L == 1 else assignment.combined_cluster_score_vs()  
        if output_columns.get("grounding_coherence", False):
            coherence_array = assignment.gr_score
            for i, value in enumerate(coherence_array):
                row[f"Coherence V{i + 1}"] = coherence_array[i]
        
        data.append(row)
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(csv_dir, f"{position}_assignment.csv")
        df.to_csv(csv_path, index=False)

        # Save to LaTeX
        latex_path = os.path.join(latex_dir, f"{position}_assignment.tex")
        with open(latex_path, "w") as f:
            f.write(df.to_latex(index=False, escape=False))

        print(f"Saved {position} assignment table to {csv_path} and {latex_path}")
        
def parse_args():
    # IMPORTANT: Default Args are specified depending on the environment in config.json

    parser = argparse.ArgumentParser(
        description="This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected. See the societies.json and algorithm_config.json files")

    general_group = parser.add_argument_group('General Parameters')
    
    general_group.add_argument('-ename', '--experiment_name', type=str,
                               default='test_experiment', required=True, help='Experiment name')
    
    general_group.add_argument('-sh', '--show', action='store_true', default=False,
                               help='Show plots calculated before saving')
    
    #subfig_multiplier
    general_group.add_argument('-subfm', '--subfig_multiplier', type=float, default=6.0,
                               help='Scales subfigs inside the plots.')
    general_group.add_argument('-pfont', '--plot_fontsize', type=int, default=12,
                               help='Font size in plots.')
    general_group.add_argument(
        '-s', '--seed', type=int, default=DEFAULT_SEED, required=False, help='Random seed')

    
    return parser.parse_args()

if __name__ == "__main__":
    # This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected
    # IMPORTANT: Default Args are specified depending on the environment in config.json
    parser_args = filter_none_args(parse_args())
    # If a config file is specified, load it and override command line args
    experiment_name = parser_args.experiment_name
    target_agent_and_vs_to_learned_ones, reward_net_pair_agent_and_vs, metrics, exp_parser_args_base, historic_assignments = load_training_results(
        experiment_name)
    config = exp_parser_args_base['config']
    society_config = exp_parser_args_base['society_config']
    exp_parser_args = exp_parser_args_base['parser_args']

    # This will look again into the config files to see if there are new fields (wont update the old ones)
    config_actual = load_json_config(exp_parser_args.config_file)
    society_config_actual = load_json_config(exp_parser_args.society_file if hasattr(exp_parser_args, 'society_file') else 'societies.json')

    def merge_dicts_recursive(base, update):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dicts_recursive(base[key], value)
            else:
                base[key] = value

    merge_dicts_recursive(config, config_actual)
    merge_dicts_recursive(society_config, society_config_actual)
        
    np.random.seed(parser_args.seed)
    torch.manual_seed(parser_args.seed)
    random.seed(parser_args.seed)
    rng_for_algorithms = np.random.default_rng(parser_args.seed)

    environment_data = config[exp_parser_args.environment]
    society_data = society_config[exp_parser_args.environment][exp_parser_args.society_name]
    alg_config = environment_data['algorithm_config'][exp_parser_args.algorithm]
    
    experiment_name = exp_parser_args.experiment_name
    dataset_name = exp_parser_args.dataset_name

    
    
    opt_kwargs, opt_class = parse_optimizer_data(environment_data, alg_config)
    
    path = os.path.join(
        DATASETS_PATH, calculate_dataset_save_path(dataset_name, environment_data, society_data, epsilon=exp_parser_args.reward_epsilon))
    dataset_train = VSLPreferenceDataset.load(
        os.path.join(path, "dataset_train.pkl"))
    dataset_test = VSLPreferenceDataset.load(
        os.path.join(path, "dataset_test.pkl"))

    example_model: LinearVSLRewardFunction = list(reward_net_pair_agent_and_vs.values())[0]
    env = example_model.remove_env()
    example_model.set_env(env)
    if exp_parser_args.algorithm == 'pc':

        # TODO: K FOLD CROSS VALIDATION. AND ALSO TEST SET EVALUATION!!!
        vsl_algo = PreferenceBasedClusteringTabularMDPVSL(
            env=example_model.remove_env(),
            reward_net=example_model,
            optimizer_cls=opt_class,
            optimizer_kwargs=opt_kwargs,
            discount=environment_data['discount'],
            discount_factor_preferences=alg_config['discount_factor_preferences'],
            dataset=dataset_train,
            training_mode=TrainingModes.SIMULTANEOUS,
            cluster_sizes=parse_cluster_sizes(
                environment_data['K'], n_values=environment_data['n_values']),
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
            assume_variable_horizon=environment_data['assume_variable_horizon']

        )
    if exp_parser_args.algorithm == 'pc':
        alg_config['train_kwargs']['experiment_name'] = experiment_name

    

    assignment_memory: ClusterAssignmentMemory = metrics['assignment_memory']
    assignment_memory.sort_lexicographic(lexicographic_vs_first=False)


    # 1: tables.
    # Put for the first, middle, and last assignment in separated tables.
    # For each assignment, put in a table the value systems of each cluster, the number of agents, the representativity of each cluster regarding value systems, average distance to other clusters, the combined score, the representativity and conciseness of the assignment, and the grouinding coherence (given by the .gr_score).
    #  Make every single column modular, i.e. to activate or deactivate it with a flag.
    # Output the tables in latex anc csv in the test_results/{experiment_name}/csv and test_results/{experiment_name}/latex folders.
    
    output_columns = {
        "value_systems": True,
        "num_agents": True,
        "representativity": True,
        "conciseness": True,
        "combined_score": True,
        "grounding_coherence": True,
    }

    # Generate tables
    generate_assignment_tables(assignment_memory, experiment_name, output_columns, output_dir='test_results', label='train_set')

    best_vs_then_gr_assignment = assignment_memory.memory[-1]
    best_gr_then_vs_assignment = assignment_memory.memory[0]

    test_assignment_memory = ClusterAssignmentMemory(assignment_memory.max_size, n_values=assignment_memory.memory[0].n_values)
    test_assignment_memory.maximum_conciseness_vs = assignment_memory.maximum_conciseness_vs
    test_assignment_memory.maximum_conciseness_gr = assignment_memory.maximum_conciseness_gr # these two are train estimations.
    for a in assignment_memory.memory:
        test_assignment = vsl_algo.evaluate_assignment(a, dataset_test)
        test_assignment_memory.memory.append(test_assignment) # just insert it.

    generate_assignment_tables(test_assignment_memory, experiment_name, output_columns, output_dir='test_results', label='test_set')


    # 2: plots.
    # 3: SHAP VALUES TODO
    # 4: CONTEXT CLUSTERS TODO
    
    print(best_gr_then_vs_assignment)
    print(best_vs_then_gr_assignment)
    print("TEST ASIGNMENT")
    print(test_assignment)
    print(len(assignment_memory))
    print(assignment_memory)

    best_vs_then_gr_assignment.plot_vs_assignments(f"test_results/{experiment_name}/plots/figure_clusters_vs_gr.pdf", 
                                                   subfig_multiplier=parser_args.subfig_multiplier,
                                                   values_color_map=environment_data['profiles_colors'], 
                                                   values_names=environment_data['values_names'], 
                                                   values_short_names=environment_data['values_short_names'],
                                                   fontsize=parser_args.plot_fontsize,)
    
    best_gr_then_vs_assignment.plot_vs_assignments(f"test_results/{experiment_name}/plots/figure_clusters_gr_vs.pdf", 
                                                   subfig_multiplier=parser_args.subfig_multiplier,
                                                   values_color_map=environment_data['profiles_colors'], 
                                                   values_names=environment_data['values_names'], 
                                                   values_short_names=environment_data['values_short_names'],
                                                   fontsize=parser_args.plot_fontsize,)
    
    
