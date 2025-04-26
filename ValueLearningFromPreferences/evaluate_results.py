import argparse
import os
import pprint
import random
from typing import Dict, List

import numpy as np
import torch
from interpret import show
from envs.routechoiceApollo import RouteChoiceEnvironmentApolloComfort
from envs.tabularVAenv import TabularVAMDP
from src.dataset_processing.utils import DATASETS_PATH, DEFAULT_SEED
from src.algorithms.clustering_utils import ClusterAssignment, ClusterAssignmentMemory
from src.algorithms.preference_based_vsl import PreferenceBasedClusteringTabularMDPVSL
from src.dataset_processing.data import VSLPreferenceDataset
from src.dataset_processing.datasets import calculate_dataset_save_path
from src.reward_nets.vsl_reward_functions import LinearVSLRewardFunction, TrainingModes
from train_vsl import find_parse_ename, load_training_results, parse_cluster_sizes, parse_optimizer_data
from src.utils import filter_none_args, load_json_config, merge_dicts_recursive
import interpret.blackbox as b
from imitation.data.types import (
    TrajectoryPair,
    Transitions,
)
import matplotlib.pyplot as plt
import pandas as pd

def parse_enames_for_learning_curve(learning_curve_from):
    """
    Parses the learning curve from the given string.
    """
    if not learning_curve_from:
        return []
    return [name.strip() for name in learning_curve_from.split(',') if name.strip()]
def generate_assignment_tables(assignment_identifier_to_assignment: Dict[str,ClusterAssignment], experiment_name, output_columns, output_dir="test_results", label='train_set'):
    
    # Ensure output directories exist
    csv_dir = os.path.join(output_dir, experiment_name, label, 'tables' , 'general', "csv")
    latex_dir = os.path.join(output_dir, experiment_name, label, 'tables', 'general', "latex")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)

    # Select the first, middle, and last assignments
    

    for position, assignment in assignment_identifier_to_assignment:
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
        csv_path = os.path.join(csv_dir, f"{position}.csv")
        df.to_csv(csv_path, index=False)

        # Save to LaTeX
        latex_path = os.path.join(latex_dir, f"{position}.tex")
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
    general_group.add_argument('-lrcfrom', '--learning_curve_from', type=parse_enames_for_learning_curve, default=None,help="Generate the learning curve for the specified experiments")
    general_group.add_argument(
        '-s', '--seed', type=int, default=DEFAULT_SEED, required=False, help='Random seed')

    
    return parser.parse_args()

def contextual_feature_analysis(experiment_name, values_names, dataset_reference: VSLPreferenceDataset, assignment: ClusterAssignment, label='train_set', assignment_identifier=''):
    all_context_features = [dataset_reference.data_per_agent[agent_id].fragments1[0].infos[0]['agent_context'] for agent_id in dataset_reference.agent_ids]
    max_context_features = np.max(all_context_features, axis=0)
    context_features_per_cluster = []

    for clust_idx, agent_group in enumerate(assignment.assignment_vs):
        if len(agent_group) == 0:
            continue
        # Extract and normalize context features for the current cluster
        context_features_cidx = np.array([dataset_reference.data_per_agent[agent_id].fragments1[0].infos[0]['agent_context'] for agent_id in agent_group])
        
        context_features_per_cluster.append(context_features_cidx)

        value_system = assignment.get_value_system(clust_idx)
        #print(clust_idx, np.mean(context_features_cidx, axis=0), value_system)

    # Plot all features in a single barplot with standard error bars for each cluster
    feature_names = ["Houshold Income", "Car available", "Conmuting", "Shopping", "Business", "Leisure"]
    cluster_data = []

    for cluster_idx, cluster_features in enumerate(context_features_per_cluster):
        value_system = assignment.get_value_system(cluster_idx)
        
        num_agents = len(cluster_features)
        perc_increase_over_mean = (np.mean(cluster_features, axis=0) - np.mean(all_context_features, axis=0)) / np.mean(all_context_features, axis=0) * 100
        means = np.mean(cluster_features, axis=0)
        std_errors = np.sqrt(np.var(cluster_features, axis=0) / len(cluster_features) + np.var(all_context_features, axis=0) / len(all_context_features)) / np.mean(all_context_features, axis=0) * 100

        # Format the value system with names and values
        value_system_str = ", ".join(f"{name}: {value:.2f}" for name, value in zip(values_names, value_system))

        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(means) + 1), perc_increase_over_mean, yerr=std_errors, capsize=5, alpha=0.7, color='skyblue')
        plt.xticks(range(1, len(means) + 1), [feature_names[i] for i in range(len(means))])
        plt.ylim(-1.4 * 100, 1.4 * 100)  # Set y-axis
        plt.yticks(np.arange(-1.4 * 100, 1.5 * 100, 0.1 * 100), rotation=45)
        plt.title(f"Barplot of Context Features for Cluster {cluster_idx + 1} (Agents: {num_agents})\n(Value System: {value_system_str})")
        plt.xlabel("Features")
        plt.ylabel("Percentage increase/decrease over average")
        plt.grid(axis='y')

        # Save the plot
        plot_dir = os.path.join('test_results', experiment_name, label, 'plots', 'context_features')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"cluster_{cluster_idx + 1}_context_features.pdf")
        plt.savefig(plot_path)
        plt.close()

        # Add cluster data to the table
        cluster_row = [f"{mean:.2f} \\pm{{{perc:+.2f}\\%}}" for mean, perc in zip(means, perc_increase_over_mean)]
        cluster_data.append([f"Cluster {cluster_idx + 1}", num_agents] + cluster_row)

    # Add overall mean data to the table
    overall_means = np.mean(all_context_features, axis=0)
    overall_row = [f"{mean:.2f}" for mean in overall_means]
    cluster_data.append(["Overall", len(dataset_reference.agent_ids)] + overall_row)

    # Save the table to CSV and LaTeX
    table_path_csv = os.path.join('test_results', experiment_name, label, 'tables', 'context_features', 'csv')
    table_path_latex = os.path.join('test_results', experiment_name, label, 'tables', 'context_features', 'latex')
    os.makedirs(table_path_csv, exist_ok=True)
    os.makedirs(table_path_latex, exist_ok=True)
    table_path_csv = os.path.join(table_path_csv, f"context_features_{assignment_identifier}.csv")
    table_path_latex = os.path.join(table_path_latex, f"context_features_{assignment_identifier}.tex")
    
    # Create DataFrame
    df = pd.DataFrame(cluster_data, columns=["Cluster", "Number of Agents"] + feature_names)

    # Save to CSV
    df.to_csv(table_path_csv, index=False)

    # Save to LaTeX
    with open(table_path_latex, "w") as f:
        f.write(df.to_latex(index=False, escape=False))

    print(f"Saved context features table to {table_path_csv} and {table_path_latex}")


def plot_metrics_for_experiments(historic_assignments_per_lre: Dict[str, List[ClusterAssignment]], experiment_names, assignment_memories: Dict[str, ClusterAssignmentMemory]):
    """
    Plots conciseness, representativity, and combined score vs for each assignment in historic assignments
    of each experiment. Each experiment has a different color, and each metric is a different line shape.
    """
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
    line_styles = {
    "Conciseness": "--",
    "Representativity": "-.",
    "Combined Score": "-x"
    }
    opacity = {
    "Conciseness": 0.5,
    "Representativity": 0.5,
    "Combined Score": 1.0
    }

    plt.figure(figsize=(10, 6))

    for idx, ename in enumerate(experiment_names):
        assignment_memory = assignment_memories[ename]
        historic_assignments = historic_assignments_per_lre[ename]
        conciseness = [min(assignment.conciseness_vs(), assignment_memory.maximum_conciseness_vs) for assignment in historic_assignments]
        representativity = [assignment.representativity_vs(aggr=np.min) for assignment in historic_assignments]
        combined_score = [assignment.combined_cluster_score_vs(aggr_repr=np.min) for assignment in historic_assignments]
        grounding_scores = [np.mean(assignment.gr_score) for assignment in historic_assignments]
        isl1= [assignment.L == 1 for assignment in historic_assignments]

        x = list(reversed(list(range(1, len(historic_assignments) + 1))))

        plt.plot(x, conciseness, line_styles["Conciseness"], color=colors[idx % len(colors)],
            alpha=opacity["Conciseness"], label=f"{ename} - Conciseness")
        plt.plot(x, representativity, line_styles["Representativity"], color=colors[idx % len(colors)],
            alpha=opacity["Representativity"], label=f"{ename} - Representativity")
        plt.plot(x, combined_score, line_styles["Combined Score"], color=colors[idx % len(colors)],
            alpha=opacity["Combined Score"], label=f"{ename} - Combined Score")
        plt.plot(x, grounding_scores, '-o', color=colors[idx % len(colors)],
            alpha=0.5, label=f"{ename} - Grounding Score")
        plt.plot(x, isl1, '-|', color=colors[idx % len(colors)],
            alpha=0.5, label=f"{ename} - Is L1 Score")

    plt.xlabel("Assignment Index")
    plt.ylabel("Metric Value")
    plt.title("Conciseness, Representativity, and Combined Score vs for Assignments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_dir = os.path.join('test_results', experiment_name, 'learning_curves')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "metrics_plot.pdf")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    print(f"Saved metrics plot to {plot_path}")


if __name__ == "__main__":
    # This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected
    # IMPORTANT: Default Args are specified depending on the environment in config.json
    parser_args = filter_none_args(parse_args())
    
    _, experiment_name = find_parse_ename(parser_args.experiment_name)
    target_agent_and_vs_to_learned_ones, reward_net_pair_agent_and_vs, metrics, exp_parser_args_base, historic_assignments, env_state = load_training_results(
        experiment_name)
    assignment_memory: ClusterAssignmentMemory = metrics['assignment_memory']
    target_agent_and_vs_to_learned_ones_per_lre, reward_net_pair_agent_and_vs_per_lre, metrics_per_lre, exp_parser_args_base_per_lre, historic_assignments_per_lre= {}, {}, {}, {}, {}
    if hasattr(parser_args, 'learning_curve_from') and parser_args.learning_curve_from is not None:
        # If learning curve from is specified, load the results for each experiment
        # This will be used to generate the learning curve

        # Process experiments and plot metrics
        enames_for_lr_curve = []
        for ename in parser_args.learning_curve_from:
            ename_clean = find_parse_ename(ename)[1]
            enames_for_lr_curve.append(ename_clean)
            target_agent_and_vs_to_learned_ones_per_lre[ename_clean], reward_net_pair_agent_and_vs_per_lre[ename_clean], metrics_per_lre[ename_clean], exp_parser_args_base_per_lre[ename_clean], historic_assignments_per_lre[ename_clean], _ = load_training_results(
            enames_for_lr_curve[-1])


        plot_metrics_for_experiments(historic_assignments_per_lre, enames_for_lr_curve, assignment_memories={ename: metrics['assignment_memory'] for ename, metrics in metrics_per_lre.items()})
    config = exp_parser_args_base['config']
    society_config = exp_parser_args_base['society_config']
    exp_parser_args = exp_parser_args_base['parser_args']

    # This will look again into the config files to see if there are new fields (wont update the old ones)
    config_actual = load_json_config(exp_parser_args.config_file)
    society_config_actual = load_json_config(exp_parser_args.society_file if hasattr(exp_parser_args, 'society_file') else 'societies.json')

    
    # If a config file is specified, load it and override command line args
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
    try:
        dataset_train = VSLPreferenceDataset.load(
            os.path.join(path, "dataset_train.pkl"))
        dataset_test = VSLPreferenceDataset.load(
            os.path.join(path, "dataset_test.pkl"))
    except FileNotFoundError:
        dataset_train = VSLPreferenceDataset.load(
            os.path.join(path, "dataset.pkl"))
        dataset_test = []

    plot_test = True
    if len(dataset_test) == 0:
        plot_test = False

    example_model: LinearVSLRewardFunction = list(reward_net_pair_agent_and_vs.values())[0]
    env = example_model.remove_env()
    example_model.set_env(env)
    if exp_parser_args.algorithm == 'pc':

        # TODO: K FOLD CROSS VALIDATION. AND ALSO TEST SET EVALUATION!!!
        vsl_algo = PreferenceBasedClusteringTabularMDPVSL(
            env=env_state,
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
    vsl_algo.init_models(10, vsl_algo.vsi_optimizer_cls, vsl_algo.vsi_optimizer_kwargs)
    pprint.pprint(config)
    
    
    assignment_memory.sort_lexicographic(lexicographic_vs_first=True)
    best_vs_then_gr_assignment = assignment_memory.memory[0]

    print(assignment_memory)
    exit(0)
    assignment_memory.sort_lexicographic(lexicographic_vs_first=False)
    best_gr_then_vs_assignment = assignment_memory.memory[0]

    assignment_memory.sort_lexicographic(lexicographic_vs_first=True)
    num_digits = len(str(len(assignment_memory.memory)))
    assignments_identifier_to_assignment = [
        (f"assign_p{str(i+1).zfill(num_digits)}_vs_first_in_train", assignment_memory.memory[i]) for i in range(0, len(assignment_memory.memory))
    ]
    if plot_test:
        test_assignment_memory = ClusterAssignmentMemory(assignment_memory.max_size, n_values=assignment_memory.memory[0].n_values)
        test_assignment_memory.maximum_conciseness_vs = assignment_memory.maximum_conciseness_vs
        test_assignment_memory.maximum_conciseness_gr = assignment_memory.maximum_conciseness_gr # these two are train estimations.
        for a in assignment_memory.memory:
            test_assignment = vsl_algo.evaluate_assignment(a, dataset_test)
            test_assignment_memory.memory.append(test_assignment) # just insert it.
        test_assignments_identifier_to_assignment = [
            (f"assign_p{str(i+1).zfill(num_digits)}_vs_first_in_train", test_assignment_memory.memory[i]) for i in range(0, len(test_assignment_memory.memory))
        ]
        best_vs_then_gr_assignment_test = vsl_algo.evaluate_assignment(best_vs_then_gr_assignment, dataset_test)
        best_gr_then_vs_assignment_test = vsl_algo.evaluate_assignment(best_gr_then_vs_assignment, dataset_test)

    # 1: Context feature analysis
    if isinstance(env, RouteChoiceEnvironmentApolloComfort):
        env: RouteChoiceEnvironmentApolloComfort
        import matplotlib.pyplot as plt

        values_names = environment_data['values_names']
        contextual_feature_analysis(experiment_name, values_names, dataset_train, best_vs_then_gr_assignment, label='train_set', assignment_identifier='best_vs_then_gr')
        contextual_feature_analysis(experiment_name, values_names, dataset_train, best_gr_then_vs_assignment, label='train_set', assignment_identifier='best_gr_then_vs')
        for aid, assignment in assignments_identifier_to_assignment:
            contextual_feature_analysis(experiment_name, values_names, dataset_train, assignment, label='train_set', assignment_identifier=aid)
        if plot_test:
            contextual_feature_analysis(experiment_name, values_names, dataset_test, best_vs_then_gr_assignment_test, label='test_set', assignment_identifier='best_vs_then_gr')
            contextual_feature_analysis(experiment_name, values_names, dataset_test, best_gr_then_vs_assignment_test, label='test_set', assignment_identifier='best_gr_then_vs')
            for aid, assignment in test_assignments_identifier_to_assignment:
                contextual_feature_analysis(experiment_name, values_names, dataset_train, assignment, label='test_set', assignment_identifier=aid)
        
    # 2: tables.
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

    generate_assignment_tables(assignments_identifier_to_assignment, experiment_name, output_columns, output_dir='test_results', label='train_set')
    if plot_test:
        generate_assignment_tables(test_assignments_identifier_to_assignment, experiment_name, output_columns, output_dir='test_results', label='test_set')


    
    # 3: Explainability TODO
    if isinstance(env, RouteChoiceEnvironmentApolloComfort):
        if plot_test:
            data_test = np.array([[*t1.obs[0], *t2.obs[0]] for (t1, t2) in zip(dataset_test.fragments1, dataset_test.fragments2)])
        data_train = np.array([[*t1.obs[0], *t2.obs[0]] for (t1, t2) in zip(dataset_train.fragments1, dataset_train.fragments2)])
    else:
        env: TabularVAMDP
        data_train =  np.concatenate([env.observation_matrix, env.observation_matrix], axis=1)
        print(data_train.shape)
        # TODO, explainability here is not straightforward...
        data_test = None

    for i, (data_train_or_test, dataset_train_or_test) in enumerate(zip([data_train, data_test], [dataset_train, dataset_test])):
        if not plot_test and i == 1:
            continue
        values_names = environment_data['values_names']
        values_short_names = environment_data['values_short_names']

        agent = dataset_train_or_test.agent_ids[0]

        feature_names = ['Cost', 'Time', 'Headway', 'Interchanges', 'Cost2', 'Time2', 'Headway2', 'Interchanges2']
        reward_model_per_value = []
        obs_acts_next_obs_idxs = {'obs': list(range(4)), 'acts': [0]}
        for value in range(dataset_train.n_values): 
            def obs_shap_to_fragment_pairs(obs):
                return (obs[..., :obs.shape[-1] // 2], obs[..., obs.shape[-1] // 2:])
            reward_model_per_value.append( lambda obs: vsl_algo.preference_model.predict_proba(fragment_pairs=obs_shap_to_fragment_pairs(obs), 
                                                                                            
                                        obs_acts_next_obs_idxs=obs_acts_next_obs_idxs, alignment=vsl_algo.env.basic_profiles[value],
                                        model=best_vs_then_gr_assignment.reward_model_per_agent_id[agent]))
            
            
            s = b.MorrisSensitivity(data=data_train_or_test, model=reward_model_per_value[value], feature_names=feature_names)
            
            #explanation = s.explain_local(data[0:5], y=dataset_test.preferences_with_grounding[0:5,value])
            explanation = s.explain_global( name="test_explanation")
            fig = explanation.visualize()

            explanations_dir = os.path.join('test_results', experiment_name, 'explanations')
            morris_dir = os.path.join(explanations_dir, "morris")
            os.makedirs(morris_dir, exist_ok=True)
            path = os.path.join(morris_dir, f"morris_{values_names[value]}.pdf")
            print("Saving morris explanation to", path)
            #os.makedirs(f"demo_images", exist_ok=True)
            fig.write_image(path)


    print(best_gr_then_vs_assignment)
    print(best_vs_then_gr_assignment)
    print("TEST ASIGNMENT")
    
    print(len(assignment_memory))
    print(assignment_memory)
    if plot_test:
        print(test_assignment_memory)
        print(best_vs_then_gr_assignment_test)
        print(best_gr_then_vs_assignment_test)

    # 4: Plots. (PIE + HISTOGRAM + VISUALIZATION)
    best_vs_then_gr_assignment.plot_vs_assignments(f"test_results/{experiment_name}/train_set/plots/figure_clusters_vs_gr.pdf", 
                                                   subfig_multiplier=parser_args.subfig_multiplier,
                                                   values_color_map=environment_data['profiles_colors'], 
                                                   values_names=environment_data['values_names'], 
                                                   values_short_names=environment_data['values_short_names'],
                                                   fontsize=parser_args.plot_fontsize,)
    
    best_gr_then_vs_assignment.plot_vs_assignments(f"test_results/{experiment_name}/train_set/plots/figure_clusters_gr_vs.pdf", 
                                                   subfig_multiplier=parser_args.subfig_multiplier,
                                                   values_color_map=environment_data['profiles_colors'], 
                                                   values_names=environment_data['values_names'], 
                                                   values_short_names=environment_data['values_short_names'],
                                                   fontsize=parser_args.plot_fontsize,)
    
    if plot_test:
        best_vs_then_gr_assignment_test.plot_vs_assignments(f"test_results/{experiment_name}/test_set/plots/figure_clusters_vs_gr.pdf", 
                                                   subfig_multiplier=parser_args.subfig_multiplier,
                                                   values_color_map=environment_data['profiles_colors'], 
                                                   values_names=environment_data['values_names'], 
                                                   values_short_names=environment_data['values_short_names'],
                                                   fontsize=parser_args.plot_fontsize,)
    
        best_gr_then_vs_assignment_test.plot_vs_assignments(f"test_results/{experiment_name}/test_set/plots/figure_clusters_gr_vs.pdf", 
                                                   subfig_multiplier=parser_args.subfig_multiplier,
                                                   values_color_map=environment_data['profiles_colors'], 
                                                   values_names=environment_data['values_names'], 
                                                   values_short_names=environment_data['values_short_names'],
                                                   fontsize=parser_args.plot_fontsize,)
    # 5 learning curves
    # 
        
    
