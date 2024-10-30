import argparse
import ast
from copy import deepcopy
import json
import pprint
import numpy as np
import torch
from env_data import FIRE_FIGHTERS_ENV_NAME, ROAD_WORLD_ENV_NAME, EnvDataForIRL, EnvDataForIRLFireFighters, EnvDataForRoadWorld, PrefLossClasses

from src.envs.firefighters_env import FeatureSelectionFFEnv

from src.vsl_algorithms.base_tabular_vsl_algorithm import PolicyApproximators
from src.vsl_algorithms.me_irl_for_vsl import MaxEntropyIRLForVSL, check_coherent_rewards
from src.vsl_algorithms.preference_model_vs import PreferenceBasedTabularMDPVSL, SupportedFragmenters
from src.vsl_algorithms.vsl_plot_utils import plot_learned_and_expert_occupancy_measures, plot_learned_and_expert_reward_pairs, plot_learned_and_expert_rewards, plot_learned_to_expert_policies, plot_learned_to_expert_policies, plot_learning_curves, plot_f1_and_jsd
from src.vsl_reward_functions import TrainingModes
from utils import filter_none_args, load_json_config

POLICY_APPROXIMATION_METHOD = PolicyApproximators.MCE_ORIGINAL
# Approximate policies using causal entropy (original MCE_IRL algorithm, up to the stablished HORIZON),
# or use normal value iteration ('value_iteration')
# or another method... (NOT IMPLEMENTED)


# IMPORTANT: Default Args are specified depending on the environment in env_data.py Using the JSON file
# you can override some of that configuration settings


def parse_args():
    # IMPORTANT: Default Args are specified depending on the environment in env_data.py

    parser = argparse.ArgumentParser(
        description="Value System Learning Script configuration. Arguments which default is None have specific values depending on the algorithm and the domain. See the env_data.py file")

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument(
        '-ename', '--experiment_name', type=str, default='', help='Experiment name')
    general_group.add_argument('-cf', '--config_file', type=str, default='cmd',
                               help='Path to JSON configuration file (overrides command line arguments)')
    general_group.add_argument('-sh', '--show', action='store_true', default=False,
                               help='Show plots calculated before saving')

    general_group.add_argument('-e', '--environment', type=str, default='roadworld', choices=[
                               'roadworld', 'firefighters'], help='environment (roadworld or firefighters)')
    general_group.add_argument('-a', '--algorithm', type=str, choices=[
                               'me', 'pc'], default='me', help='Algorithm to use (max entropy or preference comparison)')
    general_group.add_argument('-df', '--discount_factor', type=float, default=0.7,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')

    general_group.add_argument('-sexp', '--stochastic_expert', action='store_true',
                               default=None, help='Original Agents have stochastic behavior')
    general_group.add_argument('-slearn', '--learn_stochastic_policy', action='store_true',
                               default=None, help='The learned policies will be stochastic')
    general_group.add_argument('-senv', '--environment_is_stochastic', action='store_true',
                               default=None, help='Whether it is known the environment is stochastic')

    general_group.add_argument('-n', '--n_experiments', type=int,
                               default=4, help='Number of experiment repetitions')
    general_group.add_argument(
        '-s', '--seed', type=int, default=26, help='Random seed')
    general_group.add_argument('-hz', '--horizon', type=int, required=False,
                               default=None, help='Maximum environment horizon')

    # Adding arguments for the constants with both long and short options
    general_group.add_argument(
        '-li', '--log_interval', type=int, default=1, help='Log evaluation interval')

    task_group = parser.add_argument_group('Task Selection')
    task_group.add_argument('-t', '--task', type=str, required=True, default='vgl',
                            choices=['vgl', 'vsi', 'all'], help='Select task to perform')
    task_group.add_argument('-pv', '--profile_variety', type=int, required=False, default=None,
                            help="Specify profile variety to generate a number of profiles to test")

    alg_group = parser.add_argument_group('Algorithm-specific Parameters')
    pc_group = alg_group.add_argument_group(
        'Preference Comparisons Parameters')
    pc_group.add_argument('-dfp', '--discount_factor_preferences', type=float,
                          default=None, help='Discount factor for preference comparisons')
    pc_group.add_argument('-qp', '--use_quantified_preference', action='store_true',
                          default=False, help='Use quantified preference flag')
    pc_group.add_argument('-temp', '--preference_sampling_temperature',
                          type=float, default=0, help='Preference sampling temperature')
    pc_group.add_argument('-qs', '--query_schedule', type=str, default="hyperbolic", choices=[
                          'hyperbolic', 'constant'], help='Query schedule for Preference Comparisons')
    pc_group.add_argument('-fl', '--fragment_length', type=int,
                          default=None, help='Fragment length. Default is Horizon')
    pc_group.add_argument('-loss', '--loss_class', type=str,
                          default=PrefLossClasses.CROSS_ENTROPY, choices=[e.value for e in PrefLossClasses], help='Loss Class')
    pc_group.add_argument('-losskw', '--loss_kwargs', type=json.loads,
                          default={}, help='Loss Kwargs as a Python dictionary')
    pc_group.add_argument('-acfrag', '--active_fragmenter_on', type=str,
                          default=SupportedFragmenters.RANDOM_FRAGMENTER, choices=[e.value for e in SupportedFragmenters], help='Active fragmenter criterion')

    debug_params = parser.add_argument_group('Debug Parameters')
    debug_params.add_argument('-db', '--check_rewards', action='store_true',
                              default=False, help='Check rewards before learning for debugging')

    exp_group = parser.add_argument_group('Experimental Parameters')
    exp_group.add_argument('-soc', '--is_society', action='store_true', default=False,
                           help='Expert agents perform as a profile society (NOT TESTED IN PC)')
    exp_group.add_argument('-probr', '--use_probabilistic_reward',
                           action='store_true', default=False, help='Lear a probabilistic reward')

    env_group = parser.add_argument_group('environment-specific Parameters')
    env_group.add_argument('-rwdt', '--dest', type=int,
                           default=413, help='Destination for roadworld')
    env_group.add_argument('-ffpm', '--use_pmovi_expert', action='store_true',
                           default=False, help='Use PMOVI expert for firefighters')
    testing_args = parser.add_argument_group('Testing options')
    testing_args.add_argument('-tn', '--n_trajs_testing', default=100,
                              type=int, help='Number of trajectories to sample for testing')
    testing_args.add_argument('-tr', '--expert_to_random_ratios', default=[1, 0.8, 0.6, 0.4, 0.2, 0.0], type=lambda x: list(
        ast.literal_eval(x)), help='Percentages of routes that are from expert instead of from random policy for testing purposes.')

    return parser.parse_args()

import random
if __name__ == "__main__":
    # IMPORTANT: Default Args are specified depending on the environment in env_data.py
    parser_args = filter_none_args(parse_args())
    if parser_args.use_quantified_preference:
        parser_args.preference_sampling_temperature = 1
    # If a config file is specified, load it and override command line args
    if parser_args.config_file != 'cmd':
        config = load_json_config(parser_args.config_file)
        for key, value in config.items():
            setattr(parser_args, key, value)

    np.random.seed(parser_args.seed)
    torch.manual_seed(parser_args.seed)
    random.seed(parser_args.seed)

    training_data: EnvDataForIRL

    task = parser_args.task
    environment = parser_args.environment

    if task == 'vsi' and environment == 'firefighters':
        parser_args.feature_selection = FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS
        parser_args.use_one_hot_state_action = True
    if parser_args.environment == 'firefighters':
        value_names = EnvDataForIRLFireFighters.VALUES_NAMES

        training_data = EnvDataForIRLFireFighters(
            env_name=FIRE_FIGHTERS_ENV_NAME,
            **dict(parser_args._get_kwargs()))
    elif parser_args.environment == 'roadworld':
        value_names = EnvDataForRoadWorld.VALUES_NAMES

        training_data = EnvDataForRoadWorld(
            env_name=ROAD_WORLD_ENV_NAME,
            **dict(parser_args._get_kwargs()))

    if task == 'vgl':
        vgl_or_vsi = 'vgl'
        task = 'vgl'
    elif task == 'vsi':
        vgl_or_vsi = 'vsi'
        task = 'vsi'
        target_align_funcs_to_learned_align_funcs_per_round = []
    else:
        assert task == 'all'
        vgl_or_vsi = 'vsi'
        task = 'all'
        target_align_funcs_to_learned_align_funcs_per_round = []

    algorithm = parser_args.algorithm
    environment = parser_args.environment

    if algorithm == 'me':
        vsl_algo = MaxEntropyIRLForVSL(
            env=training_data.env,
            reward_net=training_data.get_reward_net(),
            log_interval=parser_args.log_interval,
            vsi_optimizer_cls=training_data.vsi_optimizer_cls,
            vgl_optimizer_cls=training_data.vgl_optimizer_cls,
            vsi_optimizer_kwargs=training_data.vsi_optimizer_kwargs,
            vgl_optimizer_kwargs=training_data.vgl_optimizer_kwargs,
            vgl_expert_policy=training_data.vgl_expert_policy,
            vsi_expert_policy=training_data.vsi_expert_policy,
            vgl_expert_sampler=training_data.vgl_expert_train_sampler,
            vsi_expert_sampler=training_data.vsi_expert_train_sampler,
            target_align_func_sampler=training_data.target_align_func_sampler,
            vgl_target_align_funcs=training_data.vgl_targets,
            vsi_target_align_funcs=training_data.vsi_targets,
            approximator_kwargs=training_data.approximator_kwargs,
            
            training_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
            policy_approximator=training_data.policy_approximation_method,
            learn_stochastic_policy=training_data.learn_stochastic_policy,
            expert_is_stochastic=training_data.stochastic_expert,
            discount=training_data.discount_factor,
            environment_is_stochastic=training_data.environment_is_stochastic,
            **training_data.me_config[vgl_or_vsi]
        )
    if algorithm == 'pc':
        vsl_algo = PreferenceBasedTabularMDPVSL(env=training_data.env,
                                                reward_net=training_data.get_reward_net(),
                                                discount=training_data.discount_factor,
                                                discount_factor_preferences=training_data.discount_factor_preferences,
                                                log_interval=parser_args.log_interval,
                                                vgl_reference_policy=training_data.vgl_reference_policy,
                                                vsi_reference_policy=training_data.vsi_reference_policy,
                                                vgl_expert_policy=training_data.vgl_expert_policy,
                                                vsi_expert_policy=training_data.vsi_expert_policy,
                                                vgl_target_align_funcs=training_data.vgl_targets,
                                                vsi_target_align_funcs=training_data.vsi_targets,
                                                rng=training_data.rng,
                                                approximator_kwargs=training_data.approximator_kwargs,
                                                policy_approximator=training_data.policy_approximation_method,
                                                expert_is_stochastic=training_data.stochastic_expert,
                                                learn_stochastic_policy=training_data.learn_stochastic_policy,
                                                use_quantified_preference=parser_args.use_quantified_preference,
                                                preference_sampling_temperature=parser_args.preference_sampling_temperature,
                                                reward_trainer_kwargs=training_data.reward_trainer_kwargs,
                                                loss_class=training_data.loss_class,
                                                loss_kwargs=training_data.loss_kwargs,
                                                active_fragmenter_on=training_data.active_fragmenter_on,
                                                **training_data.pc_config[vgl_or_vsi])
    if task == 'all':
        # TODO: for now the only option to learn grounding is Preference comparison quantitative
        vgl_before_vsi_vsl_algo = PreferenceBasedTabularMDPVSL(env=training_data.env,
                                                               reward_net=vsl_algo.reward_net,
                                                               discount=training_data.discount_factor,
                                                               discount_factor_preferences=training_data.discount_factor_preferences,
                                                               log_interval=parser_args.log_interval,
                                                               vgl_reference_policy=training_data.vgl_reference_policy,
                                                               vsi_reference_policy=training_data.vsi_reference_policy,
                                                               vgl_expert_policy=training_data.vgl_expert_policy,
                                                               vsi_expert_policy=training_data.vsi_expert_policy,
                                                               vgl_target_align_funcs=training_data.vgl_targets,
                                                               vsi_target_align_funcs=training_data.vsi_targets,
                                                               rng=training_data.rng,
                                                               policy_approximator=training_data.policy_approximation_method,
                                                               approximator_kwargs=training_data.approximator_kwargs,
                                                               learn_stochastic_policy=training_data.learn_stochastic_policy,
                                                               use_quantified_preference=True,
                                                               expert_is_stochastic=training_data.stochastic_expert,
                                                               preference_sampling_temperature=1,
                                                               reward_trainer_kwargs=training_data.reward_trainer_kwargs,
                                                               loss_class=training_data.loss_class,
                                                               loss_kwargs=training_data.loss_kwargs,
                                                               active_fragmenter_on=training_data.active_fragmenter_on,
                                                               **training_data.pc_config['vgl'])

    if parser_args.check_rewards:
        assumed_grounding = training_data.get_assumed_grounding()
        check_coherent_rewards(vsl_algo, align_funcs_to_test=training_data.vsi_targets, real_grounding=assumed_grounding,
                               policy_approx_method=training_data.policy_approximation_method,
                               stochastic_expert=training_data.stochastic_expert, stochastic_learner=training_data.learn_stochastic_policy)

    
    learned_rewards_per_round = []
    policies_per_round = []
    plot_metric_per_round = []
    reward_nets_per_round = []

    target_align_funcs_to_learned_align_funcs_per_round = (
        None if vgl_or_vsi == 'vgl' else [])

    
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(vsl_algo))
    pp.pprint(vars(vsl_algo.reward_net))

    n_experiment_reps = parser_args.n_experiments
    for rep in range(n_experiment_reps):

        if task == 'all':
            assumed_grounding, reward_net_learned_per_al_func, metrics = vgl_before_vsi_vsl_algo.train(mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                                                                                                       use_probabilistic_reward=False,
                                                                                                       n_reward_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                                                                                       **training_data.pc_train_config['vgl'])
            vsl_algo.reward_net = vgl_before_vsi_vsl_algo.reward_net
            
        elif task == 'vsi':
            assumed_grounding = training_data.get_assumed_grounding()

        pp.pprint((training_data.me_train_config[vgl_or_vsi] if algorithm ==
                  'me' else training_data.pc_train_config[vgl_or_vsi]))
        # assert parser_args.use_quantified_preference
        alg_ret = vsl_algo.train(mode=TrainingModes.VALUE_GROUNDING_LEARNING if vgl_or_vsi == 'vgl' else TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                 assumed_grounding=assumed_grounding if vgl_or_vsi == 'vsi' else None,
                                 use_probabilistic_reward=parser_args.use_probabilistic_reward if vgl_or_vsi != 'vgl' else False,
                                 n_reward_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                 **(training_data.me_train_config[vgl_or_vsi] if algorithm == 'me' else training_data.pc_train_config[vgl_or_vsi]))

        if vgl_or_vsi == 'vsi':
            target_align_funcs_to_learned_align_funcs, reward_net_learned_per_al_func, metrics = alg_ret
        else:
            learned_grounding, reward_net_learned_per_al_func, metrics = alg_ret
        if algorithm == 'me':
            name_metric = 'TVC'
            metric_per_align_func, _ = metrics['tvc'], metrics['grad']
        if algorithm == 'pc':
            name_metric = 'Accuracy'
            metric_per_align_func = metrics['accuracy']

        plot_metric_per_round.append(metric_per_align_func)
        policies_per_round.append(deepcopy(vsl_algo.learned_policy_per_va))
        reward_nets_per_round.append(reward_net_learned_per_al_func)

        learned_rewards_matrix = metrics['learned_rewards']
        learned_rewards_per_round.append(learned_rewards_matrix)
        if vgl_or_vsi == 'vsi':
            target_align_funcs_to_learned_align_funcs_per_round.append(
                target_align_funcs_to_learned_align_funcs)

    extras = ''
    if parser_args.use_probabilistic_reward:
        extras += 'prob_reward_'
    if parser_args.is_society:
        extras += 'prob_profiles_'
    if algorithm == 'pc':
        extras += "tpref"+str(parser_args.preference_sampling_temperature)+'_'
        if parser_args.use_quantified_preference:
            extras += "with_qpref_"

    plot_learning_curves(algo=vsl_algo, historic_metric=plot_metric_per_round,
                         
                         ylim=None if name_metric != 'Accuracy' else (0.0, 1.1),
                         name_metric=name_metric if algorithm == 'me' else 'Accuracy',
                         name_method=f'{parser_args.experiment_name}{algorithm}_{extras}expected_{name_metric}_over_{n_experiment_reps}_{environment}_{task}',
                         align_func_colors=training_data.align_colors)

    testing_profiles_grounding = None
    if task == 'vgl':
        testing_profiles = vsl_algo.vsi_target_align_funcs
        testing_profiles_grounding = training_data.vgl_targets
    else:
        testing_profiles = vsl_algo.vsi_target_align_funcs
    # testing_profiles = [(1.0,3.0), (0.0,1.0)]
    testing_policy_per_round = []
    random_policy_tests = vsl_algo.get_policy_from_reward_per_align_func(testing_profiles,
                                                                         reward_net_per_al=None,
                                                                         expert=False, random=True,
                                                                         n_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                                                         use_probabilistic_reward=parser_args.use_probabilistic_reward,
                                                                         use_custom_grounding=training_data.get_assumed_grounding() if task == 'vsi' else None
                                                                         )[0]
    expert_policy_tests = vsl_algo.get_policy_from_reward_per_align_func(testing_profiles,
                                                                         reward_net_per_al=None,
                                                                         expert=True, random=False,
                                                                         n_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                                                         use_probabilistic_reward=parser_args.use_probabilistic_reward,
                                                                         use_custom_grounding=training_data.get_assumed_grounding() if task == 'vsi' else None
                                                                         )[0]
    learned_reward_per_test_al_round = []

    for r in range(n_experiment_reps):

        # print(testing_profiles)
        policy_r, learned_reward_per_test_al_r = vsl_algo.get_policy_from_reward_per_align_func(align_funcs=testing_profiles,
                                                                                                use_custom_grounding=training_data.get_assumed_grounding() if task == 'vsi' else None,
                                                                                                target_to_learned=None if vgl_or_vsi == 'vgl' else target_align_funcs_to_learned_align_funcs_per_round[
                                                                                                    r],
                                                                                                reward_net_per_al={al:
                                                                                                                   (reward_nets_per_round[r][al] if al in
                                                                                                                    reward_nets_per_round[r].keys() else
                                                                                                                    reward_nets_per_round[r][list(reward_nets_per_round[r].keys())[0]
                                                                                                                                             ]) for al in testing_profiles},
                                                                                                expert=False, random=False, n_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                                                                                use_probabilistic_reward=parser_args.use_probabilistic_reward
                                                                                                )
        
        learned_reward_per_test_al_round.append(learned_reward_per_test_al_r)
        # print(learned_reward_per_test_al_r)
        testing_policy_per_round.append(policy_r)

    f1_and_jsd_expert_random, value_expectations_per_ratio, value_expectations_per_ratio_expert = vsl_algo.test_accuracy_for_align_funcs(
        learned_rewards_per_round=learned_reward_per_test_al_round,
                                                                                                    testing_policy_per_round=testing_policy_per_round,
                                                                                                    expert_policy=expert_policy_tests,
                                                                                                    random_policy=random_policy_tests,
                                                                                                    target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                                                                                    n_seeds=parser_args.n_trajs_testing,  # parser_args.n_trajs_for_testing,
                                                                                                    seed=training_data.seed+2321489,#not to have the same trajectories as in training
                                                                                                    ratios_expert_random=parser_args.expert_to_random_ratios,
                                                                                                    n_samples_per_seed=1,
                                                                                                    initial_state_distribution_for_expected_alignment_estimation=training_data.initial_state_distribution_for_expected_alignment_eval,
                                                                                                    testing_align_funcs=testing_profiles)
    plot_f1_and_jsd(f1_and_jsd_expert_random, namefig=f'{parser_args.experiment_name}{algorithm}_{extras}expected_over_{n_experiment_reps}_{environment}_{task}',
                    align_func_colors=training_data.align_colors,
                    values_names=value_names,usecmap = 'viridis',
                    target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                    show=parser_args.show, value_expectations_per_ratio=value_expectations_per_ratio,
                    value_expectations_per_ratio_expert=value_expectations_per_ratio_expert)
    if testing_profiles_grounding is not None:
        f1_and_jsd_expert_random_vgl, value_expectations_per_ratio, value_expectations_per_ratio_expert = vsl_algo.test_accuracy_for_align_funcs(learned_rewards_per_round=learned_reward_per_test_al_round,
                                                                                                            testing_policy_per_round=testing_policy_per_round,
                                                                                                            expert_policy=expert_policy_tests,
                                                                                                            random_policy=random_policy_tests,
                                                                                                            target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                                                                                            n_seeds=parser_args.n_trajs_testing,
                                                                                                            seed=training_data.seed+2321489,#not to have the same trajectories as in training
                                                                                                            ratios_expert_random=parser_args.expert_to_random_ratios,
                                                                                                            n_samples_per_seed=1,
                                                                                                            initial_state_distribution_for_expected_alignment_estimation=training_data.initial_state_distribution_for_expected_alignment_eval,
                                                                                                            testing_align_funcs=testing_profiles_grounding)
        plot_f1_and_jsd(f1_and_jsd_expert_random_vgl, namefig=f'{parser_args.experiment_name}{algorithm}_GROUNDING_ERROR_{extras}expected_over_{n_experiment_reps}_{environment}_{task}', 
                        show=parser_args.show,
                        target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                        align_func_colors=training_data.align_colors,
                        values_names=value_names,
                        value_expectations_per_ratio=value_expectations_per_ratio,
                        value_expectations_per_ratio_expert=value_expectations_per_ratio_expert
                        )
    print("Plotting learned and expert reward pairs")
    plot_learned_and_expert_reward_pairs(vsl_algo=vsl_algo, learned_rewards_per_al_func=learned_rewards_per_round, vsi_or_vgl=vgl_or_vsi,
                                         target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                         namefig=f'{parser_args.experiment_name}{algorithm}_{extras}expected_over_{n_experiment_reps}_{environment}_{task}', show=parser_args.show)
    
    print("Plotting learned and expert policies")
    plot_learned_to_expert_policies(vsl_algo=vsl_algo, expert_policy=training_data.vgl_expert_policy,
                                    target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                    vsi_or_vgl=vgl_or_vsi, namefig=f'{parser_args.experiment_name}{algorithm}_{extras}expected_over_{n_experiment_reps}_{environment}_{task}',
                                    learnt_policy=policies_per_round, show=parser_args.show)
    print("Plotting learned and expert rewards")
    plot_learned_and_expert_rewards(vsl_algo=vsl_algo,
                                    learned_rewards_per_al_func=learned_rewards_per_round,
                                    target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                    vsi_or_vgl=vgl_or_vsi,
                                    namefig=f'{parser_args.experiment_name}{algorithm}_{extras}expected_over_{n_experiment_reps}_{environment}_{task}', show=parser_args.show)
    print("Plotting learned and expert occupancy measures")
    if algorithm == 'me':
        plot_learned_and_expert_occupancy_measures(vsl_algo=vsl_algo, expert_policy=training_data.vgl_expert_policy,
                                                   target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                                   learned_rewards_per_al_func=learned_rewards_per_round, vsi_or_vgl=vgl_or_vsi,
                                                   namefig=f'{parser_args.experiment_name}{algorithm}_{extras}expected_over_{n_experiment_reps}_{environment}_{task}', show=parser_args.show)
