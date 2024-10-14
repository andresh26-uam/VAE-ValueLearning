import argparse
from copy import deepcopy

import numpy as np
import torch
from env_data import FIRE_FIGHTERS_ENV_NAME, ROAD_WORLD_ENV_NAME, EnvDataForIRL, EnvDataForIRLFireFighters, EnvDataForRoadWorld

from src.values_and_costs import BASIC_PROFILES, PROFILE_COLORS_VEC
from src.vsl_algorithms.me_irl_for_vsl import MaxEntropyIRLForVSL, check_coherent_rewards, PolicyApproximators
from src.vsl_algorithms.preference_model_vs import PreferenceBasedTabularMDPVSL
from src.vsl_algorithms.vsl_plot_utils import get_color_gradient, get_linear_combination_of_colors, plot_learned_and_expert_occupancy_measures, plot_learned_and_expert_reward_pairs, plot_learned_and_expert_rewards, plot_learned_to_expert_policies, plot_learned_to_expert_policies, plot_learning_curves
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
    general_group.add_argument('-cf', '--config_file', type=str, default='cmd',
                               help='Path to JSON configuration file (overrides command line arguments)')

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
                               default=3, help='Number of experiment repetitions')
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
    pc_group.add_argument('-qpref', '--use_quantified_preference', action='store_true',
                          default=False, help='Use quantified preference flag')
    pc_group.add_argument('-tpref', '--preference_sampling_temperature',
                          type=float, default=0, help='Preference sampling temperature')
    pc_group.add_argument('-qs', '--query_schedule', type=str, default="hyperbolic", choices=[
                          'hyperbolic', 'constant'], help='Query schedule for Preference Comparisons')
    pc_group.add_argument('-fl', '--fragment_length', type=int,
                          default=None, help='Fragment length. Default is Horizon')

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

    return parser.parse_args()


if __name__ == "__main__":
    # IMPORTANT: Default Args are specified depending on the environment in env_data.py
    parser_args = filter_none_args(parse_args())

    # If a config file is specified, load it and override command line args
    if parser_args.config_file != 'cmd':
        config = load_json_config(parser_args.config_file)
        for key, value in config.items():
            setattr(parser_args, key, value)

    np.random.seed(parser_args.seed)
    torch.manual_seed(parser_args.seed)

    training_data: EnvDataForIRL

    if parser_args.environment == 'firefighters':
        def align_colors(align_func): return get_color_gradient(
            [1, 0, 0], [0, 0, 1], align_func[0])
        training_data = EnvDataForIRLFireFighters(
            env_name=FIRE_FIGHTERS_ENV_NAME,
            **dict(parser_args._get_kwargs()))
    elif parser_args.environment == 'roadworld':
        parser_args.discount_factor = 1.0  # Needed.

        def align_colors(align_func): return get_linear_combination_of_colors(
            BASIC_PROFILES, PROFILE_COLORS_VEC, align_func)
        training_data = EnvDataForRoadWorld(
            env_name=ROAD_WORLD_ENV_NAME,
            **dict(parser_args._get_kwargs()))

    experiment = parser_args.task
    if experiment == 'vgl':
        vgl_or_vsi = 'vgl'
        experiment = 'vgl'
    elif experiment == 'vsi':
        vgl_or_vsi = 'vsi'
        experiment = 'vsi'
        target_align_funcs_to_learned_align_funcs_per_round = []
    else:
        assert experiment == 'all'
        vgl_or_vsi = 'vsi'
        experiment = 'all'
        target_align_funcs_to_learned_align_funcs_per_round = []

    algorithm = parser_args.algorithm
    environment = parser_args.environment

    if parser_args.algorithm == 'me':
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
            training_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
            policy_approximator=training_data.policy_approximation_method,
            learn_stochastic_policy=training_data.learn_stochastic_policy,
            expert_is_stochastic=training_data.stochastic_expert,
            discount=training_data.discount_factor,
            environment_is_stochastic=training_data.environment_is_stochastic,

            **training_data.me_config[vgl_or_vsi]
        )
    if parser_args.algorithm == 'pc':
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
                                                learn_stochastic_policy=training_data.learn_stochastic_policy,
                                                use_quantified_preference=parser_args.use_quantified_preference,
                                                preference_sampling_temperature=parser_args.preference_sampling_temperature,
                                                reward_trainer_kwargs=training_data.reward_trainer_kwargs,
                                                **training_data.pc_config[vgl_or_vsi])

    if parser_args.check_rewards:
        assumed_grounding = training_data.get_assumed_grounding()
        check_coherent_rewards(vsl_algo, align_funcs_to_test=training_data.vsi_targets, real_grounding=assumed_grounding,
                               policy_approx_method=training_data.policy_approximation_method,
                               stochastic_expert=training_data.stochastic_expert, stochastic_learner=training_data.learn_stochastic_policy)

    # VALUE GROUNDING LEARNING:
    learned_rewards_per_round = []
    policies_per_round = []
    plot_metric_per_round = []

    target_align_funcs_to_learned_align_funcs_per_round = None

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    print(pp.pprint(vars(vsl_algo)))
    print(pp.pprint(vars(vsl_algo.reward_net)))

    n_experiment_reps = parser_args.n_experiments
    for rep in range(parser_args.n_experiments):

        train_config_vgl = (
            training_data.me_train_config['vgl'] if algorithm == 'me' else training_data.pc_train_config['vgl'])
        if experiment == 'all':
            assumed_grounding, reward_net_learned_per_al_func, metrics = vsl_algo.train(mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                                                                                        use_probabilistic_reward=False,
                                                                                        n_reward_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                                                                        **train_config_vgl)
        elif experiment == 'vsi':
            assumed_grounding = training_data.get_assumed_grounding()

        pp.pprint((training_data.me_train_config[vgl_or_vsi] if algorithm ==
                  'me' else training_data.pc_train_config[vgl_or_vsi]))
        # assert parser_args.use_quantified_preference
        alg_ret = vsl_algo.train(mode=TrainingModes.VALUE_GROUNDING_LEARNING if vgl_or_vsi == 'vgl' else TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                 assumed_grounding=assumed_grounding if vgl_or_vsi == 'vsi' else None,
                                 use_probabilistic_reward=parser_args.use_probabilistic_reward if vgl_or_vsi != 'vgl' else False,
                                 n_reward_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                 **(training_data.me_train_config[vgl_or_vsi] if algorithm == 'me' else training_data.pc_train_config[vgl_or_vsi]))

        if vgl_or_vsi == 'vgl':
            target_align_funcs_to_learned_align_funcs, reward_net_per_target_va, metrics = alg_ret
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
                         name_metric=name_metric if algorithm == 'me' else 'Accuracy',
                         name_method=f'{algorithm}_{extras}expected_{name_metric}_over_{n_experiment_reps}_{environment}_{experiment}', align_func_colors=align_colors)
    plot_learned_and_expert_reward_pairs(vsl_algo=vsl_algo, learned_rewards_per_al_func=learned_rewards_per_round, vsi_or_vgl=vgl_or_vsi,
                                         target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                         namefig=f'{algorithm}_{extras}expected_over_{n_experiment_reps}_{environment}_{experiment}', )

    if algorithm == 'me':
        plot_learned_and_expert_occupancy_measures(vsl_algo=vsl_algo, expert_policy=training_data.vgl_expert_policy,
                                                   target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                                   learned_rewards_per_al_func=learned_rewards_per_round, vsi_or_vgl=vgl_or_vsi,
                                                   namefig=f'{algorithm}_{extras}expected_over_{n_experiment_reps}_{environment}_{experiment}')

    plot_learned_to_expert_policies(vsl_algo=vsl_algo, expert_policy=training_data.vgl_expert_policy,
                                    target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                    vsi_or_vgl=vgl_or_vsi, namefig=f'{algorithm}_{extras}expected_over_{n_experiment_reps}_{environment}_{experiment}',
                                    learnt_policy=policies_per_round)

    plot_learned_and_expert_rewards(vsl_algo=vsl_algo,
                                    learned_rewards_per_al_func=learned_rewards_per_round,
                                    target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                    vsi_or_vgl=vgl_or_vsi,
                                    amefig=f'{algorithm}_{extras}expected_over_{n_experiment_reps}_{environment}_{experiment}')
