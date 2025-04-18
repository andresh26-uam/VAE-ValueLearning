import random
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
from src.vsl_algorithms.adversarial_vsl import AdversarialVSL
from src.vsl_algorithms._inverseQlearning import DeepInverseQLearning, TabularInverseQLearning
from src.vsl_algorithms.me_irl_for_vsl import MaxEntropyIRLForVSL, check_coherent_rewards
from src.vsl_algorithms.preference_model_vs import PreferenceBasedTabularMDPVSL, SupportedFragmenters
from src.vsl_algorithms.vsl_plot_utils import plot_learned_and_expert_occupancy_measures, plot_learned_and_expert_reward_pairs, plot_learned_and_expert_rewards, plot_learned_to_expert_policies, plot_learned_to_expert_policies, plot_learning_curves, plot_vs_preference_metrics
from src.vsl_policies import LearnerValueSystemLearningPolicy, VAlignedDictDiscreteStateActionPolicyTabularMDP
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
                               'me', 'pc', 'pc-me', 'me-pc', 'pc-pc', 'me-me', 'ad', 'pc-ad', 'ad-ad', 'pc-iq', 'iq-iq', 'iq', 'ti', 'ti-ti'], default='me', help='Algorithm to use (max entropy or preference comparison)')
    general_group.add_argument('-df', '--discount_factor', type=float, default=0.7,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')

    general_group.add_argument('-stexp', '--stochastic_expert', action='store_true',
                               help='The expert/original/reference policies are stochastic. This is set by default in env_data.py, but can override here setting this flag or the opposite')
    general_group.add_argument('-detexp', '--deterministic_expert', dest='stochastic_expert', action='store_false',
                               help='The expert/original/reference policies are taken as deterministic.')
    # Set in env_data.py depending on the environment and algorithm
    general_group.set_defaults(stochastic_expert=None)

    general_group.add_argument('-stlearn', '--learn_stochastic_policy', action='store_true',
                               help='The learned policies will be stochastic. This is set by default in env_data.py, but can override here setting this flag or the opposite')
    general_group.add_argument('-detlearn', '--learn_deterministic_policy', dest='learn_stochastic_policy', action='store_false',
                               help='The learned policies will be deterministic.')
    # Set in env_data.py depending on the environment.
    general_group.set_defaults(learn_stochastic_policy=None)

    general_group.add_argument('-stenv', '--environment_is_stochastic', action='store_true',
                               help='Use this flag to say the environment has stochastic transitions. This is set by default in env_data.py, but can override here setting this flag or the opposite')
    general_group.add_argument('-detenv', '--environment_is_deterministic', dest='environment_is_stochastic', action='store_false',
                               help='Use this flag to say the environment has deterministic transitions. This can save time in some computations of some algoritms.')
    # Set in env_data.py depending on the environment.
    general_group.set_defaults(environment_is_stochastic=None)

    general_group.add_argument('-n', '--n_experiments', type=int,
                               default=4, help='Number of experiment repetitions')
    general_group.add_argument(
        '-s', '--seed', type=int, default=26, help='Random seed')
    general_group.add_argument('-hz', '--horizon', type=int, required=False,
                               default=None, help='Maximum environment horizon')

    general_group.add_argument(
        '-li', '--log_interval', type=int, default=1, help='Log evaluation interval')
    general_group.add_argument('-fixsamp', '--sampler_over_precalculated_trajs', action='store_true',
                               help='Use trajectories sampled from a fixed batch instead of using a policy. Set to true typically.')
    general_group.add_argument('-polsamp', '--sampler_over_policy', dest='sampler_over_precalculated_trajs', action='store_false',
                               help='Use trajectories sampled from the learned/expert policy directly.')
    # Set in env_data.py depending on the environment.
    general_group.set_defaults(sampler_over_precalculated_trajs=None)

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
    pc_group.add_argument('-expobs', '--expose_observations', action='store_true',
                          default=False, help='Discount factor for preference comparisons')
    pc_group.add_argument('-qp', '--use_quantified_preference', action='store_true',
                          default=True, help='Use quantified preference flag')
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
                          default=SupportedFragmenters.CONNECTED_FRAGMENTER, choices=[e.value for e in SupportedFragmenters], help='Active fragmenter criterion')
    me_group = alg_group.add_argument_group(
        'Maximum Entropy Parameters')
    me_group.add_argument('-ompol', '--demo_om_from_policy', action='store_true',
                          default=False, help='Use the policy itself to estimate state-action visitation counts, without sampling')

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
                           default=64, help='Destination for roadworld')  # 413 has problems... (need exact approximation)
    env_group.add_argument('-rt', '--retrain', action='store_true',
                           default=False, help='Retrain experts (roadworld)')
    env_group.add_argument('-appr', '--approx_expert', action='store_true',
                           default=False, help='Approximate expert (roadworld)')
    env_group.add_argument('-ffpm', '--use_pmovi_expert', action='store_true',
                           default=False, help='Use PMOVI expert for firefighters')
    testing_args = parser.add_argument_group('Testing options')
    testing_args.add_argument('-tn', '--n_trajs_testing', default=100,
                              type=int, help='Number of trajectories to sample for testing')
    testing_args.add_argument('-tr', '--expert_to_random_ratios', default=[1, 0.8, 0.6, 0.4, 0.2, 0.0], type=lambda x: list(
        ast.literal_eval(x)), help='Percentages of routes that are from expert instead of from random policy for testing purposes.')

    return parser.parse_args()


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
    algorithm = parser_args.algorithm
    environment = parser_args.environment

    # In any case, Algorithm 1 is the evaluated algorithm for tasks 'vsi' and 'vgl')
    # Algorithm 0 is only used in the full VSL (Value System Learning) pipeline
    # (algorithm[0] is the one used for VGL in that case, which defaults to 'pc')
    if len(algorithm) == 2:
        algorithm = ['pc', algorithm]  # by default, PC method is used for VSL.
    else:
        algorithm = algorithm.split('-')

    if algorithm[1] == 'ad' or algorithm[1] == 'iq' or algorithm[1] == 'ti':
        # the observations will be used only to calculate the rewards with env.obs_from_state or policy.obtain_observation_for_reward
        parser_args.expose_observations = False

    if task == 'vsi' and environment == 'firefighters':
        parser_args.feature_selection = FeatureSelectionFFEnv.ONE_HOT_FEATURES
        # if parser_args.feature_selection == FeatureSelectionFFEnv.ENCRYPTED_OBSERVATIONS else False
        parser_args.use_one_hot_state_action = False

    if parser_args.environment == 'firefighters':
        value_names = EnvDataForIRLFireFighters.VALUES_NAMES

        training_data = EnvDataForIRLFireFighters(
            env_name=FIRE_FIGHTERS_ENV_NAME,
            **dict(parser_args._get_kwargs()))
        # assert training_data.approx_expert is True
    elif parser_args.environment == 'roadworld':
        value_names = EnvDataForRoadWorld.VALUES_NAMES

        training_data = EnvDataForRoadWorld(
            env_name=ROAD_WORLD_ENV_NAME,
            **dict(parser_args._get_kwargs()))
        if not training_data.approx_expert:
            training_data.policy_approximation_method = lambda env, reward, discount, **kwargs: (
                None, None, training_data.compute_precise_policy(env, w=None, reward=reward))

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

    if algorithm[1] == 'ad':
        vsl_algo = AdversarialVSL(
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
            learn_stochastic_policy=training_data.learn_stochastic_policy,
            # policy_approximator=training_data.policy_approximation_method,
            # approximator_kwargs=training_data.approximator_kwargs,
            stochastic_expert=training_data.stochastic_expert,
            environment_is_stochastic=training_data.environment_is_stochastic,
            discount=training_data.discount_factor,
            **(training_data.ad_config['vsi'])
        )
    elif algorithm[1] == 'me':
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
    elif algorithm[1] == 'pc':
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
        if algorithm[0] == 'pc':
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
                                                                   use_quantified_preference=parser_args.use_quantified_preference,
                                                                   expert_is_stochastic=training_data.stochastic_expert,
                                                                   preference_sampling_temperature=parser_args.preference_sampling_temperature,
                                                                   reward_trainer_kwargs=training_data.reward_trainer_kwargs,
                                                                   loss_class=training_data.loss_class,
                                                                   loss_kwargs=training_data.loss_kwargs,
                                                                   active_fragmenter_on=training_data.active_fragmenter_on,
                                                                   **training_data.pc_config['vgl'])
        elif algorithm[0] == 'me':
            vgl_before_vsi_vsl_algo = MaxEntropyIRLForVSL(
                env=training_data.env,
                reward_net=vsl_algo.reward_net,
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
                training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                policy_approximator=training_data.policy_approximation_method,
                learn_stochastic_policy=training_data.learn_stochastic_policy,
                expert_is_stochastic=training_data.stochastic_expert,
                discount=training_data.discount_factor,
                environment_is_stochastic=training_data.environment_is_stochastic,
                **training_data.me_config['vgl']
            )
        elif algorithm[0] == 'ad':
            print("adversarial testing VGL")
            vgl_before_vsi_vsl_algo = AdversarialVSL(
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
                training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                learn_stochastic_policy=training_data.learn_stochastic_policy,
                # policy_approximator=training_data.policy_approximation_method,
                # approximator_kwargs=training_data.approximator_kwargs,
                stochastic_expert=training_data.stochastic_expert,
                environment_is_stochastic=training_data.environment_is_stochastic,
                discount=training_data.discount_factor,
                **(training_data.ad_config['vgl'])

            )
        else:
            raise ValueError("Unknown algorithm key : " + str(algorithm[0]))
    if parser_args.check_rewards:
        assumed_grounding = training_data.get_assumed_grounding()
        check_coherent_rewards(vsl_algo, align_funcs_to_test=training_data.vsi_targets, real_grounding=assumed_grounding,
                               policy_approx_method=training_data.policy_approximation_method,
                               stochastic_expert=training_data.stochastic_expert, stochastic_learner=training_data.learn_stochastic_policy)

    learned_rewards_per_round = []
    tabulated_policies_per_round = []
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
                                                                                                       assumed_grounding=None,
                                                                                                       use_probabilistic_reward=False,
                                                                                                       n_reward_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                                                                                       **(training_data.pc_train_config['vgl'] if algorithm[0] == 'pc' else
                                                                                                          training_data.me_train_config['vgl'] if algorithm[0] == 'me' else
                                                                                                          training_data.ad_train_config['vgl'] if algorithm[0] == 'ad' else {
                                                                                                       }
                                                                                                       ))
            vsl_algo.reward_net = vgl_before_vsi_vsl_algo.reward_net

        elif task == 'vsi':
            assumed_grounding = training_data.get_assumed_grounding()

        # assert parser_args.use_quantified_preference
        alg_ret = vsl_algo.train(mode=TrainingModes.VALUE_GROUNDING_LEARNING if vgl_or_vsi == 'vgl' else TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                 assumed_grounding=assumed_grounding if vgl_or_vsi == 'vsi' else None,
                                 use_probabilistic_reward=parser_args.use_probabilistic_reward if vgl_or_vsi != 'vgl' else False,
                                 n_reward_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                 **(training_data.me_train_config[vgl_or_vsi] if algorithm[1] == 'me' else
                                    training_data.ad_train_config[vgl_or_vsi] if algorithm[1] == 'ad' else
                                    training_data.iq_train_config[vgl_or_vsi] if algorithm[1] == 'iq' else
                                    training_data.tiq_train_config[vgl_or_vsi] if algorithm[1] == 'ti' else {
                                 }
                                 ))

        if vgl_or_vsi == 'vsi':
            target_align_funcs_to_learned_align_funcs, reward_net_learned_per_al_func, metrics = alg_ret
        else:
            learned_grounding, reward_net_learned_per_al_func, metrics = alg_ret
        if algorithm[1] == 'me':
            name_metric = 'TVC'
            metric_per_align_func, _ = metrics['tvc'], metrics['grad']
        if algorithm[1] == 'pc':
            name_metric = 'Accuracy'
            metric_per_align_func = metrics['accuracy']
        if algorithm[1] == 'ad' or algorithm[1] == 'iq':
            name_metric = 'Accuracy'
            metric_per_align_func = metrics['accuracy']

        plot_metric_per_round.append(metric_per_align_func)
        if isinstance(vsl_algo.learned_policy_per_va, LearnerValueSystemLearningPolicy):
            vsl_algo.learned_policy_per_va.save(
                f"{parser_args.experiment_name}_round_save_{rep}")
            round_policy = LearnerValueSystemLearningPolicy.load(
                vsl_algo.env, f"{parser_args.experiment_name}_round_save_{rep}")
            tabular_policy = {}
            for w, lw in target_align_funcs_to_learned_align_funcs.items() if task == 'all' or task == 'vsi' else zip(vsl_algo.vgl_target_align_funcs, vsl_algo.vgl_target_align_funcs):
                tabular_policy[w] = np.zeros(
                    (vsl_algo.env.state_dim, vsl_algo.env.action_dim))
                tabular_policy[lw] = np.zeros(
                    (vsl_algo.env.state_dim, vsl_algo.env.action_dim))

                for s in range(vsl_algo.env.state_dim):

                    tabular_policy[w][s] = round_policy.act_and_obtain_action_distribution(s,
                                                                                           stochastic=vsl_algo.learn_stochastic_policy, alignment_function=w)[2].detach().numpy()
                    tabular_policy[lw][s] = tabular_policy[w][s]
            tabulated_policies_per_round.append(VAlignedDictDiscreteStateActionPolicyTabularMDP(
                policy_per_va_dict=tabular_policy, env=vsl_algo.env))
        else:
            tabulated_policies_per_round.append(
                deepcopy(vsl_algo.learned_policy_per_va))
        reward_nets_per_round.append(reward_net_learned_per_al_func)

        learned_state_action_rewards_per_target = metrics['learned_rewards']
        learned_rewards_per_round.append(
            learned_state_action_rewards_per_target)
        if vgl_or_vsi == 'vsi':
            target_align_funcs_to_learned_align_funcs_per_round.append(
                target_align_funcs_to_learned_align_funcs)

    extras = ''
    if parser_args.use_probabilistic_reward:
        extras += 'prob_reward_'
    if parser_args.is_society:
        extras += 'prob_profiles_'
    if algorithm[1] == 'pc':
        extras += "tpref"+str(parser_args.preference_sampling_temperature)+'_'
        if parser_args.use_quantified_preference:
            extras += "with_qpref_"

    plot_learning_curves(algo=vsl_algo, historic_metric=plot_metric_per_round,
                         usecmap='viridis',
                         ylim=None if name_metric != 'Accuracy' else (
                             0.0, 1.1),
                         # if algorithm[1] == 'me' else 'Accuracy',
                         name_metric=name_metric,
                         name_method=f'{parser_args.experiment_name}{algorithm[1]}_{extras}expected_{name_metric}_over_{n_experiment_reps}_{environment}_{task}',
                         align_func_colors=training_data.align_colors)

    testing_profiles_grounding = None
    if task == 'vgl' or task == 'all':
        testing_profiles = vsl_algo.vsi_target_align_funcs
        testing_profiles_grounding = training_data.vgl_targets
    else:
        testing_profiles = vsl_algo.vsi_target_align_funcs
    # testing_profiles = [(1.0,3.0), (0.0,1.0)]
    testing_policy_per_round = []
    random_policy_tests = vsl_algo.get_tabular_policy_from_reward_per_align_func(testing_profiles,
                                                                                 reward_net_per_al=None,
                                                                                 expert=False, random=True,
                                                                                 n_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                                                                 use_probabilistic_reward=parser_args.use_probabilistic_reward,
                                                                                 use_custom_grounding=training_data.get_assumed_grounding() if task == 'vsi' else None,
                                                                                 precise_deterministic=not training_data.approx_expert
                                                                                 )[0]
    expert_policy_tests = vsl_algo.get_tabular_policy_from_reward_per_align_func(testing_profiles,
                                                                                 reward_net_per_al=None,
                                                                                 expert=True, random=False,
                                                                                 n_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                                                                 use_probabilistic_reward=parser_args.use_probabilistic_reward,
                                                                                 use_custom_grounding=training_data.get_assumed_grounding() if task == 'vsi' else None,
                                                                                 precise_deterministic=not training_data.approx_expert
                                                                                 )[0]
    learned_reward_per_test_al_round = []

    for r in range(n_experiment_reps):

        # print(testing_profiles)
        policy_r, learned_reward_per_test_al_r = vsl_algo.get_tabular_policy_from_reward_per_align_func(align_funcs=testing_profiles,
                                                                                                        use_custom_grounding=training_data.get_assumed_grounding() if task == 'vsi' else None,
                                                                                                        target_to_learned=None if vgl_or_vsi == 'vgl' else target_align_funcs_to_learned_align_funcs_per_round[
                                                                                                            r],
                                                                                                        reward_net_per_al={al:
                                                                                                                           (reward_nets_per_round[r][al] if al in
                                                                                                                            reward_nets_per_round[r].keys() else
                                                                                                                            reward_nets_per_round[r][list(reward_nets_per_round[r].keys())[0]
                                                                                                                                                     ]) for al in testing_profiles},
                                                                                                        expert=False, random=False, n_reps_if_probabilistic_reward=training_data.n_reward_samples_per_iteration,
                                                                                                        use_probabilistic_reward=parser_args.use_probabilistic_reward,
                                                                                                        precise_deterministic=not training_data.approx_expert
                                                                                                        )

        learned_reward_per_test_al_round.append(learned_reward_per_test_al_r)
        testing_policy_per_round.append(policy_r)

    preference_metrics_expert_random, value_expectations_per_ratio, value_expectations_per_ratio_expert = vsl_algo.test_accuracy_for_align_funcs(
        learned_rewards_per_round=learned_reward_per_test_al_round,
        testing_policy_per_round=testing_policy_per_round,
        expert_policy=expert_policy_tests,
        random_policy=random_policy_tests,
        target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
        n_seeds=parser_args.n_trajs_testing,  # parser_args.n_trajs_for_testing,
        # not to have the same trajectories as in training
        seed=training_data.seed+2321489,
        ratios_expert_random=parser_args.expert_to_random_ratios,
        n_samples_per_seed=1,
        initial_state_distribution_for_expected_alignment_estimation=training_data.initial_state_distribution_for_expected_alignment_eval,
        testing_align_funcs=testing_profiles)
    plot_vs_preference_metrics(preference_metrics_expert_random, namefig=f'{parser_args.experiment_name}{algorithm[1]}_{extras}expected_over_{n_experiment_reps}_{environment}_{task}',
                               align_func_colors=training_data.align_colors,
                               values_names=value_names, usecmap='viridis',
                               target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                               show=parser_args.show, value_expectations_per_ratio=value_expectations_per_ratio,
                               value_expectations_per_ratio_expert=value_expectations_per_ratio_expert)
    if testing_profiles_grounding is not None:
        preference_metrics_expert_random_vgl, value_expectations_per_ratio, value_expectations_per_ratio_expert = vsl_algo.test_accuracy_for_align_funcs(learned_rewards_per_round=learned_reward_per_test_al_round,
                                                                                                                                                         testing_policy_per_round=testing_policy_per_round,
                                                                                                                                                         expert_policy=expert_policy_tests,
                                                                                                                                                         random_policy=random_policy_tests,
                                                                                                                                                         target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                                                                                                                                         n_seeds=parser_args.n_trajs_testing,
                                                                                                                                                         # not to have the same trajectories as in training
                                                                                                                                                         seed=training_data.seed+2321489,
                                                                                                                                                         ratios_expert_random=parser_args.expert_to_random_ratios,
                                                                                                                                                         n_samples_per_seed=1,
                                                                                                                                                         initial_state_distribution_for_expected_alignment_estimation=training_data.initial_state_distribution_for_expected_alignment_eval,
                                                                                                                                                         testing_align_funcs=testing_profiles_grounding)
        plot_vs_preference_metrics(preference_metrics_expert_random_vgl, namefig=f'{parser_args.experiment_name}{algorithm[1]}_GROUNDING_ERROR_{extras}expected_over_{n_experiment_reps}_{environment}_{task}',
                                   show=parser_args.show,
                                   target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                   align_func_colors=training_data.align_colors, usecmap='viridis',
                                   values_names=value_names,
                                   value_expectations_per_ratio=value_expectations_per_ratio,
                                   value_expectations_per_ratio_expert=value_expectations_per_ratio_expert
                                   )
        if task == 'all':  # also plot the learned groundings
            plot_learned_and_expert_reward_pairs(vsl_algo=vsl_algo, learned_rewards_per_al_func=learned_rewards_per_round, vsi_or_vgl='vgl',
                                                 target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                                 namefig=f'{parser_args.experiment_name}{algorithm[0]}_VSL_ERROR_PURES_{extras}_expected_over_{n_experiment_reps}_{environment}_{task}', show=parser_args.show)

    print("Plotting learned and expert reward pairs")
    plot_learned_and_expert_reward_pairs(vsl_algo=vsl_algo, learned_rewards_per_al_func=learned_rewards_per_round, vsi_or_vgl=vgl_or_vsi,
                                         target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                         namefig=f'{parser_args.experiment_name}{algorithm[1]}_{extras}expected_over_{n_experiment_reps}_{environment}_{task}', show=parser_args.show)

    print("Plotting learned and expert policies")
    plot_learned_to_expert_policies(vsl_algo=vsl_algo, expert_policy=training_data.vgl_expert_policy,
                                    target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                    vsi_or_vgl=vgl_or_vsi, namefig=f'{parser_args.experiment_name}{algorithm[1]}_{extras}expected_over_{n_experiment_reps}_{environment}_{task}',
                                    learnt_policy=tabulated_policies_per_round, show=parser_args.show)
    print("Plotting learned and expert rewards")
    plot_learned_and_expert_rewards(vsl_algo=vsl_algo,
                                    learned_rewards_per_al_func=learned_rewards_per_round,
                                    target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                    vsi_or_vgl=vgl_or_vsi,
                                    namefig=f'{parser_args.experiment_name}{algorithm[1]}_{extras}expected_over_{n_experiment_reps}_{environment}_{task}', show=parser_args.show)
    print("Plotting learned and expert occupancy measures")
    if algorithm[1] == 'me' and environment == 'firefighters':
        plot_learned_and_expert_occupancy_measures(vsl_algo=vsl_algo, expert_policy=training_data.vgl_expert_policy,
                                                   target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round,
                                                   learned_rewards_per_al_func=learned_rewards_per_round, vsi_or_vgl=vgl_or_vsi,
                                                   namefig=f'{parser_args.experiment_name}{algorithm[1]}_{extras}expected_over_{n_experiment_reps}_{environment}_{task}', show=parser_args.show)
