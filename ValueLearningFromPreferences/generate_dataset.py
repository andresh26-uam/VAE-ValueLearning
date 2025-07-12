from functools import partial
from sb3_contrib.ppo_mask.policies import MlpPolicy as MASKEDMlpPolicy
from stable_baselines3.ppo import MlpPolicy
import argparse
from copy import deepcopy
import os
import pprint
import random
from typing import Dict, Union
import imitation
import imitation.data
import imitation.data.rollout
import imitation.util
import numpy as np
import stable_baselines3
import stable_baselines3.common
import stable_baselines3.common.evaluation
import stable_baselines3.common.policies
import torch

from envs.multivalued_car_env import MVFS, MultiValuedCarEnv
from envs.tabularVAenv import ContextualEnv, TabularVAMDP, ValueAlignedEnvironment
from src.dataset_processing.utils import DEFAULT_SEED, GROUNDINGS_PATH, calculate_expert_policy_save_path
from src.algorithms.utils import PolicyApproximators, mce_partition_fh
from envs.firefighters_env import FeatureSelectionFFEnv
from src.dataset_processing.data import TrajectoryWithValueSystemRews
from src.dataset_processing.utils import calculate_dataset_save_path
from src.dataset_processing.datasets import create_dataset
from src.dataset_processing.preferences import save_preferences
from src.dataset_processing.preferences import load_preferences
from src.dataset_processing.trajectories import compare_trajectories, load_trajectories
from src.dataset_processing.trajectories import save_trajectories
from src.policies.vsl_policies import ContextualVAlignedDictSpaceActionPolicy, LearnerValueSystemLearningPolicy, MaskedPolicySimple, VAlignedDictSpaceActionPolicy, VAlignedDiscreteDictPolicy, ValueSystemLearningPolicyCustomLearner

from use_cases.roadworld_env_use_case.network_env import FeaturePreprocess, FeatureSelection
from src.utils import filter_none_args, load_json_config
import gymnasium as gym


def parse_policy_approximator(ref_class, env_name: ValueAlignedEnvironment, society_data: Dict, environment_data: Dict, ref_policy_kwargs: Dict, environment: Union[ValueAlignedEnvironment, ContextualEnv], grounding_variant: str=None, all_agent_groundings_to_save_files=None, parser_args=None,learner_or_expert: str = 'expert'):
    if 'Discrete' in ref_class or 'VAlignedDict' in ref_class:
        eap_agg_=None
        if society_data['is_tabular']:
                
                if environment_data['is_contextual']:
                    if learner_or_expert == 'expert':
                        def reward_matrix_contextual(new_context, old_context, align_func):
                            assert environment.context == new_context
                            return environment.reward_matrix_per_align_func(
                                align_func, custom_grounding=environment.calculate_assumed_grounding(
                                    variants=grounding_variant,
                                    save_folder=grounding_path, variants_save_files=all_agent_groundings_to_save_files[
                                        grounding_variant],
                                    recalculate=parser_args.recalculate_groundings)[1])
                    else:
                        raise NotImplementedError(
                            "Learner policies for contextual environments are not implemented yet")
                    eap_agg_ = dict(
                        contextual_reward_matrix=reward_matrix_contextual,
                        contextual_policy_estimation_kwargs=dict(
                            discount=environment_data['discount'],
                            horizon=environment.horizon,
                            approximator_kwargs=ref_policy_kwargs['approximator_kwargs'],
                            policy_approximator=PolicyApproximators(
                                ref_policy_kwargs['policy_approximation_method']),
                            deterministic=not society_data['stochastic_expert']
                        ),
                        policy_per_va_dict={}, state_encoder=None)
                    eap_class = ContextualVAlignedDictSpaceActionPolicy
                else:
                    eap_agg_ = dict(
                        policy_per_va_dict={}, state_encoder=None, **ref_policy_kwargs)
                    eap_class = VAlignedDictSpaceActionPolicy
        else:
                    eap_agg_ = dict(policy_per_va_dict={},state_encoder=None, **ref_policy_kwargs)
                    eap_class = VAlignedDiscreteDictPolicy
                    
        """return lambda environment, alignment_function, grounding_function, reward, discount, **kwargs: mce_partition_fh(environment,
            discount=discount,
            reward=environment.reward_matrix_per_align_func(
                alignment_function, custom_grounding=grounding_function),
            horizon=environment.horizon,
            approximator_kwargs=kwargs['approximator_kwargs'],
            policy_approximator=PolicyApproximators(policy_approximation_method),
            deterministic=not society_data['stochastic_expert'])"""
    
    elif 'custom' in ref_class or 'Custom' in ref_class:
        if 'MultiValued' in env_name:
            learner = lambda environment, alignment_function, grounding_function, reward, discount, stochastic, **kwargs: MultiValuedCarEnv.compute_policy(environment, reward=grounding_function,
                                                                                                    discount=discount,
                                                                                                    weights=alignment_function,**kwargs)
            eap_agg_ = dict(policy_per_va_dict={},learner_method=learner, **ref_policy_kwargs)
            eap_class = ValueSystemLearningPolicyCustomLearner
            
        else:
            raise NotImplementedError(
                f"Custom policy class {expert_policy_class} not implemented for environment {env_name}")
    else:
        eap_agg_ = dict(
                learner_class=parse_learner_class(
                    ref_policy_kwargs['learner_policy_class']),
                learner_kwargs=ref_policy_kwargs['learner_policy_kwargs'],
                masked=ref_policy_kwargs.get('policy_class', None)== 'MASKEDMlpPolicy',
                observation_space=environment.observation_space,
                action_space=environment.action_space, policy_class=parse_policy_class(
                    ref_policy_kwargs.get('policy_class', 'MlpPolicy')),
                policy_kwargs=ref_policy_kwargs.get('policy_kwargs', {}),)
        eap_class = LearnerValueSystemLearningPolicy
    print("PARSED KWARGS", eap_agg_)
    #input("Press enter to continue...")
    return eap_class, eap_agg_
        

def parse_learner_policy_class(learner_policy_class):
    if learner_policy_class == 'CustomPolicy':
        return ValueSystemLearningPolicyCustomLearner
    elif learner_policy_class == 'VAlignedDictSpaceActionPolicy':
        return VAlignedDictSpaceActionPolicy
    elif learner_policy_class == 'ContextualVAlignedDictSpaceActionPolicy':
        return ContextualVAlignedDictSpaceActionPolicy
    elif learner_policy_class == 'LearnerValueSystemLearningPolicy':
        return LearnerValueSystemLearningPolicy

def parse_learner_policy_kwargs(learner_policy_class, learner_policy_kwargs): 
    if learner_policy_class == 'VAlignedDictSpaceActionPolicy':
        return {'policy_per_va_dict': {}, 'state_encoder': None, 'expose_state': learner_policy_kwargs.get('expose_state', True), 'use_checkpoints': learner_policy_kwargs.get('use_checkpoints', False)}
    elif learner_policy_class == 'ContextualVAlignedDictSpaceActionPolicy':
        raise NotImplementedError(
            "ContextualVAlignedDictSpaceActionPolicy is not implemented yet for learner policies")
        #return {'contextual_reward_matrix': learner_policy_kwargs['contextual_reward_matrix'], 'contextual_policy_estimation_kwargs': learner_policy_kwargs['contextual_policy_estimation_kwargs'], 'env': learner_policy_kwargs['env'], 'state_encoder': learner_policy_kwargs.get('state_encoder', None), 'expose_state': learner_policy_kwargs.get('expose_state', True), 'use_checkpoints': learner_policy_kwargs.get('use_checkpoints', False)}
    elif learner_policy_class == 'LearnerValueSystemLearningPolicy':
        return {'learner_class': parse_learner_class(learner_policy_kwargs['learner_class']),
                'learner_kwargs': learner_policy_kwargs['learner_kwargs'], 'masked': learner_policy_kwargs.get('masked', False),
                'observation_space': learner_policy_kwargs['observation_space'], 'action_space': learner_policy_kwargs['action_space'],
                'policy_class': parse_policy_class(learner_policy_kwargs.get('policy_class', 'MlpPolicy')),
                'policy_kwargs': learner_policy_kwargs.get('policy_kwargs', {})}
    
def parse_learner_class(learner_class):
    if learner_class == 'PPO':
        from stable_baselines3 import ppo
        return ppo.PPO
    elif learner_class == 'MaskablePPO':
        from sb3_contrib import ppo_mask
        return ppo_mask.MaskablePPO
    else:
        raise ValueError(f"Unknown learner class: {learner_class}")


def parse_policy_class(policy_class):
    if policy_class == "MASKEDMlpPolicy":
        return MASKEDMlpPolicy
    elif policy_class == "MlpPolicy":
        return MlpPolicy
    elif policy_class == "MaskedPolicySimple":
        return MaskedPolicySimple
    else:
        return None


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
    general_group.add_argument('-sf', '--society_file', type=str, default='societies.json',
                               help='Path to JSON society configuration file (overrides other defaults here, but not the command line arguments)')
    general_group.add_argument('-sname', '--society_name', type=str, default='default',
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')

    general_group.add_argument('-rg', '--recalculate_groundings', action='store_true', default=True,
                               help='Recalculate custom agent groundings')

    general_group.add_argument('-e', '--environment', type=str, required=True, choices=[
                               'rw', 'ff', 'vrw', 'mvc'], help='environment (roadworld - rw, firefighters - ff, variable dest roadworld - vrw, multivalued car - mvc)')

    general_group.add_argument('-df', '--discount_factor', type=float, default=1.0,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')

    general_group.add_argument(
        '-s', '--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    general_group.add_argument('-varhz', '--end_trajs_when_ended', action='store_true', default=False,
                               help="Allow trajectories to end when the environment says an episode is done or horizon is reached, whatever happens first, instead of forcing all trajectories to have the length of the horizon")
    general_group.add_argument('-tsize', '--test_size', type=float,
                               default=0.0, help='Ratio_of_test_versus_train_preferences')
    alg_group = parser.add_argument_group('Algorithm-specific Parameters')
    pc_group = alg_group.add_argument_group(
        'Preference Comparisons Parameters')
    pc_group.add_argument('-dfp', '--discount_factor_preferences', type=float,
                          default=1.0, help='Discount factor for preference comparisons')

    env_group = parser.add_argument_group('environment-specific Parameters')

    env_group.add_argument('-rt', '--retrain', action='store_true',
                           # TODO: might be needed if we implement non tabular environments...
                           default=False, help='Retrain experts')
    env_group.add_argument('-appr', '--is_tabular', action='store_true',
                           default=False, help='Approximate expert')
    env_group.add_argument('-reps', '--reward_epsilon', default=0.0, type=float,
                           help='Distance between the cummulative rewards of each pair of trajectories for them to be considered as equal in the comparisons')

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

    environment_data = config[parser_args.environment]
    society_data = society_config[parser_args.environment][parser_args.society_name]
    alg_config = environment_data['algorithm_config'][parser_args.algorithm]
    grounding_path = os.path.join(
        'envs', parser_args.environment, GROUNDINGS_PATH)
    dataset_name = parser_args.dataset_name

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
    if parser_args.environment == 'mvc':
        extra_kwargs = {
            'feature_selection': MVFS(environment_data['feature_selection']),
        }
    if parser_args.environment == 'rw' or parser_args.environment == 'vrw':
        extra_kwargs = {'env_kwargs': {
            'feature_selection': FeatureSelection(environment_data['feature_selection']),
            'feature_preprocessing': FeaturePreprocess(environment_data['feature_preprocessing']),

        }}
        if 'Fixed' in environment_data['name']:
            extra_kwargs['with_destination'] = 64

    print(extra_kwargs)
    environment: Union[TabularVAMDP, ContextualEnv] = gym.make(
        environment_data['name'],
        horizon=environment_data['horizon'], **extra_kwargs)
    environment.reset(seed=parser_args.seed)

    expert_policy_per_grounding_combination: Dict[tuple,
                                                  VAlignedDictSpaceActionPolicy] = dict()

    all_agent_groundings = dict()
    all_agent_groundings = dict()

    expert_policy_kwargs: Dict = alg_config['expert_policy_kwargs'][alg_config['expert_policy_class']]
    expert_policy_class = alg_config['expert_policy_class']

    for agg in all_agent_groundings_to_save_files.keys():
        os.makedirs(grounding_path, exist_ok=True)
        all_agent_groundings[agg] = environment.calculate_assumed_grounding(variants=agg,
                                    save_folder=grounding_path,
                                    variants_save_files=all_agent_groundings_to_save_files[
                                        agg],
                                    recalculate=parser_args.recalculate_groundings)
        environment.reset(seed=parser_args.seed)

        epclass, epkwargs = parse_policy_approximator(
            ref_class=expert_policy_class,
            all_agent_groundings_to_save_files=all_agent_groundings_to_save_files,
            learner_or_expert= 'expert',
                                                                env_name=environment_data['name'], 
                                                                society_data=society_data, environment_data=environment_data,
                                                                ref_policy_kwargs=expert_policy_kwargs, environment=environment, grounding_variant=agg)


        
        expert_policy_per_grounding_combination[agg] = epclass(env=environment, **epkwargs)
        for w in agent_profiles:
            eap_agg = expert_policy_per_grounding_combination[agg]
            if epclass != LearnerValueSystemLearningPolicy:
                try:
                    otrajs = eap_agg.obtain_trajectories(n_seeds=100, seed=DEFAULT_SEED, stochastic=society_data['stochastic_expert'], exploration=0.0, align_funcs_in_policy=[
                                                        w], repeat_per_seed=1, with_reward=True, with_grounding=True, alignments_in_env=[w], end_trajectories_when_ended=parser_args.end_trajs_when_ended)
                    print("O trajs mean reward: ", np.mean(
                        [t.vs_rews.sum() for t in otrajs]))
                except:
                    print("Error obtaining trajectories, skipping...")
                    
            
                
                eap_agg.learn(
                alignment_function=w, grounding_function=all_agent_groundings[agg],
                reward=None, stochastic=society_data['stochastic_expert'], **expert_policy_kwargs)

                otrajs = eap_agg.obtain_trajectories(n_seeds=100, seed=DEFAULT_SEED, stochastic=society_data['stochastic_expert'], exploration=0.0, align_funcs_in_policy=[
                                                     w], repeat_per_seed=1, with_reward=True, with_grounding=True, alignments_in_env=[w], end_trajectories_when_ended=parser_args.end_trajs_when_ended)
                print("N trajs mean reward: ", np.mean(
                    [t.vs_rews.sum() for t in otrajs]))
                

                """policy_per_va_precalc = dict()
                
                if parser_args.environment == 'rw':
                    np.testing.assert_allclose(environment.reward_matrix_per_align_func(
                        agent_profiles[0], custom_grounding=all_agent_groundings[agg])[environment.goal_states], 0.0)
                for w in agent_profiles:
                    _, _, assumed_expert_pi = policy_approximator(
                            environment, alignment_function=w, 
                            grounding_function= all_agent_groundings[agg],
                            reward = None,
                            **expert_policy_kwargs)

                    policy_per_va_precalc[w] = assumed_expert_pi

                
                
                    else:
                        expert_policy_per_grounding_combination[agg] = VAlignedDictSpaceActionPolicy(
                            policy_per_va_dict=policy_per_va_precalc,  env=environment, state_encoder=None, expose_state=True, use_checkpoints=False)
                else:
                    # TODO: need a custom policy for non tabular environments...?
                    expert_policy_per_grounding_combination[agg] = VAlignedDiscretePolicy(
                        policy_per_va=lambda va: (lambda obs: policy_per_va_precalc[va](obs)),  env=environment, state_encoder=None, 
                        expose_state=False, assume_env_produce_state=False, use_checkpoints=False )"""
                    #expert_policy_per_grounding_combination[agg].get_policy_probs = lambda policy, state_obs: 
            else:
            
                
                otrajs = eap_agg.obtain_trajectories(n_seeds=100, seed=DEFAULT_SEED, stochastic=False, exploration=0.0, align_funcs_in_policy=[
                                                     w], repeat_per_seed=1, with_reward=True, with_grounding=True, alignments_in_env=[w], end_trajectories_when_ended=parser_args.end_trajs_when_ended)
                print("O trajs mean reward: ", np.mean(
                    [t.vs_rews.sum() for t in otrajs]))
                evaluationo = stable_baselines3.common.evaluation.evaluate_policy(eap_agg.get_learner_for_alignment_function(
                    w).policy, eap_agg.get_environ(w), n_eval_episodes=50, deterministic=not society_data['stochastic_expert'], return_episode_rewards=False)

                if parser_args.retrain:
                    eap_agg.learn(
                        alignment_function=w, grounding_function=all_agent_groundings[agg], stochastic=society_data['stochastic_expert'], **expert_policy_kwargs, tb_log_name=f'expert_policy_{w}')
                    #eap_agg.save(path=f'expert_policy_{w}')

                print(f"Evaluation before for {w}: {evaluationo}")
                # print(environment.observation_space)

                #
                #
                # eap_agg.save(path=f'testing_{w}')
                # eap_agg = LearnerValueSystemLearningPolicy.load(ref_env = eap_agg.get_environ(w), path=f'testing_{w}')
                # eap_agg.save(path=f'testing_{w}')
                """eap_agg = LearnerValueSystemLearningPolicy.load(
                    ref_env=environment, path=f'expert_policy_{w}')"""
                print(eap_agg.get_learner_for_alignment_function(w).policy)
                evaluation = stable_baselines3.common.evaluation.evaluate_policy(eap_agg.get_learner_for_alignment_function(
                    w).policy, eap_agg.get_environ(w), n_eval_episodes=100, deterministic=not society_data['stochastic_expert'], return_episode_rewards=False)
                print(f"Evaluation after for {w}: {evaluation}")
                
                ntrajs = eap_agg.obtain_trajectories(n_seeds=100, seed=DEFAULT_SEED, stochastic=False, exploration=0.0, align_funcs_in_policy=[
                                                     w], repeat_per_seed=1, with_reward=True, with_grounding=True, alignments_in_env=[w], end_trajectories_when_ended=parser_args.end_trajs_when_ended)
                
                    
                print("N trajs mean reward: ", np.mean(
                    [t.vs_rews.sum() for t in ntrajs]))
                print("N trajs mean reward per value: ", np.mean(
                    [t.v_rews.sum(axis=-1) for t in ntrajs],axis=0))
                
                
        eap_agg.save(path=calculate_expert_policy_save_path(
            environment_name=parser_args.environment, 
            dataset_name=parser_args.dataset_name,
            society_name=parser_args.society_name,
            class_name=eap_agg.__class__.__name__,
            grounding_name=agg))
        eap_agg = eap_agg.__class__.load(ref_env=environment, path=calculate_expert_policy_save_path(
            environment_name=parser_args.environment, 
            dataset_name=parser_args.dataset_name,
            society_name=parser_args.society_name,
            class_name=eap_agg.__class__.__name__,
            grounding_name=agg))
        #exit(0)
        
    # TODO: Generate dataset of trajectories.
    if parser_args.end_trajs_when_ended is True and parser_args.gen_trajs is False:
        raise ValueError(
            "To use varhz flag (variable horizon trajectories), you need to generate trajectories again, i.e. also use the -gentr flag")
    if parser_args.gen_trajs:
        for iag, ag in enumerate(society_data['agents']):
            ag_grounding = ag_name_to_aggrounding[ag['name']]
            ag_policy = expert_policy_per_grounding_combination[ag_grounding]
            n_rational_trajs = int(np.ceil(ag['n_agents']*ag['data']['trajectory_pairs']*(1-ag['data']['random_traj_proportion']))
                                   ) if not society_data['same_trajectories_for_each_agent_type'] else int(np.ceil(ag['data']['trajectory_pairs']*(1-ag['data']['random_traj_proportion'])))
            n_random_trajs = int(np.floor(ag['n_agents']*ag['data']['trajectory_pairs']*ag['data']['random_traj_proportion'])
                                 ) if not society_data['same_trajectories_for_each_agent_type'] else int(np.ceil(ag['data']['trajectory_pairs']*(ag['data']['random_traj_proportion'])))
            # rationality is epsilon rationality.
            print(f"Agent {iag}")
            grounding_simple = ag_policy.env.calculate_assumed_grounding(variants=tuple(ag['grounding']),
                                                                            save_folder=grounding_path,
                                                                            variants_save_files=all_agent_groundings_to_save_files[tuple(
                                                                                ag['grounding'])],
                                                                            recalculate=False)

            print(f"Grounding Done {iag}")
            ag_rational_trajs: TrajectoryWithValueSystemRews = ag_policy.obtain_trajectories(n_seeds=n_rational_trajs,
                                                                                             stochastic=society_data[
                                                                                                 'stochastic_expert'], exploration=1.0-ag['data']['rationality'],
                                                                                             align_funcs_in_policy=[tuple(ag['value_system'])], repeat_per_seed=1, with_reward=True, with_grounding=True,
                                                                                             alignments_in_env=[tuple(ag['value_system'])], end_trajectories_when_ended=parser_args.end_trajs_when_ended, reward_dtype=np.float64)
            print(f"Rational Trajs Done {iag} ({len(ag_rational_trajs)})")

            if len(ag_rational_trajs) > 0:
                assert ag_rational_trajs[0].v_rews.dtype == np.float64
            # random policies are equivalent to having rationality 0:

            ag_random_trajs: TrajectoryWithValueSystemRews = ag_policy.obtain_trajectories(n_seeds=n_random_trajs,
                                                                                           stochastic=society_data['stochastic_expert'], exploration=1.0, align_funcs_in_policy=[tuple(ag['value_system'])],
                                                                                           repeat_per_seed=1, with_reward=True, with_grounding=True, alignments_in_env=[tuple(ag['value_system'])],
                                                                                           end_trajectories_when_ended=parser_args.end_trajs_when_ended, reward_dtype=np.float64)
            print(f"Random Trajs Done {iag} ({len(ag_random_trajs)})")

            if len(ag_random_trajs) > 0:
                assert ag_random_trajs[0].v_rews.dtype == np.float64

            all_trajs_ag = []
            all_trajs_ag.extend(ag_random_trajs)
            all_trajs_ag.extend(ag_rational_trajs)

            
            
            random.shuffle(all_trajs_ag)

            print(f"All Trajs Done {iag} ({len(all_trajs_ag)})", ag)

            if parser_args.test_size > 0.0:

                all_trajs_ag_train = all_trajs_ag[:int(
                    len(all_trajs_ag)*(1-parser_args.test_size))]
                all_trajs_ag_test = all_trajs_ag[int(
                    len(all_trajs_ag)*(1-parser_args.test_size)):]
                if society_data['same_trajectories_for_each_agent_type']:
                    for _ in range(ag['n_agents']-1):
                        all_trajs_ag_train.extend(
                            all_trajs_ag[:int(np.ceil(len(all_trajs_ag)*(1-parser_args.test_size)))])
                        t_insert = all_trajs_ag[int(np.ceil(
                            len(all_trajs_ag)*(1-parser_args.test_size))):]
                        assert len(t_insert) == np.ceil(parser_args.test_size*len(all_trajs_ag)
                                                        ), f"Expected {int(parser_args.test_size*len(all_trajs_ag))} test trajectories, got {len(t_insert)}"
                        all_trajs_ag_test.extend(t_insert)
                        print("INDEX REF??", int(
                            len(all_trajs_ag)*(1-parser_args.test_size)))
                        print("NEXT INDEX", int(
                            np.ceil(parser_args.test_size*len(all_trajs_ag))))
                        print("N_trajs??", len(all_trajs_ag))
                        print("NAGENTS?", ag['n_agents'])
                    np.testing.assert_allclose(all_trajs_ag_train[0].obs, all_trajs_ag_train[int(
                        np.ceil((1-parser_args.test_size)*len(all_trajs_ag)))].obs)
                save_trajectories(all_trajs_ag_train, dataset_name=dataset_name+'_train', ag=ag,
                                  society_data=society_data, environment_data=environment_data, dtype=parser_args.dtype)

                save_trajectories(all_trajs_ag_test, dataset_name=dataset_name+'_test', ag=ag,
                                  society_data=society_data, environment_data=environment_data, dtype=parser_args.dtype)
                trajs_test = load_trajectories(dataset_name=dataset_name+'_test', ag=ag,
                                               society_data=society_data, environment_data=environment_data, override_dtype=parser_args.dtype)
                np.testing.assert_equal(
                    trajs_test[0].obs, all_trajs_ag_test[0].obs)
                np.testing.assert_equal(
                    trajs_test[-1].obs, all_trajs_ag_test[-1].obs)
                if society_data['same_trajectories_for_each_agent_type']:
                    print(len(trajs_test), len(all_trajs_ag_test),
                          len(trajs_test)//ag['n_agents'])
                    np.testing.assert_allclose(
                        all_trajs_ag_test[0].obs, all_trajs_ag_test[len(trajs_test)//ag['n_agents']].obs)
                    np.testing.assert_allclose(
                        trajs_test[0].obs, trajs_test[len(trajs_test)//ag['n_agents']].obs)
            else:
                if society_data['same_trajectories_for_each_agent_type']:
                    for _ in range(ag['n_agents']):
                        all_trajs_ag.extend(ag_random_trajs)
                        all_trajs_ag.extend(ag_rational_trajs)

                save_trajectories(all_trajs_ag, dataset_name=dataset_name, ag=ag,
                                  society_data=society_data, environment_data=environment_data, dtype=parser_args.dtype)

    if parser_args.gen_preferences:

        for iag, ag in enumerate(society_data['agents']):
            for suffix in ['_train', '_test'] if parser_args.test_size > 0.0 else ['']:

                agg = tuple(ag['grounding'])
                all_trajs_ag = load_trajectories(
                    dataset_name=dataset_name+suffix, ag=ag, society_data=society_data, environment_data=environment_data, override_dtype=parser_args.dtype)

                print(
                    f"1AGENT {iag} ({ag['name']}) - {len(all_trajs_ag)} trajectories loaded", suffix)
                if society_data["same_trajectories_for_each_agent_type"]:
                    # This assumes the trajectories of each agent are the same, and then we will make each agent label the same pairs
                    idxs_unique = np.random.permutation(
                        len(all_trajs_ag)//ag['n_agents'])
                    print(len(idxs_unique))
                    idxs = []
                    for step in range(ag['n_agents']):
                        idxs.extend(list(idxs_unique + step *
                                    len(all_trajs_ag)//ag['n_agents']))
                    idxs = np.array(idxs, dtype=np.int64)
                    print(len(idxs), len(all_trajs_ag))
                    print("IDXS AT HERE", idxs)
                    assert len(idxs) == len(all_trajs_ag)
                else:
                    # Indicates the order of comparison. idxs[0] with idxs[1], then idxs[1] with idxs[2], etc...
                    idxs = np.random.permutation(len(all_trajs_ag))
                discounted_sums = np.zeros_like(idxs, dtype=np.float64)

                ag_grounding = all_agent_groundings[agg]
                discounted_sums_per_grounding = np.zeros(
                    (len(environment_data['basic_profiles']), discounted_sums.shape[0]), dtype=np.float64)
                for i in range((len(all_trajs_ag))):

                    discounted_sums[i] = imitation.data.rollout.discounted_sum(
                        all_trajs_ag[i].vs_rews, gamma=alg_config['discount_factor_preferences'])
                    for vi in range(discounted_sums_per_grounding.shape[0]):
                        if society_data['is_tabular']:
                            grounding_simple = np.astype(
                                # this is NOT contextualized, CAUTION!
                                ag_grounding(all_trajs_ag[i].obs[:-1], all_trajs_ag[i].acts)[:, vi], np.float64)
                        
                        list_ = []

                        if environment_data['is_contextual']:
                            environment.contextualize(
                                all_trajs_ag[i].infos[0]['context'])

                        for o, no, a, info in zip(all_trajs_ag[i].obs[:-1], all_trajs_ag[i].obs[1:], all_trajs_ag[i].acts, all_trajs_ag[i].infos):
                            list_.append(environment.get_reward_per_align_func(align_func=tuple(
                                environment_data['basic_profiles'][vi]), action=a, info=info, state=o, next_state=no, custom_grounding=ag_grounding))

                        # print("LIST", list_)
                        # print("GR SIMPLE", grounding_simple)
                        # print("VREW", all_trajs_ag[i].v_rews[vi])
                        # np.testing.assert_almost_equal(np.asarray(list_, dtype=parser_args.dtype), grounding_simple, decimal = 5 if parser_args.dtype in [np.float32, np.float64] else 3)

                        np.testing.assert_almost_equal(
                            np.asarray(list_, dtype=parser_args.dtype), all_trajs_ag[i].v_rews[vi], decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                        
                        discounted_sums_per_grounding[vi, i] = imitation.data.rollout.discounted_sum(
                            all_trajs_ag[i].v_rews[vi], gamma=alg_config['discount_factor_preferences'])
                # We save the comparison of 1 vs 2, 2 vs 3 in the order stablished in discounted_sums.
                print(
                    f"2AGENT {iag} ({ag['name']}) - {len(idxs)} idxs generated", len(discounted_sums), suffix)
                assert max(idxs) < len(all_trajs_ag)
                save_preferences(idxs=idxs, discounted_sums=discounted_sums, discounted_sums_per_grounding=discounted_sums_per_grounding,
                                 dataset_name=dataset_name+suffix, epsilon=parser_args.reward_epsilon, environment_data=environment_data, society_data=society_data, ag=ag)

    # TEST preferences load okey.
    print("TESTING DATA COHERENCE. It is safe to stop this program now...")
    for dataset_name_ in [dataset_name+'_train', dataset_name+'_test'] if parser_args.test_size > 0.0 else [dataset_name]:
        for iag, ag in enumerate(society_data['agents']):
            agg = tuple(ag['grounding'])
            #  Here idxs is the list of trajectory PAIRS of indices from the trajectory list that are compared.
            idxs, discounted_sums, discounted_sums_per_grounding, preferences, preferences_per_grounding = load_preferences(
                epsilon=parser_args.reward_epsilon, dataset_name=dataset_name_, environment_data=environment_data, society_data=society_data, ag=ag, dtype=parser_args.dtype)

            trajs_ag = load_trajectories(dataset_name=dataset_name_, ag=ag,
                                         environment_data=environment_data, society_data=society_data, override_dtype=parser_args.dtype)
            if parser_args.test_size > 0.0:
                trajs_ag_all = load_trajectories(dataset_name=dataset_name+'_train', ag=ag,
                                                 environment_data=environment_data, society_data=society_data, override_dtype=parser_args.dtype)
                trajs_ag_all.extend(load_trajectories(dataset_name=dataset_name+'_test', ag=ag,
                                                      environment_data=environment_data, society_data=society_data, override_dtype=parser_args.dtype))
            else:
                trajs_ag_all = trajs_ag
            n_pairs_per_agent_prime = len(trajs_ag)//ag['n_agents']

            if society_data["same_trajectories_for_each_agent_type"]:
                for t in range(ag['n_agents']-2):
                    np.testing.assert_allclose(idxs[0:n_pairs_per_agent_prime], (idxs[(
                        t+1)*n_pairs_per_agent_prime:(t+2)*n_pairs_per_agent_prime] - n_pairs_per_agent_prime*(t+1)))
                    # print(len(trajs_ag))
                    # print(idxs[0:n_pairs_per_agent_prime])
                    # print(idxs[(t+1)*(n_pairs_per_agent_prime):(t+2)*(n_pairs_per_agent_prime)])
                    # print("A", ag['name'], t+1, t, trajs_ag[idxs[(t+1)*(n_pairs_per_agent_prime)][0]].obs, trajs_ag[idxs[0][0]].obs)

                    for traj_i in range(n_pairs_per_agent_prime):

                        np.testing.assert_allclose(trajs_ag[traj_i + t*n_pairs_per_agent_prime].obs, trajs_ag[(
                            t+1)*n_pairs_per_agent_prime + traj_i].obs)

            for i in range((len(trajs_ag))):
                np.testing.assert_almost_equal(discounted_sums[i], imitation.data.rollout.discounted_sum(
                    trajs_ag[i].vs_rews, gamma=alg_config['discount_factor_preferences']))

            for idx, pr in zip(idxs, preferences):
                assert isinstance(discounted_sums, np.ndarray)
                assert isinstance(idx, np.ndarray)
                idx = [int(ix) for ix in idx]

                np.testing.assert_almost_equal(discounted_sums[idx[0]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                    trajs_ag[idx[0]].vs_rews, gamma=alg_config['discount_factor_preferences'])), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
                np.testing.assert_almost_equal(discounted_sums[idx[1]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                    trajs_ag[idx[1]].vs_rews, gamma=alg_config['discount_factor_preferences'])), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
                np.testing.assert_almost_equal(compare_trajectories(
                    discounted_sums[idx[0]], discounted_sums[idx[1]], epsilon=parser_args.reward_epsilon), pr, decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)
            for vi in range(len(environment_data['basic_profiles'])):
                for i in range((len(trajs_ag))):
                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, i], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        trajs_ag[i].v_rews[vi], gamma=alg_config['discount_factor_preferences'])), decimal=4 if parser_args.dtype in [np.float32, np.float64] else 3)
                    """if not environment_data['is_contextual']:
                        np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, i], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        all_agent_groundings[agg][trajs_ag[i].obs[:-1], trajs_ag[i].acts][:, vi], gamma=alg_config['discount_factor_preferences'])), decimal = 5 if parser_args.dtype in [np.float32, np.float64] else 3)
                    """
                for idx, pr in zip(idxs, preferences_per_grounding[:, vi]):
                    idx = [int(ix) for ix in idx]

                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[0]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        trajs_ag[idx[0]].v_rews[vi], gamma=alg_config['discount_factor_preferences'])), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[1]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        trajs_ag[idx[1]].v_rews[vi], gamma=alg_config['discount_factor_preferences'])), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                    """if not environment_data['is_contextual']:
                        np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[0]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                            all_agent_groundings[agg][trajs_ag[idx[0]].obs[:-1], trajs_ag[idx[0]].acts][:, vi], gamma=alg_config['discount_factor_preferences'])), decimal = 5 if parser_args.dtype in [np.float32, np.float64] else 3)
                        np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[1]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                            all_agent_groundings[agg][trajs_ag[idx[1]].obs[:-1], trajs_ag[idx[1]].acts][:, vi], gamma=alg_config['discount_factor_preferences'])), decimal = 5 if parser_args.dtype in [np.float32, np.float64] else 3)
                    """
                    np.testing.assert_almost_equal(compare_trajectories(
                        discounted_sums_per_grounding[vi, idx[0]], discounted_sums_per_grounding[vi, idx[1]], epsilon=parser_args.reward_epsilon), pr, decimal=4 if parser_args.dtype in [np.float32, np.float64] else 3)
    if parser_args.test_size > 0.0:
        print("Dataset generated correctly.")
        dataset_train = create_dataset(parser_args, config, society_data, train_or_test='train',
                                       default_groundings=society_config[parser_args.environment]['groundings'])
        print("TRAIN SIZE", len(dataset_train))
        dataset_test = create_dataset(parser_args, config, society_data, train_or_test='test',
                                      default_groundings=society_config[parser_args.environment]['groundings'])
        print("TEST SIZE", len(dataset_test))
    else:
        dataset = create_dataset(parser_args, config, society_data,
                                 default_groundings=society_config[parser_args.environment]['groundings'])
        print("Dataset generated correctly.")
        print("Dataset size", len(dataset))
