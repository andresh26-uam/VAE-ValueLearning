import argparse
import os
import pprint
import random
from typing import Dict, Union
import imitation
import imitation.data
import imitation.data.rollout
import imitation.util
import numpy as np
import torch

from envs.tabularVAenv import ContextualEnv, TabularVAMDP
from src.dataset_processing.utils import DEFAULT_SEED, GROUNDINGS_PATH
from src.algorithms.utils import PolicyApproximators, mce_partition_fh
from envs.firefighters_env import FeatureSelectionFFEnv
from src.dataset_processing.data import TrajectoryWithValueSystemRews
from src.dataset_processing.utils import calculate_dataset_save_path
from src.dataset_processing.datasets import create_dataset
from src.dataset_processing.preferences import save_preferences
from src.dataset_processing.preferences import load_preferences
from src.dataset_processing.trajectories import compare_trajectories, load_trajectories
from src.dataset_processing.trajectories import save_trajectories
from src.policies.vsl_policies import ContextualVAlignedDictSpaceActionPolicy, VAlignedDictSpaceActionPolicy

from use_cases.roadworld_env_use_case.network_env import FeaturePreprocess, FeatureSelection
from src.utils import filter_none_args, load_json_config
import gymnasium as gym




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
                               'rw', 'ff', 'vrw'], help='environment (roadworld - rw, firefighters - ff, variable dest roadworld - vrw)')

    general_group.add_argument('-df', '--discount_factor', type=float, default=1.0,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')

    general_group.add_argument(
        '-s', '--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    general_group.add_argument('-varhz', '--end_trajs_when_ended', action='store_true', default=False,
                               help="Allow trajectories to end when the environment says an episode is done or horizon is reached, whatever happens first, instead of forcing all trajectories to have the length of the horizon")

    alg_group = parser.add_argument_group('Algorithm-specific Parameters')
    pc_group = alg_group.add_argument_group(
        'Preference Comparisons Parameters')
    pc_group.add_argument('-dfp', '--discount_factor_preferences', type=float,
                          default=1.0, help='Discount factor for preference comparisons')

    env_group = parser.add_argument_group('environment-specific Parameters')

    env_group.add_argument('-rt', '--retrain', action='store_true',
                           # TODO: might be needed if we implement non tabular environments...
                           default=False, help='Retrain experts')
    env_group.add_argument('-appr', '--approx_expert', action='store_true',
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
    tabular_all_agent_groundings = dict()

    for agg in all_agent_groundings_to_save_files.keys():
        os.makedirs(grounding_path, exist_ok=True)
        all_agent_groundings[agg], tabular_all_agent_groundings[agg] = environment.calculate_assumed_grounding(variants=agg,
                                                                                                               save_folder=grounding_path,
                                                                                                               variants_save_files=all_agent_groundings_to_save_files[
                                                                                                                   agg],
                                                                                                               recalculate=parser_args.recalculate_groundings)

        if society_data['approx_expert']:
            profile_to_assumed_matrix = dict()
            environment.reset(seed=parser_args.seed)
            if parser_args.environment == 'rw':
                np.testing.assert_allclose(environment.reward_matrix_per_align_func(
                    agent_profiles[0], custom_grounding=tabular_all_agent_groundings[agg])[environment.goal_states], 0.0)
            for w in agent_profiles:

                _, _, assumed_expert_pi = mce_partition_fh(environment, discount=environment_data['discount'],
                                                           reward=environment.reward_matrix_per_align_func(
                    w, custom_grounding=tabular_all_agent_groundings[agg]),
                    horizon=environment.horizon,
                    approximator_kwargs=society_data['approximator_kwargs'],
                    policy_approximator=PolicyApproximators(
                                                               society_data['policy_approximation_method']),
                    deterministic=not society_data['stochastic_expert'])

                profile_to_assumed_matrix[w] = assumed_expert_pi

            def reward_matrix_contextual(new_context, old_context, align_func):
                assert environment.context == new_context
                return environment.reward_matrix_per_align_func(
                    align_func, custom_grounding=environment.calculate_assumed_grounding(
                        variants=agg,
                        save_folder=grounding_path, variants_save_files=all_agent_groundings_to_save_files[
                            agg],
                        recalculate=parser_args.recalculate_groundings)[1])

            if environment_data['is_contextual']:
                expert_policy_per_grounding_combination[agg] = ContextualVAlignedDictSpaceActionPolicy(
                    contextual_reward_matrix=reward_matrix_contextual,
                    contextual_policy_estimation_kwargs=dict(
                        discount=environment_data['discount'],
                        horizon=environment.horizon,
                        approximator_kwargs=society_data['approximator_kwargs'],
                        policy_approximator=PolicyApproximators(
                            society_data['policy_approximation_method']),
                        deterministic=not society_data['stochastic_expert']
                    ),
                    policy_per_va_dict=profile_to_assumed_matrix, env=environment, state_encoder=None, expose_state=True, use_checkpoints=False)
            else:
                expert_policy_per_grounding_combination[agg] = VAlignedDictSpaceActionPolicy(
                    policy_per_va_dict=profile_to_assumed_matrix,  env=environment, state_encoder=None, expose_state=True, use_checkpoints=False)
        else:
            raise NotImplementedError(
                "not tested a PPO or other deep learning technique for expert RL generator yet")

    # TODO: Generate dataset of trajectories.
    if parser_args.end_trajs_when_ended is True and parser_args.gen_trajs is False:
        raise ValueError(
            "To use varhz flag (variable horizon trajectories), you need to generate trajectories again, i.e. also use the -gentr flag")
    if parser_args.gen_trajs:
        for i, ag in enumerate(society_data['agents']):
            ag_grounding = ag_name_to_aggrounding[ag['name']]
            ag_policy = expert_policy_per_grounding_combination[ag_grounding]
            n_rational_trajs = int(np.ceil(ag['n_agents']*ag['data']['trajectory_pairs']*(1-ag['data']['random_traj_proportion']))
                                   ) if not society_data['same_trajectories_for_each_agent_type'] else int(np.ceil(ag['data']['trajectory_pairs']*(1-ag['data']['random_traj_proportion'])))
            n_random_trajs = int(np.floor(ag['n_agents']*ag['data']['trajectory_pairs']*ag['data']['random_traj_proportion'])
                                 ) if not society_data['same_trajectories_for_each_agent_type'] else int(np.ceil(ag['data']['trajectory_pairs']*(ag['data']['random_traj_proportion'])))
            # rationality is epsilon rationality.
            print(f"Agent {i}")
            _, grounding_simple = ag_policy.env.calculate_assumed_grounding(variants=tuple(ag['grounding']),
                                                                            save_folder=grounding_path,
                                                                            variants_save_files=all_agent_groundings_to_save_files[tuple(
                                                                                ag['grounding'])],
                                                                            recalculate=False)

            print(f"Grounding Done {i}")
            ag_rational_trajs: TrajectoryWithValueSystemRews = ag_policy.obtain_trajectories(n_seeds=n_rational_trajs,
                                                                                             stochastic=society_data[
                                                                                                 'stochastic_expert'], exploration=1.0-ag['data']['rationality'],
                                                                                             align_funcs_in_policy=[tuple(ag['value_system'])], repeat_per_seed=1, with_reward=True, with_grounding=True,
                                                                                             alignments_in_env=[tuple(ag['value_system'])], end_trajectories_when_ended=parser_args.end_trajs_when_ended, reward_dtype=np.float64)
            print(f"Rational Trajs Done {i} ({len(ag_rational_trajs)})")

            if len(ag_rational_trajs) > 0:
                assert ag_rational_trajs[0].v_rews.dtype == np.float64
            # random policies are equivalent to having rationality 0:

            ag_random_trajs: TrajectoryWithValueSystemRews = ag_policy.obtain_trajectories(n_seeds=n_random_trajs,
                                                                                           stochastic=society_data['stochastic_expert'], exploration=1.0, align_funcs_in_policy=[tuple(ag['value_system'])],
                                                                                           repeat_per_seed=1, with_reward=True, with_grounding=True, alignments_in_env=[tuple(ag['value_system'])],
                                                                                           end_trajectories_when_ended=parser_args.end_trajs_when_ended, reward_dtype=np.float64)
            print(f"Random Trajs Done {i} ({len(ag_random_trajs)})")
            if len(ag_random_trajs) > 0:
                assert ag_random_trajs[0].v_rews.dtype == np.float64
            # TODO: ADD FOR EACH TRAJECTORY the Value rews.

            all_trajs_ag = []
            if not society_data['same_trajectories_for_each_agent_type']:
                all_trajs_ag.extend(ag_random_trajs)
                all_trajs_ag.extend(ag_rational_trajs)
            else:
                for _ in range(ag['n_agents']):
                    all_trajs_ag.extend(ag_random_trajs)
                    all_trajs_ag.extend(ag_rational_trajs)
            save_trajectories(all_trajs_ag, dataset_name=dataset_name, ag=ag,
                              society_data=society_data, environment_data=environment_data, dtype=parser_args.dtype)

    if parser_args.gen_preferences:
        for i, ag in enumerate(society_data['agents']):
            agg = tuple(ag['grounding'])
            all_trajs_ag = load_trajectories(
                dataset_name=dataset_name, ag=ag, society_data=society_data, environment_data=environment_data, override_dtype=parser_args.dtype)
            if society_data["same_trajectories_for_each_agent_type"]:
                # This assumes the trajectories of each agent are the same, and then we will make each agent label the same pairs
                idxs_unique = np.random.permutation(
                    len(all_trajs_ag)//ag['n_agents'])
                idxs = []
                for step in range(ag['n_agents']):
                    idxs.extend(list(idxs_unique + step*len(idxs_unique)))
                idxs = np.array(idxs, dtype=np.int64)
                assert len(idxs) == len(all_trajs_ag)
            else:
                # Indicates the order of comparison. idxs[0] with idxs[1], then idxs[1] with idxs[2], etc...
                idxs = np.random.permutation(len(all_trajs_ag))
            discounted_sums = np.zeros_like(idxs, dtype=np.float64)

            ag_grounding = tabular_all_agent_groundings[agg]
            discounted_sums_per_grounding = np.zeros(
                (len(environment_data['basic_profiles']), discounted_sums.shape[0]), dtype=np.float64)
            for i in range((len(all_trajs_ag))):

                discounted_sums[i] = imitation.data.rollout.discounted_sum(
                    all_trajs_ag[i].vs_rews, gamma=alg_config['discount_factor_preferences'])
                for vi in range(discounted_sums_per_grounding.shape[0]):
                    grounding_simple = np.astype(
                        # this is NOT contextualized, CAUTION!
                        ag_grounding[all_trajs_ag[i].obs[:-1], all_trajs_ag[i].acts][:, vi], np.float64)
                    list_ = []

                    if environment_data['is_contextual']:
                        environment.contextualize(
                            all_trajs_ag[i].infos[0]['context'])

                    for o, no, a, info in zip(all_trajs_ag[i].obs[:-1], all_trajs_ag[i].obs[1:], all_trajs_ag[i].acts, all_trajs_ag[i].infos):
                        list_.append(environment.get_reward_per_align_func(align_func=tuple(
                            environment_data['basic_profiles'][vi]), action=a, info=info, obs=o, next_obs=no, custom_grounding=ag_grounding))

                    # print("LIST", list_)
                    # print("GR SIMPLE", grounding_simple)
                    # print("VREW", all_trajs_ag[i].v_rews[vi])
                    # np.testing.assert_almost_equal(np.asarray(list_, dtype=parser_args.dtype), grounding_simple, decimal = 5 if parser_args.dtype in [np.float32, np.float64] else 3)

                    np.testing.assert_almost_equal(
                        np.asarray(list_, dtype=parser_args.dtype), all_trajs_ag[i].v_rews[vi], decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                    assert len(grounding_simple) == len(all_trajs_ag[i].acts)
                    discounted_sums_per_grounding[vi, i] = imitation.data.rollout.discounted_sum(
                        all_trajs_ag[i].v_rews[vi], gamma=alg_config['discount_factor_preferences'])
            # We save the comparison of 1 vs 2, 2 vs 3 in the order stablished in discounted_sums.

            save_preferences(idxs=idxs, discounted_sums=discounted_sums, discounted_sums_per_grounding=discounted_sums_per_grounding,
                             dataset_name=dataset_name, epsilon=parser_args.reward_epsilon, environment_data=environment_data, society_data=society_data, ag=ag)

    # TEST preferences load okey.
    print("TESTING DATA COHERENCE. It is safe to stop this program now...")
    for i, ag in enumerate(society_data['agents']):
        agg = tuple(ag['grounding'])
        #  Here idxs is the list of trajectory PAIRS of indices from the trajectory list that are compared.
        idxs, discounted_sums, discounted_sums_per_grounding, preferences, preferences_per_grounding = load_preferences(
            epsilon=parser_args.reward_epsilon, dataset_name=dataset_name, environment_data=environment_data, society_data=society_data, ag=ag, dtype=parser_args.dtype)

        trajs_ag = load_trajectories(dataset_name=dataset_name, ag=ag,
                                     environment_data=environment_data, society_data=society_data, override_dtype=parser_args.dtype)

        if society_data["same_trajectories_for_each_agent_type"]:
            for t in range(ag['n_agents']-2):
                np.testing.assert_allclose(idxs[0:ag['data']['trajectory_pairs']], (idxs[(
                    t+1)*ag['data']['trajectory_pairs']:(t+2)*ag['data']['trajectory_pairs']] - ag['data']['trajectory_pairs']*(t+1)))

                for traj_i in range(ag['data']['trajectory_pairs']):

                    np.testing.assert_allclose(trajs_ag[traj_i + t*ag['data']['trajectory_pairs']].obs, trajs_ag[(
                        t+1)*ag['data']['trajectory_pairs'] + traj_i].obs)

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
                    tabular_all_agent_groundings[agg][trajs_ag[i].obs[:-1], trajs_ag[i].acts][:, vi], gamma=alg_config['discount_factor_preferences'])), decimal = 5 if parser_args.dtype in [np.float32, np.float64] else 3)
                """
            for idx, pr in zip(idxs, preferences_per_grounding[:, vi]):
                idx = [int(ix) for ix in idx]

                np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[0]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                    trajs_ag[idx[0]].v_rews[vi], gamma=alg_config['discount_factor_preferences'])), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[1]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                    trajs_ag[idx[1]].v_rews[vi], gamma=alg_config['discount_factor_preferences'])), decimal=5 if parser_args.dtype in [np.float32, np.float64] else 3)

                """if not environment_data['is_contextual']:
                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[0]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        tabular_all_agent_groundings[agg][trajs_ag[idx[0]].obs[:-1], trajs_ag[idx[0]].acts][:, vi], gamma=alg_config['discount_factor_preferences'])), decimal = 5 if parser_args.dtype in [np.float32, np.float64] else 3)
                    np.testing.assert_almost_equal(discounted_sums_per_grounding[vi, idx[1]], parser_args.dtype(imitation.data.rollout.discounted_sum(
                        tabular_all_agent_groundings[agg][trajs_ag[idx[1]].obs[:-1], trajs_ag[idx[1]].acts][:, vi], gamma=alg_config['discount_factor_preferences'])), decimal = 5 if parser_args.dtype in [np.float32, np.float64] else 3)
                """
                np.testing.assert_almost_equal(compare_trajectories(
                    discounted_sums_per_grounding[vi, idx[0]], discounted_sums_per_grounding[vi, idx[1]], epsilon=parser_args.reward_epsilon), pr, decimal=4 if parser_args.dtype in [np.float32, np.float64] else 3)

    print("Dataset generated correctly.")
    dataset = create_dataset(parser_args, config, society_data,
                             default_groundings=society_config[parser_args.environment]['groundings'],save=True)
    
