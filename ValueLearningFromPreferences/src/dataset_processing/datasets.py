from collections import defaultdict
import os

from src.dataset_processing.utils import COMPARISONS_DATASETS_PATH, DATASETS_PATH, calculate_dataset_save_path
from src.dataset_processing.data import TrajectoryWithValueSystemRewsPair, VSLPreferenceDataset
from src.dataset_processing.preferences import load_preferences
from src.dataset_processing.trajectories import load_trajectories

import numpy as np
from typing import Sequence



def create_dataset(parser_args, config, society_data={'name': "default", "same_trajectories_for_each_agent_type": False}, train_or_test=None, default_groundings=None, debug_grounding=False,save=True,
                   split_ratio=0.25):
    environment_data = config[parser_args.environment]

    dataset_name = parser_args.dataset_name
    dataset_name_pure = dataset_name
    if train_or_test is not None:
        dataset_name += '_'
        assert train_or_test == 'train' or train_or_test == 'test'
        dataset_name += train_or_test

    dataset = VSLPreferenceDataset(n_values=environment_data['n_values'])

    if 'agents' not in society_data.keys():
        agents = []
        # TODO: HERE; THIS PATH IS NOT CORRECT!!
        folder_path = os.path.join(
            COMPARISONS_DATASETS_PATH, f"{environment_data['name']}/{society_data['name']}/{dataset_name}")
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                if dir_name.startswith("prefs_ag_"):
                    ag_name = dir_name.split("_")[2]
                    agents.append(ag_name)
    else:
        agents = society_data['agents']
    print("CRETING DATASET FOR AGENTS: ", agents)
    for i, ag in enumerate(agents):
        if 'agents' not in society_data.keys():
            ag = {'agent_id': ag, 'name': ag, 'value_system': 'unk', 'data': defaultdict(
                lambda: 'nd'), 'n_agents': 1, 'grounding': list(default_groundings.keys())}
        #  Here idxs is the list of trajectory PAIRS of indices from the trajectory list that are compared.
        idxs, discounted_sums, discounted_sums_per_grounding, preferences, preferences_per_grounding = load_preferences(
            epsilon=parser_args.reward_epsilon, dataset_name=dataset_name, environment_data=environment_data, society_data=society_data, ag=ag, dtype=parser_args.dtype, debug_grounding=debug_grounding)
        trajs_ag = np.asarray(load_trajectories(dataset_name=dataset_name,
                              ag=ag, environment_data=environment_data, society_data=society_data,  override_dtype=parser_args.dtype))
        
        if train_or_test is None:
            trajs_ag_all = trajs_ag 
        else:
            ttrain = load_trajectories(dataset_name=dataset_name_pure+'_train',
                              ag=ag, environment_data=environment_data, society_data=society_data,  override_dtype=parser_args.dtype)
            ttest = load_trajectories(dataset_name=dataset_name_pure+'_test',
                              ag=ag, environment_data=environment_data, society_data=society_data,  override_dtype=parser_args.dtype)
            trajs_ag_all = np.concatenate((ttrain, ttest), axis=0)
        n_pairs_per_agent = len(trajs_ag)//ag['n_agents']
        n_pairs_prime = n_pairs_per_agent
        #print(ag['data']['trajectory_pairs'], n_pairs_per_agent, len(trajs_ag))
        #exit(0)

        if society_data['same_trajectories_for_each_agent_type']:
            for t in range(ag['n_agents']-2):
                np.testing.assert_allclose(idxs[0:n_pairs_prime], (idxs[(
                    t+1)*n_pairs_prime:(t+2)*n_pairs_prime] - n_pairs_prime*(t+1)))

                for traj_i in range(n_pairs_prime):

                            np.testing.assert_allclose(trajs_ag[traj_i + t*n_pairs_prime].obs, trajs_ag[(
                                t+1)*n_pairs_prime + traj_i].obs)

        ag_point = 0
        
        for id in range(ag['n_agents']):
            agent_id = ag['name']+'_'+str(id)
            ag_idxs = idxs[ag_point:ag_point+n_pairs_per_agent]

            trajectory_pairs: Sequence[TrajectoryWithValueSystemRewsPair] = trajs_ag[ag_idxs]
            dataset.push(trajectory_pairs, preferences[ag_point:ag_point+n_pairs_per_agent], preferences_per_grounding[ag_point:(
                ag_point+n_pairs_per_agent)], agent_ids=[agent_id]*n_pairs_per_agent, agent_data={agent_id: ag})
            """if society_data["same_trajectories_for_each_agent_type"] and ag_point > 0:
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
"""
            ag_point += n_pairs_per_agent
            # last_agent_id = agent_id
            # las_agent_name = ag['name']

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
    if save:
        path = os.path.join(
        DATASETS_PATH, calculate_dataset_save_path(dataset_name_pure, environment_data, society_data, epsilon=parser_args.reward_epsilon))
        os.makedirs(path, exist_ok=True)
        dataset.save(os.path.join(path, f"{"dataset_train" if train_or_test == "train" else "dataset_test" if train_or_test=='test' else "dataset"}.pkl"))
    return dataset