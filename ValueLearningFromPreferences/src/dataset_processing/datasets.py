from collections import defaultdict
import os

from src.dataset_processing.utils import COMPARISONS_DATASETS_PATH, DATASETS_PATH, calculate_dataset_save_path
from src.dataset_processing.data import TrajectoryWithValueSystemRewsPair, VSLPreferenceDataset
from src.dataset_processing.preferences import load_preferences
from src.dataset_processing.trajectories import load_trajectories

import numpy as np
from typing import Sequence


def create_dataset(parser_args, config, society_data={'name': "default", "same_trajectories_for_each_agent_type": False}, train_or_test=None, default_groundings=None, debug_grounding=False, save=True):
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
        
    for i, ag in enumerate(agents):
        if 'agents' not in society_data.keys():
            ag = {'agent_id': ag, 'name': ag, 'value_system': 'unk', 'data': defaultdict(
                lambda: 'nd'), 'n_agents': 1, 'grounding': list(default_groundings.keys())}
        #  Here idxs is the list of trajectory PAIRS of indices from the trajectory list that are compared.
        idxs, discounted_sums, discounted_sums_per_grounding, preferences, preferences_per_grounding = load_preferences(
            epsilon=parser_args.reward_epsilon, dataset_name=dataset_name, environment_data=environment_data, society_data=society_data, ag=ag, dtype=parser_args.dtype, debug_grounding=debug_grounding)
        trajs_ag = np.asarray(load_trajectories(dataset_name=dataset_name,
                              ag=ag, environment_data=environment_data, society_data=society_data,  override_dtype=parser_args.dtype))

        for t in range(ag['n_agents']-2):
            np.testing.assert_allclose(idxs[0:ag['data']['trajectory_pairs']], (idxs[(
                t+1)*ag['data']['trajectory_pairs']:(t+2)*ag['data']['trajectory_pairs']] - ag['data']['trajectory_pairs']*(t+1)))

            for traj_i in range(ag['data']['trajectory_pairs']):

                np.testing.assert_allclose(trajs_ag[traj_i + t*ag['data']['trajectory_pairs']].obs, trajs_ag[(
                    t+1)*ag['data']['trajectory_pairs'] + traj_i].obs)

        ag_point = 0
        n_pairs_per_agent = len(idxs)//ag['n_agents']
        for id in range(ag['n_agents']):
            agent_id = ag['name']+'_'+str(id)
            ag_idxs = idxs[ag_point:ag_point+n_pairs_per_agent]

            trajectory_pairs: Sequence[TrajectoryWithValueSystemRewsPair] = trajs_ag[ag_idxs]
            dataset.push(trajectory_pairs, preferences[ag_point:ag_point+n_pairs_per_agent], preferences_per_grounding[ag_point:(
                ag_point+n_pairs_per_agent)], agent_ids=[agent_id]*n_pairs_per_agent, agent_data={agent_id: ag})
            
            ag_point += n_pairs_per_agent
            # last_agent_id = agent_id
            # las_agent_name = ag['name']
    if save:
        path = os.path.join(
            DATASETS_PATH, calculate_dataset_save_path(dataset_name_pure, environment_data, society_data, epsilon=parser_args.reward_epsilon))
        os.makedirs(path, exist_ok=True)
        dataset.save(os.path.join(
            path, f"{'dataset_train' if train_or_test == 'train' else 'dataset_test' if train_or_test == 'test' else 'dataset'}.pkl"))
        print(f"Dataset saved at {path}")
    return dataset
