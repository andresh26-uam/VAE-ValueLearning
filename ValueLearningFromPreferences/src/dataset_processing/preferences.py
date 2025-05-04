import csv
from typing import Tuple
import torch
from src.dataset_processing.utils import COMPARISONS_DATASETS_PATH, calculate_preferences_save_path
import numpy as np
import os
from src.dataset_processing.trajectories import compare_trajectories


def save_preferences(idxs: np.ndarray, discounted_sums: np.ndarray, discounted_sums_per_grounding: np.ndarray, epsilon: float, dataset_name, ag, environment_data, society_data, real_preference=None, real_grounding_preference=None):
    path = calculate_preferences_save_path(
        dataset_name, ag, environment_data, society_data, epsilon)
    path = os.path.join(COMPARISONS_DATASETS_PATH, path)
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, 'agent_preferences_file.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Traj1', 'Traj2', 'CR1', 'CR2', 'Flag']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(discounted_sums)-1):
            if real_preference is not None:
                if (idxs[i], idxs[i+1]) not in real_preference.keys():
                    continue
            traj_i = discounted_sums[idxs[i]]
            traj_j = discounted_sums[idxs[i+1]]
            comparison_flag = compare_trajectories(
                traj_i, traj_j, epsilon=epsilon)
            if real_preference is not None:
                comparison_flag_real = float(
                    real_preference[(idxs[i], idxs[i+1])])

                assert comparison_flag_real == comparison_flag
            writer.writerow({'Traj1': idxs[i], 'Traj2': idxs[(
                i+1)], 'CR1': traj_i, 'CR2': traj_j, 'Flag': comparison_flag})

    for vi in range(discounted_sums_per_grounding.shape[0]):
        csv_path = os.path.join(path, f'value_{vi}_preferences_file.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Traj1', 'Traj2', 'CR1', 'CR2', 'Flag']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(discounted_sums_per_grounding[vi])-1):
                traj_i = discounted_sums_per_grounding[vi][idxs[i]]
                traj_j = discounted_sums_per_grounding[vi][idxs[(
                    i+1)]]
                comparison_flag = compare_trajectories(
                    traj_i, traj_j, epsilon=epsilon)

                if real_grounding_preference is not None:
                    if (idxs[i], idxs[i+1]) in real_grounding_preference[vi].keys():
                        comparison_flag_real = float(
                            real_grounding_preference[vi][(idxs[i], idxs[i+1])])
                        writer.writerow({'Traj1': idxs[i], 'Traj2': idxs[(
                            i+1)], 'CR1': traj_i, 'CR2': traj_j, 'Flag': comparison_flag_real})
                else:
                    writer.writerow({'Traj1': idxs[i], 'Traj2': idxs[(
                        i+1)], 'CR1': traj_i, 'CR2': traj_j, 'Flag': comparison_flag})


def load_preferences(dataset_name, ag, environment_data, society_data, epsilon, dtype=np.float32, debug_grounding=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = calculate_preferences_save_path(
        dataset_name, ag, environment_data, society_data, epsilon=epsilon)
    path = os.path.join(COMPARISONS_DATASETS_PATH, path)
    csv_path = os.path.join(path, 'agent_preferences_file.csv')
    idxs = []
    preferences = []

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        max_traj_idx = 0
        for row in reader:
            traj1 = int(row['Traj1'])
            traj2 = int(row['Traj2'])
            max_traj_idx = max(max(traj1, traj2), max_traj_idx)
            comparison_flag = float(row['Flag'])
            idxs.append([traj1, traj2])
            preferences.append(comparison_flag)

        if isinstance(dtype, torch.dtype):
            discounted_sums = torch.zeros(
                (max_traj_idx+1,), dtype=dtype, requires_grad=False)
        elif np.issubdtype(dtype, np.floating):
            discounted_sums = np.zeros((max_traj_idx+1,), dtype=dtype)
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for ir, row in enumerate(reader):
            traj1 = int(row['Traj1'])
            traj2 = int(row['Traj2'])
            cr1 = float(row['CR1'])
            cr2 = float(row['CR2'])
            comparison_flag = float(row['Flag'])

            if discounted_sums[traj1] != 0.0:
                assert discounted_sums[traj1] == cr1
            discounted_sums[traj1] = cr1

            if discounted_sums[traj2] != 0.0:
                assert discounted_sums[traj2] == cr2
            assert comparison_flag == preferences[ir]
            discounted_sums[traj2] = cr2
            assert compare_trajectories(cr1, cr2, epsilon) == preferences[ir]

    if isinstance(dtype, torch.dtype):
        discounted_sums_per_grounding = torch.zeros(
            (len(environment_data['basic_profiles']), discounted_sums.shape[0]), dtype=dtype)
        preferences_per_grounding = torch.zeros((len(preferences),
                                                 len(environment_data['basic_profiles'])), dtype=dtype)
    elif np.issubdtype(dtype, np.floating):
        discounted_sums_per_grounding = np.zeros(
            (len(environment_data['basic_profiles']), discounted_sums.shape[0]), dtype=dtype)
        preferences_per_grounding = np.zeros(
            (len(preferences), len(environment_data['basic_profiles'])), dtype=dtype)

    for vi in range(len(environment_data['basic_profiles'])):

        csv_path = os.path.join(path, f'value_{vi}_preferences_file.csv')
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for ir, row in enumerate(reader):
                traj1 = int(row['Traj1'])
                traj2 = int(row['Traj2'])
                cr1 = float(row['CR1'])
                cr2 = float(row['CR2'])
                comparison_flag = float(row['Flag'])

                preferences_per_grounding[ir, vi] = comparison_flag
                if discounted_sums_per_grounding[vi, traj1] != 0.0:
                    assert discounted_sums_per_grounding[vi, traj1] == cr1
                discounted_sums_per_grounding[vi, traj1] = cr1

                if discounted_sums_per_grounding[vi, traj2] != 0.0:
                    assert discounted_sums_per_grounding[vi, traj2] == cr2

                discounted_sums_per_grounding[vi, traj2] = cr2

                if debug_grounding:
                    assert compare_trajectories(
                    cr1, cr2, epsilon) == preferences_per_grounding[ir, vi]
    if isinstance(dtype, torch.dtype):
        return np.array(idxs, dtype=np.int_), discounted_sums, discounted_sums_per_grounding, torch.tensor(preferences, dtype=dtype, requires_grad=False), preferences_per_grounding
    else:
        return np.array(idxs, dtype=np.int_), discounted_sums, discounted_sums_per_grounding, np.array(preferences, dtype=dtype), preferences_per_grounding