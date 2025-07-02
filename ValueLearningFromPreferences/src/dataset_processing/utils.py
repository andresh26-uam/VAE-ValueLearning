import os

DATASETS_PATH = 'datasets/complete_datasets/'
TRAJECTORIES_DATASETS_PATH = 'datasets/trajectories/'
COMPARISONS_DATASETS_PATH = 'datasets/comparisons/'

GROUNDINGS_PATH = 'groundings/'

DEFAULT_SEED = 26

USEINFO = True


def calculate_dataset_save_path(dataset_name, environment_data, society_data, epsilon=None):
    path = f"{environment_data['name']}/{society_data['name']}/{dataset_name}/"
    if epsilon is not None:
        path = os.path.join(path, f"reps_{epsilon}/")
    return path

def calculate_preferences_save_path(dataset_name, ag, environment_data, society_data, epsilon):

    return os.path.join(calculate_dataset_save_path(dataset_name, environment_data, society_data, epsilon), f"prefs_ag_{ag['name']}_{ag['value_system']}")


def calculate_trajectory_save_path(dataset_name, ag, environment_data, society_data):
    return os.path.join(calculate_dataset_save_path(dataset_name, environment_data, society_data), f"trajs_ag_{ag['name']}_{ag['value_system']}_rp_{ag['data']['random_traj_proportion']}_rat_{ag['data']['rationality']}")


def calculate_expert_policy_save_path(environment_name, society_name, dataset_name, class_name, grounding_name):
    return f'{environment_name}/{society_name}/{dataset_name}/{class_name}/expert_policy_G_{grounding_name}'
def calculate_learned_policy_save_path(environment_name, society_name, dataset_name, class_name, grounding_name):
    return f'{environment_name}/{society_name}/{dataset_name}/{class_name}/learned_policy_G_{grounding_name}'
