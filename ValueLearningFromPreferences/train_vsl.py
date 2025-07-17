import argparse
import time
import dill
import pprint
import random
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
from defines import CHECKPOINTS, TRAIN_RESULTS_PATH
from envs.tabularVAenv import TabularVAMDP
from generate_dataset import parse_dtype_torch
from src.algorithms.preference_based_vsl_lib import ConstrainedOptimizer, SobaOptimizer
from src.dataset_processing.utils import DATASETS_PATH, DEFAULT_SEED, GROUNDINGS_PATH
from generate_dataset import PICKLED_ENVS
from src.algorithms.clustering_utils import ClusterAssignment
from src.algorithms.preference_based_vsl import PreferenceBasedClusteringTabularMDPVSL, load_historic_assignments
from src.dataset_processing.datasets import calculate_dataset_save_path
from src.feature_extractors import ContextualFeatureExtractorFromVAEnv, FeatureExtractorFromVAEnv, OneHotFeatureExtractor
from src.reward_nets.vsl_reward_functions import AbstractVSLRewardFunction, ConvexAlignmentLayer, LinearVSLRewardFunction, TrainingModes, parse_layer_name
from src.dataset_processing.data import TrajectoryWithValueSystemRews, VSLPreferenceDataset
import os
from src.utils import filter_none_args, load_json_config


def parse_feature_extractors(environment, environment_data, dtype=torch.float32):
    # Dummy implementation, replace with actual logic
    if environment_data['reward_feature_extractor'] == "FeatureExtractorFromVAEnv":
        reward_net_features_extractor_class = FeatureExtractorFromVAEnv
        reward_net_features_extractor_kwargs = dict(
            env=environment,
            dtype=dtype,
        )
    elif environment_data['reward_feature_extractor'] == "ContextualFeatureExtractorFromVAEnv":
        reward_net_features_extractor_class = ContextualFeatureExtractorFromVAEnv
        reward_net_features_extractor_kwargs = dict(
            env=environment,
            dtype=dtype,
        )

    else:
        raise ValueError(
            f"Unknown reward feature extractor {environment_data['reward_feature_extractor']}")
    # "reward_feature_extractor": "FeatureExtractorFromVAEnv",
    # "policy_state_feature_extractor": "OneHotFeatureExtractor",
    if environment_data['policy_state_feature_extractor'] == "OneHotFeatureExtractor":
        policy_features_extractor_class = OneHotFeatureExtractor
        policy_features_extractor_kwargs = dict(
            n_categories=environment.action_dim,
            dtype=dtype)
    else:
        raise ValueError(
            f"Unknown policy feature extractor {environment_data['reward_feature_extractor']}")
    return reward_net_features_extractor_class, policy_features_extractor_class, reward_net_features_extractor_kwargs, policy_features_extractor_kwargs


def parse_optimizer_data(environment_data, alg_config):
    opt_kwargs = environment_data['default_optimizer_kwargs']
    opt_kwargs = opt_kwargs if alg_config['optimizer_kwargs'] == "default" else alg_config['optimizer_kwargs']

    opt_class = environment_data['default_optimizer_class']
    opt_class = opt_class if alg_config['optimizer_class'] == "default" else alg_config['optimizer_class']

    if opt_class == 'Adam':
        opt_class = torch.optim.Adam
    elif opt_class == 'Soba':
        opt_class = SobaOptimizer
    elif opt_class == 'lagrange':
        opt_class = ConstrainedOptimizer
    else:
        raise ValueError(f"Unknown optimizer class {opt_class}")
    return opt_kwargs, opt_class


def save_training_results(experiment_name, target_agent_and_vs_to_learned_ones, reward_net_pair_agent_and_vs, metrics, parser_args):
    # Save the training results to a file
    os.makedirs(TRAIN_RESULTS_PATH, exist_ok=True)

    with open(os.path.join(TRAIN_RESULTS_PATH, f"{experiment_name}.pkl"), 'wb') as f:
        dill.dump({
            "target_agent_and_vs_to_learned_ones": target_agent_and_vs_to_learned_ones,
            "reward_net_pair_agent_and_vs": reward_net_pair_agent_and_vs,
            "metrics": metrics,
            "parser_args": parser_args,
        }, f)

    print(
        f"Training results saved to {os.path.join(CHECKPOINTS, f'{experiment_name}.pkl')}")


def load_training_results(experiment_name) -> Tuple[Tuple[Dict[Tuple[str, Tuple], Tuple],
                                                          Dict[Tuple[str, Tuple],
                                                               AbstractVSLRewardFunction],
                                                          Dict[str, Any]],
                                                    List[ClusterAssignment],
                                                    TabularVAMDP,
                                                    int]:
    # Load the training results from a file
    file_path, experiment_name = find_parse_ename(experiment_name)
    with open(file_path, 'rb') as f:
        data = dill.load(f)
    print(f"Training results loaded from {file_path}")

    returned_tuple = [None, None, None, None]
    for k in data.keys():
        if k == "target_agent_and_vs_to_learned_ones":
            returned_tuple[0] = data[k]
        elif k == "reward_net_pair_agent_and_vs":
            returned_tuple[1] = data[k]
        elif k == "metrics":
            returned_tuple[2] = data[k]
        elif k == "parser_args":
            returned_tuple[3] = data[k]
    returned_tuple = tuple(returned_tuple)
    # Get the saved best assignments per iteration
    historic_assignments, env_state, n_iterations_real = load_historic_assignments(
        experiment_name, sample=20)
    return *returned_tuple, historic_assignments, env_state, n_iterations_real


def find_parse_ename(experiment_name: str):

    file_path = os.path.join(TRAIN_RESULTS_PATH, (
        f"{experiment_name}.pkl" if 'pkl' not in experiment_name else f"{experiment_name}"))
    if not os.path.exists(file_path):
        matching_files = [f for f in os.listdir(
            TRAIN_RESULTS_PATH) if f.startswith(experiment_name)]
        if not matching_files:
            raise FileNotFoundError(
                f"Training results file not found: {file_path} or any file starting with {experiment_name}")
        file_path = os.path.join(TRAIN_RESULTS_PATH, matching_files[0])
        experiment_name = matching_files[0].strip('.pkl')
    return file_path, experiment_name


def parse_cluster_sizes(k, n_values):
    if isinstance(k, int):
        return [k]*n_values
    elif isinstance(k, list):
        assert len(k) == n_values
        return k
    else:
        raise ValueError(f"Number of clusters not identifiable {k}")


def parse_args():
    # IMPORTANT: Default Args are specified depending on the environment in config.json

    parser = argparse.ArgumentParser(
        description="This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected. See the societies.json and algorithm_config.json files")

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument('-dname', '--dataset_name', type=str,
                               default='test_dataset', required=True, help='Dataset name')
    """general_group.add_argument('-sname', '--society_name', type=str, default='default',
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')
"""
    general_group.add_argument('-ename', '--experiment_name', type=str,
                               default='test_experiment', required=True, help='Experiment name')

    general_group.add_argument('-dtype', '--dtype', type=parse_dtype_torch, default=torch.float32, choices=[torch.float32, torch.float64],
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')

    general_group.add_argument(
        '-s', '--seed', type=int, default=DEFAULT_SEED, required=False, help='Random seed')

    general_group.add_argument('-a', '--algorithm', type=str, choices=[
                               'pc','rlhf'], default='pc', help='Algorithm to use (preference comparison - pc) or vanilla RLHF based loss fit with 1 cluster - rlhf')

    general_group.add_argument('-cf', '--config_file', type=str, default='algorithm_config.json',
                               help='Path to JSON general configuration file (overrides other defaults here, but not the command line arguments)')
    """ general_group.add_argument('-sf', '--society_file', type=str, default='societies.json',
                               help='Path to JSON society configuration file (overrides other defaults here, but not the command line arguments)')"""

    general_group.add_argument('-sh', '--show', action='store_true', default=False,
                               help='Show plots calculated before saving')

    general_group.add_argument('-e', '--environment', type=str, default='apollo', choices=[
                               'apollo'], help='environment (apollo)')

    general_group.add_argument('-df', '--discount_factor', type=float, default=1.0,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')

    general_group.add_argument('-sp', '--split_ratio', type=float,
                               default=0.0, help='Split ratio for train/test set. 0.2 means 80% train, 20% test')
    alg_group = parser.add_argument_group('Algorithm-specific Parameters')
    alg_group.add_argument('-k', '--k_clusters', type=Union[int, list], default=-1,
                           help="Number of clusters per value (overriging configuration file)")

    debug_params = parser.add_argument_group('Debug Parameters')
    debug_params.add_argument('-db', '--debug_mode', action='store_true',
                              default=False, help='Debug Mode')

    env_group = parser.add_argument_group('environment-specific Parameters')

    env_group.add_argument('-appr', '--approx_expert', action='store_true',
                           default=False, help='Approximate expert (roadworld)')
    env_group.add_argument('-reps', '--reward_epsilon', default=0.000, type=float,
                           help='Distance between the cummulative rewards of each pair of trajectories for them to be considered as equal in the comparisons')

    return parser.parse_args()


def rlhfFit(dataset_train: VSLPreferenceDataset, v, lr, hidden_layer_sizes=(16,24,16), maxiter=1000000, net_vs=None):
    matrix = list()
    items = set()
    f1: TrajectoryWithValueSystemRews
    agent_ids = dataset_train.agent_ids
    
    preferences = list( dataset_train.preferences_with_grounding[:,v]) if v != 'vs' else dataset_train.preferences
    pair_to_agent_ids = [agent_ids[if1] for if1, p1 in enumerate(preferences)]
    print("NPREFS", len(preferences))

    for f1,f2 , pr in zip(list(dataset_train.fragments1), list(dataset_train.fragments2),preferences):
        assert isinstance(pr, float)
                #print(dataset_train.preferences_with_grounding.shape)
                #print(pr)
        if pr == 0.5:
            continue
        items.add(f1.infos[0]['state'])
        items.add(f2.infos[0]['state'])
        
        if pr <= 0.5:
            matrix.append((f2.infos[0]['state'], f1.infos[0]['state']))
                    
        elif pr >= 0.5:
            matrix.append((f1.infos[0]['state'], f2.infos[0]['state']))
        #matrixnp = np.array(matrix, dtype=object)
                #print(matrixnp.shape)
    params = choix.mm_pairwise(max(items)+1, data=matrix, alpha=1.0, max_iter=maxiter,tol=1e-16)
    print(params.shape)
    pair_to_agent_ids = [agent_ids[if1] for if1, p1 in enumerate(dataset_train.preferences)]
    acc = accuracy(matrix, params, pair_to_agent_ids)
    print(f"Accuracy for {v} is {acc}")
            # Prepare data for regression
    X = np.zeros((len(dataset_train.fragments1) + len(dataset_train.fragments2), 4))
    y = np.zeros((len(dataset_train.fragments1) + len(dataset_train.fragments2),))
    for f in list(dataset_train.fragments1) + list(dataset_train.fragments2):
                # Extract features from f.obs (flatten if needed)
        obs = np.array(f.obs).flatten()
        state_idx = f.infos[0]['state']
        X[state_idx] = obs[0:4]
                # Target is the corresponding value in params for the fragment's state
        
        y[state_idx] = params[state_idx]

    X = np.array(X)
    y = np.array(y)

            # Fit regression model
            
            # Fit a simple neural network regressor
    import torch.nn as nn
    import torch.optim as optim

    class PreferenceNet(nn.Module):
        def __init__(self, input_dim, v, hidden_layer_sizes, nets_grounding=None, freeze_grounding=False):
            super().__init__()
            if v != 'vs':
                layers = []
                prev_dim = input_dim
                for ih, h in enumerate(hidden_layer_sizes):
                    layers.append(nn.Linear(prev_dim, h))
                    if h == dataset_train.n_values and v=='vs':
                        layers.append(nn.Softplus())
                    else:
                        layers.append(nn.Tanh())
                    prev_dim = h
            
                layers.append(nn.Linear(prev_dim, 1))
                layers.append(nn.Softplus())
                self.net = nn.Sequential(*layers)
                self.negative=True
            else:
                if nets_grounding is None:
                    nets_grounding = []
                    for v_ in range(dataset_train.n_values):
    
                        nets_grounding.append(PreferenceNet(input_dim, v=v_, hidden_layer_sizes=hidden_layer_sizes[:-1],nets_grounding=None, freeze_grounding=False))
                assert len(nets_grounding) == dataset_train.n_values, "Number of nets must match the number of value systems"
                self.nets_grounding = nn.ModuleList(nets_grounding)
                if freeze_grounding:
                        for net in self.nets_grounding:
                            for param in net.parameters():
                                param.requires_grad = False
                self.final_linear = ConvexAlignmentLayer(len(self.nets_grounding), 1, bias=False, dtype=torch.float32)
                # Compose self.net: outputs of net_vs -> final_linear
                def combined_net(x):
                    outputs = [net(x) for net in self.nets_grounding]
                    stacked = torch.stack(outputs, dim=1)
                    return self.final_linear(stacked)
                self.net = combined_net
                self.negative=False
                

        def forward(self, x):
            return -self.net(x).squeeze(-1) if self.negative else self.net(x).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    matrix_tensor = torch.tensor(matrix, dtype=torch.long, device=device)

    net = PreferenceNet(X.shape[1], v=v, hidden_layer_sizes=hidden_layer_sizes, nets_grounding=net_vs, freeze_grounding=(net_vs is not None)).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # Targets are always 1.0: p1 should be preferred over p2
    targets = torch.ones(len(matrix), dtype=torch.float32, device=device)

    for epoch in range(maxiter):
        optimizer.zero_grad()
        p1_idx = matrix_tensor[:, 0]
        p2_idx = matrix_tensor[:, 1]
        out1 = net(X_tensor[p1_idx])
        out2 = net(X_tensor[p2_idx])
        logits = out1 - out2
        preds = torch.sigmoid(logits)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Use the trained network to estimate params
    estimated_params_v = None
    with torch.no_grad():
        estimated_params = net(X_tensor).detach().cpu().numpy()
    if v=='vs':
        with torch.no_grad():
            estimated_params_v = [net_v(X_tensor).detach().cpu().numpy()  for net_v in net.nets_grounding]
            print("Estimated params vector for v:", estimated_params_v)
    print("Estimated params vector:", estimated_params)

    acc = accuracy(matrix, estimated_params, pair_to_agent_ids)
    print(f"Accuracy estimated for {v} is {acc}")

            #print("Regression coefficients:", reg.coefs_)
            #print("Regression intercept:", reg.intercept_)
            # Create the estimated version of the params vector using the regression model
    """reg = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=0.001, activation='tanh', max_iter=200000, random_state=parser_args.seed)
    reg.fit(X, y)
    estimated_params = reg.predict(X)
    print("Estimated params vector:", estimated_params)
    counts = 0
    for p1, p2 in matrix:
        p =  choix.probabilities((p1,p2), estimated_params)
        correct = p[0] > 0.5
        if correct:
            counts += 1
    print(f"Accuracy estimated for {v} is {counts/len(matrix)}")
    print (params)"""
    return net, matrix, estimated_params, estimated_params_v

def accuracy(matrix, estimated_params, pair_to_agent_ids):
    counts = 0
    counts_by_agent = {agent_id: 0 for agent_id in set(pair_to_agent_ids)}
    n_by_agent = {agent_id: 0 for agent_id in set(pair_to_agent_ids)}
    
    for ip, (p1, p2) in enumerate(matrix):
        agent_id = pair_to_agent_ids[ip]
        p = choix.probabilities((p1, p2), estimated_params)
        correct = p[0] > 0.5
        if correct:
            counts += 1
            counts_by_agent[agent_id] += 1
        n_by_agent[agent_id] += 1
        
    #acc = counts/len(matrix)
    representativeness = np.mean([counts_by_agent[agent_id] / n_by_agent[agent_id] for agent_id in n_by_agent.keys() if n_by_agent[agent_id] > 0])
    # Because each agent ranks the same number of pairs, the representativeness (per agent avg accuracy) is the same as full-batch accuracy
    return representativeness

if __name__ == "__main__":
    # This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected
    # IMPORTANT: Default Args are specified depending on the environment in config.json
    parser_args = filter_none_args(parse_args())
    # If a config file is specified, load it and override command line args
    config = load_json_config(parser_args.config_file)
    society_config = load_json_config('societies.json')

    pprint.pprint(parser_args)
    np.random.seed(parser_args.seed)
    torch.manual_seed(parser_args.seed)
    random.seed(parser_args.seed)
    rng_for_algorithms = np.random.default_rng(parser_args.seed)

    environment_data = config[parser_args.environment]
    society_data = society_config[parser_args.environment]['default']
    parser_args.society_name = 'default'
    grounding_path = os.path.join(
        'envs', parser_args.environment, GROUNDINGS_PATH)
    dataset_name = parser_args.dataset_name
    experiment_name = parser_args.experiment_name

    experiment_name = experiment_name  # + '_' + str(parser_args.split_ratio)

    """agent_profiles = [tuple(ag['value_system'])
                      for ag in society_data['agents']]
    agent_groundings = [tuple(ag['grounding'])
                        for ag in society_data['agents']]
    ag_name_to_aggrounding = {ag['name']: tuple(
        ag['grounding']) for ag in society_data['agents']}
    grounding_files = society_config[parser_args.environment]['groundings']"""
    """all_agent_groundings_to_save_files = dict(
        {agg: [grounding_files[agg[i]] for i in range(len(agg))] for agg in set(agent_groundings)})"""

    extra_kwargs = {}

    if parser_args.environment == 'apollo':
        extra_kwargs = {
            'test_size': parser_args.split_ratio
        }

    try:
        f = open(os.path.join(os.path.join(
            PICKLED_ENVS, environment_data['name'], dataset_name), f"env_kw_{extra_kwargs}.pkl"), 'rb')
        environment = dill.load(f)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print("TESTING DATA COHERENCE. It is safe to stop this program now...")

    rewward_net_features_extractor_class, policy_features_extractor_class, features_extractor_kwargs, policy_features_extractor_kwargs = parse_feature_extractors(
        environment, environment_data, dtype=parser_args.dtype)

    alg_config = environment_data['algorithm_config'][parser_args.algorithm]

    data_reward_net = environment_data['default_reward_net']
    data_reward_net.update(alg_config['reward_net'])

    reward_net = LinearVSLRewardFunction(
        environment=environment,
        use_state=data_reward_net['use_state'],
        use_action=data_reward_net['use_action'],
        use_next_state=data_reward_net['use_next_state'],
        use_done=data_reward_net['use_done'],
        hid_sizes=data_reward_net['hid_sizes'],
        reward_bias=0,
        basic_layer_classes=[parse_layer_name(
            l) for l in data_reward_net['basic_layer_classes']],
        use_one_hot_state_action=False,
        activations=[parse_layer_name(l)
                     for l in data_reward_net['activations']],
        negative_grounding_layer=data_reward_net['negative_grounding_layer'],
        use_bias=data_reward_net['use_bias'],
        clamp_rewards=data_reward_net['clamp_rewards'],
        mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
        features_extractor_class=rewward_net_features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
        action_features_extractor_class=policy_features_extractor_class,
        action_features_extractor_kwargs=policy_features_extractor_kwargs,
        dtype=parser_args.dtype

    )
    opt_kwargs, opt_class = parse_optimizer_data(environment_data, alg_config)

    path = os.path.join(
        DATASETS_PATH, calculate_dataset_save_path(dataset_name, environment_data, society_data, epsilon=parser_args.reward_epsilon))
    dataset_train = VSLPreferenceDataset.load(
        os.path.join(path, "dataset_train.pkl"))
    dataset_test = VSLPreferenceDataset.load(
        os.path.join(path, "dataset_test.pkl"))
    if parser_args.algorithm == 'rlhf':
        import choix
        n_repeats = 10
        align_bypassed = []
        align_full = []
        acc_vs_bypassed = []
        acc_gr_bypassed = []
        acc_vs_full = []
        acc_gr_full = []

        for repeat in range(n_repeats):
            torch.manual_seed(parser_args.seed + repeat)
            np.random.seed(parser_args.seed + repeat)
            random.seed(parser_args.seed + repeat)  

            print(f"Repeat {repeat+1}/{n_repeats}")
            net_vs = []
            matrix_v = []
            for v in range(dataset_train.n_values):
                net_v, matrix, _, _ = rlhfFit(dataset_train, v, lr=0.005, hidden_layer_sizes=(16, 24, 16), maxiter=2000, net_vs=None)
                net_vs.append(net_v)
                matrix_v.append(matrix)

            net_bypassed, matrix_b, estimated_params_b, estimated_params_v_b  = rlhfFit( dataset_train, 'vs', lr=0.005, hidden_layer_sizes=(16, 24, 16,3), maxiter=20000, net_vs=None)
            net_full, matrix_vs, estimated_params, estimated_params_v = rlhfFit( dataset_train, 'vs', lr=0.01, hidden_layer_sizes=(16, 24, 16,3), maxiter=20000, net_vs=net_vs)

            # Collect alignment layers
            align_bypassed.append(net_bypassed.final_linear.get_alignment_layer()[0].detach().cpu().numpy().flatten())
            align_full.append(net_full.final_linear.get_alignment_layer()[0].detach().cpu().numpy().flatten())

            # Collect accuracies
            pair_to_agent_ids = [dataset_train.agent_ids[if1] for if1, p1 in enumerate(dataset_train.preferences)]
            acc_vs_bypassed.append(accuracy(matrix_b, estimated_params_b, pair_to_agent_ids))
            acc_gr_bypassed.append([accuracy(matrix_v[i], estimated_params_v_b[i], pair_to_agent_ids) for i in range(len(estimated_params_v_b))])
            acc_vs_full.append(accuracy(matrix_vs, estimated_params, pair_to_agent_ids))
            acc_gr_full.append([accuracy(matrix_v[i], estimated_params_v[i], pair_to_agent_ids) for i in range(len(estimated_params_v))])

        align_bypassed = np.array(align_bypassed)
        align_full = np.array(align_full)
        acc_vs_bypassed = np.array(acc_vs_bypassed)
        acc_gr_bypassed = np.array(acc_gr_bypassed)
        acc_vs_full = np.array(acc_vs_full)
        acc_gr_full = np.array(acc_gr_full)


        print("\n=== Alignment Layer (RLHF for original preferences with the same network architecture) ===")
        print("Average Value System:", np.mean(align_bypassed, axis=0))
        print("Std Value System:", np.std(align_bypassed, axis=0))

        print("\n=== Alignment Layer (Sequential RLHF: first fit grounding, then value system) ===")
        print("Average Value System:", np.mean(align_full, axis=0))
        print("Std Value System:", np.std(align_full, axis=0))

        print("\n=== Accuracies (RLHF for original preferences with the same network architecture) ===")
        print("VS representativeness Mean:", np.mean(acc_vs_bypassed, axis=0))
        print("VS representativeness Std:", np.std(acc_vs_bypassed, axis=0))
        print(" GR coherence Mean:", np.mean(acc_gr_bypassed, axis=0))
        print(" GR coherence Std:", np.std(acc_gr_bypassed, axis=0))

        print("\n=== Accuracies (Sequential RLHF: first fit grounding, then value system) ===")
        print("VS representativeness Mean:", np.mean(acc_vs_full, axis=0))
        print("VS representativeness Std:", np.std(acc_vs_full, axis=0))
        print(" GR coherence Mean:", np.mean(acc_gr_full, axis=0))
        print(" GR coherence Std:", np.std(acc_gr_full, axis=0))

        results_filename = f'supplementary_RLHFtest_and_sequential_SVSL_{n_repeats}_{parser_args.experiment_name}.txt'
        with open(results_filename, 'w') as f:
            f.write("\n=== Alignment Layer (RLHF for original preferences with the same network architecture) ===\n")
            f.write(f"Average Value System: {np.mean(align_bypassed, axis=0)}\n")
            f.write(f"Std Value System: {np.std(align_bypassed, axis=0)}\n")

            f.write("\n=== Accuracies (RLHF for original preferences with the same network architecture) ===\n")
            f.write(f"VS representativeness Mean: {np.mean(acc_vs_bypassed)}\n")
            f.write(f"VS representativeness Std: {np.std(acc_vs_bypassed)}\n")
            f.write(f" GR coherence Mean: {np.mean(acc_gr_bypassed, axis=0)}\n")
            f.write(f" GR coherence Std: {np.std(acc_gr_bypassed, axis=0)}\n")

            f.write("\n=== Alignment Layer (Sequential RLHF: first fit grounding, then value system) ===\n")
            f.write(f"Average Value System: {np.mean(align_full, axis=0)}\n")
            f.write(f"Std Value System: {np.std(align_full, axis=0)}\n")

            f.write("\n=== Accuracies (Sequential RLHF: first fit grounding, then value system) ===\n")
            f.write(f"VS representativeness Mean: {np.mean(acc_vs_full)}\n")
            f.write(f"VS representativeness Std: {np.std(acc_vs_full)}\n")
            f.write(f" GR coherence Mean: {np.mean(acc_gr_full, axis=0)}\n")
            f.write(f" GR coherence Std: {np.std(acc_gr_full, axis=0)}\n")
        print(f"Results written to {results_filename}")
        exit(0)    
            
    if parser_args.algorithm == 'pc':

        vsl_algo = PreferenceBasedClusteringTabularMDPVSL(
            env=environment,
            reward_net=reward_net,
            optimizer_cls=opt_class,
            optimizer_kwargs=opt_kwargs,
            discount=environment_data['discount'],
            discount_factor_preferences=alg_config['discount_factor_preferences'],
            dataset=dataset_train,
            training_mode=TrainingModes.SIMULTANEOUS,
            cluster_sizes=parse_cluster_sizes(
                environment_data['K'] if parser_args.k_clusters == -1 else parser_args.k_clusters, n_values=environment_data['n_values']),
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
            custom_logger='disable',
            debug_mode=parser_args.debug_mode,
            assume_variable_horizon=environment_data['assume_variable_horizon']

        )
    if parser_args.algorithm == 'pc':
        alg_config['train_kwargs']['experiment_name'] = experiment_name
    
    train_time = time.time()
    target_agent_and_vs_to_learned_ones_s, reward_net_pair_agent_and_vs_s, metrics_s, historic_assignments_s = vsl_algo.train(mode=TrainingModes.SIMULTANEOUS,
                                                                                                                              assumed_grounding=None, **alg_config['train_kwargs'])
    train_time = time.time() - train_time
    
    # Now we need to train.
    save_training_results(experiment_name, target_agent_and_vs_to_learned_ones_s,
                          reward_net_pair_agent_and_vs_s, metrics_s, parser_args={'parser_args': parser_args, 'config': config, 'society_config': society_config})
    print("FINAL MEMORY OF SOLUTIONS",metrics_s['assignment_memory'])
    target_agent_and_vs_to_learned_ones, reward_net_pair_agent_and_vs, metrics, parser_args, historic_assignments, env_state, n_iterations = load_training_results(
        experiment_name)

    assignment: ClusterAssignment = historic_assignments[-1]

    assert target_agent_and_vs_to_learned_ones == target_agent_and_vs_to_learned_ones_s, "Mismatch in target_agent_and_vs_to_learned_ones"
    assert reward_net_pair_agent_and_vs.keys() == reward_net_pair_agent_and_vs_s.keys(
    ), "Mismatch in reward_net_pair_agent_and_vs"
    assert metrics.keys() == metrics_s.keys(), "Mismatch in metrics"

    print(f"Training time: {train_time:.2f} seconds")