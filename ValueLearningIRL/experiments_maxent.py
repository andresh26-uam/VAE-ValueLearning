# %%
from copy import deepcopy
from functools import partial
from math import ceil
import time

from seals import base_envs
from seals.diagnostics.cliff_world import CliffWorldEnv
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from src.mce_irl_for_road_network import (
    MCEIRL_RoadNetwork,
    mce_occupancy_measures,
    mce_partition_fh,
    TabularPolicyPerProfile
)
from imitation.data import rollout
from imitation.rewards import reward_nets
import torch

from src.policies import SimplePolicy, ValueIterationPolicy, check_policy_gives_optimal_paths
from src.network_env import DATA_FOLDER, NO_NORMALIZATION_OR_STANDARIZATION, NO_ONE_HOT, NORMALIZATION, ONE_HOT_ALL, ONE_HOT_ORIGIN_ONLY, ONE_HOT_ORIGIN_AND_DEST, ONLY_COSTS, STANDARIZATION, TRAINED_MODELS, FeaturePreprocess, FeatureSelection, RoadWorld, RoadWorldExpert, RoadWorldPOMDPStateAsTuple, RoadWorldExpertPOMDPStateDesPF
from src.reward_functions import ProfiledRewardFunction
from src.src_rl.aggregations import SumScore
from src.values_and_costs import BASIC_PROFILE_NAMES, BASIC_PROFILES, PROFILE_COLORS
from src.utils.load_data import ini_od_dist
from utils import create_expert_trajectories, split_od_train_test

from torch import nn
rng = np.random.default_rng(0)

# CUSTOM

log_interval = 10  # interval between training status logs

seed = 260 # random seed for parameter initialization
rng = np.random.default_rng(260)

size = 100  # size of training data [100, 1000, 10000]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""seeding"""

np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

"""environment"""
edge_p = f"{DATA_FOLDER}/edge.txt"
network_p = f"{DATA_FOLDER}/transit.npy"
path_feature_p = f"{DATA_FOLDER}/feature_od.npy"
train_p = f"{DATA_FOLDER}/cross_validation/train_CV%d_size%d.csv" % (0, size)
#test_p = f"{DATA_FOLDER}/cross_validation/test_CV%d.csv" % 0 # Test set is taken as part of the train set directly (of course then that part is not used in training phase)
node_p = f"{DATA_FOLDER}/node.txt"


# %%
"""inialize road environment"""


PROFILE = (1, 0, 0)

HORIZON = 100 # Maximum trajectory length

USE_OPTIMAL_REWARD_AS_FEATURE = False

EXAMPLE_PROFILES: set = set(BASIC_PROFILES)


PREPROCESSING = FeaturePreprocess.NORMALIZATION
PROFILE_VARIETY_TEST = 2

USE_OM = False
STOCHASTIC = False
USE_DIJSTRA = False if USE_OM else False if STOCHASTIC else True # change True only when USE_OM is not True.

N_EXPERT_SAMPLES_PER_OD = 1 if USE_OM is True else 30 if STOCHASTIC else 3# change True only when USE_OM is not True.
FEATURE_SELECTION = FeatureSelection.ONLY_COSTS

"""inialize road environment"""

od_list, od_dist = ini_od_dist(train_p)
print("DEBUG MODE", __debug__)
#od_list, od_dist = [od_list[0],], [od_dist[0], ]
od_list = od_list[0:15] # For debug only
od_dist = od_dist[0:15]


env_creator = partial(RoadWorldPOMDPStateAsTuple, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path = path_feature_p, 
                      pre_reset=(od_list, od_dist), profile=PROFILE, visualize_example=False, horizon=HORIZON,
                      feature_selection=FEATURE_SELECTION,
                        feature_preprocessing=PREPROCESSING, 
                        use_optimal_reward_per_profile=USE_OPTIMAL_REWARD_AS_FEATURE)
env_single = env_creator()

start_vi = time.time()
pi_with_d_per_profile = {
    pr: ValueIterationPolicy(env_single).value_iteration(0.000001, verbose=True, custom_reward=lambda s,a,d: env_single.get_reward(s,a,d,tuple(pr))) for pr in EXAMPLE_PROFILES
}
end_vi = time.time()

print("VI TIME: ", end_vi - start_vi)

expert_policyAlgo: SimplePolicy = SimplePolicy.from_policy_matrix(pi_with_d_per_profile, real_env = env_single)

expert_demonstrations_all_profiles = expert_policyAlgo.sample_trajectories(stochastic=False, repeat_per_od=N_EXPERT_SAMPLES_PER_OD, profile=EXAMPLE_PROFILES)

N_EXPERIMENTS = 3
N_ITER_PER_EXPERIMENT = 5

rewards_per_experiment = []
train_stats_per_experiment = []
test_stats_per_experiment = []


SAVED_REWARD_NET_FILE = "profiled_reward_function_trained.pt"

for repeat in range(N_EXPERIMENTS):
    od_list_train, od_list_test, _, _= split_od_train_test(od_list, od_dist, split=0.8, to_od_list_int=True)
    # TODO: TESTEAR ESTO.
    
    # TODO: PLOT DESVIACIONES TIPICAS en train_stats['op'], train_stats['vc'], train_stats['fd'] y test_stats (similar)
    # TODO: ANOTAR MEAN media de train_stats['op'][-1] y todos esos para ver que tal va.
    # TODO: UTILIZAR OTROS DATOS CADA VEZ (?)
    # TODO: PESOS ALEATORIOS (?)
     
    reward_net = ProfiledRewardFunction(
        environment=env_single,
        use_state=False,
        use_action=False,
        use_next_state=True,
        use_done=False,
        hid_sizes=[3,],
        reward_bias=-0.1, # Aprender este bias (?)
        activations=[partial(nn.Softplus, beta=1, threshold=5)
                     , nn.Identity]
    )
    reward_net.set_profile(PROFILE)

    
    
    print(reward_net.values_net)


    st = time.time()
    mce_irl = MCEIRL_RoadNetwork(
        expert_policy=pi_with_d_per_profile,
        expert_trajectories=expert_demonstrations_all_profiles, # los rollout no me fio en absoluto.
        env=env_single,
        reward_net=reward_net,
        log_interval=1,
        optimizer_kwargs={"lr": 0.2, "weight_decay": 1e-2},
        mean_vc_diff_eps=0.001,
        rng=rng,
        overlaping_percentage=1.0,
        use_expert_policy_oms_instead_of_monte_carlo=USE_OM,
        n_repeat_per_od_monte_carlo = N_EXPERT_SAMPLES_PER_OD,
        grad_l2_eps=0.0,
        fd_lambda=0.0,
        use_dijkstra=USE_DIJSTRA,
        stochastic_expert=False,
        od_list_train = od_list_train,
        od_list_test = od_list_test
    )
    print(list(mce_irl._reward_net.parameters()))
    print("TRAINING STARTED")

    predicted_rewards_per_profile, train_stats, test_stats = mce_irl.train(max_iter=N_ITER_PER_EXPERIMENT,render_partial_plots=False)
    end_time = time.time()
    print("TIME: ", end_time - st)
    print("FINAL PARAMETERS: ")
    print(list(mce_irl._reward_net.parameters()))
    

    rewards_per_experiment.append(predicted_rewards_per_profile)
    train_stats_per_experiment.append(train_stats)
    test_stats_per_experiment.append(test_stats)



average_reward_per_profile = {pr: np.zeros_like(predicted_rewards_per_profile[pr]) for pr in EXAMPLE_PROFILES}
std_reward_per_profile = {pr: np.zeros_like(predicted_rewards_per_profile[pr]) for pr in EXAMPLE_PROFILES}

average_train_stats = {pr: {k: np.zeros_like(np.asarray(v_per_pr[pr])) for k,v_per_pr in train_stats.items()} for pr in EXAMPLE_PROFILES}
std_train_stats = {pr: {k: np.zeros_like(np.asarray(v_per_pr[pr])) for k,v_per_pr in train_stats.items()} for pr in EXAMPLE_PROFILES}

average_test_stats = {pr: {k: np.zeros_like(np.asarray(v_per_pr[pr])) for k,v_per_pr in test_stats.items()} for pr in EXAMPLE_PROFILES}
std_test_stats = {pr: {k: np.zeros_like(np.asarray(v_per_pr[pr])) for  k,v_per_pr in test_stats.items()} for pr in EXAMPLE_PROFILES}

for reward_per_profile, train_stats, test_stats in zip(rewards_per_experiment, train_stats_per_experiment, test_stats_per_experiment):
    for pr in EXAMPLE_PROFILES:
        average_reward_per_profile[pr] += reward_per_profile[pr]/N_EXPERIMENTS
        for k, v_per_pr in train_stats.items():
            average_train_stats[pr][k] += np.asarray(v_per_pr[pr])/N_EXPERIMENTS
        for k, v_per_pr in test_stats.items():
            average_test_stats[pr][k] += np.asarray(v_per_pr[pr])/N_EXPERIMENTS

for reward_per_profile, train_stats, test_stats in zip(rewards_per_experiment, train_stats_per_experiment, test_stats_per_experiment):
    for pr in EXAMPLE_PROFILES:
        std_reward_per_profile[pr] += ((reward_per_profile[pr] - average_reward_per_profile[pr])**2)/(N_EXPERIMENTS-1)
        for k, v_per_pr in train_stats.items():
            std_train_stats[pr][k] += ((np.asarray(v_per_pr[pr]) - average_train_stats[pr][k])**2)/(N_EXPERIMENTS-1)
        for k, v_per_pr in test_stats.items():
            std_test_stats[pr][k] += ((np.asarray(v_per_pr[pr]) - average_test_stats[pr][k])**2)/(N_EXPERIMENTS-1)
import matplotlib.pyplot as plt


plt.figure(figsize=[18, 12])
#print(mean_absolute_difference_in_visitation_counts_per_profile)
plt.subplot(2, 3, 1)
plt.title(f"TRAIN: Average trajectory overlap with profiles")
plt.xlabel("Training Iteration")
plt.ylabel("Average trajectory overlap proportion")
for _, pr in enumerate(EXAMPLE_PROFILES):
    #plt.plot(mean_absolute_difference_in_visitation_counts_per_profile[pr], color=mean_absolute_difference_in_visitation_counts_per_profile_colors[pi], label='Maximum difference in visitation counts')
    #print(overlapping_proportions_per_iteration_per_profile_test[pr])
    avg = average_train_stats[pr]['op']
    std_dev = std_train_stats[pr]['op']

    plt.plot(avg,
                color=PROFILE_COLORS[pr], 
                label=f'Pr: \'{BASIC_PROFILE_NAMES[pr]}\'\nLast: {float(avg[-1]):0.3f}'
                )
    
    plt.fill_between([i for i in range(len(avg))], avg-std_dev, avg+std_dev,edgecolor=PROFILE_COLORS[pr],alpha=0.1, facecolor=PROFILE_COLORS[pr])
plt.legend()
plt.grid()

#print(mean_absolute_difference_in_visitation_counts_per_profile)
plt.subplot(2, 3, 1+3)
plt.title(f"TEST: Average trajectory overlap with profiles")
plt.xlabel("Training Iteration")
plt.ylabel("Average trajectory overlap proportion")
for _, pr in enumerate(EXAMPLE_PROFILES):
    #plt.plot(mean_absolute_difference_in_visitation_counts_per_profile[pr], color=mean_absolute_difference_in_visitation_counts_per_profile_colors[pi], label='Maximum difference in visitation counts')
    #print(overlapping_proportions_per_iteration_per_profile_test[pr])
    avg = average_test_stats[pr]['op']
    std_dev = std_test_stats[pr]['op']
    plt.plot(avg,
                color=PROFILE_COLORS[pr], 
                label=f'Pr: \'{BASIC_PROFILE_NAMES[pr]}\'\nLast: {float(avg[-1]):0.3f}'
                )
    
    plt.fill_between([i for i in range(len(avg))], avg-std_dev, avg+std_dev,edgecolor=PROFILE_COLORS[pr],alpha=0.1, facecolor=PROFILE_COLORS[pr])
plt.legend()
plt.grid()


plt.subplot(2, 3, 2)
plt.title(f"TRAIN: Expected visitation count difference")
plt.xlabel("Training Iteration")
plt.ylabel("Expected absolute difference in visitation counts")
for _, pr in enumerate(EXAMPLE_PROFILES):
    #plt.plot(mean_absolute_difference_in_visitation_counts_per_profile[pr], color=mean_absolute_difference_in_visitation_counts_per_profile_colors[pi], label='Maximum difference in visitation counts')
    #print(overlapping_proportions_per_iteration_per_profile_test[pr])
    avg = average_train_stats[pr]['vc']
    std_dev = std_train_stats[pr]['vc']

    plt.plot(avg,
                color=PROFILE_COLORS[pr], 
                label=f'Pr: \'{BASIC_PROFILE_NAMES[pr]}\'\nLast: {float(avg[-1]):0.3f}'
                )
    
    plt.fill_between([i for i in range(len(avg))], avg-std_dev, avg+std_dev,edgecolor=PROFILE_COLORS[pr],alpha=0.1, facecolor=PROFILE_COLORS[pr])
plt.legend()
plt.grid()

#print(mean_absolute_difference_in_visitation_counts_per_profile)
plt.subplot(2, 3, 2+3)
plt.title(f"TEST: Expected visitation count difference")
plt.xlabel("Training Iteration")
plt.ylabel("Expected absolute difference in visitation counts")
for _, pr in enumerate(EXAMPLE_PROFILES):
    #plt.plot(mean_absolute_difference_in_visitation_counts_per_profile[pr], color=mean_absolute_difference_in_visitation_counts_per_profile_colors[pi], label='Maximum difference in visitation counts')
    #print(overlapping_proportions_per_iteration_per_profile_test[pr])
    avg = average_test_stats[pr]['vc']
    std_dev = std_test_stats[pr]['vc']
    plt.plot(avg,
                color=PROFILE_COLORS[pr], 
                label=f'Pr: \'{BASIC_PROFILE_NAMES[pr]}\'\nLast: {float(avg[-1]):0.3f}'
                )
    
    plt.fill_between([i for i in range(len(avg))], avg-std_dev, avg+std_dev,edgecolor=PROFILE_COLORS[pr],alpha=0.1, facecolor=PROFILE_COLORS[pr])
plt.legend()
plt.grid()



plt.subplot(2, 3, 3)
plt.title(f"TRAIN: Expected profiled cost difference per profile")
plt.xlabel("Training Iteration")
plt.ylabel("Expected trajectory profiled cost difference")
for _, pr in enumerate(EXAMPLE_PROFILES):
    #plt.plot(mean_absolute_difference_in_visitation_counts_per_profile[pr], color=mean_absolute_difference_in_visitation_counts_per_profile_colors[pi], label='Maximum difference in visitation counts')
    #print(overlapping_proportions_per_iteration_per_profile_test[pr])
    avg = average_train_stats[pr]['fd']
    std_dev = std_train_stats[pr]['fd']

    plt.plot(avg,
                color=PROFILE_COLORS[pr], 
                label=f'Pr: \'{BASIC_PROFILE_NAMES[pr]}\'\nLast: {float(avg[-1]):0.3f}'
                )
    
    plt.fill_between([i for i in range(len(avg))], avg-std_dev, avg+std_dev,edgecolor=PROFILE_COLORS[pr],alpha=0.1, facecolor=PROFILE_COLORS[pr])
plt.legend()
plt.grid()

#print(mean_absolute_difference_in_visitation_counts_per_profile)
plt.subplot(2, 3, 3+3)
plt.title(f"TEST: Expected profiled cost difference per profile")
plt.xlabel("Training Iteration")
plt.ylabel("Expected trajectory profiled cost difference")
for _, pr in enumerate(EXAMPLE_PROFILES):
    #plt.plot(mean_absolute_difference_in_visitation_counts_per_profile[pr], color=mean_absolute_difference_in_visitation_counts_per_profile_colors[pi], label='Maximum difference in visitation counts')
    #print(overlapping_proportions_per_iteration_per_profile_test[pr])
    avg = average_test_stats[pr]['fd']
    std_dev = std_test_stats[pr]['fd']
    plt.plot(avg,
                color=PROFILE_COLORS[pr], 
                label=f'Pr: \'{BASIC_PROFILE_NAMES[pr]}\'\nLast: {float(avg[-1]):0.3f}'
                )
    
    plt.fill_between([i for i in range(len(avg))], avg-std_dev, avg+std_dev,edgecolor=PROFILE_COLORS[pr],alpha=0.1, facecolor=PROFILE_COLORS[pr])
plt.legend()
plt.grid()

plt.savefig(f'plots/ME_experiment_results_{N_EXPERIMENTS}x{N_ITER_PER_EXPERIMENT}.png')
plt.show(block=False)
plt.close()

env_single.render(caminos_by_value={}, file='me_learned_costs_w_label.png', show=False, show_edge_weights=True, custom_weights={pr: -predicted_rewards_per_profile[pr] for pr in predicted_rewards_per_profile.keys()}, custom_weights_dest=413)
env_single.render(caminos_by_value={}, file='me_learned_costs.png', show=False, show_edge_weights=False, custom_weights={pr: -predicted_rewards_per_profile[pr] for pr in predicted_rewards_per_profile.keys()}, custom_weights_dest=413)
env_single.render(caminos_by_value={}, file='me_real_costs.png', show=False, show_edge_weights=False)


mce_irl.policy.set_profile(PROFILE)

sampler: SimplePolicy = SimplePolicy.from_sb3_policy(mce_irl.policy, real_env = env_single)
path, edge_path = sampler.sample_path(start=107, des = 413, stochastic=False, profile=PROFILE,t_max=HORIZON)

env_single.render(caminos_by_value={'eco': [path,]}, file="me_learned_paths.png", show=False,show_edge_weights=True)

