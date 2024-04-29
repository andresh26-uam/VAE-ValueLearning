from functools import partial
import time

from seals import base_envs
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from deep_maxent_irl import SAVED_REWARD_NET_FILE
from src.mce_irl_for_road_network import (
    TrainingModes,
    TrainingSetModes,
    MCEIRL_RoadNetwork
)
import torch

from src.policies import SimplePolicy, check_policy_gives_optimal_paths
from src.network_env import DATA_FOLDER, FeaturePreprocess, FeatureSelection, RoadWorldPOMDPStateAsTuple
from src.reward_functions import PositiveBoundedLinearModule, ProfiledRewardFunction

from src.values_and_costs import BASIC_PROFILES
from src.utils.load_data import ini_od_dist
from utils import split_od_train_test



# CUSTOM
PROFILE_VARIETY_TRAIN = 9

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


NEW_PROFILE = (0.7, 0.3, 0.0)

HORIZON = 100 # Maximum trajectory length

USE_OPTIMAL_REWARD_AS_FEATURE = False

EXAMPLE_PROFILES: set = set(BASIC_PROFILES)
#EXAMPLE_PROFILES.update(sample_example_profiles(profile_variety=PROFILE_VARIETY_TEST,n_values=3))
EXAMPLE_PROFILES.add(NEW_PROFILE)

PREPROCESSING = FeaturePreprocess.NORMALIZATION
PROFILE_VARIETY_TEST = 2

USE_OM = False
STOCHASTIC = False
USE_DIJSTRA = False if USE_OM else False if STOCHASTIC else True # change True only when USE_OM is not True.

N_EXPERT_SAMPLES_PER_OD = 1 if USE_OM is True else 30 if STOCHASTIC else 3# change True only when USE_OM is not True.
FEATURE_SELECTION = FeatureSelection.ONLY_COSTS

if __name__ == "__main__":
    od_list, od_dist = ini_od_dist(train_p)
    print("DEBUG MODE", __debug__)
    print("Learning/using profiles: ", EXAMPLE_PROFILES)
    print("Profile of society: ", NEW_PROFILE)
    env_creator = partial(RoadWorldPOMDPStateAsTuple, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path = path_feature_p, 
                        pre_reset=(od_list, od_dist), profile=NEW_PROFILE, 
                        visualize_example=True, horizon=HORIZON,
                        feature_selection=FEATURE_SELECTION,
                        feature_preprocessing=PREPROCESSING, 
                        use_optimal_reward_per_profile=USE_OPTIMAL_REWARD_AS_FEATURE)
    env_single = env_creator()


    pre_created_env = base_envs.ExposePOMDPStateWrapper(env_single)

    state_env_creator = lambda: pre_created_env

    state_venv = DummyVecEnv([state_env_creator] * 1)

    reward_net: ProfiledRewardFunction = ProfiledRewardFunction.from_checkpoint(SAVED_REWARD_NET_FILE)
    #reward_net.trained_profile_net = PositiveBoundedLinearModule(3,1, bias=False) # ?? 
    reward_net.set_mode(TrainingModes.PROFILE_LEARNING)
    learned_profile = reward_net.get_learned_profile(with_bias=False)
    print("Checkpointed profile: ", learned_profile)

    assert reward_net.action_space == env_single.action_space
    assert reward_net.hid_sizes[0] == env_single.process_features(state_des=torch.tensor([env_single.od_list_int[0], ]), feature_selection=FEATURE_SELECTION, feature_preprocessing=PREPROCESSING, use_real_env_rewards_as_feature=USE_OPTIMAL_REWARD_AS_FEATURE).shape[-1]


    """start_vi = time.time()
    
    example_vi = ValueIterationPolicy(env_single)
    example_vi.value_iteration(0.000001, verbose=True, custom_reward=lambda s,a,d: env_single.get_reward(s,a,d,tuple(BASIC_PROFILES[0])))
    pi_with_d_per_profile = {
        pr: ValueIterationPolicy(env_single).value_iteration(0.000001, verbose=True, 
                                                                                         custom_reward=lambda s,a,d: env_single.get_reward(s,a,d,tuple(pr)), 
                                                                                         reset_with_values_and_qs=(example_vi.values.copy(), example_vi.q_vals.copy())
                                                                                         ) for pr in EXAMPLE_PROFILES
    }

    end_vi = time.time()

    print("VI TIME: ", end_vi - start_vi)

    expert_policyAlgo: PolicyAlgo = PolicyAlgo.from_policy_matrix(pi_with_d_per_profile, real_env = env_single)

    expert_demonstrations_all_profiles = expert_policyAlgo.sample_trajectories(stochastic=False, repeat_per_od=N_EXPERT_SAMPLES_PER_OD, with_profiles=EXAMPLE_PROFILES)
"""

    expert_sampler: SimplePolicy = SimplePolicy.from_environment_expert(env_single, profiles=EXAMPLE_PROFILES)
    expert_demonstrations_all_profiles = expert_sampler.sample_trajectories(stochastic=False, repeat_per_od=N_EXPERT_SAMPLES_PER_OD, with_profiles=EXAMPLE_PROFILES)
    st = time.time()

    od_list_train, od_list_test, _, _= split_od_train_test(od_list, od_dist, split=0.8, to_od_list_int=True)
    
    mce_irl = MCEIRL_RoadNetwork(
        expert_policy=expert_sampler,
        expert_trajectories=expert_demonstrations_all_profiles, # los rollout no me fio en absoluto.
        env=env_single,
        reward_net=reward_net,
        log_interval=10,
        optimizer_kwargs={"lr": 0.1, "weight_decay": 0},
        mean_vc_diff_eps=0.001,
        rng=rng,
        overlaping_percentage=0.99,
        use_expert_policy_oms_instead_of_monte_carlo=USE_OM,
        n_repeat_per_od_monte_carlo = N_EXPERT_SAMPLES_PER_OD,
        training_profiles=EXAMPLE_PROFILES,
        grad_l2_eps=0.0000001,
        fd_lambda=0.0,
        use_dijkstra=USE_DIJSTRA,
        stochastic_expert=STOCHASTIC,
        od_list_test=od_list_test,
        od_list_train=od_list_train,
        training_mode=TrainingModes.PROFILE_LEARNING,
        training_set_mode=TrainingSetModes.PROFILED_SOCIETY
    )
    print("REWARD PARAMS: ")
    print(list(mce_irl.reward_net.parameters()))
    print("VALUE_MATRIX: ")
    print(mce_irl.reward_net.value_matrix())

    
    

    # NOW TRAINING ON PROFILED AGENT

    # Without retraining, check whether the sampled trajs are consistent with the profile.
    mce_irl.adapt_policy_to_profile(NEW_PROFILE)
    # TODO: Sacar a funcion los graficos guapos esos de overlapping proportion y tambien los correspondientes para FEATURE DIFF Y VISITATION COUNT DIFFS.
    sampler: SimplePolicy = SimplePolicy.from_sb3_policy(mce_irl.policy, real_env = env_single)
    path, edge_path = sampler.sample_path(start=107, des = 413, stochastic=False, profile=NEW_PROFILE,t_max=HORIZON)
    
    #mce_irl.expert_policy.fit_to_profile(profile=NEW_PROFILE, new_pi = ValueIterationPolicy(env_single, score_calculator=SumScore()).value_iteration(profile=NEW_PROFILE, reset_with_values=example_vi.values))
    real_path, real_edge_path = mce_irl.expert_policy.sample_path(start=107, des = 413, stochastic=False, profile=NEW_PROFILE,t_max=HORIZON)
    print(f"INDIRECTLY learned path for profile {NEW_PROFILE}: {edge_path}")
    print(f"Real path for profile {NEW_PROFILE}: {real_edge_path}")

    env_single.render(caminos_by_value={'eco': [path,], 'eff': [real_path]}, file="me_learned_paths.png", show=False,show_edge_weights=True)
    
    
    
    df, train_data, test_data = mce_irl.expected_trajectory_cost_calculation(on_profiles=EXAMPLE_PROFILES, stochastic_sampling=False, n_samples_per_od=None, custom_cost_preprocessing=None)
    
    df.to_csv("statistics_for_new_profiles.csv")

    # LEARNING.

    
    # Opcion 1. 70% with 1,0,0, 30% with 0,1,0. Weighting the loss of the 1,0,0 by 0.7 with 70% ODs and 0,1,0 by 0.3 with remaining 30% ODs.
    mce_irl.training_profiles = [NEW_PROFILE, ]
    mce_irl.training_set_mode = TrainingSetModes.PROFILED_SOCIETY
    mce_irl.train(1000, training_mode = TrainingModes.PROFILE_LEARNING)
    learned_profile, learned_bias = mce_irl.reward_net.get_learned_profile(with_bias=True)
    print(learned_profile, learned_bias)
    mce_irl.adapt_policy_to_profile(learned_profile)
    sampler: SimplePolicy = SimplePolicy.from_sb3_policy(mce_irl.policy, real_env = env_single)
    path, edge_path = sampler.sample_path(start=107, des = 413, stochastic=False, profile=learned_profile,t_max=HORIZON)
    real_path, real_edge_path = mce_irl.expert_policy.sample_path(start=107, des = 413, stochastic=False, profile=NEW_PROFILE,t_max=HORIZON)
    print(f"Learned path with learned profile {learned_profile} (from profile {NEW_PROFILE}) : {edge_path}")
    print(f"Real path for profile {NEW_PROFILE}: {real_edge_path}")

    df_l, train_data_l, test_data_l = mce_irl.expected_trajectory_cost_calculation(on_profiles=EXAMPLE_PROFILES, stochastic_sampling=False, n_samples_per_od=None, custom_cost_preprocessing=None)
    

    # Opcion 2 (perfect)

    mce_irl.training_profiles = [NEW_PROFILE, ]
    mce_irl.training_set_mode = TrainingSetModes.COST_MODEL_SOCIETY
    mce_irl.train(1000, training_mode = TrainingModes.PROFILE_LEARNING)
    learned_profile, learned_bias = mce_irl.reward_net.get_learned_profile(with_bias=True)
    print(learned_profile, learned_bias)
    mce_irl.adapt_policy_to_profile(learned_profile)
    sampler: SimplePolicy = SimplePolicy.from_sb3_policy(mce_irl.policy, real_env = env_single)
    path, edge_path = sampler.sample_path(start=107, des = 413, stochastic=False, profile=learned_profile,t_max=HORIZON)
    real_path, real_edge_path = mce_irl.expert_policy.sample_path(start=107, des = 413, stochastic=False, profile=NEW_PROFILE,t_max=HORIZON)
    print(f"Learned path with learned profile {learned_profile} (from profile {NEW_PROFILE}) : {edge_path}")
    print(f"Real path for profile {NEW_PROFILE}: {real_edge_path}")

    df_l, train_data_l, test_data_l = mce_irl.expected_trajectory_cost_calculation(on_profiles=EXAMPLE_PROFILES, stochastic_sampling=False, n_samples_per_od=None, custom_cost_preprocessing=None)
    



