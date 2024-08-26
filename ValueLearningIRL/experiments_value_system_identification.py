from copy import deepcopy
from functools import partial
import random
import time

import pandas as pd
from seals import base_envs
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from deep_maxent_value_grounding_learning import SAVED_REWARD_NET_FILE
from src.mce_irl_for_road_network import (
    TrainingModes,
    TrainingSetModes,
    MCEIRL_RoadNetwork
)
import torch

from src.road_network_policies import SimplePolicy, check_policy_gives_optimal_paths
from src.network_env import DATA_FOLDER, FeaturePreprocess, FeatureSelection, RoadWorldPOMDPStateAsTuple
from src.reward_functions import PositiveBoundedLinearModule, ProfiledRewardFunction

from src.values_and_costs import BASIC_PROFILES
from src.utils.load_data import ini_od_dist
from utils import sample_example_profiles, split_od_train_test

# CUSTOM

log_interval = 10  # interval between training status logs

seed = 260 # random seed for parameter initialization
rng = np.random.default_rng(260)

size = 100  # size of training data [100, 1000, 10000]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""seeding"""



"""environment"""
edge_p = f"{DATA_FOLDER}/edge.txt"
network_p = f"{DATA_FOLDER}/transit.npy"
path_feature_p = f"{DATA_FOLDER}/feature_od.npy"
train_p = f"{DATA_FOLDER}/cross_validation/train_CV%d_size%d.csv" % (0.0, size)
#test_p = f"{DATA_FOLDER}/cross_validation/test_CV%d.csv" % 0 # Test set is taken as part of the train set directly (of course then that part is not used in training phase)
node_p = f"{DATA_FOLDER}/node.txt"


"""inialize road environment"""


if __name__ == "__main__":
    new_test_data = None
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    NEW_PROFILE = (0.7, 0.3, 0.0)

    HORIZON = 100 # Maximum trajectory length
    LOGINTERVAL = 20

    USE_OPTIMAL_REWARD_AS_FEATURE = False

    #combinations={((1.0,0.0,0.0),(0.0,1.0,0.0), (0.0,0.0,1.0)): (5, 'all_5_'), ((1.0,0.0,0.0), (0.0,1.0,0.0)): (11, 'sus_sec_11_'), ((1.0,0.0,0.0), (0.0,0.0,1.0)): (11, 'sus_eff_11_'), ((0.0,1.0,0.0), (0.0,0.0,1.0)): (11, 'sec_eff_11_'),}

    combinations={((1.0,0.0,0.0), (0.0,0.0,1.0)): (11, 'sus_eff_11_'),((0.0,1.0,0.0), (0.0,0.0,1.0)): (11, 'sec_eff_11_'),((1.0,0.0,0.0), (0.0,1.0,0.0)): (11, 'sus_sec_11_'), ((1.0,0.0,0.0),(0.0,1.0,0.0), (0.0,0.0,1.0)): (5, 'all_5_'),}
    #combinations={((1.0,0.0,0.0), (0.0,1.0,0.0)): (11, 'cor_sus_sec_11_'), ((1.0,0.0,0.0), (0.0,0.0,1.0)): (11, 'cor_sus_eff_11_'),((0.0,1.0,0.0), (0.0,0.0,1.0)): (11, 'cor_sec_eff_11_'),((1.0,0.0,0.0),(0.0,1.0,0.0), (0.0,0.0,1.0)): (5, 'cor_all_5_'),}

    for combination, n_profiles_and_name in combinations.items():
        n_profiles, name_of_files = n_profiles_and_name
        PROFILE_VARIETY_TEST  = n_profiles
        EXAMPLE_PROFILES = sample_example_profiles(profile_variety=PROFILE_VARIETY_TEST,n_values=len(BASIC_PROFILES))
        
        
        invalid_component_indexes = set(BASIC_PROFILES.index(p) for p in BASIC_PROFILES if p not in combination)
        
        for p_with_eff in deepcopy(EXAMPLE_PROFILES):
            for ind in invalid_component_indexes:
                if p_with_eff[ind] > 0.0:
                    EXAMPLE_PROFILES.remove(p_with_eff)
        
        a = np.array(EXAMPLE_PROFILES, dtype=np.dtype([('x', float), ('y', float), ('z', float)]))
        sortedprofiles =a[np.argsort(a, axis=-1, order=('x', 'y', 'z'), )]
        EXAMPLE_PROFILES = list(tuple(t) for t in sortedprofiles.tolist())

        PREPROCESSING = FeaturePreprocess.NORMALIZATION

        USE_OM = False
        STOCHASTIC = False
        USE_DIJKSTRA = False if USE_OM else False if STOCHASTIC else True # change True only when USE_OM is not True.

        N_EXPERT_SAMPLES_PER_OD = 1 if USE_OM is True else 30 if STOCHASTIC else 1# change True only when USE_OM is not True.
        FEATURE_SELECTION = FeatureSelection.ONLY_COSTS

        LEARNING_ITERATIONS = 80
        BATCH_SIZE_PS = 200 # In profile society, batch size is vital, for sampling routes with random profiles and destinations with enough variety

        N_OD_SPLITS_FOR_SIMULATING_SOCIETY = 10
        N_NEW_TEST_DATA = 100

        PLOT_HISTS = False

        reward_net: ProfiledRewardFunction = ProfiledRewardFunction.from_checkpoint(SAVED_REWARD_NET_FILE)

        od_list, od_dist = ini_od_dist(train_p)
        print("DEBUG MODE", __debug__)
        print("Learning/using profiles: ", EXAMPLE_PROFILES)
        #print("Profile of society: ", NEW_PROFILE)
        env_creator = partial(RoadWorldPOMDPStateAsTuple, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path = path_feature_p, 
                            pre_reset=(od_list, od_dist), profile=EXAMPLE_PROFILES[0], 
                            visualize_example=True, horizon=HORIZON,
                            feature_selection=FEATURE_SELECTION,
                            feature_preprocessing=PREPROCESSING, 
                            use_optimal_reward_per_profile=USE_OPTIMAL_REWARD_AS_FEATURE)
        env_single = env_creator()

        
        pre_created_env = base_envs.ExposePOMDPStateWrapper(env_single)

        state_env_creator = lambda: pre_created_env

        state_venv = DummyVecEnv([state_env_creator] * 1)

        
        #reward_net.reset_learning_profile()
        reward_net.set_mode(TrainingModes.VALUE_SYSTEM_IDENTIFICATION)
        checkpointed_learned_profile = reward_net.get_learned_profile(with_bias=False)
        reward_net.reset_learning_profile(checkpointed_learned_profile)
        print("Checkpointed profile: ", reward_net.get_learned_profile(), np.sum(reward_net.get_learned_profile()), reward_net.get_learned_profile())

        assert reward_net.action_space == env_single.action_space
        assert reward_net.hid_sizes[0] == env_single.process_features(state_des=torch.tensor([env_single.od_list_int[0], ]), feature_selection=FEATURE_SELECTION, feature_preprocessing=PREPROCESSING, use_real_env_rewards_as_feature=USE_OPTIMAL_REWARD_AS_FEATURE).shape[-1]

        expert_sampler: SimplePolicy = SimplePolicy.from_environment_expert(env_single, profiles=EXAMPLE_PROFILES)
        expert_demonstrations_all_profiles = expert_sampler.sample_trajectories(stochastic=False, repeat_per_od=N_EXPERT_SAMPLES_PER_OD, with_profiles=EXAMPLE_PROFILES)
        

        

        if new_test_data is None:
            od_list_train, od_list_test, _, _= split_od_train_test(od_list, od_dist, split=0.8, to_od_list_int=True)
            def select_random_pairs(input_list, n, min_distance=5):
                # Make sure there are at least two unique elements in the input list
                if len(set(input_list)) < 2:
                    raise ValueError("Input list must contain at least two unique elements")

                # Generate unique pairs
                pairs = set()
                while len(pairs) < n:
                    pair = random.sample(input_list, 2)
                    
                    if pair[0] != pair[1] and len(env_single.shortest_path_edges(profile=(0.3,0.3,0.3), from_state=pair[0], to_state=pair[1])) >= min_distance:
                        pairs.add(tuple(pair))

                return list(pairs)

            new_test_data = set(od_list_test)

            new_test_data.update(set(select_random_pairs(list(env_single.valid_edges), N_NEW_TEST_DATA, min_distance=5)))

        mce_irl = MCEIRL_RoadNetwork(
            expert_policy=expert_sampler,
            expert_trajectories=expert_demonstrations_all_profiles,
            env=env_single,
            reward_net=reward_net,
            log_interval=LOGINTERVAL,
            optimizer_kwargs={"lr": 0.2, "weight_decay": 0.0},
            mean_vc_diff_eps=0.0001,
            rng=rng,
            overlaping_percentage=0.99,
            use_expert_policy_oms_instead_of_monte_carlo=USE_OM,
            n_repeat_per_od_monte_carlo = N_EXPERT_SAMPLES_PER_OD,
            training_profiles=EXAMPLE_PROFILES,
            grad_l2_eps=0.0000001,
            fd_lambda=0.0,
            use_dijkstra=USE_DIJKSTRA,
            stochastic_expert=STOCHASTIC,
            od_list_test=od_list_test,
            od_list_train=od_list_train,
            training_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
            training_set_mode=TrainingSetModes.PROFILED_SOCIETY,
            name_method=name_of_files,
        )
        print("REWARD PARAMS: ")
        print(list(mce_irl.get_reward_net().parameters()))
        print("VALUE_MATRIX: ")
        print(mce_irl.get_reward_net().value_matrix())

        
        # Opcion 2 (learn from a single expert with a mixed profile)
        learned_profiles_to_targets = list()

        mce_irl.policies_per_profile.clear()
        
        for npr in EXAMPLE_PROFILES:
            mce_irl.name_method = name_of_files+str(npr)
            mce_irl.training_profiles = [npr, ]
            new_reward_net = mce_irl.get_reward_net()
            new_reward_net.reset_learning_profile(checkpointed_learned_profile)
            mce_irl.set_reward_net(new_reward_net)

            #mce_irl.reward_net.reset_learning_profile(checkpointed_learned_profile)

            mce_irl.train(LEARNING_ITERATIONS, training_mode = TrainingModes.VALUE_SYSTEM_IDENTIFICATION, training_set_mode=TrainingSetModes.PROFILED_EXPERT, render_partial_plots=False, batch_size=None)
            learned_profile, learned_bias = mce_irl.get_reward_net().get_learned_profile(with_bias=True)
            
            # 0.019875993952155113, 0.009943496435880661, 0.9701805710792542 EXPERT 001.
            # (0.017024695873260498, 0.008413741365075111, 0.9745615720748901) EXPERT 0109.
            learned_profiles_to_targets.append((learned_profile, npr))
            mce_irl.adapt_policy_to_profile(learned_profile, use_cached_policies=False)
            sampler: SimplePolicy = SimplePolicy.from_sb3_policy(mce_irl.policy, real_env = env_single)
            path, edge_path = sampler.sample_path(start=env_single.od_list_int[0][0], des = env_single.od_list_int[0][1], stochastic=False, profile=learned_profile,t_max=HORIZON)
            real_path, real_edge_path = mce_irl.expert_policy.sample_path(start=env_single.od_list_int[0][0], des = env_single.od_list_int[0][1], stochastic=False, profile=npr,t_max=HORIZON)
            print(f"Learned path with learned profile {learned_profile} (from profile {npr}) : {edge_path}")
            print(f"Real path for profile {npr}: {real_edge_path}")

            env_single.render(caminos_by_value={'sus': [path,], 'eff': [real_path]}, file=f"{name_of_files}_me_learned_paths_from_expert_{npr}_{learned_profile}.png", show=False,show_edge_weights=False)
            
        
        df_train, df_test, train_data, test_data, similarities_train, similarities_test = mce_irl.expected_trajectory_cost_calculation(on_profiles=EXAMPLE_PROFILES, learned_profiles=learned_profiles_to_targets, stochastic_sampling=False, n_samples_per_od=None, custom_cost_preprocessing=FeaturePreprocess.NORMALIZATION, repeat_society=N_OD_SPLITS_FOR_SIMULATING_SOCIETY, new_test_data=new_test_data, name_method=name_of_files+'learned_from_expert', plot_histograms=PLOT_HISTS)
        
        df_train.to_csv(f"results/value_system_identification/{name_of_files}_statistics_learning_from_expert_train.csv")
        df_test.to_csv(f"results/value_system_identification/{name_of_files}_statistics_learning_from_expert_test.csv")
        #df_train.to_markdown(f"results/value_system_identification/{name_of_files}_statistics_learning_from_expert_train.md")
        #df_test.to_markdown(f"results/value_system_identification/{name_of_files}_statistics_learning_from_expert_test.md")
        for metric, df in similarities_train.items():

            df.to_csv(f"results/value_system_identification/{name_of_files}_similarities_{metric}_learning_from_expert_train.csv")
            #df.to_markdown(f"results/value_system_identification/{name_of_files}_similarities_{metric}_learning_from_expert_train.md")
        for metric, df in similarities_test.items():

            df.to_csv(f"results/value_system_identification/{name_of_files}_similarities_{metric}_learning_from_expert_test.csv")
            #df.to_markdown(f"results/value_system_identification/{name_of_files}_similarities_{metric}_learning_from_expert_test.md")
                    
    

        

        # Without retraining, check whether the sampled trajs are consistent with the profile.


        learned_profiles_to_targets = list()
        mce_irl.policies_per_profile.clear()
        for npr in EXAMPLE_PROFILES:
            mce_irl.name_method = name_of_files+str(npr)
            mce_irl.adapt_policy_to_profile(npr, use_cached_policies=False)
            learned_profiles_to_targets.append((npr, npr))
            sampler: SimplePolicy = SimplePolicy.from_sb3_policy(mce_irl.policy, real_env = env_single)
            path, edge_path = sampler.sample_path(start=env_single.od_list_int[0][0], des = env_single.od_list_int[0][1], stochastic=False, profile=npr,t_max=HORIZON)
            
            #mce_irl.expert_policy.fit_to_profile(profile=NEW_PROFILE, new_pi = ValueIterationPolicy(env_single).value_iteration(profile=NEW_PROFILE, reset_with_values=example_vi.values))
            real_path, real_edge_path = mce_irl.expert_policy.sample_path(start=env_single.od_list_int[0][0], des = env_single.od_list_int[0][1], stochastic=False, profile=npr,t_max=HORIZON)
            print(f"INDIRECTLY learned path for profile {npr}: {edge_path}")
            print(f"Real path for profile {npr}: {real_edge_path}")

            env_single.render(caminos_by_value={'sus': [path,], 'eff': [real_path]}, file=f"{name_of_files}_me_learned_paths_{npr}_vl.png", show=False,show_edge_weights=False)
            
        df_train, df_test, train_data, test_data, similarities_train, similarities_test = mce_irl.expected_trajectory_cost_calculation(on_profiles=EXAMPLE_PROFILES, learned_profiles=learned_profiles_to_targets, stochastic_sampling=False, n_samples_per_od=None, custom_cost_preprocessing=FeaturePreprocess.NORMALIZATION, new_test_data=new_test_data, name_method=name_of_files+'given_profile', plot_histograms=PLOT_HISTS)
        
        df_train.to_csv(f"results/value_system_identification/{name_of_files}_statistics_for_unseen_profile_train.csv")
        df_test.to_csv(f"results/value_system_identification/{name_of_files}_statistics_for_unseen_profile_test.csv")
        df_train.to_markdown(f"results/value_system_identification/{name_of_files}_statistics_for_unseen_profile_train_train.md")
        df_test.to_markdown(f"results/value_system_identification/{name_of_files}_statistics_for_unseen_profile_test_test.md")

        for metric, df in similarities_train.items():

            df.to_csv(f"results/value_system_identification/{name_of_files}_similarities_{metric}_for_unseen_profile_train.csv")
            df.to_markdown(f"results/value_system_identification/{name_of_files}_similarities_{metric}_for_unseen_profile_train.md")
        for metric, df in similarities_test.items():

            df.to_csv(f"results/value_system_identification/{name_of_files}_similarities_{metric}_for_unseen_profile_test.csv")
            df.to_markdown(f"results/value_system_identification/{name_of_files}_similarities_{metric}_for_unseen_profile_test.md")
                
        

        # Opcion 1. probabilities 70% with 1.0,0.0,0, 30% with 0,1.0,0. (A "Society")
        mce_irl.policies_per_profile.clear()
        learned_profiles_to_targets = list()

        
        for npr in EXAMPLE_PROFILES:
            # TODO: Support for unknown societies... Define a society only with a list of profiles and the probability of choosing one or the other. 
            mce_irl.training_profiles = [npr, ]
            mce_irl.name_method = name_of_files+str(npr)
            
            new_reward_net = mce_irl.get_reward_net()
            new_reward_net.reset_learning_profile(checkpointed_learned_profile)
            mce_irl.set_reward_net(new_reward_net)
            
            #mce_irl.reward_net.reset_learning_profile(checkpointed_learned_profile)

            mce_irl.train(LEARNING_ITERATIONS, training_mode = TrainingModes.VALUE_SYSTEM_IDENTIFICATION, training_set_mode=TrainingSetModes.PROFILED_SOCIETY, render_partial_plots=False, batch_size=BATCH_SIZE_PS)
            learned_profile, learned_bias = mce_irl.get_reward_net().get_learned_profile(with_bias=True)
            #print(learned_profile, learned_bias)

            learned_profiles_to_targets.append((learned_profile, npr))

            mce_irl.adapt_policy_to_profile(learned_profile, use_cached_policies=False)
            sampler: SimplePolicy = SimplePolicy.from_sb3_policy(mce_irl.policy, real_env = env_single)
            path, edge_path = sampler.sample_path(start=env_single.od_list_int[0][0], des = env_single.od_list_int[0][1], stochastic=False, profile=learned_profile,t_max=HORIZON)
            real_path, real_edge_path = mce_irl.expert_policy.sample_path(start=env_single.od_list_int[0][0], des = env_single.od_list_int[0][1], stochastic=False, profile=npr,t_max=HORIZON)
            print(f"Learned path with learned profile {learned_profile} (from profile {npr}) : {edge_path}")
            print(f"Real path for profile {npr}: {real_edge_path}")
            env_single.render(caminos_by_value={'sus': [path,], 'eff': [real_path]}, file=f"{name_of_files}_me_learned_paths_from_society_{npr}_{learned_profile}.png", show=False,show_edge_weights=False)
            
        
            
        df_train, df_test, train_data, test_data, similarities_train, similarities_test  = mce_irl.expected_trajectory_cost_calculation(on_profiles=EXAMPLE_PROFILES, learned_profiles=learned_profiles_to_targets, stochastic_sampling=False, n_samples_per_od=None, custom_cost_preprocessing=FeaturePreprocess.NORMALIZATION, repeat_society=N_OD_SPLITS_FOR_SIMULATING_SOCIETY, new_test_data=new_test_data, name_method=name_of_files+'learned_from_society', plot_histograms=PLOT_HISTS)
        
        df_train.to_csv(f"results/value_system_identification/{name_of_files}_statistics_learning_from_society_train.csv")
        df_train.to_markdown(f"results/value_system_identification/{name_of_files}_statistics_learning_from_society_train.md")
        df_test.to_csv(f"results/value_system_identification/{name_of_files}_statistics_learning_from_society_test.csv")
        df_test.to_markdown(f"results/value_system_identification/{name_of_files}_statistics_learning_from_society_test.md")

        for metric, df in similarities_train.items():

            df.to_csv(f"results/value_system_identification/{name_of_files}_similarities_{metric}_learning_from_society_train.csv")
            df.to_markdown(f"results/value_system_identification/{name_of_files}_similarities_{metric}_learning_from_society_train.md")
        for metric, df in similarities_test.items():

            df.to_csv(f"results/value_system_identification/{name_of_files}_similarities_{metric}_learning_from_society_test.csv")
            df.to_markdown(f"results/value_system_identification/{name_of_files}_similarities_{metric}_learning_from_society_test.md")
            
        mce_irl.policies_per_profile.clear()
        
            
            # LEARNING.


        
                    
            

