import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

from src.values_and_costs import FULL_NAME_VALUES
combinations={((0.0,1.0,0.0), (0.0,0.0,1.0)): (11, 'sec_eff_11_'),((1.0,0.0,0.0), (0.0,1.0,0.0)): (11, 'sus_sec_11_'), ((1.0,0.0,0.0), (0.0,0.0,1.0)): (11, 'sus_eff_11_'),((1.0,0.0,0.0),(0.0,1.0,0.0), (0.0,0.0,1.0)): (5, 'all_5_'),}
    #combinations={((1.0,0.0,0.0), (0.0,1.0,0.0)): (11, 'cor_sus_sec_11_'), ((1.0,0.0,0.0), (0.0,0.0,1.0)): (11, 'cor_sus_eff_11_'),((0.0,1.0,0.0), (0.0,0.0,1.0)): (11, 'cor_sec_eff_11_'),((1.0,0.0,0.0),(0.0,1.0,0.0), (0.0,0.0,1.0)): (5, 'cor_all_5_'),}
PUT_OD_STD = False

OBTAIN_CORRECTED_CSVS = False # Last execution was faulty in some of the results shown in the tables, we needed to correct the results. 
# These corrected results are named as def_{nameoftheoriginalfile}.csv under the results/value_system_identification/ folder.
if OBTAIN_CORRECTED_CSVS:
    from ast import literal_eval
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

    from src.policies import SimplePolicy, check_policy_gives_optimal_paths
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
            expert_trajectories=expert_demonstrations_all_profiles, # los rollout no me fio en absoluto.
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

         # Value System Identification from society

        origdf = pd.read_csv(f"results/value_system_identification/{name_of_files}_similarities_agou_learning_from_society_train.csv")
        origdf['Learned Profile'] = origdf['Learned Profile'].apply(lambda x: literal_eval(x))
        origdf['Target Profile'] = origdf['Target Profile'].apply(lambda x: literal_eval(x))
        learned_profiles_to_targets = list(zip( origdf['Learned Profile'], origdf['Target Profile']))
        mce_irl.policies_per_profile.clear()
        
        
        for learned_profile, npr in learned_profiles_to_targets:
            #origdf = pd.read_csv(f"results/value_system_identification/def_{name_of_files}_similarities_agou_learning_from_expert_train.csv")
            mce_irl.name_method = name_of_files+str(npr)
            mce_irl.training_profiles = [npr, ]
            mce_irl.adapt_policy_to_profile(learned_profile, use_cached_policies=False)

        df_train, df_test, train_data, test_data, similarities_train, similarities_test = mce_irl.expected_trajectory_cost_calculation(on_profiles=EXAMPLE_PROFILES, learned_profiles=learned_profiles_to_targets, stochastic_sampling=False, n_samples_per_od=None, custom_cost_preprocessing=FeaturePreprocess.NORMALIZATION, repeat_society=N_OD_SPLITS_FOR_SIMULATING_SOCIETY, new_test_data=new_test_data, name_method=name_of_files+'learning_from_society', plot_histograms=PLOT_HISTS)
        
        df_train.to_csv(f"results/value_system_identification/def_{name_of_files}_statistics_learning_from_society_train.csv")
        df_test.to_csv(f"results/value_system_identification/def_{name_of_files}_statistics_learning_from_society_test.csv")
        #df_train.to_markdown(f"results/value_system_identification/def_{name_of_files}_statistics_learning_from_expert_train.md")
        #df_test.to_markdown(f"results/value_system_identification/def_{name_of_files}_statistics_learning_from_expert_test.md")
        for metric, df in similarities_train.items():

            df.to_csv(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_learning_from_society_train.csv")
            #df.to_markdown(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_learning_from_expert_train.md")
        for metric, df in similarities_test.items():

            df.to_csv(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_learning_from_society_test.csv")
            #df.to_markdown(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_learning_from_expert_test.md")
           
        
        # Value System Identification from expert

        origdf = pd.read_csv(f"results/value_system_identification/{name_of_files}_similarities_agou_learning_from_expert_train.csv")
        origdf['Learned Profile'] = origdf['Learned Profile'].apply(lambda x: literal_eval(x))
        origdf['Target Profile'] = origdf['Target Profile'].apply(lambda x: literal_eval(x))
        learned_profiles_to_targets = list(zip( origdf['Learned Profile'], origdf['Target Profile']))
        mce_irl.policies_per_profile.clear()
        
        
        for learned_profile, npr in learned_profiles_to_targets:
            #origdf = pd.read_csv(f"results/value_system_identification/def_{name_of_files}_similarities_agou_learning_from_expert_train.csv")
            mce_irl.name_method = name_of_files+str(npr)
            mce_irl.training_profiles = [npr, ]
            mce_irl.adapt_policy_to_profile(learned_profile, use_cached_policies=False)

        df_train, df_test, train_data, test_data, similarities_train, similarities_test = mce_irl.expected_trajectory_cost_calculation(on_profiles=EXAMPLE_PROFILES, learned_profiles=learned_profiles_to_targets, stochastic_sampling=False, n_samples_per_od=None, custom_cost_preprocessing=FeaturePreprocess.NORMALIZATION, repeat_society=N_OD_SPLITS_FOR_SIMULATING_SOCIETY, new_test_data=new_test_data, name_method=name_of_files+'learning_from_expert', plot_histograms=PLOT_HISTS)
        
        df_train.to_csv(f"results/value_system_identification/def_{name_of_files}_statistics_learning_from_expert_train.csv")
        df_test.to_csv(f"results/value_system_identification/def_{name_of_files}_statistics_learning_from_expert_test.csv")
        #df_train.to_markdown(f"results/value_system_identification/def_{name_of_files}_statistics_learning_from_expert_train.md")
        #df_test.to_markdown(f"results/value_system_identification/def_{name_of_files}_statistics_learning_from_expert_test.md")
        for metric, df in similarities_train.items():

            df.to_csv(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_learning_from_expert_train.csv")
            #df.to_markdown(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_learning_from_expert_train.md")
        for metric, df in similarities_test.items():

            df.to_csv(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_learning_from_expert_test.csv")
            #df.to_markdown(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_learning_from_expert_test.md")
                  
        # Without retraining, check whether the sampled trajs are consistent with the profile.

        origdf = pd.read_csv(f"results/value_system_identification/{name_of_files}_similarities_agou_learning_from_expert_train.csv")
        origdf['Learned Profile'] = origdf['Learned Profile'].apply(lambda x: literal_eval(x))
        origdf['Target Profile'] = origdf['Target Profile'].apply(lambda x: literal_eval(x))
        learned_profiles_to_targets = list(zip( origdf['Target Profile'], origdf['Target Profile']))
        mce_irl.policies_per_profile.clear()
        
        
        for learned_profile, npr in learned_profiles_to_targets:
            #origdf = pd.read_csv(f"results/value_system_identification/def_{name_of_files}_similarities_agou_learning_from_expert_train.csv")
            mce_irl.name_method = name_of_files+str(npr)
            mce_irl.training_profiles = [npr, ]
            mce_irl.adapt_policy_to_profile(npr, use_cached_policies=False)

        df_train, df_test, train_data, test_data, similarities_train, similarities_test = mce_irl.expected_trajectory_cost_calculation(on_profiles=EXAMPLE_PROFILES, learned_profiles=learned_profiles_to_targets, stochastic_sampling=False, n_samples_per_od=None, custom_cost_preprocessing=FeaturePreprocess.NORMALIZATION, new_test_data=new_test_data, name_method=name_of_files+'given_profile', plot_histograms=PLOT_HISTS)
        
        df_train.to_csv(f"results/value_system_identification/def_{name_of_files}_statistics_for_unseen_profile_train.csv")
        df_test.to_csv(f"results/value_system_identification/def_{name_of_files}_statistics_for_unseen_profile_test.csv")
        #df_train.to_markdown(f"results/value_system_identification/def_{name_of_files}_statistics_for_unseen_profile_train_train.md")
        #df_test.to_markdown(f"results/value_system_identification/def_{name_of_files}_statistics_for_unseen_profile_test_test.md")

        for metric, df in similarities_train.items():

            df.to_csv(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_for_unseen_profile_train.csv")
            #df.to_markdown(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_for_unseen_profile_train.md")
        for metric, df in similarities_test.items():

            df.to_csv(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_for_unseen_profile_test.csv")
            #df.to_markdown(f"results/value_system_identification/def_{name_of_files}_similarities_{metric}_for_unseen_profile_test.md")
       
# Define colors
colors = {
    'sus': ('darkgreen', 'green'),
    'eff': ('red', 'magenta'),
    'sec': ('blue', 'cyan')
}
similarities = (
    'agou',
    'jaccard',
    'visitation_count'
)

def swap_columns(df):
    columns = df.columns.tolist()
    columns[1], columns[2] = columns[2], columns[1]
    return df[columns]

for society_or_expert in ['expert', 'society']:
    for test_or_train in ['test','train']:
        for combination, n_profiles_and_name in combinations.items():
            n_profiles, name_of_files = n_profiles_and_name
            dfs_sim = dict()
            for similarity in similarities:
                
                df = pd.read_csv(f"results/value_system_identification/def_{name_of_files}_similarities_{similarity}_learning_from_{society_or_expert}_{test_or_train}.csv")
                df_only_means = df.copy()
                print(df.columns[3:])

                for col in df.columns[3:]:
                    
                    df_only_means[f'{col}'] = df[col].apply(lambda x: literal_eval(x)[0] if isinstance(x,str) else x)
                
                df_only_means[df.columns[1]] = df[df.columns[1]].apply(lambda x: tuple([float(f'{a:.2f}') for a in literal_eval(x)]))
                expert_df_swapped = swap_columns(df_only_means).iloc[:, 1:]
                expert_df_swapped.to_latex(f'results/value_system_identification/latex/{name_of_files}_similarities_{similarity}_learning_from_{society_or_expert}_{test_or_train}.tex',index=False,float_format='%.2f')
                
                dfs_sim[similarity] = expert_df_swapped
            #header = " & ".join([r"{\makecell{" + col.replace("_", r"\\") + "}}" if ic >= 2 else r"\makecell{" + col.replace("_", r"\\") + "}" for ic, col in enumerate(expert_df_swapped.columns[0:5])])
            header = " & ".join([r"{\makecell{" + "Original Profile" + "}}", r"{\makecell{" + "Learned Profile" + "}}"])
            #header = header + " & " + r"{\makecell{" + "Original Sus" + "}}"
            #header = header + " & " + r"{\makecell{" + "Originial Sec" + "}}"
            #header = header + " & " + r"{\makecell{" + "Learned Sus" + "}}"
            #header = header + " & " + r"{\makecell{" + "Original Eff" + "}}"
            #header = header + " & " + r"{\makecell{" + "Learned Sec" + "}}"
            #header = header + " & " + r"{\makecell{" + "Learned Eff" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{PS}_{P}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{PS}_{sus}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{PS}_{sec}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{PS}_{eff}$" + "}}"

            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{JAC}_{P}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{JAC}_{sus}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{JAC}_{sec}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{JAC}_{eff}$" + "}}"

            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{TVC}$" + "}}"
            #header = header + " & " + r"{\makecell{" + r"$\textit{TVC}_{sec}$ sim." + "}}"
            #header = header + " & " + r"{\makecell{" + r"$\textit{TVC}_{eff}$ sim." + "}}"

            #header = header + " & ".join([r"{\makecell{" + col.replace("_", r"\\") + "}}" if ic >= 2 else r"\makecell{" + col.replace("_", r"\\") + "}" for ic, col in enumerate(expert_df_swapped.columns[2:5])])
            
            # Export to LaTeX

            df = pd.read_csv(f"results/value_system_identification/def_{name_of_files}_statistics_learning_from_{society_or_expert}_{test_or_train}.csv")
            
            print(df.columns[1])
            policies = df[df.columns[1]]
            
            expert_df = df[df['Policy'].str.contains('Expert' if society_or_expert == 'expert' else 'Society', case=False, na=False)]
            print(expert_df['sec'])
            
            learned_df = df[df['Policy'].str.contains('learn', case=False, na=False)]
            print(learned_df['sec'])

            for col in ['sus', 'eff', 'sec']:
                
                expert_df[f'{col}_mean'] = expert_df[col].apply(lambda x: literal_eval(x)[0])
                
                expert_df[f'{col}_std'] = expert_df[col].apply(lambda x: literal_eval(x)[1])
            
            # Separate the averages and standard deviations for learned_df
            for col in ['sus', 'eff', 'sec']:
                learned_df[f'{col}_mean'] = learned_df[col].apply(lambda x: literal_eval(x)[0])
                learned_df[f'{col}_std'] = learned_df[col].apply(lambda x: literal_eval(x)[1])
            
            #header = " & ".join([r"{\makecell{" + col.rep
            if test_or_train == 'test' and 'all' in name_of_files:
                with open(f'results/value_system_identification/latex/TFM_{name_of_files}_{test_or_train}_{society_or_expert}.tex', 'w') as f:
                    f.write(r'\begin{table*}[h!]' + '\n')
                    f.write(r'\centering' + '\n')
                    f.write(r'\resizebox{\columnwidth}{!}{')
                    f.write(r'\begin{tabularx}{1.3\textwidth}{' +  'll' + "|YYYY|YYYY|Y" + '}' + '\n')
                    f.write(r'\hline' + '\n')
                    f.write(header + r' \\' + '\n')
                    f.write(r'\hline' + '\n')
                    for index, row in dfs_sim[similarity].iterrows():
                        row_data = " & ".join([f"{val:.2f}" if isinstance(val, float) else str(val) for val in row[0:2]])
                        print(learned_df['sus_mean'].to_list()[index])

                        for similarity in similarities:
                            row_data = row_data + " & " + f"{dfs_sim[similarity]['Similarity with target profile agent' if society_or_expert == 'expert' else 'Similarity with target profile society'].to_list()[index]:.2f}"
                            
                            if similarity != 'visitation_count':
                                row_data = row_data + " & " + f"{dfs_sim[similarity]['Similarity in terms of Sus'].to_list()[index]:.2f}"
                                row_data = row_data + " & " + f"{dfs_sim[similarity]['Similarity in terms of Sec'].to_list()[index]:.2f}"
                                row_data = row_data + " & " + f"{dfs_sim[similarity]['Similarity in terms of Eff'].to_list()[index]:.2f}"
                                
                            
                        #row_data = row_data + " & " + " & ".join([f"{val:.2f}" if isinstance(val, float) else str(val) for val in row[2:5]])
                        
                        f.write(row_data + r' \\' + '\n')
                    f.write(r'\hline' + '\n')
                    f.write(r'\end{tabularx}' + '\n')
                    f.write(r'}')
                    #f.write(r'\caption{Results for value system identification, learning from' + (r' expert agents with different profiles' if society_or_expert == 'expert' else r' different profiled societies') + r'. The learned profile represents a linear combination value system alignment function that the algorithm found the most coherent with observed behaviour. \textit{' + f'{similarity}' + r'} similarities with different route datasets of the learned trajectories are shown. The first three columns are similarities with the observed trajectories ' + f'(from the {test_or_train} set)' + r', The next three show the similarity with routes taken by agents with pure profiles, each in terms of their target value. The last two show similarity in terms of the target profile with the routes taken by an agent with that profile, and by a society with that profile, respectively.}' + '\n')
                    f.write(r'\caption{Results for value system identification, learning from' + (r' individual agents with different profiles' if society_or_expert == 'expert' else r' different profiled societies') + r'. The learned profile represents a linear combination value system alignment function that the algorithm found the most coherent with observed behaviour. The expected value alignments of the routes sampled with the original profile and the learned profile are shown in the first 6 columns. The last three columns represent the \textit{' + f'{similarity}' + r'} similarities with the observed trajectories ' + f'(from the {test_or_train} set)' + r'  according to the three values.}' + '\n')
                    if society_or_expert == 'expert':
                        f.write(r'\label{tab:simfromexpert}' + '\n')
                    if society_or_expert == 'society':
                        f.write(r'\label{tab:simfromsociety}' + '\n')
                    f.write(r'\end{table*}' + '\n')

            

for test_or_train in ['test','train']:
        for combination, n_profiles_and_name in combinations.items():
            dfs_sim = dict()
            n_profiles, name_of_files = n_profiles_and_name
            for similarity in similarities:
                
                
                df = pd.read_csv(f"results/value_system_identification/def_{name_of_files}_similarities_{similarity}_for_unseen_profile_{test_or_train}.csv")
                df_only_means = df.copy()
                print(df.columns[3:])

                for col in df.columns[3:]:
                    
                    df_only_means[f'{col}'] = df[col].apply(lambda x: literal_eval(x)[0] if isinstance(x,str) else x)
                
                df_only_means[df.columns[1]] = df[df.columns[1]].apply(lambda x: tuple([float(f'{a:.2f}') for a in literal_eval(x)]))
                expert_df_swapped = swap_columns(df_only_means).iloc[:, 1:]
                expert_df_swapped.to_latex(f'results/value_system_identification/latex/{name_of_files}_similarities_{similarity}_for_unseen_profile_{test_or_train}.tex',index=False,float_format='%.2f')
                
                dfs_sim[similarity] = expert_df_swapped
            #header = " & ".join([r"{\makecell{" + col.replace("_", r"\\") + "}}" if ic >= 2 else r"\makecell{" + col.replace("_", r"\\") + "}" for ic, col in enumerate(expert_df_swapped.columns[0:5])])
            header = r"{\makecell{" + r"Profile" + "}}"
            #header = header + " & " + r"{\makecell{" + "Original Sus" + "}}"
            #header = header + " & " + r"{\makecell{" + "Originial Sec" + "}}"
            #header = header + " & " + r"{\makecell{" + "Learned Sus" + "}}"
            #header = header + " & " + r"{\makecell{" + "Original Eff" + "}}"
            #header = header + " & " + r"{\makecell{" + "Learned Sec" + "}}"
            #header = header + " & " + r"{\makecell{" + "Learned Eff" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{PS}_{P}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{PS}_{sus}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{PS}_{sec}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{PS}_{eff}$" + "}}"

            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{JAC}_{P}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{JAC}_{sus}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{JAC}_{sec}$" + "}}"
            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{JAC}_{eff}$" + "}}"

            header = header + " & " + r"\rotatebox{0}{\makecell{" + r"$\textit{TVC}$" + "}}"
            #header = header + " & " + r"{\makecell{" + r"$\textit{TVC}_{sec}$ sim." + "}}"
            #header = header + " & " + r"{\makecell{" + r"$\textit{TVC}_{eff}$ sim." + "}}"

            #header = header + " & ".join([r"{\makecell{" + col.replace("_", r"\\") + "}}" if ic >= 2 else r"\makecell{" + col.replace("_", r"\\") + "}" for ic, col in enumerate(expert_df_swapped.columns[2:5])])
            
            # Export to LaTeX

            df = pd.read_csv(f"results/value_system_identification/def_{name_of_files}_statistics_for_unseen_profile_{test_or_train}.csv")
            
            print(df.columns[1])
            policies = df[df.columns[1]]
            
            expert_df = df[df['Policy'].str.contains('Expert' if society_or_expert == 'expert' else 'Society', case=False, na=False)]
            print(expert_df['sec'])
            
            learned_df = df[df['Policy'].str.contains('learn', case=False, na=False)]
            print(learned_df['sec'])

            for col in ['sus', 'eff', 'sec']:
                
                expert_df[f'{col}_mean'] = expert_df[col].apply(lambda x: literal_eval(x)[0])
                
                expert_df[f'{col}_std'] = expert_df[col].apply(lambda x: literal_eval(x)[1])
            
            # Separate the averages and standard deviations for learned_df
            for col in ['sus', 'eff', 'sec']:
                learned_df[f'{col}_mean'] = learned_df[col].apply(lambda x: literal_eval(x)[0])
                learned_df[f'{col}_std'] = learned_df[col].apply(lambda x: literal_eval(x)[1])
            if test_or_train == 'test' and 'all' in name_of_files:
                with open(f'results/value_system_identification/latex/TFM_{name_of_files}_{test_or_train}_for_unseen_profile.tex', 'w') as f:
                    f.write(r'\begin{table*}[h!]' + '\n')
                    f.write(r'\centering' + '\n')
                    f.write(r'\resizebox{\columnwidth}{!}{')
                    f.write(r'\begin{tabularx}{1.1\textwidth}{' +  'l' + "|YYYY|YYYY|Y" + '}' + '\n')
                    f.write(r'\hline' + '\n')
                    f.write(header + r' \\' + '\n')
                    f.write(r'\hline' + '\n')
                    for index, row in dfs_sim[similarity].iterrows():
                        print(row, similarity, )
                        row_data = " & ".join([f"{val:.2f}" if isinstance(val, float) else str(val) for val in row[0:1]])
                        

                        for similarity in similarities:
                            row_data = row_data + " & " + f"{dfs_sim[similarity]['Similarity with target profile agent'].to_list()[index]:.2f}"
                            
                            if similarity != 'visitation_count':
                                row_data = row_data + " & " + f"{dfs_sim[similarity]['Similarity in terms of Sus'].to_list()[index]:.2f}"
                                row_data = row_data + " & " + f"{dfs_sim[similarity]['Similarity in terms of Sec'].to_list()[index]:.2f}"
                                row_data = row_data + " & " + f"{dfs_sim[similarity]['Similarity in terms of Eff'].to_list()[index]:.2f}"
                                
                            
                        #row_data = row_data + " & " + " & ".join([f"{val:.2f}" if isinstance(val, float) else str(val) for val in row[2:5]])
                        
                        f.write(row_data + r' \\' + '\n')
                    f.write(r'\hline' + '\n')
                    f.write(r'\end{tabularx}' + '\n')
                    f.write(r'}')
                    #f.write(r'\caption{Results for value system identification, learning from' + (r' expert agents with different profiles' if society_or_expert == 'expert' else r' different profiled societies') + r'. The learned profile represents a linear combination value system alignment function that the algorithm found the most coherent with observed behaviour. \textit{' + f'{similarity}' + r'} similarities with different route datasets of the learned trajectories are shown. The first three columns are similarities with the observed trajectories ' + f'(from the {test_or_train} set)' + r', The next three show the similarity with routes taken by agents with pure profiles, each in terms of their target value. The last two show similarity in terms of the target profile with the routes taken by an agent with that profile, and by a society with that profile, respectively.}' + '\n')
                    f.write(r'\caption{Results for value system identification, learning from' + (r' individual agents with different profiles' if society_or_expert == 'expert' else r' different profiled societies') + r'. The learned profile represents a linear combination value system alignment function that the algorithm found the most coherent with observed behaviour. The expected value alignments of the routes sampled with the original profile and the learned profile are shown in the first 6 columns. The last three columns represent the \textit{' + f'{similarity}' + r'} similarities with the observed trajectories ' + f'(from the {test_or_train} set)' + r'  according to the three values.}' + '\n')
                    
                    f.write(r'\label{tab:simgivenprofile}' + '\n')
                    f.write(r'\end{table*}' + '\n')
            

