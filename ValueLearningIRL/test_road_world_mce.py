from functools import partial
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
from deep_maxent_value_grounding_learning import DEST
from src.envs.roadworld_env import FixedDestRoadWorldGymPOMDP
from src.me_irl_for_vsl import MaxEntropyIRLForVSL, PolicyApproximators, check_coherent_rewards, mce_partition_fh
from src.me_irl_for_vsl_plot_utils import plot_learned_and_expert_occupancy_measures, plot_learned_and_expert_rewards, plot_learned_to_expert_policies
from src.network_env import DATA_FOLDER, FeaturePreprocess, FeatureSelection, RoadWorldGymPOMDP
from src.reward_functions import ProfiledRewardFunction, TrainingModes
from src.utils.load_data import ini_od_dist
from src.values_and_costs import BASIC_PROFILES
from src.vsl_policies import VAlignedDictSpaceActionPolicy, profiled_society_sampler, profiled_society_traj_sampler_from_policy, random_sampler_among_trajs, sampler_from_policy
from utils import sample_example_profiles
from torch import nn
PREPROCESSING = FeaturePreprocess.NORMALIZATION
USE_OM = True
STOCHASTIC_EXPERT = False
LEARN_STOCHASTIC_POLICY= False


EXPERT_FIXED_TRAJECTORIES = False

N_EXPERT_SAMPLES_PER_OD = 1 if USE_OM is True else 10 if STOCHASTIC_EXPERT else 1# change True only when USE_OM is not True.
FEATURE_SELECTION = FeatureSelection.ONLY_COSTS
USE_OPTIMAL_REWARD_AS_FEATURE =False
POLICY_APPROXIMATION_METHOD = PolicyApproximators.MCE_ORIGINAL 
SINGLE_DEST = True
PROFILE = (1.0,0.0,0.0)
SEED = 26

#INITIAL_STATE_DISTRIBUTION = 'uniform' # or 'default' or a specfic probability distribution on the encrypted states.

SOCIETY_EXPERT  =True

N_SEEDS = 100
N_SEEDS_MINIBATCH=20
N_EXPERT_SAMPLES_PER_SEED_MINIBATCH = 10
N_REWARD_SAMPLES_PER_ITERATION =30

LOGINTERVAL=1
HORIZON = 30


if __name__ == '__main__':

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    log_interval = 10  # interval between training status logs

    cv = 0  # cross validation process [0, 1, 2, 3, 4] # TODO (?)
    size = 100  # size of training data [100, 1000, 10000]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    """environment"""
    edge_p = f"{DATA_FOLDER}/edge.txt"
    network_p = f"{DATA_FOLDER}/transit.npy"
    path_feature_p = f"{DATA_FOLDER}/feature_od.npy"
    train_p = f"{DATA_FOLDER}/cross_validation/train_CV%d_size%d.csv" % (cv, size)
    test_p = f"{DATA_FOLDER}/cross_validation/test_CV%d.csv" % cv
    node_p = f"{DATA_FOLDER}/node.txt"


    od_list, od_dist = ini_od_dist(train_p)
    
    env_creator = partial(RoadWorldGymPOMDP, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path = path_feature_p, 
                        pre_reset=(od_list, od_dist), profile=PROFILE, visualize_example=False, horizon=HORIZON,
                        feature_selection=FEATURE_SELECTION,
                        feature_preprocessing=PREPROCESSING, 
                        use_optimal_reward_per_profile=USE_OPTIMAL_REWARD_AS_FEATURE)
    env_single = env_creator()

    od_list = [str(state) + '_' + str(DEST) for state in env_single.valid_edges]


    if SINGLE_DEST:
        od_list = [str(state) + '_' + str(DEST) for state in env_single.valid_edges]
        env_creator = partial(RoadWorldGymPOMDP, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path = path_feature_p, pre_reset=(od_list, od_dist), profile=PROFILE, visualize_example=False, horizon=HORIZON,
                        feature_selection=FEATURE_SELECTION,
                        feature_preprocessing=PREPROCESSING, 
                        use_optimal_reward_per_profile=USE_OPTIMAL_REWARD_AS_FEATURE)
        env_single = env_creator()

    
    env_real = FixedDestRoadWorldGymPOMDP(env=env_single, with_destination=DEST)
    env_real.reset(seed=SEED)

    profiles  = sample_example_profiles(profile_variety=5,n_values=3)
    profile_to_matrix = {}
    profile_to_assumed_matrix = {}
    
    for w in profiles:
        reward = env_real.reward_matrix_per_align_func(w)
        
        _,_, assumed_expert_pi = mce_partition_fh(env_real, discount=1.0,
                                             reward=reward,
                                             approximator_kwargs={'value_iteration_tolerance': 0.00001, 'iterations': 1000},
                                             policy_approximator=POLICY_APPROXIMATION_METHOD,deterministic= not STOCHASTIC_EXPERT )
        profile_to_assumed_matrix[w] = assumed_expert_pi

    expert_policy_train = VAlignedDictSpaceActionPolicy(policy_per_va_dict = profile_to_assumed_matrix, env = env_real, state_encoder=None)
    expert_trajs_train = expert_policy_train.obtain_trajectories(n_seeds=N_SEEDS, seed=SEED, stochastic=STOCHASTIC_EXPERT, repeat_per_seed=N_EXPERT_SAMPLES_PER_OD, with_alignfunctions=profiles, t_max=HORIZON)
    expert_policy = expert_policy_train

    if EXPERT_FIXED_TRAJECTORIES:
        vgl_expert_train_sampler = partial(random_sampler_among_trajs, expert_trajs_train)
        vsi_expert_train_sampler = partial(random_sampler_among_trajs, expert_trajs_train)
    else:
        vgl_expert_train_sampler = partial(sampler_from_policy, expert_policy_train, stochastic=STOCHASTIC_EXPERT, horizon=HORIZON)
        vsi_expert_train_sampler = partial(sampler_from_policy, expert_policy_train, stochastic=STOCHASTIC_EXPERT, horizon=HORIZON)
    
    #vsi_expert_train_sampler = partial(profiled_society_traj_sampler_from_policy, expert_policy_train, stochastic=STOCHASTIC_EXPERT, horizon=HORIZON)


    reward_net = ProfiledRewardFunction(
        environment=env_real,
        use_state=False,
        use_action=False,
        use_next_state=True,
        use_done=False,
        use_one_hot_state_action=False,
        hid_sizes=[3,],
        reward_bias=-0.0,
        negative_grounding_layer=True,
        activations=[nn.Identity, nn.Identity],
    )

    max_entropy_algo = MaxEntropyIRLForVSL(
            env=env_real,
            reward_net=reward_net,
            log_interval=LOGINTERVAL,
            vsi_optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0000},
            vgl_optimizer_kwargs={"lr": 0.3, "weight_decay": 0.0000},
            vgl_expert_policy=expert_policy_train,
            vsi_expert_policy=expert_policy_train,
            vgl_expert_sampler=vgl_expert_train_sampler,
            vsi_expert_sampler=vsi_expert_train_sampler,
            target_align_func_sampler=profiled_society_sampler if SOCIETY_EXPERT else lambda al_func: al_func,

            demo_om_from_policy=USE_OM,
    
            vgl_target_align_funcs=BASIC_PROFILES,
            vsi_target_align_funcs=profiles,

            vc_diff_epsilon=1e-3,
            gradient_norm_epsilon=1e-6,
            training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
            initial_state_distribution_train=env_real.initial_state_dist,
            initial_state_distribution_test=env_real.initial_state_dist,
            policy_approximator = POLICY_APPROXIMATION_METHOD,
            learn_stochastic_policy = LEARN_STOCHASTIC_POLICY,
            expert_is_stochastic=STOCHASTIC_EXPERT,
            
            environment_is_stochastic=False,
            discount=1.0,
            
            )
    check_coherent_rewards(max_entropy_algo, align_funcs_to_test=profiles, real_grounding=nn.Identity(), policy_approx_method=POLICY_APPROXIMATION_METHOD, stochastic_expert=STOCHASTIC_EXPERT, stochastic_learner=LEARN_STOCHASTIC_POLICY)
    
    
    learned_grounding, learned_rewards, reward_net_learned, linf_delta_per_align_fun, grad_norm_per_align_func = max_entropy_algo.train(max_iter=50, 
                                                            mode=TrainingModes.VALUE_GROUNDING_LEARNING,n_seeds_for_sampled_trajectories=N_SEEDS_MINIBATCH,
                                                            n_sampled_trajs_per_seed=N_EXPERT_SAMPLES_PER_SEED_MINIBATCH,
                                                            use_probabilistic_reward=False,n_reward_reps_if_probabilistic_reward=N_REWARD_SAMPLES_PER_ITERATION)
        
    plot_learned_to_expert_policies(expert_policy, max_entropy_algo, vsi_or_vgl='vgl', namefig='test_roadworld_vgl',show=False)

    plot_learned_and_expert_rewards(env_real, max_entropy_algo, learned_rewards, vsi_or_vgl='vgl', namefig='test_roadworld_vgl')
    
    plot_learned_and_expert_occupancy_measures(env_real,max_entropy_algo,expert_policy,learned_rewards,vsi_or_vgl='vgl', namefig='test_roadworld_vgl')
    
    learned = max_entropy_algo.learned_policy_per_va.obtain_trajectory(alignment_function=profiles[0], seed=5686, stochastic=LEARN_STOCHASTIC_POLICY, exploration=0, only_states=True)
    real = expert_policy.obtain_trajectory(alignment_function=profiles[0], seed=5686, stochastic=STOCHASTIC_EXPERT, exploration=0, only_states=True)
    print("EXAMPLE TRAJS:")
    print(learned)
    print(real)
    print("LEARNED GROUNDING:")
    print(learned_grounding)


    target_align_funcs_to_learned_align_funcs, learned_rewards, reward_net_per_target_va, linf_delta_per_align_fun, grad_norm_per_align_func = max_entropy_algo.train(max_iter=50, 
                                                        mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                                        assumed_grounding=nn.Identity().requires_grad_(False),
                                                        n_seeds_for_sampled_trajectories=N_SEEDS_MINIBATCH,
                                                        n_sampled_trajs_per_seed=N_EXPERT_SAMPLES_PER_SEED_MINIBATCH,
                                                        use_probabilistic_reward=True,n_reward_reps_if_probabilistic_reward=N_REWARD_SAMPLES_PER_ITERATION)
    
    
    
    plot_learned_to_expert_policies(expert_policy, max_entropy_algo, target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs, vsi_or_vgl='vsi', namefig='test_roadworld_vsi')

    plot_learned_and_expert_rewards(env_real, max_entropy_algo, learned_rewards, vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs, namefig='test_roadworld_vsi')

    plot_learned_and_expert_occupancy_measures(env_real,max_entropy_algo,expert_policy,learned_rewards,vsi_or_vgl='vsi',target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs, namefig='test_roadworld_vsi')
    
    if False: # TODO Test the abve two functions correspond to the following code,
        fig, axes = plt.subplots(2, len(max_entropy_algo.vsi_target_align_funcs), figsize=(16, 8))
        for i, al in enumerate(max_entropy_algo.vsi_target_align_funcs):
            # Plot the first matrix
            
            im1 = axes[0,i].imshow(max_entropy_algo.learned_policy_per_va.policy_per_va(target_align_funcs_to_learned_align_funcs[al]), cmap='viridis', vmin=0, vmax=1, interpolation='none', aspect=expert_policy.policy_per_va(al).shape[1]/expert_policy.policy_per_va(al).shape[0])
            axes[0,i].set_title(f'VSI: Predicted Policy Matrix ({target_align_funcs_to_learned_align_funcs[al]})')
            axes[0,i].set_xlabel('Dimension M')
            axes[0,i].set_ylabel('Dimension N')
            fig.colorbar(im1, ax=axes[0,i], orientation='vertical', label='Value')

            # Plot the second matrix
            #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
            im2 = axes[1,i].imshow(expert_policy.policy_per_va(al), cmap='viridis', interpolation='none', vmin=0, vmax=1, aspect=expert_policy.policy_per_va(al).shape[1]/expert_policy.policy_per_va(al).shape[0])
            axes[1,i].set_title(f'VSI: Real Policy Matrix ({al})')
            axes[1,i].set_xlabel('Dimension M')
            axes[1,i].set_ylabel('Dimension N')
            fig.colorbar(im2, ax=axes[1,i], orientation='vertical', label='Value')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()

        fig, axes = plt.subplots(2, len(max_entropy_algo.vsi_target_align_funcs), figsize=(16, 8))
        for i, al in enumerate(max_entropy_algo.vsi_target_align_funcs):
            # Plot the first matrix
            
            im1 = axes[0,i].imshow(learned_rewards(al), cmap='viridis', interpolation='none', aspect=learned_rewards(al).shape[1]/learned_rewards(al).shape[0])
            axes[0,i].set_title(f'VSI: Predicted Reward Matrix ({target_align_funcs_to_learned_align_funcs[al]})')
            axes[0,i].set_xlabel('Dimension M')
            axes[0,i].set_ylabel('Dimension N')
            fig.colorbar(im1, ax=axes[0,i], orientation='vertical', label='Value')

            # Plot the second matrix
            #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
            w = al
            reward = np.sum([w[i]*env_real.reward_matrix_per_align_func(bp) for i,bp in enumerate(BASIC_PROFILES)], axis=0)
            im2 = axes[1,i].imshow(reward, cmap='viridis', interpolation='none', aspect=learned_rewards(al).shape[1]/learned_rewards(al).shape[0])
            axes[1,i].set_title(f'VSI: Real Reward Matrix ({al})')
            axes[1,i].set_xlabel('Dimension M')
            axes[1,i].set_ylabel('Dimension N')
            fig.colorbar(im2, ax=axes[1,i], orientation='vertical', label='Value')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()
        
    # TODO: Vender explicabilidad por el metodo de las probabilidades.
    # TODO: Vender precisión en términos de expected visitation counts y de expected feature counts
    # TODO: Podemos aprender el decision making con funciones de alineamiento con el VS (f^j) o bien las probabilidades que reflejan la importancia positiva de cada valor.
    # TODO: PASAR A STATE_ACTION visitation counts.
