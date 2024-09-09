from functools import partial
import pickle
import gymnasium as gym
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch
from firefighters_use_case.env import HighRiseFireEnv
from firefighters_use_case.execution import example_execution
from firefighters_use_case.scalarisation import deterministic_optimal_policy_calculator, stochastic_optimal_policy_calculator
import src.envs
from firefighters_use_case.constants import ACTION_AGGRESSIVE_FIRE_SUPPRESSION
from firefighters_use_case.pmovi import get_particular_policy, learn_and_do, pareto_multi_objective_value_iteration, scalarise_q_function
from src.envs.firefighters_env import FeatureSelectionFFEnv, FireFightersEnv
from src.me_irl_for_vsl import MaxEntropyIRLForVSL, mce_partition_fh
from src.reward_functions import PositiveBoundedLinearModule, ProfileLayer, ProfiledRewardFunction, TrainingModes
from src.vsl_policies import VAlignedDictSpaceActionPolicy, ValueSystemLearningPolicy, profiled_society_sampler, profiled_society_traj_sampler_from_policy, random_sampler_among_trajs, sampler_from_policy
from test_firefighters import *
from utils import sample_example_profiles, train_test_split_initial_state_distributions
from torch import nn


N_REWARD_SAMPLES_PER_ITERATION = 20
N_SEEDS_MINIBATCH = 50
N_EXPERT_SAMPLES_PER_SEED_MINIBATCH = 2

STOCHASTIC_EXPERT = True
LEARN_STOCHASTIC_POLICY = True

SOCIETY_EXPERT = True
DEMO_OM_FROM_POLICY = True
EXPERT_FIXED_TRAJECTORIES = False

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    discount_factor = 0.7
    # Example usage
    
    env_real: FireFightersEnv = gym.make('FireFighters-v0', feature_selection = FEATURE_SELECTION, horizon=HORIZON, initial_state_distribution=INITIAL_STATE_DISTRIBUTION)
    env_real.reset(seed=SEED)

    train_init_state_distribution, test_init_state_distribution = train_test_split_initial_state_distributions(env_real.state_dim, 0.7)
    env_training: FireFightersEnv #=gym.make('FireFighters-v0', feature_selection = FEATURE_SELECTION, horizon=HORIZON, initial_state_distribution=train_init_state_distribution)
    env_training = env_real

    state, info = env_training.reset()
    print("Initial State:", state, info)
    action = ACTION_AGGRESSIVE_FIRE_SUPPRESSION
    next_state, rewards, done, trunc, info = env_training.step(action)
    print("Next State:", next_state)
    print("Rewards:", rewards)
    print("Done:", done)
    print("Info:", info)
    env_training.reset(seed=SEED)
    # Set to false if you already have a pretrained protocol in .pickle format
    learn = True

    if learn:
        v, q = pareto_multi_objective_value_iteration(env_real.real_env, discount_factor=discount_factor, model_used=None, pareto=True, normalized=False)
        with open(r"v_hull.pickle", "wb") as output_file:
            pickle.dump(v, output_file)

        with open(r"q_hull.pickle", "wb") as output_file:
            pickle.dump(q, output_file)

    # Returns pareto front of initial state in V-table format
    with open(r"v_hull.pickle", "rb") as input_file:
        v_func = pickle.load(input_file)

    # Returns pareto front of initial state in Q-table format
    with open(r"q_hull.pickle", "rb") as input_file:
        q_func = pickle.load(input_file)

    print("--")

    initial_state = 323
    normalisation_factor = 1 - discount_factor  # because discount factor is 1 - 0.3 = 0.7
    # Shows (normalised) Pareto front of initial state
    print("Pareto front of initial state : ", v_func[initial_state] * normalisation_factor)
    


    profiles  = sample_example_profiles(profile_variety=6,n_values=2)
    profile_to_matrix = {}
    profile_to_assumed_matrix = {}
    for w in profiles:
        scalarised_q = scalarise_q_function(q_func, objectives=2, weights=w)

        pi_matrix = stochastic_optimal_policy_calculator(scalarised_q, w, deterministic= not STOCHASTIC_EXPERT)
        profile_to_matrix[w] = pi_matrix

        _,_, assumed_expert_pi = mce_partition_fh(env_real, discount=discount_factor,
                                             reward=w[0]*env_real.reward_matrix_per_align_func((1.0,0.0))+w[1]*env_real.reward_matrix_per_align_func((0.0,1.0)),
                                             approximator_kwargs={'value_iteration_tolerance': 0.00001, 'iterations': 1000},
                                             policy_approximator=POLICY_APPROXIMATION_METHOD,deterministic= not STOCHASTIC_EXPERT )
        profile_to_assumed_matrix[w] = assumed_expert_pi

    expert_policy = VAlignedDictSpaceActionPolicy(policy_per_va_dict = profile_to_matrix if USE_PMOVI_EXPERT else profile_to_assumed_matrix, env = env_real, state_encoder=None)
    expert_policy_train = VAlignedDictSpaceActionPolicy(policy_per_va_dict = profile_to_matrix if USE_PMOVI_EXPERT else profile_to_assumed_matrix, env = env_training, state_encoder=None)
    expert_trajs = expert_policy.obtain_trajectories(n_seeds=N_SEEDS, seed=SEED, stochastic=STOCHASTIC_EXPERT, repeat_per_seed=N_EXPERT_SAMPLES_PER_SEED, with_alignfunctions=profiles, t_max=HORIZON)
    expert_trajs_train = expert_policy_train.obtain_trajectories(n_seeds=N_SEEDS, seed=SEED, stochastic=STOCHASTIC_EXPERT, repeat_per_seed=N_EXPERT_SAMPLES_PER_SEED, with_alignfunctions=profiles, t_max=HORIZON)
    
    if EXPERT_FIXED_TRAJECTORIES:
        vgl_expert_train_sampler = partial(random_sampler_among_trajs, expert_trajs_train)
        vsi_expert_train_sampler = partial(random_sampler_among_trajs, expert_trajs_train)
    else:
        vgl_expert_train_sampler = partial(sampler_from_policy, expert_policy_train, stochastic=STOCHASTIC_EXPERT, horizon=HORIZON)
        vsi_expert_train_sampler = partial(sampler_from_policy, expert_policy_train, stochastic=STOCHASTIC_EXPERT, horizon=HORIZON)
    
    reward_net = ProfiledRewardFunction(
        environment=env_training,
        use_state=True,
        use_action=USE_ACTION,
        use_next_state=False,
        use_done=False,
        hid_sizes=[2,],
        reward_bias=-0.0,
        basic_layer_classes= [nn.Linear, ProfileLayer],
        use_one_hot_state_action=USE_ACTION and FEATURE_SELECTION == FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS and USE_ONE_HOT_STATE_ACTION,
        activations=[nn.Tanh, nn.Identity],
        negative_grounding_layer=False
    )
    
    max_entropy_algo = MaxEntropyIRLForVSL(
            env=env_training,
            reward_net=reward_net,
            log_interval=LOG_EVAL_INTERVAL,
            vsi_optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0000},
            vgl_optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0000},
            vgl_expert_policy=expert_policy_train,
            vsi_expert_policy=expert_policy_train,
            vgl_expert_sampler=vgl_expert_train_sampler,
            vsi_expert_sampler=vsi_expert_train_sampler,
            target_align_func_sampler=profiled_society_sampler if SOCIETY_EXPERT else lambda al_func: al_func,

            demo_om_from_policy=DEMO_OM_FROM_POLICY,
    
            vgl_target_align_funcs=[profiles[0],profiles[-1]],
            vsi_target_align_funcs=profiles,

            vc_diff_epsilon=1e-3,
            gradient_norm_epsilon=1e-6,
            initial_state_distribution_train=train_init_state_distribution,
            initial_state_distribution_test=test_init_state_distribution,
            training_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,

            policy_approximator = POLICY_APPROXIMATION_METHOD,
            learn_stochastic_policy = LEARN_STOCHASTIC_POLICY,
            expert_is_stochastic= STOCHASTIC_EXPERT,
            discount=discount_factor,
            environment_is_stochastic=False
            )
    
    #vg_learned, learned_rewards = max_entropy_algo.train(max_iter=200, mode=TrainingModes.VALUE_GROUNDING_LEARNING)
    assumed_grounding = np.zeros((reward_net.input_size, 2))
    assumed_grounding[:,0] = np.reshape(env_training.reward_matrix_per_align_func(profiles[-1]), reward_net.input_size)
    assumed_grounding[:,1] = np.reshape(env_training.reward_matrix_per_align_func(profiles[0]), reward_net.input_size)
    
    # VALUE SYSTEM IDENTIFICATION:

    target_align_funcs_to_learned_align_funcs, learned_rewards, reward_net_per_target_va, linf_delta_per_align_fun, grad_norm_per_align_func = max_entropy_algo.train(max_iter=200, 
                                                        mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                                        assumed_grounding=assumed_grounding,
                                                        n_seeds_for_sampled_trajectories=N_SEEDS_MINIBATCH,
                                                        n_sampled_trajs_per_seed=N_EXPERT_SAMPLES_PER_SEED_MINIBATCH,
                                                        use_probabilistic_reward=True,n_reward_reps_if_probabilistic_reward=N_REWARD_SAMPLES_PER_ITERATION)
    
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
        pol = expert_policy.policy_per_va(al)
        if len(pol.shape) == 3:
            pol = pol[0,:,:]
        im2 = axes[1,i].imshow(pol, cmap='viridis', interpolation='none', vmin=0, vmax=1, aspect=pol.shape[1]/pol.shape[0])
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
        learned_rewards(al)
        im1 = axes[0,i].imshow(learned_rewards(al), cmap='viridis', interpolation='none', aspect=learned_rewards(al).shape[1]/learned_rewards(al).shape[0])
        axes[0,i].set_title(f'VSI: Predicted Reward Matrix ({target_align_funcs_to_learned_align_funcs[al]})')
        axes[0,i].set_xlabel('Dimension M')
        axes[0,i].set_ylabel('Dimension N')
        fig.colorbar(im1, ax=axes[0,i], orientation='vertical', label='Value')

        # Plot the second matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        w = al
        im2 = axes[1,i].imshow(w[0]*env_real.reward_matrix_per_align_func((1.0,0.0))+w[1]*env_real.reward_matrix_per_align_func((0.0,1.0)), cmap='viridis', interpolation='none', aspect=learned_rewards(al).shape[1]/learned_rewards(al).shape[0])
        axes[1,i].set_title(f'VSI: Real Reward Matrix ({al})')
        axes[1,i].set_xlabel('Dimension M')
        axes[1,i].set_ylabel('Dimension N')
        fig.colorbar(im2, ax=axes[1,i], orientation='vertical', label='Value')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

