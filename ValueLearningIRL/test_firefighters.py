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
from src.envs.firefighters_env import FeatureSelection, FireFightersEnv
from src.me_irl_for_vsl import MaxEntropyIRLForVSL, TrainingSetModes, mce_partition_fh
from src.reward_functions import PositiveBoundedLinearModule, ProfileLayer, ProfiledRewardFunction, TrainingModes
from src.vsl_policies import VAlignedDictSpaceActionPolicy
from utils import sample_example_profiles
from torch import nn

LOGINTERVAL = 1
SEED = 26
STOCHASTIC_EXPERT = False
LEARN_STOCHASTIC_POLICY = False

HORIZON = 200
USE_ACTION = True
USE_ONE_HOT_STATE_ACTION = True
DEMO_OM_FROM_POLICY = True

N_EXPERT_SAMPLES_PER_SEED = 5
FEATURE_SELECTION = FeatureSelection.ONE_HOT_OBSERVATIONS

USE_CAUSAL_ENTROPY = True

if __name__ == "__main__":
    PROFILE = (1.0, 0.0)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Example usage
    env: FireFightersEnv = gym.make('FireFighters-v0', feature_selection = FEATURE_SELECTION, horizon=HORIZON)
    state, info = env.reset()
    print("Initial State:", state, info)
    action = ACTION_AGGRESSIVE_FIRE_SUPPRESSION
    next_state, rewards, done, trunc, info = env.step(action)
    print("Next State:", next_state)
    print("Rewards:", rewards)
    print("Done:", done)
    print("Info:", info)
    env.reset(seed=SEED)
    # Set to false if you already have a pretrained protocol in .pickle format
    learn = True

    if learn:
        v, q = pareto_multi_objective_value_iteration(env.real_env, discount_factor=0.7, model_used=None, pareto=True, normalized=False)
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
    discount_factor = 0.7
    normalisation_factor = 1 - discount_factor  # because discount factor is 1 - 0.3 = 0.7
    # Shows (normalised) Pareto front of initial state
    print("Pareto front of initial state : ", v_func[initial_state] * normalisation_factor)
    


    profiles  = sample_example_profiles(profile_variety=11,n_values=2)
    profile_to_matrix = {}
    profile_to_assumed_matrix = {}
    for w in profiles:
        scalarised_q = scalarise_q_function(q_func, objectives=2, weights=w)

        pi_matrix = stochastic_optimal_policy_calculator(scalarised_q, w, deterministic= not STOCHASTIC_EXPERT)
        profile_to_matrix[w] = pi_matrix

        _,_, assumed_expert_pi = mce_partition_fh(env, discount=0.7,
                                             reward=w[0]*env.reward_matrix_per_va((1.0,0.0))+w[1]*env.reward_matrix_per_va((0.0,1.0)),value_iteration_tolerance=0.0000001,use_causal_entropy=False,deterministic= not STOCHASTIC_EXPERT )
        profile_to_assumed_matrix[w] = assumed_expert_pi
    expert_policy = VAlignedDictSpaceActionPolicy(policy_per_va_dict = profile_to_matrix, env = env, state_encoder=None)
    
    expert_trajs = expert_policy.obtain_trajectories(n_seeds=50, seed=SEED, stochastic=STOCHASTIC_EXPERT, repeat_per_seed=N_EXPERT_SAMPLES_PER_SEED, with_alignfunctions=profiles, t_max=HORIZON)
    max_len = max([len(t) for t in expert_trajs])
    
    reward_net = ProfiledRewardFunction(
        environment=env,
        use_state=True,
        use_action=USE_ACTION,
        use_next_state=False,
        use_done=False,
        hid_sizes=[2,],
        reward_bias=-0.0,
        basic_layer_classes= [nn.Linear, ProfileLayer],
        use_one_hot_state_action=USE_ACTION and FEATURE_SELECTION == FeatureSelection.ONE_HOT_OBSERVATIONS and USE_ONE_HOT_STATE_ACTION,
        activations=[nn.Tanh, nn.Identity]
    )
    
    reward_net.set_alignment_function(PROFILE)
    
    max_entropy_algo = MaxEntropyIRLForVSL(env=env,reward_net=reward_net,
                        log_interval=LOGINTERVAL,
            optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0000},
            expert_trajectories=expert_trajs,
            expert_policy=expert_policy,
            demo_om_from_policy=DEMO_OM_FROM_POLICY,
            mean_vc_diff_eps=0.01,
            training_align_funcs=[profiles[0],profiles[-1]],
            grad_l2_eps=0.00000,
            initial_state_distribution_train=None,
            initial_state_distribution_test=None,
            training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
            training_set_mode=TrainingSetModes.PROFILED_EXPERT,
            use_causal_entropy = USE_CAUSAL_ENTROPY,
            learn_stochastic_policy = LEARN_STOCHASTIC_POLICY,
            discount=discount_factor,
            name_method="MaxEntropyNew",
            vsi_expert_trajectories=[traj for traj in expert_trajs if traj.infos[0]['align_func'] not in [profiles[0], profiles[-1]]],
            vsi_expert_policy=expert_policy,
            vsi_target_align_funcs=[p for p in profiles if p not in set([profiles[0], profiles[-1]])]
            )
    
    #vg_learned, learned_rewards = max_entropy_algo.train(max_iter=200, mode=TrainingModes.VALUE_GROUNDING_LEARNING)
    assumed_grounding = np.zeros((reward_net.input_size, 2))
    assumed_grounding[:,0] = np.reshape(env.reward_matrix_per_va(profiles[0]), reward_net.input_size)
    assumed_grounding[:,1] = np.reshape(env.reward_matrix_per_va(profiles[-1]), reward_net.input_size)
    """fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    
    # Reward similarities
    for i, al in enumerate(max_entropy_algo.vsi_target_align_funcs):
        # Ensure that you do not exceed the number of subplots
        if i >= 2:
            break

        # Plot the first matrix
        im1 = axes[i, 0].imshow(profile_to_assumed_matrix[al], cmap='viridis', interpolation='none', aspect=env.reward_matrix_per_va(profiles[0]).shape[1]/env.reward_matrix_per_va(profiles[0]).shape[0])
        axes[i, 0].set_title(f'Predicted Reward Matrix ({al})')
        axes[i, 0].set_xlabel('Dimension M')
        axes[i, 0].set_ylabel('Dimension N')
        fig.colorbar(im1, ax=axes[i, 0], orientation='vertical', label='Value')

        # Plot the second matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        im2 = axes[i, 1].imshow(profile_to_matrix[al], cmap='viridis', interpolation='none', aspect=env.reward_matrix_per_va(profiles[0]).shape[1]/env.reward_matrix_per_va(profiles[0]).shape[0])
        axes[i, 1].set_title(f'Real Matrix ({al})')
        axes[i, 1].set_xlabel('Dimension M')
        axes[i, 1].set_ylabel('Dimension N')
        fig.colorbar(im2, ax=axes[i, 1], orientation='vertical', label='Value')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
    exit(0)"""
    vg_learned, learned_rewards = max_entropy_algo.train(max_iter=500, mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,assumed_grounding=
                                                         assumed_grounding)
    
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    
    # Reward similarities
    for i, al in enumerate(max_entropy_algo.training_align_funcs):
        # Ensure that you do not exceed the number of subplots
        if i >= 2:
            break

        # Plot the first matrix
        im1 = axes[i, 0].imshow(learned_rewards(al), cmap='viridis', interpolation='none', aspect=learned_rewards(al).shape[1]/learned_rewards(al).shape[0])
        axes[i, 0].set_title(f'Predicted Reward Matrix ({al})')
        axes[i, 0].set_xlabel('Dimension M')
        axes[i, 0].set_ylabel('Dimension N')
        fig.colorbar(im1, ax=axes[i, 0], orientation='vertical', label='Value')

        # Plot the second matrix
        print(env.reward_matrix_per_va(al)[152,0])
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        im2 = axes[i, 1].imshow(env.reward_matrix_per_va(al), cmap='viridis', interpolation='none', aspect=learned_rewards(al).shape[1]/learned_rewards(al).shape[0])
        axes[i, 1].set_title(f'Real Matrix ({al})')
        axes[i, 1].set_xlabel('Dimension M')
        axes[i, 1].set_ylabel('Dimension N')
        fig.colorbar(im2, ax=axes[i, 1], orientation='vertical', label='Value')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

    

    # TODO: POLICY SIMILARITIES AL MENOS!!!!!!!!?
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for i, al in enumerate(max_entropy_algo.training_align_funcs):
        # Ensure that you do not exceed the number of subplots
        if i >= 2:
            break

        # Plot the first matrix
        im1 = axes[i, 0].imshow(max_entropy_algo.learned_policy_per_va.policy_per_va(al), cmap='viridis', vmin=0, vmax=1, interpolation='none', aspect=expert_policy.policy_per_va(al).shape[1]/expert_policy.policy_per_va(al).shape[0])
        axes[i, 0].set_title(f'Predicted Policy Matrix ({al})')
        axes[i, 0].set_xlabel('Dimension M')
        axes[i, 0].set_ylabel('Dimension N')
        fig.colorbar(im1, ax=axes[i, 0], orientation='vertical', label='Value')

        # Plot the second matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        im2 = axes[i, 1].imshow(expert_policy.policy_per_va(al), cmap='viridis', interpolation='none', vmin=0, vmax=1, aspect=expert_policy.policy_per_va(al).shape[1]/expert_policy.policy_per_va(al).shape[0])
        axes[i, 1].set_title(f'Real Policy Matrix ({al})')
        axes[i, 1].set_xlabel('Dimension M')
        axes[i, 1].set_ylabel('Dimension N')
        fig.colorbar(im2, ax=axes[i, 1], orientation='vertical', label='Value')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


    # TODO: POLICY SIMILARITIES AL MENOS!!!!!!!!?
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for i, al in enumerate(max_entropy_algo.training_align_funcs):
        # Ensure that you do not exceed the number of subplots
        if i >= 2:
            break

        # Plot the first matrix
        im1 = axes[i, 0].imshow(max_entropy_algo.learned_policy_per_va.policy_per_va(al), cmap='viridis', vmin=0, vmax=1, interpolation='none', aspect=expert_policy.policy_per_va(al).shape[1]/expert_policy.policy_per_va(al).shape[0])
        axes[i, 0].set_title(f'Predicted Policy Matrix ({al})')
        axes[i, 0].set_xlabel('Dimension M')
        axes[i, 0].set_ylabel('Dimension N')
        fig.colorbar(im1, ax=axes[i, 0], orientation='vertical', label='Value')

        # Plot the second matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        im2 = axes[i, 1].imshow(expert_policy.policy_per_va(al), cmap='viridis', interpolation='none', vmin=0, vmax=1, aspect=expert_policy.policy_per_va(al).shape[1]/expert_policy.policy_per_va(al).shape[0])
        axes[i, 1].set_title(f'Real Policy Matrix ({al})')
        axes[i, 1].set_xlabel('Dimension M')
        axes[i, 1].set_ylabel('Dimension N')
        fig.colorbar(im2, ax=axes[i, 1], orientation='vertical', label='Value')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

    # TODO: POLICY SIMILARITIES AL MENOS!!!!!!!!?
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for i, al in enumerate(max_entropy_algo.training_align_funcs):
        # Ensure that you do not exceed the number of subplots
        if i >= 2:
            break

        # Plot the first matrix
        im1 = axes[i, 0].imshow(max_entropy_algo.mce_occupancy_measures(reward_matrix=learned_rewards(al))[1][:,None], cmap='viridis', interpolation='none', aspect=expert_policy.policy_per_va(al).shape[1]/expert_policy.policy_per_va(al).shape[0])
        axes[i, 0].set_title(f'Occupancy Matrix ({al})')
        axes[i, 0].set_xlabel('Dimension M')
        axes[i, 0].set_ylabel('Dimension N')
        fig.colorbar(im1, ax=axes[i, 0], orientation='vertical', label='Value')

        # Plot the second matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        im2 = axes[i, 1].imshow(max_entropy_algo.mce_occupancy_measures(reward_matrix=env.reward_matrix_per_va(al))[1][:,None], cmap='viridis', interpolation='none', aspect=expert_policy.policy_per_va(al).shape[1]/expert_policy.policy_per_va(al).shape[0])
        axes[i, 1].set_title(f'Real Occupancy Matrix ({al})')
        axes[i, 1].set_xlabel('Dimension M')
        axes[i, 1].set_ylabel('Dimension N')
        fig.colorbar(im2, ax=axes[i, 1], orientation='vertical', label='Value')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
    learned = max_entropy_algo.learned_policy_per_va.obtain_trajectory(alignment_function=profiles[0], seed=5686, stochastic=LEARN_STOCHASTIC_POLICY, exploration=0, only_states=True)
    real = expert_policy.obtain_trajectory(alignment_function=profiles[0], seed=5686, stochastic=STOCHASTIC_EXPERT, exploration=0, only_states=True)

    


    # TODO: Develop Reward Network that works here (reuse imitation package) Process Features function... O bien a lo bestia reward para cada estado.
    # En tal caso, hay que mostrar bien las rewards originales versus las aprendidas.
    # TODO: Use MEIRL to get a reward for these weight in the Pareto Front. 
    # TODO: Value System Learning con probabilistic rewards

