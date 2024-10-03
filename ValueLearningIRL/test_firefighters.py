from copy import deepcopy
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
from src.me_irl_for_vsl import MaxEntropyIRLForVSL, check_coherent_rewards, mce_partition_fh, PolicyApproximators
from src.me_irl_for_vsl_plot_utils import plot_learned_and_expert_occupancy_measures, plot_learned_and_expert_rewards, plot_learned_to_expert_policies, plot_learned_to_expert_policies
from src.reward_functions import PositiveBoundedLinearModule, ProfileLayer, ProfiledRewardFunction, TrainingModes
from src.vsl_policies import VAlignedDictSpaceActionPolicy, ValueSystemLearningPolicy, profiled_society_sampler, profiled_society_traj_sampler_from_policy, random_sampler_among_trajs, sampler_from_policy
from utils import sample_example_profiles, train_test_split_initial_state_distributions
from torch import nn
LOG_EVAL_INTERVAL = 1
SEED = 26

STOCHASTIC_EXPERT = True
LEARN_STOCHASTIC_POLICY = True

USE_ACTION = True # Use action to approximate the reward.
USE_ONE_HOT_STATE_ACTION = True # One-Hot encode all state-action pairs or rather, concatenate state and action features.
DEMO_OM_FROM_POLICY = True # If False, use the sampled trajectories as data to calculate occcupancy measures of the expert agents,
                            # instead use the original policy probabilities directly 
# TODO: different batch size for evaluating termination condition?.
N_SEEDS = 100
N_EXPERT_SAMPLES_PER_SEED = 1 if STOCHASTIC_EXPERT is False else 10

N_REWARD_SAMPLES_PER_ITERATION = 10
N_SEEDS_MINIBATCH = 30
N_EXPERT_SAMPLES_PER_SEED_MINIBATCH = 1 if STOCHASTIC_EXPERT is False else 5
FEATURE_SELECTION = FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS

POLICY_APPROXIMATION_METHOD = PolicyApproximators.MCE_ORIGINAL  # Approximate policies using causal entropy (original MCE_IRL algorithm, up to the stablished HORIZON), 
                            # or use normal value iteration ('value_iteration')
                            # or another method... (NOT IMPLEMENTED)

HORIZON = 20

USE_PMOVI_EXPERT = False 

INITIAL_STATE_DISTRIBUTION = 'uniform' # or 'default' or a specfic probability distribution on the encrypted states.

EXPERT_FIXED_TRAJECTORIES = True
USE_VA_EXPECTATIONS_INSTEAD_OF_OCC_MEASURES = True
N_EXPERIMENT_REPETITON = 10

FIREFIGHTER_ALFUNC_COLORS = lambda align_func: get_color_gradient([1,0,0], [0,0,1], align_func[0])

PERFORM_VGL = False
PERFORM_VSI_FROM_VGL = False
PERFORM_VSI_FROM_REAL_GROUNDING = True

def get_color_gradient(c1, c2, mix):
    """
    Given two hex colors, returns a color gradient corresponding to a given [0,1] value
    """
    c1_rgb = np.array(c1)
    c2_rgb = np.array(c2)
    
    return ((1-mix)*c1_rgb + (mix*c2_rgb))

def pad(array, length):
    new_arr = np.zeros((length,))
    new_arr[0:len(array)] = np.asarray(array)
    new_arr[len(array):] = array[-1]
    return new_arr

def plot_learning_curves(historic_linf_delta_per_rep, historic_grad_norms_per_rep, name_method='test_lc', align_func_colors=lambda al: 'black'):
    plt.figure(figsize=(6,6))
    plt.title(f"Learning Curves over {historic_linf_delta_per_rep} repetitions")
    plt.xlabel("Training Iteration")
    plt.ylabel("Training error")
    
    for al in historic_linf_delta_per_rep[0].keys():  
        max_length = np.max([len(historic_linf_delta_per_rep[rep][al]) for rep in range(len(historic_linf_delta_per_rep))])

        linf_delta = np.asarray([pad(historic_linf_delta_per_rep[rep][al], max_length) for rep in range(len(historic_linf_delta_per_rep))])
        grad_norms = np.asarray([pad(historic_linf_delta_per_rep[rep][al], max_length) for rep in range(len(historic_linf_delta_per_rep))])
    
        avg_linf_delta = np.mean(linf_delta, axis=0)
        
        std_linf_delta = np.std(linf_delta, axis=0)

        avg_grad_norms = np.mean(grad_norms, axis=0)
        std_grad_norms = np.std(grad_norms, axis=0)
        color=align_func_colors(tuple(al))
        plt.plot(avg_linf_delta,
                        color=color, 
                        label=f'{tuple([float("{0:.3f}".format(t)) for t in al])} Last: {float(avg_linf_delta[-1]):0.2f}'
                        )
        #plt.plot(avg_grad_norms,color=align_func_colors.get(tuple(al), 'black'), label=f'Grad Norm: {float(avg_grad_norms[-1]):0.2f}'
                        
        
        plt.fill_between([i for i in range(len(avg_linf_delta))], avg_linf_delta-std_linf_delta, avg_linf_delta+std_linf_delta,edgecolor=color,alpha=0.3, facecolor=color)
        #plt.fill_between([i for i in range(len(avg_grad_norms))], avg_grad_norms-std_grad_norms, avg_grad_norms+std_grad_norms,edgecolor='black',alpha=0.1, facecolor='black')
        
    plt.legend()
    plt.grid()
    plt.savefig(f'plots/Maxent_learning_curves_{name_method}.pdf')

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Example usage
    

    env_real: FireFightersEnv = gym.make('FireFighters-v0', feature_selection = FEATURE_SELECTION, horizon=HORIZON, initial_state_distribution=INITIAL_STATE_DISTRIBUTION)
    env_real.reset(seed=SEED)

    print(env_real.goal_states)
    
    
    #train_init_state_distribution, test_init_state_distribution = train_test_split_initial_state_distributions(env_real.state_dim, 0.7)
    #env_training: FireFightersEnv = gym.make('FireFighters-v0', feature_selection = FEATURE_SELECTION, horizon=HORIZON, initial_state_distribution=train_init_state_distribution)
    env_training = env_real
    train_init_state_distribution, test_init_state_distribution = env_real.initial_state_dist, env_real.initial_state_dist

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
        v, q = pareto_multi_objective_value_iteration(env_real.real_env, discount_factor=0.7, model_used=None, pareto=True, normalized=False)
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
    


    profiles  = sample_example_profiles(profile_variety=6,n_values=2)
    profile_to_matrix = {}
    profile_to_assumed_matrix = {}
    for w in profiles:
        scalarised_q = scalarise_q_function(q_func, objectives=2, weights=w)

        pi_matrix = stochastic_optimal_policy_calculator(scalarised_q, w, deterministic= not STOCHASTIC_EXPERT)
        profile_to_matrix[w] = pi_matrix

        _,_, assumed_expert_pi = mce_partition_fh(env_real, discount=discount_factor,
                                             reward=env_real.reward_matrix_per_align_func(w),
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
            vsi_optimizer_kwargs={"lr": 0.15, "weight_decay": 0.0000},
            vgl_optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0000},
            vgl_expert_policy=expert_policy_train,
            vsi_expert_policy=expert_policy_train,
            vgl_expert_sampler=vgl_expert_train_sampler,
            vsi_expert_sampler=vsi_expert_train_sampler,
            target_align_func_sampler=lambda al_func: al_func,

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
            expert_is_stochastic=STOCHASTIC_EXPERT,
            discount=discount_factor,
            environment_is_stochastic=False,
            use_feature_expectations_for_vsi=USE_VA_EXPECTATIONS_INSTEAD_OF_OCC_MEASURES,
            
            )
    

    
    #vg_learned, learned_rewards = max_entropy_algo.train(max_iter=200, mode=TrainingModes.VALUE_GROUNDING_LEARNING)

    if USE_ONE_HOT_STATE_ACTION:
        assumed_grounding = np.zeros((env_real.n_states*env_real.action_dim, 2))
        assumed_grounding[:,0] = np.reshape(env_training.reward_matrix_per_align_func(profiles[-1]), (env_real.n_states*env_real.action_dim))
        assumed_grounding[:,1] = np.reshape(env_training.reward_matrix_per_align_func(profiles[0]),(env_real.n_states*env_real.action_dim))
        check_coherent_rewards(max_entropy_algo, align_funcs_to_test=profiles, real_grounding=assumed_grounding, policy_approx_method=POLICY_APPROXIMATION_METHOD, stochastic_expert=STOCHASTIC_EXPERT, stochastic_learner=LEARN_STOCHASTIC_POLICY)
    
    

    # VALUE GROUNDING LEARNING:
    if PERFORM_VGL:
        learned_rewards_per_round = []
        policies_per_round = []
        linf_delta_per_round = []
        grad_norm_per_round = []
        for rep in range(N_EXPERIMENT_REPETITON):
            learned_grounding, learned_rewards, reward_net_learned, linf_delta_per_align_fun, grad_norm_per_align_func = max_entropy_algo.train(max_iter=500, 
                                                                mode=TrainingModes.VALUE_GROUNDING_LEARNING,n_seeds_for_sampled_trajectories=N_SEEDS_MINIBATCH,
                                                                n_sampled_trajs_per_seed=N_EXPERT_SAMPLES_PER_SEED_MINIBATCH,
                                                                use_probabilistic_reward=False,n_reward_reps_if_probabilistic_reward=N_REWARD_SAMPLES_PER_ITERATION)
            
            
            learned_rewards_per_round.append(learned_rewards)
            policies_per_round.append(deepcopy(max_entropy_algo.learned_policy_per_va))
            linf_delta_per_round.append(linf_delta_per_align_fun)
            grad_norm_per_round.append(grad_norm_per_align_func)
            print(learned_grounding)
            
        plot_learning_curves(linf_delta_per_round, grad_norm_per_round, name_method=f'expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vgl', align_func_colors=FIREFIGHTER_ALFUNC_COLORS)

        plot_learned_and_expert_occupancy_measures(env_real,max_entropy_algo,expert_policy,learned_rewards_per_round,vsi_or_vgl='vgl', namefig=f'test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vgl')
        
        plot_learned_to_expert_policies(expert_policy, max_entropy_algo, vsi_or_vgl='vgl', namefig=f'test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vgl', learnt_policy=policies_per_round)
            
        plot_learned_and_expert_rewards(env_real, max_entropy_algo, learned_rewards_per_round, vsi_or_vgl='vgl', namefig=f'test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vgl')
            
            
        
        learned = max_entropy_algo.learned_policy_per_va.obtain_trajectory(alignment_function=profiles[0], seed=5686, stochastic=LEARN_STOCHASTIC_POLICY, exploration=0, only_states=True)
        real = expert_policy.obtain_trajectory(alignment_function=profiles[0], seed=5686, stochastic=STOCHASTIC_EXPERT, exploration=0, only_states=True)
        print("EXAMPLE TRAJS:")
        print(learned)
        print(real)
        
    # VALUE SYSTEM IDENTIFICATION: Correct grounding.
    if PERFORM_VSI_FROM_REAL_GROUNDING:
        assumed_grounding = np.zeros((env_real.n_states*env_real.action_dim, 2))
        assumed_grounding[:,0] = np.reshape(env_training.reward_matrix_per_align_func(profiles[-1]), (env_real.n_states*env_real.action_dim))
        assumed_grounding[:,1] = np.reshape(env_training.reward_matrix_per_align_func(profiles[0]),(env_real.n_states*env_real.action_dim))

        
        learned_rewards_per_round = []
        policies_per_round = []
        target_align_funcs_to_learned_align_funcs_per_round = []
        linf_delta_per_round = []
        grad_norm_per_round = []
        for rep in range(N_EXPERIMENT_REPETITON):
            target_align_funcs_to_learned_align_funcs, learned_rewards, reward_net_per_target_va, linf_delta_per_align_fun, grad_norm_per_align_func = max_entropy_algo.train(max_iter=200, 
                                                                mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                                                assumed_grounding=assumed_grounding,
                                                                n_seeds_for_sampled_trajectories=N_SEEDS_MINIBATCH,
                                                                n_sampled_trajs_per_seed=N_EXPERT_SAMPLES_PER_SEED_MINIBATCH,
                                                                use_probabilistic_reward=False,n_reward_reps_if_probabilistic_reward=N_REWARD_SAMPLES_PER_ITERATION)
            #plt.plot(linf_delta_per_align_fun)
            learned_rewards_per_round.append(learned_rewards)
            policies_per_round.append(deepcopy(max_entropy_algo.learned_policy_per_va)) 
            target_align_funcs_to_learned_align_funcs_per_round.append(target_align_funcs_to_learned_align_funcs)
            linf_delta_per_round.append(linf_delta_per_align_fun)
            grad_norm_per_round.append(grad_norm_per_align_func)

            plot_learned_to_expert_policies(expert_policy, max_entropy_algo, target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs, vsi_or_vgl='vsi', namefig=f'test{rep}_firefighters_vsi')

            plot_learned_and_expert_rewards(env_real, max_entropy_algo, learned_rewards, vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs, namefig=f'test{rep}_firefighters_vsi')

            plot_learned_and_expert_occupancy_measures(env_real,max_entropy_algo,expert_policy,learned_rewards,vsi_or_vgl='vsi',target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs, namefig=f'test{rep}_firefighters_vsi')
            # VALUE SYSTEM IDENTIFICATION: Learned grounding.
        plot_learning_curves(linf_delta_per_round, grad_norm_per_round, name_method=f'expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi', align_func_colors=FIREFIGHTER_ALFUNC_COLORS)

        plot_learned_and_expert_occupancy_measures(env_real,max_entropy_algo,expert_policy,learned_rewards_per_round,vsi_or_vgl='vsi', namefig=f'test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi', target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round)
        
        plot_learned_to_expert_policies(expert_policy, max_entropy_algo, vsi_or_vgl='vsi', namefig=f'test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi', learnt_policy=policies_per_round,target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round)
            
        plot_learned_and_expert_rewards(env_real, max_entropy_algo, learned_rewards_per_round, vsi_or_vgl='vsi', namefig=f'test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi',target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round)
            
    if PERFORM_VSI_FROM_VGL:
        learned_rewards_per_round = []
        policies_per_round = []
        target_align_funcs_to_learned_align_funcs_per_round = []
        linf_delta_per_round = []
        grad_norm_per_round = []
        for rep in range(N_EXPERIMENT_REPETITON):    
            target_align_funcs_to_learned_align_funcs, learned_rewards, reward_net_per_target_va, linf_delta_per_align_fun, grad_norm_per_align_func = max_entropy_algo.train(max_iter=200, 
                                                                mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                                                assumed_grounding=reward_net.values_net,
                                                                n_seeds_for_sampled_trajectories=N_SEEDS_MINIBATCH,
                                                                n_sampled_trajs_per_seed=N_EXPERT_SAMPLES_PER_SEED_MINIBATCH,
                                                                use_probabilistic_reward=False,n_reward_reps_if_probabilistic_reward=N_REWARD_SAMPLES_PER_ITERATION)
            learned_rewards_per_round.append(learned_rewards)
            policies_per_round.append(deepcopy(max_entropy_algo.learned_policy_per_va))
            target_align_funcs_to_learned_align_funcs_per_round.append(target_align_funcs_to_learned_align_funcs)
            linf_delta_per_round.append(linf_delta_per_align_fun)
            grad_norm_per_round.append(grad_norm_per_align_func)

            plot_learned_to_expert_policies(expert_policy, max_entropy_algo, target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs, vsi_or_vgl='vsi', namefig=f'test{rep}_firefighters_vsi_from_vgl')

            plot_learned_and_expert_rewards(env_real, max_entropy_algo, learned_rewards, vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs, namefig=f'test{rep}_firefighters_vsi')

            plot_learned_and_expert_occupancy_measures(env_real,max_entropy_algo,expert_policy,learned_rewards,vsi_or_vgl='vsi',target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs, namefig=f'test{rep}_firefighters_vsi_from_vgl')
        plot_learning_curves(linf_delta_per_round, grad_norm_per_round, name_method=f'expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi_from_vgl', align_func_colors=FIREFIGHTER_ALFUNC_COLORS)

        plot_learned_and_expert_occupancy_measures(env_real,max_entropy_algo,expert_policy,learned_rewards_per_round,vsi_or_vgl='vsi', namefig=f'test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi_from_vgl', target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round)
        
        plot_learned_to_expert_policies(expert_policy, max_entropy_algo, vsi_or_vgl='vsi', namefig=f'test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi_from_vgl', learnt_policy=policies_per_round,target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round)
            
        plot_learned_and_expert_rewards(env_real, max_entropy_algo, learned_rewards_per_round, vsi_or_vgl='vsi', namefig=f'test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi_from_vgl',target_align_funcs_to_learned_align_funcs=target_align_funcs_to_learned_align_funcs_per_round)
            