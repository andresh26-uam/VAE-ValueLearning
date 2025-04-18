from copy import deepcopy
import pickle
from imitation.util.util import make_vec_env
import gymnasium as gym
import numpy as np
import torch
from torch import nn

from env_data import FIREFIGHTER_ALFUNC_COLORS
from firefighters_use_case.pmovi import pareto_multi_objective_value_iteration, scalarise_q_function
from firefighters_use_case.scalarisation import stochastic_optimal_policy_calculator

from src.envs.firefighters_env import FeatureSelectionFFEnv, FireFightersEnv
from src.vsl_algorithms.me_irl_for_vsl import mce_partition_fh
from src.vsl_algorithms.vsl_plot_utils import plot_learned_and_expert_occupancy_measures, plot_learned_and_expert_reward_pairs, plot_learned_and_expert_rewards, plot_learned_to_expert_policies, plot_learning_curves
from src.vsl_reward_functions import LinearAlignmentLayer, LinearVSLRewardFunction, TrainingModes
from src.vsl_policies import VAlignedDictSpaceActionPolicy
from train_vsl import POLICY_APPROXIMATION_METHOD
from utils import sample_example_profiles

from src.vsl_algorithms.preference_model_vs import PreferenceBasedTabularMDPVSL

SEED = 0
LOGINTERVAL = 1

HORIZON = 30
USE_ACTION = True
EXPOSE_STATE = True
USE_ONE_HOT_STATE_ACTION = True
INITIAL_STATE_DISTRIBUTION = 'random'


FEATURE_SELECTION = FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS
STOCHASTIC_EXPERT = True

N_SEEDS = 200
N_EXPERT_SAMPLES_PER_SEED = 10
USE_PMOVI_EXPERT = False

discount_factor = 0.7
discount_factor_preferences = 0.7

if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    venv = make_vec_env('FireFighters-v0', rng=rng, env_make_kwargs={
                        'feature_selection': FEATURE_SELECTION, 'horizon': HORIZON, 'initial_state_distribution': INITIAL_STATE_DISTRIBUTION})
    learn = False

    env_real: FireFightersEnv = gym.make('FireFighters-v0', feature_selection=FEATURE_SELECTION,
                                         horizon=HORIZON, initial_state_distribution=INITIAL_STATE_DISTRIBUTION)
    env_real.reset(seed=SEED)
    if learn:
        v, q = pareto_multi_objective_value_iteration(
            env_real.real_env, discount_factor=discount_factor, model_used=None, pareto=True, normalized=False)
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

    reward_net = LinearVSLRewardFunction(
        environment=env_real,
        use_state=False,
        use_action=True,
        use_next_state=False,
        use_done=False,
        hid_sizes=[2],
        reward_bias=-0.0,
        basic_layer_classes=[nn.Linear, LinearAlignmentLayer],
        use_one_hot_state_action=USE_ONE_HOT_STATE_ACTION,
        activations=[nn.Tanh, nn.Identity],
        negative_grounding_layer=False,
        mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION
    )
    # reward_net.set_alignment_function(PROFILE)
    reward_net.set_mode(TrainingModes.VALUE_SYSTEM_IDENTIFICATION)
    profiles = sample_example_profiles(profile_variety=6, n_values=2)

    PROFILE = (-90.0, 3.0)  # 1, 0
    profiles.append(PROFILE)

    profile_to_matrix = {}
    profile_to_assumed_matrix = {}
    for w in profiles:
        scalarised_q = scalarise_q_function(q_func, objectives=2, weights=w)

        pi_matrix = stochastic_optimal_policy_calculator(
            scalarised_q, w, deterministic=not STOCHASTIC_EXPERT)
        profile_to_matrix[w] = pi_matrix

        _, _, assumed_expert_pi = mce_partition_fh(env_real, discount=discount_factor,
                                                   reward=env_real.reward_matrix_per_align_func(
                                                       w),
                                                   approximator_kwargs={
                                                       'value_iteration_tolerance': 0.00001, 'iterations': 1000},
                                                   policy_approximator=POLICY_APPROXIMATION_METHOD, deterministic=not STOCHASTIC_EXPERT)

        profile_to_assumed_matrix[w] = assumed_expert_pi

    assumed_grounding = torch.zeros((env_real.n_states*env_real.action_dim, 2),
                                    dtype=reward_net.dtype, device=reward_net.device, requires_grad=False)
    assumed_grounding[:, 0] = torch.reshape(torch.as_tensor(env_real.reward_matrix_per_align_func(
        (1.0, 0.0)), device=reward_net.device, dtype=reward_net.dtype), (env_real.n_states*env_real.action_dim,))
    assumed_grounding[:, 1] = torch.reshape(torch.as_tensor(env_real.reward_matrix_per_align_func(
        (0.0, 1.0)), device=reward_net.device, dtype=reward_net.dtype), (env_real.n_states*env_real.action_dim,))
    # check_coherent_rewards(max_entropy_algo, align_funcs_to_test=profiles, real_grounding=assumed_grounding, policy_approx_method=POLICY_APPROXIMATION_METHOD, stochastic_expert=STOCHASTIC_EXPERT, stochastic_learner=LEARN_STOCHASTIC_POLICY)
    grounding = nn.Linear(env_real.n_states*env_real.action_dim,
                          bias=False, out_features=2, device=reward_net.device,
                          dtype=reward_net.dtype)

    state_dict = grounding.state_dict()
    state_dict['weight'] = assumed_grounding.T
    grounding.load_state_dict(state_dict)

    reward_net.set_grounding_function(grounding)

    expert_policy = VAlignedDictSpaceActionPolicy(policy_per_va_dict=profile_to_matrix if USE_PMOVI_EXPERT else profile_to_assumed_matrix, env=env_real,
                                                  expose_state=EXPOSE_STATE, state_encoder=None)

    pref_algo: PreferenceBasedTabularMDPVSL = PreferenceBasedTabularMDPVSL(env=env_real, reward_net=reward_net,
                                                                           discount=discount_factor,
                                                                           discount_factor_preferences=discount_factor,
                                                                           log_interval=LOGINTERVAL,
                                                                           vgl_reference_policy='random',
                                                                           vsi_reference_policy='random',
                                                                           vgl_expert_policy=expert_policy,
                                                                           vsi_expert_policy=expert_policy,
                                                                           vgl_target_align_funcs=[
                                                                               (0.0, 1.0), (1.0, 0.0)],
                                                                           vsi_target_align_funcs=profiles,
                                                                           rng=rng,
                                                                           use_quantified_preference=False,
                                                                           preference_sampling_temperature=0,
                                                                           stochastic_sampling_in_reference_policy=True,
                                                                           learn_stochastic_policy=True,
                                                                           query_schedule="hyperbolic",
                                                                           reward_trainer_kwargs={
                                                                               'epochs': 1,
                                                                               'lr': 0.08,
                                                                               'batch_size': 512,
                                                                               'minibatch_size': 32,
                                                                           },)
    import pprint as pp
    pp.pprint(vars(pref_algo))
    exit(0)
    N_EXPERIMENT_REPETITON = 2

    PERFORM_VSI = True

    if PERFORM_VSI:
        learned_rewards_per_round = []
        policies_per_round = []
        accuracy_per_round = []
        grad_norm_per_round = []
        for rep in range(N_EXPERIMENT_REPETITON):
            target_align_funcs_to_learned_align_funcs, reward_net_per_target_va, metrics = pref_algo.train(
                max_iter=20000,
                mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                assumed_grounding=grounding,
                resample_trajectories_if_not_already_sampled=True, new_rng=None,
                fragment_length=HORIZON, interactive_imitation_iterations=100,
                total_comparisons=2000, initial_comparison_frac=0.1,
                initial_epoch_multiplier=1, transition_oversampling=5,
                n_sampled_trajs_per_seed=N_EXPERT_SAMPLES_PER_SEED,
                n_seeds_for_sampled_trajectories=N_SEEDS,
                use_probabilistic_reward=False,
            )
            learned_rewards_matrix = metrics['learned_rewards']
            acc = metrics['accuracy']
            
            learned_rewards_per_round.append(learned_rewards_matrix)
            policies_per_round.append(
                deepcopy(pref_algo.learned_policy_per_va))
            accuracy_per_round.append(acc)

        plot_learned_and_expert_occupancy_measures(env_real, pref_algo, expert_policy, learned_rewards_per_round,
                                                   vsi_or_vgl='vgl', namefig=f'pref_test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi')

        plot_learned_to_expert_policies(expert_policy, pref_algo, vsi_or_vgl='vgl',
                                        namefig=f'pref_test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi', learnt_policy=policies_per_round)

        plot_learned_and_expert_rewards(env_real, pref_algo, learned_rewards_per_round, vsi_or_vgl='vgl',
                                        namefig=f'pref_test_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi')
        plot_learning_curves(pref_algo, historic_metric=accuracy_per_round, name_metric='TVC',
                             name_method=f'pref_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi', align_func_colors=FIREFIGHTER_ALFUNC_COLORS)

        plot_learned_and_expert_reward_pairs(pref_algo, learned_rewards_per_round, vsi_or_vgl='vsi',
                                             target_align_funcs_to_learned_align_funcs=None, namefig='pref_expected_over_{N_EXPERIMENT_REPETITON}_firefighters_vsi', )
