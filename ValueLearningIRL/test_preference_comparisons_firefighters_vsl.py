from collections import defaultdict
import pickle
import random
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import torch
from torch import nn

from firefighters_use_case.pmovi import pareto_multi_objective_value_iteration, scalarise_q_function
from firefighters_use_case.scalarisation import stochastic_optimal_policy_calculator
from src.envs.firefighters_env import FeatureSelectionFFEnv, FireFightersEnv
from src.me_irl_for_vsl import check_coherent_rewards, mce_partition_fh
from src.reward_functions import ProfileLayer, ProfiledRewardFunction, TrainingModes
from src.vsl_policies import VAlignedDictSpaceActionPolicy
from test_firefighters import INITIAL_STATE_DISTRIBUTION, POLICY_APPROXIMATION_METHOD, USE_ONE_HOT_STATE_ACTION, USE_PMOVI_EXPERT
from utils import sample_example_profiles

SEED = 26
rng = np.random.default_rng(0)
HORIZON = 50
USE_ACTION = True
EXPOSE_STATE = True
USE_ONE_HOT_STATE_ACTION = EXPOSE_STATE
np.random.seed(SEED)
torch.manual_seed(SEED)

FEATURE_SELECTION = FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS
STOCHASTIC_EXPERT = True

discount_factor = 0.7
discount_factor_preferences = 0.7

venv = make_vec_env('FireFighters-v0', rng=rng, env_make_kwargs={'feature_selection': FEATURE_SELECTION, 'horizon':HORIZON, 'initial_state_distribution':INITIAL_STATE_DISTRIBUTION})
learn = False

env_real: FireFightersEnv = gym.make('FireFighters-v0', feature_selection = FEATURE_SELECTION, horizon=HORIZON, initial_state_distribution=INITIAL_STATE_DISTRIBUTION)
env_real.reset(seed=SEED)
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


reward_net = ProfiledRewardFunction(
        environment=env_real,
        use_state=True,
        use_action=True,
        use_next_state=False,
        use_done=False,
        hid_sizes=[2],
        reward_bias=-0.0,
        basic_layer_classes= [nn.Linear, ProfileLayer],
        use_one_hot_state_action=USE_ONE_HOT_STATE_ACTION,#USE_ACTION and FEATURE_SELECTION == FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS and USE_ONE_HOT_STATE_ACTION,
        activations=[nn.Tanh, nn.Identity],
        negative_grounding_layer=False,
        mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION
    )
#reward_net.set_alignment_function((1.0,0.0))
reward_net.set_mode(TrainingModes.VALUE_SYSTEM_IDENTIFICATION)
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

assumed_grounding = torch.zeros((env_real.n_states*env_real.action_dim, 2), dtype=reward_net.dtype, device=reward_net.device, requires_grad=False)
assumed_grounding[:,0] = torch.reshape(torch.as_tensor(env_real.reward_matrix_per_align_func(profiles[-1]), device=reward_net.device, dtype=reward_net.dtype), (env_real.n_states*env_real.action_dim,))
assumed_grounding[:,1] = torch.reshape(torch.as_tensor(env_real.reward_matrix_per_align_func(profiles[0]), device=reward_net.device, dtype=reward_net.dtype),(env_real.n_states*env_real.action_dim,))
#check_coherent_rewards(max_entropy_algo, align_funcs_to_test=profiles, real_grounding=assumed_grounding, policy_approx_method=POLICY_APPROXIMATION_METHOD, stochastic_expert=STOCHASTIC_EXPERT, stochastic_learner=LEARN_STOCHASTIC_POLICY)
grounding = nn.Linear(env_real.n_states*env_real.action_dim, 
                      bias=False, out_features=2, device=reward_net.device, 
                      dtype=reward_net.dtype)

state_dict = grounding.state_dict()
state_dict['weight'] = assumed_grounding.T
grounding.load_state_dict(state_dict)

reward_net.set_grounding_function(grounding)
# TODO: What happens with grounding and value system identification?
#exit(0)
fragmenter = preference_comparisons.RandomFragmenter(
    warning_threshold=1,
    rng=rng,
)
gatherer = preference_comparisons.SyntheticGatherer(rng=rng, discount_factor=discount_factor_preferences,sample=True)
preference_model = preference_comparisons.PreferenceModel(reward_net, discount_factor=discount_factor_preferences)
reward_trainer = preference_comparisons.BasicRewardTrainer(
    preference_model=preference_model,
    loss=preference_comparisons.CrossEntropyRewardLoss(),
    epochs=5,
    rng=rng,
    lr = 0.01
)


# Several hyperparameters (reward_epochs, ppo_clip_range, ppo_ent_coef,
# ppo_gae_lambda, ppo_n_epochs, discount_factor, use_sde, sde_sample_freq,
# ppo_lr, exploration_frac, num_iterations, initial_comparison_frac,
# initial_epoch_multiplier, query_schedule) used in this example have been
# approximately fine-tuned to reach a reasonable level of performance.


N_SEEDS = 500
N_EXPERT_SAMPLES_PER_SEED = 10

#env_real_with_observations =gym.make('FireFightersEnvWithObservation-v0', feature_selection = FEATURE_SELECTION, horizon=HORIZON, initial_state_distribution=INITIAL_STATE_DISTRIBUTION)
expert_policy = VAlignedDictSpaceActionPolicy(policy_per_va_dict = profile_to_matrix if USE_PMOVI_EXPERT else profile_to_assumed_matrix, env = env_real, 
                                              expose_state=EXPOSE_STATE, state_encoder=None)

expert_trajs_with_rew_per_profile = {
    pr: expert_policy.obtain_trajectories(n_seeds=N_SEEDS, seed=SEED, stochastic=STOCHASTIC_EXPERT, repeat_per_seed=N_EXPERT_SAMPLES_PER_SEED, with_alignfunctions=[pr,], t_max=HORIZON,
                                                 with_reward=True,
                                                 alignments_in_env=[pr,])
                                                 for pr in (profiles[0], profiles[-1])}
random_policy = VAlignedDictSpaceActionPolicy(env=env_real, policy_per_va_dict=defaultdict(lambda: np.ones_like(profile_to_matrix[(1.0,0.0)])/env_real.action_dim), expose_state=EXPOSE_STATE)
random_trajs_with_rew_per_profile = {
    pr: random_policy.obtain_trajectories(n_seeds=N_SEEDS, seed=SEED, stochastic=STOCHASTIC_EXPERT, repeat_per_seed=N_EXPERT_SAMPLES_PER_SEED, with_alignfunctions=[pr,], t_max=HORIZON,
                                                 with_reward=True,
                                                 alignments_in_env=[pr,])
                                                 for pr in (profiles[0], profiles[-1])}
print(expert_trajs_with_rew_per_profile[profiles[0]][0])
traj_dataset = preference_comparisons.TrajectoryDataset(
    expert_trajs_with_rew_per_profile[profiles[0]],
    rng=rng
)
traj_dataset_random  = preference_comparisons.TrajectoryDataset(
    random_trajs_with_rew_per_profile[profiles[0]],
    rng=rng
)
pref_comparisons = preference_comparisons.PreferenceComparisons(
    trajectory_generator=traj_dataset_random,
    reward_model=reward_net,
    num_iterations=60,  # Set to 60 for better performance
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    fragment_length=HORIZON,
    transition_oversampling=5,
    initial_comparison_frac=0.1,
    allow_variable_horizon=False,
    initial_epoch_multiplier=500,
    query_schedule="hyperbolic",
)

pref_comparisons.train(5000, total_comparisons=500, callback=None)

print(reward_net.get_learned_grounding())


# TODO: Modificar Preference Comparisons reward model para usar todo lo de mce (calculate rewards!). Comprobar similitud entre politicas.
# TODO: Usar una política aleatoria simple para conseguir trayectorias ? 

#trained_policy = mce_partition_fh(env=env_real, reward=reward_net(env_real.observation_matrix, env_real.ac))