from abc import abstractmethod
from copy import deepcopy
import enum
from functools import partial
import itertools
import pickle
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO

from firefighters_use_case.constants import ACTION_AGGRESSIVE_FIRE_SUPPRESSION
from firefighters_use_case.pmovi import pareto_multi_objective_value_iteration, scalarise_q_function
from firefighters_use_case.scalarisation import stochastic_optimal_policy_calculator
from src.envs.firefighters_env import FeatureSelectionFFEnv, FireFightersEnv
from src.envs.tabularVAenv import TabularVAMDP, encrypt_state, translate_state
from src.envs.roadworld_env import FixedDestRoadWorldGymPOMDP
from roadworld_env_use_case.values_and_costs import PROFILE_COLORS_VEC
from roadworld_env_use_case.network_env import DATA_FOLDER, FeaturePreprocess, FeatureSelection, RoadWorldGymPOMDP
from roadworld_env_use_case.utils.load_data import ini_od_dist
from roadworld_env_use_case.values_and_costs import BASIC_PROFILES, BASIC_PROFILE_NAMES
from src.vsl_algorithms.base_tabular_vsl_algorithm import PolicyApproximators, concentrate_on_max_policy
from src.vsl_algorithms.me_irl_for_vsl import mce_partition_fh
from torch import nn, optim

from src.vsl_algorithms.preference_model_vs import CrossEntropyRewardLossForQualitativePreferences, SupportedFragmenters
from src.vsl_algorithms.vsl_plot_utils import get_color_gradient, get_linear_combination_of_colors
from src.vsl_policies import VAlignedDictSpaceActionPolicy, LearnerValueSystemLearningPolicy, profiled_society_sampler, random_sampler_among_trajs, sampler_from_policy
from src.vsl_reward_functions import ConvexAlignmentLayer, ConvexLinearModule, LinearAlignmentLayer, LinearVSLRewardFunction, TrainingModes
from utils import sample_example_profiles
from imitation.algorithms.preference_comparisons import CrossEntropyRewardLoss
import torch 


USE_PMOVI_EXPERT = False
FIRE_FIGHTERS_ENV_NAME = 'FireFighters-v0'
ROAD_WORLD_ENV_NAME = 'FixedDestRoadWorld-v0'

def custom_cost_from_reward(environment:RoadWorldGymPOMDP, reward, state_des, profile):
        rews = [reward[s,a] for a,s in zip(*environment.pre_acts_and_pre_states[state_des[0]]) if s != environment.cur_des]
        state_acts = [(s,a) for a,s in zip(*environment.pre_acts_and_pre_states[state_des[0]]) if s != environment.cur_des]
        if len(rews) == 0.0:
            rews = [reward[state_des[0],0]]
        np.testing.assert_almost_equal(rews, rews[0])
        return rews[0]
    
def one_hot_encoding(a, n, dtype=np.float32):
    v = np.zeros(shape=(n,),dtype=dtype)
    v[a] = 1.0
    return v

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

class OneHotFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that performs one-hot encoding on integer observations.
    """
    def __init__(self, observation_space, n_categories):
        # The output of the extractor is the size of one-hot encoded vectors
        super(OneHotFeatureExtractor, self).__init__(observation_space, features_dim=n_categories)
        self.n_categories = n_categories

    def forward(self, observations):
        # Convert observations to integers (if needed) and perform one-hot encoding
        """batch_size = observations.shape[0]
        one_hot = torch.zeros((batch_size, self.n_categories), device=observations.device)
        
        one_hot.scatter_(1, observations.long(), 1)"""
        with torch.no_grad():
            if len(observations.shape) > 2:
                observations = torch.squeeze(observations, dim=1)
            if observations.shape[-1] != int(self.features_dim):

                ret = torch.functional.F.one_hot(observations.long(), num_classes=int(self.features_dim)).float()
            else:
                ret = observations
            return ret

class ObservationMatrixFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that performs one-hot encoding on integer observations.
    """
    def __init__(self, observation_space, observation_matrix):
        # The output of the extractor is the size of one-hot encoded vectors
        super(ObservationMatrixFeatureExtractor, self).__init__(observation_space, features_dim=observation_matrix.shape[1])
        self.observation_matrix = torch.tensor(observation_matrix, dtype=torch.float32, requires_grad=False)
        
    def forward(self, observations):
        # Convert observations to integers (if needed) and perform one-hot encoding
        """batch_size = observations.shape[0]
        one_hot = torch.zeros((batch_size, self.n_categories), device=observations.device)
        
        one_hot.scatter_(1, observations.long(), 1)"""
        with torch.no_grad():
            idx = observations
            """if idx.shape[-1] > 1:
                ret =  torch.vstack([self.observation_matrix[id] for id in idx.bool()])
            else:
                ret =  torch.vstack([self.observation_matrix[id] for id in idx.long()])"""
            if idx.shape[-1] > 1:
                # Convert idx to a boolean mask and use it to index the observation_matrix
                mask = idx.bool()
                selected_indices = mask.nonzero(as_tuple=True)[-1]  # Get indices where mask is True
                assert len(selected_indices) == observations.shape[0]
                # Select the first 32 True indices to maintain the output shape
                ret = self.observation_matrix[selected_indices] 
            else:
                # Directly index the observation_matrix using long indices
                ret = self.observation_matrix[idx.view(-1).long()]
        return ret
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from seals import base_envs
class FeatureExtractorFromVAEnv(BaseFeaturesExtractor):
    def __init__(self, observation_space,  **kwargs) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        
        self.env = kwargs['env']
        self.dtype = kwargs['dtype']
        self.torch_obs_mat = torch.tensor(self.env.observation_matrix, dtype=torch.float32)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.torch_obs_mat[observations.long()]
        
class PrefLossClasses(enum.Enum):
    CROSS_ENTROPY = 'cross_entropy'
    CROSS_ENTROPY_MODIFIED = 'cross_entropy_modified'

def calculate_s_trans_ONE_HOT_FEATURES(vec, state_space, action_space):
    
    # Calculate the cumulative indices to split vec into segments
    splits = np.cumsum([*state_space.nvec, action_space.n][:-1])
    
    # Split vec into segments for each dimension
    split_vec = np.split(vec, splits, axis=-1)
    
    # Use argmax to find the indices of 1 in each segment for all rows
    s_trans = np.stack([np.argmax(segment, axis=-1) for segment in split_vec], axis=-1)
    encrypted_s_trans = np.apply_along_axis(lambda row: encrypt_state(row[:-1], state_space), axis=1, arr=s_trans)
    actions = np.apply_along_axis(lambda row: row[-1], axis=1, arr=s_trans)
    return encrypted_s_trans, actions




class EnvDataForIRL():
    DEFAULT_HORIZON = 50
    DEFAULT_INITIAL_STATE_DISTRIBUTION = 'random'
    DEFAULT_N_SEEDS = 120
    DEFAULT_N_SEEDS_MINIBATCH = 30
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED_MINIBATCH = 10
    DEFAULT_N_REWARD_SAMPLES_PER_ITERATION = 10
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED = 5

    def __init__(self, env_name, discount_factor, **kwargs):

        self.env = None
        self.env_name = env_name
        self.discount_factor = discount_factor

        self.vgl_targets = None
        self.vsi_targets = None
        self.vgl_optimizer_cls = optim.Adam
        self.vsi_optimizer_cls = optim.Adam
        self.vsi_optimizer_kwargs = {"lr": 0.1, "weight_decay": 0.0000}
        self.vgl_optimizer_kwargs = {"lr": 0.1, "weight_decay": 0.0000}
        self.vgl_expert_policy = None
        self.vsi_expert_policy = None
        self.vgl_expert_train_sampler = None
        self.vsi_expert_train_sampler = None
        self.vgl_reference_policy = None
        self.vsi_reference_policy = None
        self.profile_variety = 1
        self.n_values = 0
        self.expose_observations = False
        
        self.target_align_func_sampler = lambda al_func: al_func
        self.reward_trainer_kwargs = {
            'epochs': 5,
            'lr': 0.08,
            'batch_size': 512,
            'minibatch_size': 32,
        }
        self.horizon = self.__class__.DEFAULT_HORIZON
        self.initial_state_distribution = 'random'
        self.stochastic_expert = False
        self.learn_stochastic_policy = True
        self.environment_is_stochastic = False  # If known to be False,
        # some processes can be made faster for some algorithms.
        self.seed = 0
        self.rng = np.random.default_rng(self.seed)

        # default params for reward net.
        self.use_action = False
        self.use_state = False
        self.use_one_hot_state_action = True
        self.use_next_state = False
        self.use_done = False
        self.negative_grounding_layer = False
        self.hid_sizes = [self.n_values,]
        self.basic_layer_classes = [nn.Linear, LinearAlignmentLayer]
        self.activations = [nn.Tanh, nn.Identity]
        self.use_bias = False
        self.reward_bias = False
        self.clamp_rewards = None
        self.discount_factor_preferences = None
        self.testing_profiles = None

        # Override defaults with specific method for other environments
        self.n_seeds_total = self.__class__.DEFAULT_N_SEEDS
        self.n_expert_samples_per_seed = 1 if self.stochastic_expert == False else self.__class__.DEFAULT_N_EXPERT_SAMPLES_PER_SEED
        self.n_reward_samples_per_iteration = self.__class__.DEFAULT_N_REWARD_SAMPLES_PER_ITERATION
        self.n_expert_samples_per_seed_minibatch = self.__class__.DEFAULT_N_EXPERT_SAMPLES_PER_SEED_MINIBATCH
        self.n_seeds_minibatch = self.__class__.DEFAULT_N_SEEDS_MINIBATCH
        
        self.initial_state_distribution_for_expected_alignment_eval = None
        self.active_fragmenter_on = SupportedFragmenters.CONNECTED_FRAGMENTER
        
        self.set_defaults()
        for kw, kwv in kwargs.items():
            setattr(self, kw, kwv)

        assert len(self.activations) == len(self.basic_layer_classes)
        assert len(self.hid_sizes) == len(self.basic_layer_classes) - 1
        self.custom_reward_net_initializer = None
        self.discount_factor_preferences = self.discount_factor if self.discount_factor_preferences is None else self.discount_factor_preferences

        if 'loss_class' in vars(self).keys():
            if self.loss_class == PrefLossClasses.CROSS_ENTROPY_MODIFIED.value:
                self.loss_class = CrossEntropyRewardLossForQualitativePreferences
            elif self.loss_class == PrefLossClasses.CROSS_ENTROPY.value:
                self.loss_class = CrossEntropyRewardLoss
            else:
                self.loss_class = CrossEntropyRewardLoss
            self.loss_kwargs = self.loss_kwargs

    @abstractmethod
    def set_defaults(self):
        pass

    def get_reward_net(self, algorithm='me'):
        features_extractor_class = FeatureExtractorFromVAEnv
        features_extractor_kwargs = dict(
            env = self.env,
            dtype=torch.float32,
        )
        action_features_extractor_class = OneHotFeatureExtractor
        action_features_extractor_kwargs = dict(n_categories=self.env.action_dim)
        return LinearVSLRewardFunction(
            environment=self.env,
            use_state=self.use_state,
            use_action=self.use_action,
            use_next_state=self.use_next_state,
            use_done=self.use_done,
            hid_sizes=self.hid_sizes,
            reward_bias=self.reward_bias,
            basic_layer_classes=self.basic_layer_classes,
            use_one_hot_state_action=self.use_one_hot_state_action,
            activations=self.activations,
            negative_grounding_layer=self.negative_grounding_layer,
            use_bias=self.use_bias,
            clamp_rewards=self.clamp_rewards,
            mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs =features_extractor_kwargs,
            action_features_extractor_class=action_features_extractor_class,
            action_features_extractor_kwargs=action_features_extractor_kwargs
            
        )

    @abstractmethod
    def get_assumed_grounding(self):
        return nn.Identity()

    @property
    def pc_config(self):
        return {'vgl': dict(query_schedule='hyperbolic',
                            stochastic_sampling_in_reference_policy=True),
                'vsi': dict(
                query_schedule='hyperbolic',
            stochastic_sampling_in_reference_policy=True,)}
    
    @property
    def tad_config(self):
        return self.ad_config
    @property
    def tad_train_config(self):
        return self.ad_train_config
    @property
    def ad_config(self):
        """
        learner_kwargs=dict(
        feature_extractor_class=OneHotFeatureExtractor,
        feature_extractor_kwargs=dict(n_categories=self.env.state_dim))"""

        policy_kwargs=dict(
        features_extractor_class=OneHotFeatureExtractor,
        features_extractor_kwargs=dict(n_categories=self.env.state_dim),
        net_arch=[], # net arch is input to output linearly,
            )
        
        learner_kwargs=dict(gamma=self.discount_factor, tensorboard_log="./unknown_tensorboard_expert/"
        )

        return {'vgl': dict(policy_kwargs=policy_kwargs, learner_kwargs=learner_kwargs),
                'vsi': dict(policy_kwargs=policy_kwargs, learner_kwargs=learner_kwargs)}
    @property
    def iq_config(self):
        qnetwork_kwargs={
                     'net_arch': [],
                     'activation_fn': torch.nn.Identity,
                     'features_extractor_class': OneHotFeatureExtractor,
                     'features_extractor_kwargs': dict(n_categories=self.env.state_dim),
                 }
        pred_network_kwargs = {
                     'net_arch': [],
                     'activation_fn': torch.nn.Identity,
                     'features_extractor_class': OneHotFeatureExtractor,
                     'features_extractor_kwargs': dict(n_categories=self.env.state_dim),
                 }
        return {'vgl': 
                {
                    'qnetwork_kwargs': qnetwork_kwargs,
                    'pred_network_kwargs': pred_network_kwargs,
                }, 'vsi': {
                    'qnetwork_kwargs': qnetwork_kwargs,
                    'pred_network_kwargs': pred_network_kwargs,
                }
        }
    @property
    def iq_train_config(self):
        q_optimizer_kwargs = {
                     'lr': self.vsi_optimizer_kwargs['lr'],
                     'weight_decay': 0.0
                 }
        pred_optimizer_kwargs = {
            'lr': self.vsi_optimizer_kwargs['lr'],
            'weight_decay': 0.0
        }
        #self.vsi_optimizer_kwargs = {"lr": 0.001, "weight_decay": 0.0000, "betas": (0.5, 0.999)} # FOR DEMO_OM_TRUE 0.05 before
        #q_optimizer_kwargs = self.vsi_optimizer_kwargs
        #pred_optimizer_kwargs = self.vsi_optimizer_kwargs # TODO test other parameters
        n_seeds_for_sampled_trajectories=self.n_seeds_total # 1000 was too few with full variable length trajectories
        n_sampled_trajs_per_seed=self.n_expert_samples_per_seed
            
        return {'vgl': 
                {
                    'q_optimizer_kwargs': q_optimizer_kwargs,
                    'pred_optimizer_kwargs': pred_optimizer_kwargs,
                    'n_seeds_for_sampled_trajectories': n_seeds_for_sampled_trajectories,
                    'n_sampled_trajs_per_seed': n_sampled_trajs_per_seed,
                    'demo_batch_size': 100
                }, 'vsi': {
                    'q_optimizer_kwargs': pred_optimizer_kwargs,
                    'pred_optimizer_kwargs': pred_optimizer_kwargs,
                    'n_seeds_for_sampled_trajectories': n_seeds_for_sampled_trajectories,
                    'n_sampled_trajs_per_seed': n_sampled_trajs_per_seed,
                    'demo_batch_size': 100
                }}
    @property
    def tiq_config(self):
        return {'vgl': {}, 'vsi': {}}
    @property
    def tiq_train_config(self):
        base = {'vgl': dict(n_seeds_for_sampled_trajectories=self.n_seeds_total,
                            n_sampled_trajs_per_seed=self.n_expert_samples_per_seed,), 
                            'vsi': dict(n_seeds_for_sampled_trajectories=self.n_seeds_total,
                                    n_sampled_trajs_per_seed=self.n_expert_samples_per_seed,)}

        base['vgl'].update(dict(
            max_iter=10000,
            alpha_q = 0.01,
            alpha_sh = 0.01,
            
        ))
        base['vsi'].update(dict(
            max_iter=10000,
            alpha_q = 0.01,
            alpha_sh = 0.01,
        ))
        return base
        
    
    @property
    def me_config(self):
        return {'vgl': dict(
            vc_diff_epsilon=1e-5,
            gradient_norm_epsilon=1e-9,
            use_feature_expectations_for_vsi=False,
            demo_om_from_policy=True
        ), 'vsi': dict(
            vc_diff_epsilon=1e-5,
            gradient_norm_epsilon=1e-9,
            use_feature_expectations_for_vsi=False,
            demo_om_from_policy=True
        )}

    @property
    def me_train_config(self):
        return {'vgl': dict(n_seeds_for_sampled_trajectories=self.n_seeds_minibatch,
                            n_sampled_trajs_per_seed=self.n_expert_samples_per_seed_minibatch,), 'vsi': dict(n_seeds_for_sampled_trajectories=self.n_seeds_minibatch,
                                                                                                             n_sampled_trajs_per_seed=self.n_expert_samples_per_seed_minibatch,)}

    @property
    def pc_train_config(self):
        return {'vgl': dict(
            resample_trajectories_if_not_already_sampled=False, new_rng=None,
            random_trajs_proportion=0.5,
        ), 'vsi': dict(resample_trajectories_if_not_already_sampled=False, new_rng=None,
                       random_trajs_proportion=0.5
                       )}
    
    @property
    def ad_train_config(self):
        base = dict(vgl={}, vsi={})
        base['vgl'].update(dict(
            max_iter=1000,
            n_seeds_for_sampled_trajectories=self.n_seeds_total, # 1000 was too few with full variable length trajectories
            n_sampled_trajs_per_seed=self.n_expert_samples_per_seed,
            demo_batch_size=512, # 512
            gen_replay_buffer_capacity=2048, # 2048 for RW 
            n_disc_updates_per_round=2
        ))
        base['vsi'].update(dict(
            max_iter=1000,
            n_seeds_for_sampled_trajectories=self.n_seeds_total, 
            n_sampled_trajs_per_seed=self.n_expert_samples_per_seed,
            demo_batch_size=512, # 512
            gen_replay_buffer_capacity=2048, # 2048 for RW 
            n_disc_updates_per_round=2
        ))
        return base
    @abstractmethod
    def align_colors(self, align_func): pass


    def compute_precise_policy(self, env_real: FixedDestRoadWorldGymPOMDP, w, reward):
        pass


class EnvDataForIRLFireFighters(EnvDataForIRL):
    DEFAULT_HORIZON = 50
    DEFAULT_INITIAL_STATE_DISTRIBUTION = 'random'
    DEFAULT_N_SEEDS = 1000 
    DEFAULT_N_SEEDS_MINIBATCH = 200
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED_MINIBATCH = 1
    DEFAULT_N_REWARD_SAMPLES_PER_ITERATION = 1
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED = 1
    DEFAULT_FEATURE_SELECTION = FeatureSelectionFFEnv.ONE_HOT_FEATURES
    VALUES_NAMES = {(1.0, 0.0): 'Prof', (0.0, 1.0): 'Prox'}
    POLICY_SAVE_PATH = 'EXPERT_PPO_FF'

    def align_colors(self, align_func): return get_color_gradient(
        [1, 0, 0], [0, 0, 1], align_func)

    def __init__(self, env_name, discount_factor=1.0, feature_selection=DEFAULT_FEATURE_SELECTION, horizon=DEFAULT_HORIZON, is_society=False, initial_state_dist=DEFAULT_INITIAL_STATE_DISTRIBUTION, learn=False, use_pmovi_expert=USE_PMOVI_EXPERT, n_seeds_for_samplers=DEFAULT_N_SEEDS,
                 sampler_over_precalculated_trajs=True, **kwargs):
        super().__init__(env_name, discount_factor, **kwargs)

        self.horizon = horizon
        self.n_seeds_total = n_seeds_for_samplers
        self.target_align_func_sampler = profiled_society_sampler if is_society else lambda al_func: al_func
        env_real: FireFightersEnv = gym.make(
            self.env_name, feature_selection=feature_selection, horizon=self.horizon, initial_state_distribution=initial_state_dist)
        env_real.reset(seed=self.seed)
        self.feature_selection = feature_selection
        # train_init_state_distribution, test_init_state_distribution = train_test_split_initial_state_distributions(env_real.state_dim, 0.7)
        # env_training: FireFightersEnv = gym.make('FireFighters-v0', feature_selection = FEATURE_SELECTION, horizon=HORIZON, initial_state_distribution=train_init_state_distribution)
        env_training = env_real
        # train_init_state_distribution, test_init_state_distribution = env_real.initial_state_dist, env_real.initial_state_dist

        state, info = env_training.reset()
        action = ACTION_AGGRESSIVE_FIRE_SUPPRESSION
        next_state, rewards, done, trunc, info = env_training.step(action)
        env_training.reset(seed=self.seed)
        # Set to false if you already have a pretrained protocol in .pickle format
        learn = True

        if learn:
            v, q = pareto_multi_objective_value_iteration(
                env_real.real_env, discount_factor=self.discount_factor, model_used=None, pareto=True, normalized=False)
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
        normalisation_factor = 1 - discount_factor
        # Shows (normalised) Pareto front of initial state
        print("Pareto front of initial state : ",
              v_func[initial_state] * normalisation_factor)

        profiles = sample_example_profiles(
            profile_variety=self.profile_variety, n_values=self.n_values)
        profile_to_matrix = {}
        profile_to_assumed_matrix = {}
        for w in profiles:
            scalarised_q = scalarise_q_function(
                q_func, objectives=2, weights=w)

            pi_matrix = stochastic_optimal_policy_calculator(
                scalarised_q, w, deterministic=not self.stochastic_expert)
            profile_to_matrix[w] = pi_matrix

            _, _, assumed_expert_pi = mce_partition_fh(env_real, discount=discount_factor,
                                                       reward=env_real.reward_matrix_per_align_func(
                                                           w),
                                                       horizon = env_real.horizon,
                                                       approximator_kwargs=self.approximator_kwargs,
                                                       policy_approximator=self.policy_approximation_method,
                                                       deterministic=not self.stochastic_expert)

            profile_to_assumed_matrix[w] = assumed_expert_pi

        expert_policy_train = VAlignedDictSpaceActionPolicy(
            policy_per_va_dict=profile_to_matrix if use_pmovi_expert else profile_to_assumed_matrix, env=env_training, state_encoder=None, expose_state=not self.expose_observations)
        
        expert_kwargs =dict(batch_size=32,
            # For Roadworld. 
            n_steps=128,
            ent_coef=0.1, 
            learning_rate=0.002,  
            gamma=self.discount_factor,
            gae_lambda=0.999,
            clip_range=0.1,
            vf_coef=0.001, 
            n_epochs=10, 
            normalize_advantage = True, 
            tensorboard_log="./ppo_tensorboard_expert_ff/"
        )

        
        new_policy_kwargs = dict(
        features_extractor_class=OneHotFeatureExtractor,
        features_extractor_kwargs=dict(n_categories=env_real.state_dim),
        net_arch=[], # net arch is input to output linearly,
            )
        """new_policy_kwargs=dict(
            features_extractor_class=ObservationMatrixFeatureExtractor,
            features_extractor_kwargs=dict(observation_matrix=env_real.observation_matrix),
            net_arch=dict(pi=[30,30,30,20], vf=[30,30,30,20]), # net arch is input to output linearly,
        )""" 
        if self.approx_expert is False:
            no_previous = False
            try:
                expert_policy_train = LearnerValueSystemLearningPolicy.load(ref_env=env_training, path=EnvDataForIRLFireFighters.POLICY_SAVE_PATH)
            except FileNotFoundError:
                no_previous = True
            finally:
                if self.retrain or no_previous:
                    expert_policy_train = LearnerValueSystemLearningPolicy(learner_class=PPO,env=env_training,
                                                                        learner_kwargs=expert_kwargs,
                                                                        policy_class='MlpPolicy',
                                                                        policy_kwargs=new_policy_kwargs,
                                                                        masked=False,
                                                                        use_checkpoints=False, 
                                                                        state_encoder=None,
                                                                        observation_space=env_training.state_space,
                                                                        action_space=env_training.action_space
                                                                        )
                    for alignment in profiles:
                        timesteps = 1000000
                        expert_policy_train.learn(alignment_function=alignment, total_timesteps=timesteps, tb_log_name=f'EXPERT_PPO_{alignment}_{timesteps}')
                        expert_policy_train.save(path=EnvDataForIRLFireFighters.POLICY_SAVE_PATH)
                
            expert_policy_train = LearnerValueSystemLearningPolicy.load(ref_env=env_training, path=EnvDataForIRLFireFighters.POLICY_SAVE_PATH)    
        self.env = env_real
        
        

        self.vsi_targets = profiles

        self.initial_state_distribution_for_expected_alignment_eval = np.zeros((self.env.n_states,), dtype=np.float64)
        self.initial_state_distribution_for_expected_alignment_eval[self.env.real_env.encrypt(
                np.array([0, 3, 4, 0, 0, 3]))] = 1.0 # This testing state 
        # Fire in floor level 0
        # Severe fire(3), 
        # occupancy very high (4)
        # equipent readiness NONE
        # Visibility Poor (0)
        # Firefighter is in perfect condition (3)
        
        if sampler_over_precalculated_trajs:
            expert_trajs_train = expert_policy_train.obtain_trajectories(n_seeds=self.n_seeds_total, seed=self.seed, stochastic=self.stochastic_expert, repeat_per_seed=self.n_expert_samples_per_seed,
                                                                        align_funcs_in_policy=[profiles[0]], t_max=self.horizon,with_reward=True)
            
            initial_states_of_expert_trajs = [t.obs[0] for t in expert_trajs_train]
            initial_state_dist_expert_trajs = np.zeros_like(self.env.initial_state_dist)
            initial_state_dist_expert_trajs[initial_states_of_expert_trajs] = 1/len(initial_states_of_expert_trajs)
            initial_state_dist_intersected =  self.env.initial_state_dist * initial_state_dist_expert_trajs
            initial_state_dist_intersected /= np.sum(initial_state_dist_intersected)
            assert np.allclose(np.sum(initial_state_dist_intersected), 1.0)
            
            self.env.set_initial_state_distribution(initial_state_dist_intersected)
            assert len(np.where(initial_state_dist_intersected > 0.0)[0]) <= self.n_seeds_total
            expert_policy_train.env.set_initial_state_distribution(initial_state_dist_intersected)
            
            # TODO 20000? Maybe a big enough sample is enough to imitate correctly (to test?)
            expert_trajs_train = expert_policy_train.obtain_trajectories(n_seeds=self.n_seeds_total, seed=self.seed, stochastic=self.stochastic_expert, repeat_per_seed=self.n_expert_samples_per_seed,
                                                                    align_funcs_in_policy=profiles, t_max=self.horizon,with_reward=True)
        
            self.vgl_expert_train_sampler = partial(
                partial(random_sampler_among_trajs, replace=False), expert_trajs_train)
            self.vsi_expert_train_sampler = partial(
                partial(random_sampler_among_trajs, replace=False), expert_trajs_train)
            
        else:
            self.vgl_expert_train_sampler = partial(
                sampler_from_policy, expert_policy_train, stochastic=self.stochastic_expert, horizon=self.horizon, with_reward=True)
            self.vsi_expert_train_sampler = partial(
                sampler_from_policy, expert_policy_train, stochastic=self.stochastic_expert, horizon=self.horizon, with_reward=True)
        self.vgl_expert_policy = expert_policy_train
        self.vsi_expert_policy = expert_policy_train

        self.vgl_reference_policy = expert_policy_train
        self.vsi_reference_policy = expert_policy_train
    def set_defaults(self):

        self.stochastic_expert = False # TODO STOCHASTIC FF?
        self.learn_stochastic_policy = False
        self.environment_is_stochastic = False

        self.vgl_targets = [(1.0, 0.0), (0.0, 1.0)]
        #self.vsi_optimizer_kwargs = {"lr": 0.1, "weight_decay": 0.0000, "betas": (0.9, 0.999)} # ME DEMO_OM_TRUE 0.05 before
        self.vsi_optimizer_kwargs = {"lr": 0.01, "weight_decay": 0.0000} # ME DEMO_OM_FALSE
        #self.vsi_optimizer_kwargs = {"lr": 0.01, "weight_decay": 0.0000, "betas": (0.0, 0.0)} # if IQL TODO
        #self.vsi_optimizer_kwargs = {"lr": 0.0001, "weight_decay": 0.0000, "betas": (0.0,0.0)} # if TIQL
        #self.vsi_optimizer_kwargs = {"lr": 0.005, "weight_decay": 0.0000, "betas": (0.0, 0.0)} # if ADL TODO
        
        #self.vgl_optimizer_kwargs = {"lr": 0.03, "weight_decay": 0.000} # ME DEMO_OM FALSE
        self.vgl_optimizer_kwargs = {"lr": 0.1, "weight_decay": 0.000} # IN ME THIS IS NOT USED, SEE CUSTOM_OPTIMIZER_KWARGS.

        self.use_state = True
        self.use_action = True
        self.use_one_hot_state_action = True if EnvDataForIRLFireFighters.DEFAULT_FEATURE_SELECTION == FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS else False
        self.use_next_state = False
        self.use_done = False
        self.n_values = 2
        self.testing_profiles = list(itertools.product(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        self.testing_profiles.remove((0.0, 0.0))
        # self.clamp_rewards = [-1.0,1.0]

        # self.basic_layer_classes = [nn.Linear, ConvexAlignmentLayer]

        self.profile_variety = 6

        self.policy_approximation_method = PolicyApproximators.MCE_ORIGINAL
        self.approximator_kwargs = {
            'value_iteration_tolerance': 0.0000001, 'iterations': 2000}
        # self.vgl_reference_policy = 'random'
        # self.vsi_reference_policy = 'random'

        self.reward_trainer_kwargs = {
            'epochs': 5, # 1, 3 
            'lr': 0.001, # 0.001 0.0005
            'batch_size': 512, # 4096
        }

    @property
    def pc_config(self):
        
        base = super().pc_config
        base['vgl'].update(dict(query_schedule='hyperbolic',
                            stochastic_sampling_in_reference_policy=True))
        base['vsi'].update(dict(
                query_schedule='hyperbolic', #need constant
            stochastic_sampling_in_reference_policy=True,))
        return base
    
    
    @property
    def pc_train_config(self):
        base = super().pc_train_config
        base['vgl'].update(dict(
            max_iter=10000,
            random_trajs_proportion=0.8,
            n_seeds_for_sampled_trajectories=4500, # 2600, 3000, 3500 : PREV: 4500
            n_sampled_trajs_per_seed=2, #10, 2
            fragment_length=self.horizon, interactive_imitation_iterations=200, #total | 200, 150
            total_comparisons=10000, initial_comparison_frac=0.25,  #50000, 20000
            initial_epoch_multiplier=40, transition_oversampling=1 #15,5 | 4,1: PREV 40.
        ))
        base['vsi'].update(dict(
            max_iter=10000,
            random_trajs_proportion=0.8,
            n_seeds_for_sampled_trajectories=4500, # 2600, 3000, 3500 : PREV: 4500
            n_sampled_trajs_per_seed=2, #10, 2
            fragment_length=self.horizon, interactive_imitation_iterations=200, #total | 200, 150
            total_comparisons=10000, initial_comparison_frac=0.25,  #50000, 20000
            initial_epoch_multiplier=40, transition_oversampling=1 #15,5 | 4,1: PREV 40.
        ))
        return base

    @property
    def me_config(self):
        base = super().me_config
        base['vsi'].update(dict(
            vc_diff_epsilon=1e-10,
            gradient_norm_epsilon=1e-10,
            use_feature_expectations_for_vsi=False,
            demo_om_from_policy=False
        ))
        base['vgl'].update(dict(
            vc_diff_epsilon=1e-10,
            gradient_norm_epsilon=1e-10,
            use_feature_expectations_for_vsi=False,
            demo_om_from_policy=False
        ))
        return base

    @property
    def me_train_config(self):
        base = super().me_train_config
        base['vgl'].update(dict(max_iter=150,
                                custom_optimizer_kwargs = {
                                    "lr": 0.002, 
                                    "weight_decay": 0.0000, 
                                    "betas": (0.5, 0.99)} if self.me_config['vgl']['demo_om_from_policy'] is False 
                                    else {
                                    "lr": 0.002, 
                                    "weight_decay": 0.0000, 
                                    "betas": (0.5, 0.99)}
                                ))
        base['vsi'].update(dict(max_iter=150,
                                custom_optimizer_kwargs = {
                                    "lr": 0.03, 
                                    "weight_decay": 0.0000, 
                                    "betas": (0.5, 0.99)} if self.me_config['vsi']['demo_om_from_policy'] is False 
                                    else {
                                    "lr": 0.05, 
                                    "weight_decay": 0.0000, 
                                    "betas": (0.9, 0.99)}, 
            ))#200
        return base

    def get_assumed_grounding(self):
        
        
        if self.use_one_hot_state_action:
            assumed_grounding = np.zeros((self.env.n_states*self.env.action_dim, 2), dtype=np.float64)
            assumed_grounding[:, 0] = np.reshape(self.env.reward_matrix_per_align_func(
                (1.0, 0.0)), (self.env.n_states*self.env.action_dim,))
            assumed_grounding[:, 1] = np.reshape(self.env.reward_matrix_per_align_func(
                (0.0, 1.0)), (self.env.n_states*self.env.action_dim,))
        
            return assumed_grounding
        elif self.feature_selection == FeatureSelectionFFEnv.ONE_HOT_FEATURES:
            assumed_grounding = np.zeros(
            (self.env.n_states,self.env.action_dim, 2), dtype=np.float64)
            assumed_grounding[:, :, 0] = self.env.reward_matrix_per_align_func(
                (1.0, 0.0))
            assumed_grounding[:, :, 1] = self.env.reward_matrix_per_align_func(
                (0.0, 1.0))
            
            t_assumed_grounding = torch.tensor(assumed_grounding, dtype=torch.float32 ).requires_grad_(False)
            def processing_obs(torch_obs):
                states, actions = calculate_s_trans_ONE_HOT_FEATURES(torch_obs, self.env.real_env.state_space, self.env.real_env.action_space)
                
                
                ret = t_assumed_grounding[states,actions]
                """
                assert ret[0,0] == self.env.reward_matrix_per_align_func(
                (1.0, 0.0))[states[0], actions[0]]
                assert ret[0,1] == self.env.reward_matrix_per_align_func(
                (0.0, 1.0))[states[0], actions[0]]"""
                return ret
        
            return processing_obs
            
            raise NotImplementedError(
                "No knwon closed-form grounding with other feature configuration than one-hot encoded state-action pairs")

    def get_reward_net(self, algorithm='me'):
        if not self.use_one_hot_state_action:

            self.use_bias = [True, True, True, False, False]
            self.hid_sizes = [50, 100, 50, self.n_values,]
            self.basic_layer_classes = [
                nn.Linear, nn.Linear, nn.Linear, nn.Linear, ConvexAlignmentLayer]
            self.activations = [nn.LeakyReLU, nn.LeakyReLU,
                                nn.Tanh, nn.Tanh, nn.Identity]
        else:
            self.use_bias = False
            self.hid_sizes = [self.n_values,]
            self.basic_layer_classes = [nn.Linear, LinearAlignmentLayer]
            self.activations = [nn.Tanh, nn.Identity]

        return super().get_reward_net(algorithm)
    @property
    def ad_train_config(self):
        base = super().ad_train_config

        base['vgl'].update(dict(
            max_iter=20000,
            n_seeds_for_sampled_trajectories=self.n_seeds_total*self.n_expert_samples_per_seed, 
            n_sampled_trajs_per_seed=1,
            demo_batch_size=2048, 
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=1
            
        ))
        base['vsi'].update(dict(
            max_iter=2000000,
            n_seeds_for_sampled_trajectories=self.n_seeds_total*self.n_expert_samples_per_seed, 
            n_sampled_trajs_per_seed=1,
            demo_batch_size=500, # 256
            gen_replay_buffer_capacity=1000, # 2048 for RW # 
            n_disc_updates_per_round=1 
        ))
        return base
    @property
    def ad_config(self):
        
        base = super().ad_config
        base['vsi']['learner_class'] = PPO
        base['vgl']['learner_class'] = PPO
        base['vsi']['policy_class'] = 'MlpPolicy'
        base['vgl']['policy_class'] = 'MlpPolicy'
        base['vsi']['masked_policy'] = False
        base['vgl']['masked_policy'] = False

        if base['vsi']['learner_class'] == PPO:
            new_learner_kwargs =dict(batch_size=25,
            # For Roadworld. 
            n_steps=self.horizon, # n_steps should be the real horizon
            ent_coef=0.1, 
            learning_rate=0.02,
            gamma=self.discount_factor,
            gae_lambda=0.999,
            clip_range=0.05, 
            vf_coef=0.01, 
            n_epochs=5, 
            normalize_advantage = True, 
            tensorboard_log="./ppo_tensorboard_expert_ff/"
            )
            new_policy_kwargs = dict(
                features_extractor_class=OneHotFeatureExtractor,
                features_extractor_kwargs=dict(n_categories=self.env.state_dim),
                net_arch=[], # net arch is input to output linearly,
                    )
            """new_policy_kwargs=dict(
                features_extractor_class=ObservationMatrixFeatureExtractor,
                features_extractor_kwargs=dict(observation_matrix=self.env.observation_matrix),
                net_arch=dict(pi=[30,30,30,20], vf=[30,30,30,20]), # net arch is input to output linearly,
            )"""
            """new_policy_kwargs=dict(
                features_extractor_class=ObservationMatrixFeatureExtractor,
                features_extractor_kwargs=dict(observation_matrix=self.env.observation_matrix),
                net_arch=dict(pi=self.hid_sizes, vf=self.hid_sizes), # net arch is input to output linearly,
            ) """

        else:
            new_learner_kwargs = dict()
        base['vsi']['learner_kwargs'].update(new_learner_kwargs)
        base['vgl']['learner_kwargs'].update(new_learner_kwargs)

        base['vsi']['policy_kwargs'].update(new_policy_kwargs)
        base['vgl']['policy_kwargs'].update(new_policy_kwargs)
        
        return base
class EnvDataForRoadWorld(EnvDataForIRL):
    DEFAULT_HORIZON = 50
    DEFAULT_INITIAL_STATE_DISTRIBUTION = 'random'
    DEFAULT_N_SEEDS = 200 # TODO 200 is enough normally
    DEFAULT_N_SEEDS_MINIBATCH = 100
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED_MINIBATCH = 1
    DEFAULT_N_REWARD_SAMPLES_PER_ITERATION = 30
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED = 1
    DEFAULT_DEST = 64 # 413

    VALUES_NAMES = BASIC_PROFILE_NAMES

    def __init__(self, env_name=ROAD_WORLD_ENV_NAME, horizon=DEFAULT_HORIZON, dest=DEFAULT_DEST, n_seeds_for_samplers=DEFAULT_N_SEEDS, 
                 sampler_over_precalculated_trajs=True, **kwargs):
        super().__init__(env_name=env_name, **kwargs)
        self.horizon = horizon
        assert self.discount_factor == 1.0

        self.n_seeds_total = n_seeds_for_samplers
        self.dest = dest
        cv = 0  # cross validation process [0, 1, 2, 3, 4] 
        size = 100  # size of training data [100, 1000, 10000]

        self.stochastic_expert = False
        self.learn_stochastic_policy = False

        """environment"""
        edge_p = f"{DATA_FOLDER}/edge.txt"
        network_p = f"{DATA_FOLDER}/transit.npy"
        path_feature_p = f"{DATA_FOLDER}/feature_od.npy"
        train_p = f"{DATA_FOLDER}/cross_validation/train_CV%d_size%d.csv" % (
            cv, size)
        test_p = f"{DATA_FOLDER}/cross_validation/test_CV%d.csv" % cv
        node_p = f"{DATA_FOLDER}/node.txt"

        od_list, od_dist = ini_od_dist(train_p)

        env_creator = partial(RoadWorldGymPOMDP, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path=path_feature_p,
                              pre_reset=(od_list, od_dist), profile=(1.0, 0.0, 0.0), visualize_example=False, horizon=self.horizon,
                              feature_selection=FeatureSelection.ONLY_COSTS,
                              feature_preprocessing=FeaturePreprocess.NORMALIZATION,
                              use_optimal_reward_per_profile=False)
        env_single_all_goals = env_creator()

        od_list = [str(state) + '_' + str(dest)
                   for state in env_single_all_goals.valid_edges]

        env_creator = partial(RoadWorldGymPOMDP, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path=path_feature_p,
                              pre_reset=(od_list, od_dist),
                              profile=(1.0, 0.0, 0.0), visualize_example=True, horizon=self.horizon,
                              feature_selection=FeatureSelection.ONLY_COSTS,
                              feature_preprocessing=FeaturePreprocess.NORMALIZATION,
                              use_optimal_reward_per_profile=False)
        env_single = env_creator()

        env_real = FixedDestRoadWorldGymPOMDP(
            env=env_single, with_destination=dest)
        env_real.reset(seed=self.seed)


        

        profiles = sample_example_profiles(
            profile_variety=self.profile_variety, n_values=self.n_values)
        #profiles = [(1.0,0.0,0.0), (0.75,0.25,0.0), (0.5,0.5,0.0), (0.25, 0.75, 0.0), (0.0,1.0,0.0), (0.25,0.5,0.25), (0.0,0.75,0.25), (0.0,0.5,0.5), (0.25,0.5,0.25), (0.0, 0.25, 0.75), (0.0, 0.0, 1.0)]
        profile_to_assumed_matrix = {}
        if not self.approx_expert :
            self._state_to_action = dict()
        
        
        for w in profiles:
            reward = env_real.reward_matrix_per_align_func(w)
            if self.approx_expert == False:
                if self.retrain:
                    assumed_expert_pi = self.compute_precise_policy(env_real, w, reward)
                    #assumed_expert_pi2 = self.compute_precise_policy(env_real, w, None)
                    #np.testing.assert_almost_equal(assumed_expert_pi,assumed_expert_pi2)
                    np.save(f'roadworld_env_use_case/expert_policy_{w}.npy', assumed_expert_pi)
                else:
                    try:
                        assumed_expert_pi = np.load(f'roadworld_env_use_case/expert_policy_{w}.npy')
                    except:
                        assumed_expert_pi = self.compute_precise_policy(env_real, w, reward)
                        np.save(f'roadworld_env_use_case/expert_policy_{w}.npy', assumed_expert_pi)
                
                
                
                
                
            else:
                modified_kwargs = deepcopy(self.approximator_kwargs)
                modified_kwargs['iterations'] = 1000000
                _, _, assumed_expert_pi = mce_partition_fh(env_real, discount=self.discount_factor,
                                                       reward=reward,
                                                       horizon = env_real.horizon,
                                                       approximator_kwargs=modified_kwargs,
                                                       policy_approximator=self.policy_approximation_method,
                                                       deterministic=not self.stochastic_expert)
            profile_to_assumed_matrix[w] = assumed_expert_pi


        
        expert_policy_train = VAlignedDictSpaceActionPolicy(
            policy_per_va_dict=profile_to_assumed_matrix, env=env_real, state_encoder=None, expose_state=not self.expose_observations)
        

        self.env = env_real
        self.vgl_expert_policy = expert_policy_train
        self.vsi_expert_policy = expert_policy_train
        self.vgl_reference_policy = expert_policy_train
        self.vsi_reference_policy = expert_policy_train

        self.vsi_targets = profiles
        #self.initial_state_distribution_for_expected_alignment_eval = np.zeros((self.env.real_environ.n_states,), dtype=np.float64)
        #self.initial_state_distribution_for_expected_alignment_eval[49] = 1.0 
        
        expert_trajs_train = expert_policy_train.obtain_trajectories(n_seeds=self.n_seeds_total,
                                                                     seed=self.seed, stochastic=self.stochastic_expert,
                                                                     repeat_per_seed=self.n_expert_samples_per_seed, 
                                                                     align_funcs_in_policy=profiles,
                                                                     alignments_in_env=profiles,
                                                                     t_max=self.horizon,with_reward=True)
        
        for tr in expert_trajs_train:
            if self.expose_observations == False:
                assert tr.obs[-1] == self.dest
        
        if sampler_over_precalculated_trajs:
            for tr in expert_trajs_train:
                assert tr.obs[-1] == self.dest
            self.vgl_expert_train_sampler = partial(
                partial(random_sampler_among_trajs, replace=False), expert_trajs_train)
            self.vsi_expert_train_sampler = partial(
                partial(random_sampler_among_trajs, replace=False), expert_trajs_train)
        else:
            self.vgl_expert_train_sampler = partial(
                sampler_from_policy, expert_policy_train, stochastic=self.stochastic_expert, horizon=self.horizon)
            self.vsi_expert_train_sampler = partial(
                sampler_from_policy, expert_policy_train, stochastic=self.stochastic_expert, horizon=self.horizon)

        # vsi_expert_train_sampler = partial(profiled_society_traj_sampler_from_policy, expert_policy_train, stochastic=STOCHASTIC_EXPERT, horizon=HORIZON)

    def get_assumed_grounding(self):
        return nn.Identity().requires_grad_(False)

    
    def compute_precise_policy(self, env_real: FixedDestRoadWorldGymPOMDP, w=None, reward=None):
        custom_cost=None
        environment = env_real.real_environ
        if reward is not None:
            custom_cost = lambda stdes, profile: custom_cost_from_reward(environment, -reward, stdes,profile)
        policy = np.zeros((env_real.state_dim, env_real.action_dim),dtype=np.float32)

        
        for s in np.arange(env_real.state_dim):
            if s in env_real.invalid_states:
                continue
            edge_state = (s,env_real.goal_states[0])
            
            av_actions = np.asarray(env_real.real_environ.get_available_actions_from_state(edge_state))
            
            actions = None
            profile = None
            if w is not None:
                actions = self._state_to_action.get((s, w), None)
                profile = w
            if actions is None:
                if edge_state[0] == edge_state[1]:
                    actions =  environment.get_available_actions_from_state((edge_state[1], edge_state[1]))
                    if profile is not None:
                        self._state_to_action[((edge_state[1], edge_state[1]),profile)] = actions
                    
                else:
                    actions = []
                    path = environment.shortest_path_edges(profile=profile, to_state=edge_state[1], 
                                                           from_state=edge_state[0], with_length=False, 
                                                           all_alternatives=False, 
                                                        custom_cost=custom_cost, flattened=False)
                    
                    for iedge in range(1,len(path)):
                        prev_edge = path[iedge-1]
                        edge = path[iedge]
                        prev_state = (prev_edge, edge_state[1])
                        av_actions_prev = np.asarray(environment.get_available_actions_from_state(prev_state))
                        av_states = [environment.get_state_des_transition(prev_state, av_c) for av_c in av_actions_prev]
                        estate = (edge, edge_state[1])
                        istate_or_act = av_states.index(estate)
                        optimal_action = av_actions_prev[istate_or_act]

                        if iedge == 1:
                            actions.append(optimal_action)
                        if profile is not None:
                            self._state_to_action[(prev_state, profile)] = [optimal_action,]
            if actions is not None or len(actions) > 0:
                probs = np.zeros_like(np.asarray(av_actions), dtype=np.float64)
                for a in actions:
                    probs[np.where(av_actions==a)[0][0]] = 1.0/float(len(actions))
            else: 
                actions = [np.random.choice(av_actions),]
                probs = np.zeros_like(av_actions)
                probs[0] = 1.0
            policy[s,av_actions] = probs

        policy = concentrate_on_max_policy(policy, valid_action_checker = lambda s: env_real.valid_actions(s, None))
        if __debug__:
            for i in range(policy.shape[0]):
                assert np.allclose(np.sum(policy[i]), 1)
        return policy
    
    @property
    def pc_config(self):
        base = super().pc_config

        base['vgl'].update(
            query_schedule='hyperbolic',
            stochastic_sampling_in_reference_policy=False,)
        base['vsi'].update(
            query_schedule='hyperbolic',
            stochastic_sampling_in_reference_policy=False
        )
        return base

    @property
    def pc_train_config(self):

        base = super().pc_train_config

        base['vgl'].update(dict(
            max_iter=10000,
            n_seeds_for_sampled_trajectories=2500, # 1000 was too few with full variable length trajectories
            n_sampled_trajs_per_seed=1,
            fragment_length=self.horizon, interactive_imitation_iterations=200, 
            total_comparisons=7000, initial_comparison_frac=0.15,
            initial_epoch_multiplier=100, transition_oversampling=1,
            random_trajs_proportion=0.8
        ))
        base['vsi'].update(dict(
            max_iter=10000,
            n_seeds_for_sampled_trajectories=2500, # 1000 was too few with full variable length trajectories
            n_sampled_trajs_per_seed=1,
            fragment_length=self.horizon, interactive_imitation_iterations=200,
            total_comparisons=7000, initial_comparison_frac=0.15,
            initial_epoch_multiplier=100, transition_oversampling=1,
            random_trajs_proportion=0.8
        ))
        return base

    @property
    def ad_config(self):
        base = super().ad_config
        base['vsi']['learner_class'] = PPO
        base['vgl']['learner_class'] = PPO
        base['vsi']['policy_class'] = 'MlpPolicy'
        base['vgl']['policy_class'] = 'MlpPolicy'
        base['vsi']['masked_policy'] = False # will use the MaskedPPO library. FALSE is better with AIRL
        base['vgl']['masked_policy'] = False


        if base['vsi']['learner_class'] == MaskablePPO:
            new_learner_kwargs =dict(batch_size=32,
                # For Roadworld. 
                n_steps=128, 
                ent_coef=0.1, 
                learning_rate=0.004,  
                gae_lambda=0.999,
                clip_range=0.05, 
                vf_coef=0.4, 
                n_epochs=5, 
                normalize_advantage = False, 
                tensorboard_log="./mppo_tensorboard_expert_rw/"
            )
            new_policy_kwargs=dict(
        features_extractor_class=OneHotFeatureExtractor,
        features_extractor_kwargs=dict(n_categories=self.env.state_dim),
        net_arch=dict(pi=[], vf=[]), # net arch is input to output linearly,
            )
        elif base['vsi']['learner_class'] == PPO:
            new_learner_kwargs =dict(batch_size=50,
                # For Roadworld. 
                n_steps=self.horizon, 
                ent_coef=0.01, 
                learning_rate=0.001,  
                gae_lambda=0.999,
                clip_range=0.05, 
                vf_coef=0.01, 
                n_epochs=5,
                normalize_advantage = False, 
                tensorboard_log="./ppo_tensorboard_expert_rw/"
            )
            new_policy_kwargs=dict(
            features_extractor_class=OneHotFeatureExtractor,
            features_extractor_kwargs=dict(n_categories=self.env.state_dim),
            net_arch=dict(pi=[], vf=[]), # net arch is input to output linearly,
                )
        else:
            new_learner_kwargs = dict()
        base['vsi']['learner_kwargs'].update(new_learner_kwargs)
        base['vgl']['learner_kwargs'].update(new_learner_kwargs)
        
        base['vsi']['policy_kwargs'].update(new_policy_kwargs)
        base['vgl']['policy_kwargs'].update(new_policy_kwargs)
        return base
    @property
    def me_config(self):
        base = super().me_config
        base['vgl'].update(dict(vc_diff_epsilon=1e-10,
            gradient_norm_epsilon=1e-10,
            use_feature_expectations_for_vsi=False,
            demo_om_from_policy=False))
        base['vsi'].update(dict(vc_diff_epsilon=1e-10,
            gradient_norm_epsilon=1e-10,
            use_feature_expectations_for_vsi=False,
            demo_om_from_policy=False))
        return base

    @property
    def me_train_config(self):
        base = super().me_train_config
        base['vgl'].update(dict(
            max_iter=100, custom_optimizer_kwargs = {
                                    "lr": 0.1, 
                                    "weight_decay": 0.0000, 
                                    "betas": (0.5, 0.99)} if self.me_config['vgl']['demo_om_from_policy'] is False 
                                    else {
                                    "lr": 0.1, 
                                    "weight_decay": 0.0000, 
                                    "betas": (0.9, 0.99)}
                                ))
        base['vsi'].update(dict(
            max_iter=200, custom_optimizer_kwargs = {
                                    "lr": 0.07, 
                                    "weight_decay": 0.0000, 
                                    "betas": (0.5, 0.99)} if self.me_config['vsi']['demo_om_from_policy'] is False 
                                    else {
                                    "lr": 0.1, 
                                    "weight_decay": 0.0000, 
                                    "betas": (0.9, 0.99)}
                                ))

        return base
    @property
    def ad_train_config(self):
        base = super().ad_train_config
        base['vgl'].update(dict(
            max_iter=50000, # TODO 200000
            n_seeds_for_sampled_trajectories=self.n_seeds_total, 
            n_sampled_trajs_per_seed=1,
            demo_batch_size=500, # 512
            gen_replay_buffer_capacity=2000, # 2048 for RW 
            n_disc_updates_per_round=2,
            custom_optimizer_kwargs={"lr": 0.005, "weight_decay": 0.0000, 'betas': (0.0,0.0)} # TODO TEST.
            
        ))
        base['vsi'].update(dict(
            max_iter=100000, # TODO 200000
            n_seeds_for_sampled_trajectories=self.n_seeds_total, 
            n_sampled_trajs_per_seed=1,
            demo_batch_size=500, # 512
            gen_replay_buffer_capacity=2000, # 2048 for RW 
            n_disc_updates_per_round=2,
            custom_optimizer_kwargs={"lr": 0.005, "weight_decay": 0.0000, 'betas': (0.0,0.0)} 
        ))
        return base

    def align_colors(self, align_func): return get_linear_combination_of_colors(
        BASIC_PROFILES, PROFILE_COLORS_VEC, align_func)

    def set_defaults(self):
        super().set_defaults()
        self.use_action = False
        self.stochastic_expert = False
        self.learn_stochastic_policy = False
        
        self.environment_is_stochastic = False
        self.use_state = False
        self.use_one_hot_state_action = False
        self.use_next_state = True
        self.use_done = False
        self.hid_sizes = [3,]
        self.use_bias = False
        self.basic_layer_classes = [ConvexLinearModule, ConvexAlignmentLayer]
        self.activations = [nn.Identity, nn.Identity]
        self.vgl_targets = BASIC_PROFILES
        self.profile_variety = 4 # If 4 it is 0.67, 0.33...
        self.n_values = 3
        self.negative_grounding_layer = True
        
        #self.vsi_optimizer_kwargs = {"lr": 0.2, "weight_decay": 0.0000} # if MCL

        self.vsi_optimizer_kwargs = {"lr": 0.005, "weight_decay": 0.0000, 'betas': (0.0,0.0)} # if AIRL
        
        #self.vsi_optimizer_kwargs = {"lr": 0.5, "weight_decay": 0.0000} # if IQL
        
        #self.vsi_optimizer_kwargs = {"lr": 0.001, "weight_decay": 0.0000} # if TIQL
        self.vgl_optimizer_kwargs = {"lr": 0.05, "weight_decay": 0.0000, 'betas': (0.0,0.0)} # if AIRL
        
        self.policy_approximation_method = PolicyApproximators.NEW_SOFT_VALUE_ITERATION
        self.approximator_kwargs = {
            'value_iteration_tolerance': 0.00001, 'iterations': 2000}
        # self.vgl_reference_policy = 'random' # SEE __INIT__!
        # self.vsi_reference_policy = 'random' # SEE __INIT__!

        self.testing_profiles = list(itertools.product(
            [0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]))
        # self.testing_profiles.remove([0.0, 0.0,0.0])
        self.testing_profiles.remove((0.0, 0.0, 0.0))
        self.reward_trainer_kwargs = {
            'epochs': 1,
            'lr': 0.03,
            'batch_size': 32,
        }
