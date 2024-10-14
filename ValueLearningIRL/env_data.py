

        
from abc import abstractmethod
import enum
from functools import partial
import pickle
import gymnasium as gym
import numpy as np
from firefighters_use_case.constants import ACTION_AGGRESSIVE_FIRE_SUPPRESSION
from firefighters_use_case.pmovi import pareto_multi_objective_value_iteration, scalarise_q_function
from firefighters_use_case.scalarisation import stochastic_optimal_policy_calculator
from src.envs.firefighters_env import FeatureSelectionFFEnv, FireFightersEnv
from src.envs.roadworld_env import FixedDestRoadWorldGymPOMDP
from src.network_env import DATA_FOLDER, FeaturePreprocess, FeatureSelection, RoadWorldGymPOMDP
from src.utils.load_data import ini_od_dist
from src.values_and_costs import BASIC_PROFILES
from src.vsl_algorithms.me_irl_for_vsl import PolicyApproximators, mce_partition_fh
from torch import nn, optim

from src.vsl_algorithms.preference_model_vs import CrossEntropyRewardLossForQualitativePreferences
from src.vsl_algorithms.vsl_plot_utils import get_color_gradient
from src.vsl_policies import VAlignedDictSpaceActionPolicy, profiled_society_sampler, random_sampler_among_trajs, sampler_from_policy
from src.vsl_reward_functions import ConvexAlignmentLayer, LinearAlignmentLayer, LinearVSLRewardFunction, TrainingModes
from utils import sample_example_profiles
from imitation.algorithms.preference_comparisons import CrossEntropyRewardLoss

USE_PMOVI_EXPERT = False
FIRE_FIGHTERS_ENV_NAME = 'FireFighters-v0'
ROAD_WORLD_ENV_NAME = 'FixedDestRoadWorld-v0'


FIREFIGHTER_ALFUNC_COLORS = lambda align_func: get_color_gradient([1,0,0], [0,0,1], align_func[0])

class PrefLossClasses(enum.Enum):
    CROSS_ENTROPY = 'cross_entropy'
    CROSS_ENTROPY_MODIFIED = 'cross_entropy_modified'

class EnvDataForIRL():
    DEFAULT_HORIZON = 50
    DEFAULT_INITIAL_STATE_DISTRIBUTION = 'random'
    DEFAULT_N_SEEDS = 100
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
        self.vsi_optimizer_kwargs={"lr": 0.15, "weight_decay": 0.0000}
        self.vgl_optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0000}
        self.vgl_expert_policy = None
        self.vsi_expert_policy = None
        self.vgl_expert_train_sampler = None
        self.vsi_expert_train_sampler = None
        self.vgl_reference_policy = None
        self.vsi_reference_policy = None
        self.profile_variety = 1
        self.n_values = 0

        self.target_align_func_sampler = lambda al_func: al_func
        self.reward_trainer_kwargs = {
                                 'epochs': 1,
                                 'lr': 0.08,
                                 'batch_size': 512,
                                 'minibatch_size': 32,
                             }
        self.horizon = self.__class__.DEFAULT_HORIZON
        self.initial_state_distribution = 'random'
        self.stochastic_expert = True
        self.learn_stochastic_policy = True
        self.environment_is_stochastic = True # If known to be False, 
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
        self.discount_factor_preferences = None

        # Override defaults with specific method for other environments
        self.n_seeds_total = 100
        self.n_expert_samples_per_seed = 1 if self.stochastic_expert is False else self.__class__.DEFAULT_N_EXPERT_SAMPLES_PER_SEED
        self.n_reward_samples_per_iteration = self.__class__.DEFAULT_N_REWARD_SAMPLES_PER_ITERATION
        self.n_expert_samples_per_seed_minibatch = self.__class__.DEFAULT_N_EXPERT_SAMPLES_PER_SEED_MINIBATCH
        self.n_seeds_minibatch = self.__class__.DEFAULT_N_SEEDS_MINIBATCH
    


        
        self.set_defaults()
        for kw, kwv in kwargs.items():
            setattr(self, kw, kwv)
            
        assert len(self.activations) == len(self.basic_layer_classes)
        assert len(self.hid_sizes) == len(self.basic_layer_classes) -1
        self.custom_reward_net_initializer = None 
        self.discount_factor_preferences = self.discount_factor if self.discount_factor_preferences is None else self.discount_factor_preferences
        if 'loss_class' in vars(self).keys():
            if self.loss_class == PrefLossClasses.CROSS_ENTROPY_MODIFIED: 
                self.loss_class = CrossEntropyRewardLossForQualitativePreferences 
            elif self.loss_class == PrefLossClasses.CROSS_ENTROPY:
                self.loss_class = CrossEntropyRewardLoss 
            else:
                self.loss_class = CrossEntropyRewardLoss 
            self.loss_kwargs = self.loss_kwargs


    @abstractmethod
    def set_defaults(self):
        pass
    
    def get_reward_net(self, algorithm='me'):
        return LinearVSLRewardFunction(
            environment=self.env,
            use_state=self.use_state,
            use_action=self.use_action,
            use_next_state=self.use_next_state,
            use_done=self.use_done,
            hid_sizes=self.hid_sizes,
            reward_bias=self.reward_bias,
            basic_layer_classes= self.basic_layer_classes,
            use_one_hot_state_action=self.use_one_hot_state_action,
            activations=self.activations,
            negative_grounding_layer=self.negative_grounding_layer,
            use_bias = self.use_bias,
            mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION
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
    def me_config(self):
        return {'vgl': dict(
            vc_diff_epsilon=1e-5,
            gradient_norm_epsilon=1e-6,
            use_feature_expectations_for_vsi=False,
            demo_om_from_policy = True
        ), 'vsi': dict(
            vc_diff_epsilon=1e-5,
            gradient_norm_epsilon=1e-6,
            use_feature_expectations_for_vsi=False,
            demo_om_from_policy = True
        )}
    
    @property
    def me_train_config(self):
        return {'vgl': dict(n_seeds_for_sampled_trajectories=self.n_seeds_minibatch,
            n_sampled_trajs_per_seed=self.n_expert_samples_per_seed_minibatch,), 'vsi': dict(n_seeds_for_sampled_trajectories=self.n_seeds_minibatch,
            n_sampled_trajs_per_seed=self.n_expert_samples_per_seed_minibatch,)}
    
    @property
    def pc_train_config(self):
        return {'vgl': dict(
            resample_trajectories_if_not_already_sampled=True, new_rng=None,
        ), 'vsi': dict(resample_trajectories_if_not_already_sampled=True, new_rng=None,
                       )}
    
    
class EnvDataForIRLFireFighters(EnvDataForIRL):
    DEFAULT_HORIZON = 50
    DEFAULT_INITIAL_STATE_DISTRIBUTION = 'random'
    DEFAULT_N_SEEDS = 100
    DEFAULT_N_SEEDS_MINIBATCH = 30
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED_MINIBATCH = 10
    DEFAULT_N_REWARD_SAMPLES_PER_ITERATION = 10
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED = 5

    def __init__(self, env_name, discount_factor=0.7, feature_selection= FeatureSelectionFFEnv.ONE_HOT_OBSERVATIONS, horizon=DEFAULT_HORIZON, is_society=False, initial_state_dist=DEFAULT_INITIAL_STATE_DISTRIBUTION, learn=False, use_pmovi_expert=USE_PMOVI_EXPERT, n_seeds_for_samplers = DEFAULT_N_SEEDS, 
                 sampler_over_precalculated_trajs = False, **kwargs):
        super().__init__(env_name, discount_factor, **kwargs)
        self.horizon = horizon
        self.n_seeds_total = n_seeds_for_samplers
        self.target_align_func_sampler=profiled_society_sampler if is_society else lambda al_func: al_func

        env_real: FireFightersEnv = gym.make(self.env_name, feature_selection = feature_selection, horizon=self.horizon, initial_state_distribution=initial_state_dist)
        env_real.reset(seed=self.seed)
        
        #train_init_state_distribution, test_init_state_distribution = train_test_split_initial_state_distributions(env_real.state_dim, 0.7)
        #env_training: FireFightersEnv = gym.make('FireFighters-v0', feature_selection = FEATURE_SELECTION, horizon=HORIZON, initial_state_distribution=train_init_state_distribution)
        env_training = env_real
        #train_init_state_distribution, test_init_state_distribution = env_real.initial_state_dist, env_real.initial_state_dist

        state, info = env_training.reset()
        print("Initial State:", state, info)
        action = ACTION_AGGRESSIVE_FIRE_SUPPRESSION
        next_state, rewards, done, trunc, info = env_training.step(action)
        print("Next State:", next_state)
        print("Rewards:", rewards)
        print("Done:", done)
        print("Info:", info)
        env_training.reset(seed=self.seed)
        # Set to false if you already have a pretrained protocol in .pickle format
        learn = True

        if learn:
            v, q = pareto_multi_objective_value_iteration(env_real.real_env, discount_factor=self.discount_factor, model_used=None, pareto=True, normalized=False)
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
        print("Pareto front of initial state : ", v_func[initial_state] * normalisation_factor)
        


        profiles  = sample_example_profiles(profile_variety=self.profile_variety,n_values=self.n_values)
        profile_to_matrix = {}
        profile_to_assumed_matrix = {}
        for w in profiles:
            scalarised_q = scalarise_q_function(q_func, objectives=2, weights=w)

            pi_matrix = stochastic_optimal_policy_calculator(scalarised_q, w, deterministic= not self.stochastic_expert)
            profile_to_matrix[w] = pi_matrix

            _,_, assumed_expert_pi = mce_partition_fh(env_real, discount=discount_factor,
                                                reward=env_real.reward_matrix_per_align_func(w),
                                                approximator_kwargs=self.approximator_kwargs,
                                                policy_approximator=self.policy_approximation_method,
                                                deterministic= not self.stochastic_expert )
            
            profile_to_assumed_matrix[w] = assumed_expert_pi
        
        

        expert_policy_train = VAlignedDictSpaceActionPolicy(policy_per_va_dict = profile_to_matrix if use_pmovi_expert else profile_to_assumed_matrix, env = env_training, state_encoder=None)
        expert_trajs_train = expert_policy_train.obtain_trajectories(n_seeds=n_seeds_for_samplers, seed=self.seed, stochastic=self.stochastic_expert, repeat_per_seed=self.n_expert_samples_per_seed, 
                                                                     with_alignfunctions=profiles, t_max=self.horizon)
        
        self.env = env_real
        self.vgl_expert_policy = expert_policy_train
        self.vsi_expert_policy = expert_policy_train
        self.vsi_targets = profiles

        if sampler_over_precalculated_trajs:
            self.vgl_expert_train_sampler = partial(random_sampler_among_trajs, expert_trajs_train)
            self.vsi_expert_train_sampler = partial(random_sampler_among_trajs, expert_trajs_train)
        else:
            self.vgl_expert_train_sampler = partial(sampler_from_policy, expert_policy_train, stochastic=self.stochastic_expert, horizon=self.horizon)
            self.vsi_expert_train_sampler = partial(sampler_from_policy, expert_policy_train, stochastic=self.stochastic_expert, horizon=self.horizon)
        

    def set_defaults(self):
        
        self.stochastic_expert = True
        self.learn_stochastic_policy = True
        
        self.stochastic_expert =True
        self.learn_stochastic_policy = True
        self.environment_is_stochastic = False


        self.vgl_targets = [(1.0, 0.0), (0.0,1.0)]
        self.vsi_optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0000}
        self.vgl_optimizer_kwargs={"lr": 0.3, "weight_decay": 0.0000}
        
        self.use_state = True
        self.use_action = True
        self.use_one_hot_state_action = True
        self.use_next_state = False
        self.use_done = False
        self.n_values = 2
        self.hid_sizes = [self.n_values,]
        #self.basic_layer_classes = [nn.Linear, ConvexAlignmentLayer]
        self.basic_layer_classes = [nn.Linear, LinearAlignmentLayer]
        self.activations=[nn.Tanh, nn.Identity]
        self.profile_variety = 6

        self.policy_approximation_method = PolicyApproximators.MCE_ORIGINAL
        self.approximator_kwargs={'value_iteration_tolerance': 0.0000001, 'iterations': 100}
        self.vgl_reference_policy = 'random'
        self.vsi_reference_policy = 'random'

        self.reward_trainer_kwargs = {
                                 'epochs': 1,
                                 'lr': 0.08,
                                 'batch_size': 512,
                                 'minibatch_size': 32,
                             }
        
    @property
    def pc_config(self):
        return super().pc_config
    @property
    def pc_train_config(self):
        base = super().pc_train_config

        base['vgl'].update(dict(
            max_iter=20000,
            
            n_seeds_for_sampled_trajectories=200,
            n_sampled_trajs_per_seed=10,
            fragment_length=self.horizon, interactive_imitation_iterations=100, 
            total_comparisons=2000, initial_comparison_frac=0.1, 
                    initial_epoch_multiplier=1, transition_oversampling=5,
        ))
        base['vsi'].update(dict(
            max_iter=20000,
            n_seeds_for_sampled_trajectories=200,
            n_sampled_trajs_per_seed=10,
            fragment_length=self.horizon, interactive_imitation_iterations=100, 
            total_comparisons=2000, initial_comparison_frac=0.1, 
                    initial_epoch_multiplier=1, transition_oversampling=5,
        )
        )
        return base
    
    
    @property
    def me_config(self):
        base = super().me_config
        return base
    
    @property
    def me_train_config(self):
        base = super().me_train_config
        base['vgl'].update(dict(max_iter=200,))
        base['vsi'].update(dict(max_iter=200))
        return base

    def get_assumed_grounding(self):
        if self.use_one_hot_state_action:
            assumed_grounding = np.zeros((self.env.n_states*self.env.action_dim, 2), dtype=np.float64)
            assumed_grounding[:,0] = np.reshape(self.env.reward_matrix_per_align_func((1.0,0.0)), (self.env.n_states*self.env.action_dim))
            assumed_grounding[:,1] = np.reshape(self.env.reward_matrix_per_align_func((0.0,1.0)),(self.env.n_states*self.env.action_dim))

            return assumed_grounding
        else:
            raise NotImplementedError("No knwon closed-form grounding with other feature configuration than one-hot encoded state-action pairs")
    
class EnvDataForRoadWorld(EnvDataForIRL):
    DEFAULT_HORIZON = 40
    DEFAULT_INITIAL_STATE_DISTRIBUTION = 'random'
    DEFAULT_N_SEEDS = 100
    DEFAULT_N_SEEDS_MINIBATCH = 20
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED_MINIBATCH = 10
    DEFAULT_N_REWARD_SAMPLES_PER_ITERATION = 30
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED = 5
    DEFAULT_DEST = 413

    def __init__(self, env_name=ROAD_WORLD_ENV_NAME, horizon=DEFAULT_HORIZON, dest=DEFAULT_DEST, n_seeds_for_samplers=DEFAULT_N_SEEDS, sampler_over_precalculated_trajs=False, **kwargs):
        super().__init__(env_name=env_name, **kwargs)
        self.horizon = horizon
        assert self.discount_factor == 1.0

        self.n_seeds_total = n_seeds_for_samplers

        cv = 0  # cross validation process [0, 1, 2, 3, 4] # TODO (?)
        size = 100  # size of training data [100, 1000, 10000]


        self.stochastic_expert = False
        self.learn_stochastic_policy = False

        """environment"""
        edge_p = f"{DATA_FOLDER}/edge.txt"
        network_p = f"{DATA_FOLDER}/transit.npy"
        path_feature_p = f"{DATA_FOLDER}/feature_od.npy"
        train_p = f"{DATA_FOLDER}/cross_validation/train_CV%d_size%d.csv" % (cv, size)
        test_p = f"{DATA_FOLDER}/cross_validation/test_CV%d.csv" % cv
        node_p = f"{DATA_FOLDER}/node.txt"


        od_list, od_dist = ini_od_dist(train_p)
        
        env_creator = partial(RoadWorldGymPOMDP, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path = path_feature_p, 
                            pre_reset=(od_list, od_dist), profile=(1.0,0.0,0.0), visualize_example=False, horizon=self.horizon,
                            feature_selection=FeatureSelection.ONLY_COSTS,
                            feature_preprocessing=FeaturePreprocess.NORMALIZATION, 
                            use_optimal_reward_per_profile=False)
        env_single_all_goals = env_creator()
        
        od_list = [str(state) + '_' + str(dest) for state in env_single_all_goals.valid_edges]


        env_creator = partial(RoadWorldGymPOMDP, network_path=network_p, edge_path=edge_p, node_path=node_p, path_feature_path = path_feature_p, 
                              pre_reset=(od_list, od_dist), 
                              profile=(1.0,0.0,0.0), visualize_example=False, horizon=self.horizon,
                        feature_selection=FeatureSelection.ONLY_COSTS,
                        feature_preprocessing=FeaturePreprocess.NORMALIZATION, 
                        use_optimal_reward_per_profile=False)
        env_single = env_creator()

        
        env_real = FixedDestRoadWorldGymPOMDP(env=env_single, with_destination=dest)
        env_real.reset(seed=self.seed)
        

        profiles  = sample_example_profiles(profile_variety=self.profile_variety,n_values=self.n_values)
        profile_to_matrix = {}
        profile_to_assumed_matrix = {}
        
        for w in profiles:
            reward = env_real.reward_matrix_per_align_func(w)
            
            _,_, assumed_expert_pi = mce_partition_fh(env_real, discount=self.discount_factor,
                                                reward=reward,
                                                approximator_kwargs=self.approximator_kwargs,
                                                policy_approximator=self.policy_approximation_method,
                                                deterministic= not self.stochastic_expert )
            profile_to_assumed_matrix[w] = assumed_expert_pi
        
        expert_policy_train = VAlignedDictSpaceActionPolicy(policy_per_va_dict = profile_to_assumed_matrix, env = env_real, state_encoder=None)
        expert_trajs_train = expert_policy_train.obtain_trajectories(n_seeds=self.n_seeds_total, 
            seed=self.seed, stochastic=self.stochastic_expert, 
            repeat_per_seed=self.n_expert_samples_per_seed, with_alignfunctions=profiles, 
            t_max=self.horizon)

        self.env = env_real
        self.vgl_expert_policy = expert_policy_train
        self.vsi_expert_policy = expert_policy_train
        # TODO: construir politicas aleatorias dentro de la legalidad de estados.

        self.vgl_reference_policy = expert_policy_train
        self.vsi_reference_policy = expert_policy_train

        self.vsi_targets = profiles

        if sampler_over_precalculated_trajs:
            self.vgl_expert_train_sampler = partial(random_sampler_among_trajs, expert_trajs_train)
            self.vsi_expert_train_sampler = partial(random_sampler_among_trajs, expert_trajs_train)
        else:
            self.vgl_expert_train_sampler = partial(sampler_from_policy, expert_policy_train, stochastic=self.stochastic_expert, horizon=self.horizon)
            self.vsi_expert_train_sampler = partial(sampler_from_policy, expert_policy_train, stochastic=self.stochastic_expert, horizon=self.horizon)
        
        #vsi_expert_train_sampler = partial(profiled_society_traj_sampler_from_policy, expert_policy_train, stochastic=STOCHASTIC_EXPERT, horizon=HORIZON)


    def get_assumed_grounding(self):
        return nn.Identity().requires_grad_(False)
    
    @property
    def pc_config(self):
        base = super().pc_config
        
        base['vgl'].update(
            query_schedule = 'constant',
            stochastic_sampling_in_reference_policy=False,)
        base['vsi'].update(
            query_schedule = 'constant',
            stochastic_sampling_in_reference_policy=False
        )
        return base

    @property
    def pc_train_config(self):

        base = super().pc_train_config

        base['vgl'].update(dict(
            max_iter=10000,
            
            n_seeds_for_sampled_trajectories=1500,
            n_sampled_trajs_per_seed=1,
            fragment_length=10, interactive_imitation_iterations=150, 
            total_comparisons=1000, initial_comparison_frac=0.1, 
                    initial_epoch_multiplier=1, transition_oversampling=3,
        ))
        base['vsi'].update(dict(
            max_iter=10000,
            n_seeds_for_sampled_trajectories=1500,
            n_sampled_trajs_per_seed=1,
            fragment_length=10, interactive_imitation_iterations=150, 
            total_comparisons=1000, initial_comparison_frac=0.1, 
                    initial_epoch_multiplier=1, transition_oversampling=3,
        )
        )
        return base
    @property
    def me_config(self):
        return super().me_config
    
    @property
    def me_train_config(self):
        base= super().me_train_config
        base['vgl'].update(dict(
            max_iter=500))
        base['vsi'].update(dict(
            max_iter=500))
        
        return base

    def set_defaults(self):
        super().set_defaults()
        self.use_action = False
        self.stochastic_expert = False
        self.learn_stochastic_policy = False
        self.use_state = False
        self.use_one_hot_state_action = False
        self.use_next_state = True
        self.use_done = False
        self.environment_is_stochastic = False
        self.hid_sizes = [3,]
        
        self.basic_layer_classes = [nn.Linear, ConvexAlignmentLayer]
        self.activations=[nn.Identity, nn.Identity]
        self.vgl_targets = BASIC_PROFILES
        self.profile_variety = 4
        self.n_values = 3
        self.negative_grounding_layer = True
        
        self.vsi_optimizer_kwargs={"lr": 0.15, "weight_decay": 0.0000}
        self.vgl_optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0000}

        self.policy_approximation_method = PolicyApproximators.MCE_ORIGINAL
        self.approximator_kwargs={'value_iteration_tolerance': 0.0000001, 'iterations': 1000}
        self.vgl_reference_policy = 'random'
        self.vsi_reference_policy = 'random'
        
        self.reward_trainer_kwargs = {
                                 'epochs': 1,
                                 'lr': 0.02,
                                 'batch_size': 1024,
                             }
   
