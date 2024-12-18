from abc import abstractmethod
import enum
from functools import partial
import itertools
import pickle
import gymnasium as gym
import numpy as np
from firefighters_use_case.constants import ACTION_AGGRESSIVE_FIRE_SUPPRESSION
from firefighters_use_case.pmovi import pareto_multi_objective_value_iteration, scalarise_q_function
from firefighters_use_case.scalarisation import stochastic_optimal_policy_calculator
from src.envs.firefighters_env import FeatureSelectionFFEnv, FireFightersEnv
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
from src.vsl_policies import VAlignedDictSpaceActionPolicy, profiled_society_sampler, random_sampler_among_trajs, sampler_from_policy
from src.vsl_reward_functions import ConvexAlignmentLayer, ConvexLinearModule, LinearAlignmentLayer, LinearVSLRewardFunction, TrainingModes
from utils import sample_example_profiles
from imitation.algorithms.preference_comparisons import CrossEntropyRewardLoss

USE_PMOVI_EXPERT = False
FIRE_FIGHTERS_ENV_NAME = 'FireFighters-v0'
ROAD_WORLD_ENV_NAME = 'FixedDestRoadWorld-v0'

def custom_cost_from_reward(environment:RoadWorldGymPOMDP, reward, state_des, profile):
        #print(environment.pre_acts_and_pre_states[0])
        rews = [reward[s,a] for a,s in zip(*environment.pre_acts_and_pre_states[state_des[0]]) if s != environment.cur_des]
        state_acts = [(s,a) for a,s in zip(*environment.pre_acts_and_pre_states[state_des[0]]) if s != environment.cur_des]
        #print(rews, state_des)
        #print(state_acts)
        #print([environment.get_state_des_transition((s, 413), a) for s,a in state_acts])
        
        if len(rews) == 0.0:
            rews = [reward[state_des[0],0]]
        np.testing.assert_almost_equal(rews, rews[0])
        return rews[0]
    

class PrefLossClasses(enum.Enum):
    CROSS_ENTROPY = 'cross_entropy'
    CROSS_ENTROPY_MODIFIED = 'cross_entropy_modified'


class EnvDataForIRL():
    DEFAULT_HORIZON = 50
    DEFAULT_INITIAL_STATE_DISTRIBUTION = 'random'
    DEFAULT_N_SEEDS = 200
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

        self.target_align_func_sampler = lambda al_func: al_func
        self.reward_trainer_kwargs = {
            'epochs': 5,
            'lr': 0.08,
            'batch_size': 512,
            'minibatch_size': 32,
        }
        self.horizon = self.__class__.DEFAULT_HORIZON
        self.initial_state_distribution = 'random'
        self.stochastic_expert = True
        self.learn_stochastic_policy = True
        self.environment_is_stochastic = True  # If known to be False,
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
        self.n_expert_samples_per_seed = 1 if self.stochastic_expert is False else self.__class__.DEFAULT_N_EXPERT_SAMPLES_PER_SEED
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

    @abstractmethod
    def align_colors(self, align_func): pass


    def compute_precise_policy(self, env_real: FixedDestRoadWorldGymPOMDP, w, reward):
        pass


class EnvDataForIRLFireFighters(EnvDataForIRL):
    DEFAULT_HORIZON = 50
    DEFAULT_INITIAL_STATE_DISTRIBUTION = 'random'
    DEFAULT_N_SEEDS = 200
    DEFAULT_N_SEEDS_MINIBATCH = 10
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED_MINIBATCH = 10
    DEFAULT_N_REWARD_SAMPLES_PER_ITERATION = 10
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED = 10
    DEFAULT_FEATURE_SELECTION = FeatureSelectionFFEnv.ONE_HOT_FEATURES
    VALUES_NAMES = {(1.0, 0.0): 'Prof', (0.0, 1.0): 'Prox'}

    def align_colors(self, align_func): return get_color_gradient(
        [1, 0, 0], [0, 0, 1], align_func)

    def __init__(self, env_name, discount_factor=1.0, feature_selection=DEFAULT_FEATURE_SELECTION, horizon=DEFAULT_HORIZON, is_society=False, initial_state_dist=DEFAULT_INITIAL_STATE_DISTRIBUTION, learn=False, use_pmovi_expert=USE_PMOVI_EXPERT, n_seeds_for_samplers=DEFAULT_N_SEEDS,
                 sampler_over_precalculated_trajs=False, **kwargs):
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
            policy_per_va_dict=profile_to_matrix if use_pmovi_expert else profile_to_assumed_matrix, env=env_training, state_encoder=None)
        
        self.env = env_real
        
        self.vgl_expert_policy = expert_policy_train
        self.vsi_expert_policy = expert_policy_train

        self.vgl_reference_policy = expert_policy_train
        self.vsi_reference_policy = expert_policy_train

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
            expert_trajs_train = expert_policy_train.obtain_trajectories(n_seeds=n_seeds_for_samplers, seed=self.seed, stochastic=self.stochastic_expert, repeat_per_seed=self.n_expert_samples_per_seed,
                                                                     align_funcs_in_policy=profiles, t_max=self.horizon)

            self.vgl_expert_train_sampler = partial(
                random_sampler_among_trajs, expert_trajs_train)
            self.vsi_expert_train_sampler = partial(
                random_sampler_among_trajs, expert_trajs_train)
        else:
            self.vgl_expert_train_sampler = partial(
                sampler_from_policy, expert_policy_train, stochastic=self.stochastic_expert, horizon=self.horizon)
            self.vsi_expert_train_sampler = partial(
                sampler_from_policy, expert_policy_train, stochastic=self.stochastic_expert, horizon=self.horizon)

    def set_defaults(self):

        self.stochastic_expert = True
        self.learn_stochastic_policy = True
        self.environment_is_stochastic = False

        self.vgl_targets = [(1.0, 0.0), (0.0, 1.0)]
        self.vsi_optimizer_kwargs = {"lr": 0.015, "weight_decay": 0.0000} # FOR DEMO_OM_TRUE 0.05 before
        #self.vsi_optimizer_kwargs = {"lr": 0.01, "weight_decay": 0.0000} # DEMO_OM_FALSE
        self.vgl_optimizer_kwargs = {"lr": 0.1, "weight_decay": 0.000}

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
            'value_iteration_tolerance': 0.0000001, 'iterations': 1000000}
        # self.vgl_reference_policy = 'random'
        # self.vsi_reference_policy = 'random'

        self.reward_trainer_kwargs = {
            'epochs': 5, # 1, 3 TODO: PREV: 5
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
            n_seeds_for_sampled_trajectories=4500, # 2600, 3000, 3500 TODO : PREV: 4500
            n_sampled_trajs_per_seed=2, #10, 2
            fragment_length=self.horizon, interactive_imitation_iterations=200, #total | 200, 150
            total_comparisons=10000, initial_comparison_frac=0.25,  #50000, 20000
            initial_epoch_multiplier=40, transition_oversampling=1 #15,5 | 4,1 TODO: PREV 40.
        ))
        base['vsi'].update(dict(
            max_iter=10000,
            random_trajs_proportion=0.8,
            n_seeds_for_sampled_trajectories=4500, # 2600, 3000, 3500 TODO : PREV: 4500
            n_sampled_trajs_per_seed=2, #10, 2
            fragment_length=self.horizon, interactive_imitation_iterations=200, #total | 200, 150
            total_comparisons=10000, initial_comparison_frac=0.25,  #50000, 20000
            initial_epoch_multiplier=40, transition_oversampling=1 #15,5 | 4,1 TODO: PREV 40.
        ))
        return base

    @property
    def me_config(self):
        base = super().me_config
        base['vsi'].update(dict(
            vc_diff_epsilon=1e-5,
            gradient_norm_epsilon=1e-9,
            use_feature_expectations_for_vsi=False,
            demo_om_from_policy=True
        ))
        return base

    @property
    def me_train_config(self):
        base = super().me_train_config
        base['vgl'].update(dict(max_iter=200,))
        base['vsi'].update(dict(max_iter=200))#200 TODO
        return base

    def get_assumed_grounding(self):
        if self.use_one_hot_state_action:
            assumed_grounding = np.zeros(
                (self.env.n_states*self.env.action_dim, 2), dtype=np.float64)
            assumed_grounding[:, 0] = np.reshape(self.env.reward_matrix_per_align_func(
                (1.0, 0.0)), (self.env.n_states*self.env.action_dim,))
            assumed_grounding[:, 1] = np.reshape(self.env.reward_matrix_per_align_func(
                (0.0, 1.0)), (self.env.n_states*self.env.action_dim,))

            return assumed_grounding
        else:
            raise NotImplementedError(
                "No knwon closed-form grounding with other feature configuration than one-hot encoded state-action pairs")

    def get_reward_net(self, algorithm='me'):
        if not self.use_one_hot_state_action:

            self.use_bias = True
            self.hid_sizes = [50, 100, 50, self.n_values,]
            self.basic_layer_classes = [
                nn.Linear, nn.Linear, nn.Linear, nn.Linear, LinearAlignmentLayer]
            self.activations = [nn.LeakyReLU, nn.LeakyReLU,
                                nn.LeakyReLU, nn.Tanh, nn.Identity]
        else:
            self.use_bias = False
            self.hid_sizes = [self.n_values,]
            self.basic_layer_classes = [nn.Linear, LinearAlignmentLayer]
            self.activations = [nn.Tanh, nn.Identity]

        return super().get_reward_net(algorithm)


class EnvDataForRoadWorld(EnvDataForIRL):
    DEFAULT_HORIZON = 50
    DEFAULT_INITIAL_STATE_DISTRIBUTION = 'random'
    DEFAULT_N_SEEDS = 100
    DEFAULT_N_SEEDS_MINIBATCH = 20
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED_MINIBATCH = 10
    DEFAULT_N_REWARD_SAMPLES_PER_ITERATION = 30
    DEFAULT_N_EXPERT_SAMPLES_PER_SEED = 5
    DEFAULT_DEST = 64 # 413

    VALUES_NAMES = BASIC_PROFILE_NAMES

    def __init__(self, env_name=ROAD_WORLD_ENV_NAME, horizon=DEFAULT_HORIZON, dest=DEFAULT_DEST, n_seeds_for_samplers=DEFAULT_N_SEEDS, sampler_over_precalculated_trajs=False, **kwargs):
        super().__init__(env_name=env_name, **kwargs)
        self.horizon = horizon
        assert self.discount_factor == 1.0

        self.n_seeds_total = n_seeds_for_samplers
        self.dest = dest
        cv = 0  # cross validation process [0, 1, 2, 3, 4] # TODO (?)
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
            #print(env_real.valid_actions(318))
            #print(env_real.valid_actions(321))
            #print(w, reward[318],reward[321])
            assert self.approx_expert == True # TODO NEed more testing
            if not self.approx_expert:
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
                
                
                # TODO: MCE does not work... The approximation is bad, state visitation count cannot do good.
                # Need another method....
                
                
            else:
                _, _, assumed_expert_pi = mce_partition_fh(env_real, discount=self.discount_factor,
                                                       reward=reward,
                                                       horizon = env_real.horizon,
                                                       approximator_kwargs=self.approximator_kwargs,
                                                       policy_approximator=PolicyApproximators.MCE_ORIGINAL,
                                                       deterministic=not self.stochastic_expert)
            profile_to_assumed_matrix[w] = assumed_expert_pi


        
        expert_policy_train = VAlignedDictSpaceActionPolicy(
            policy_per_va_dict=profile_to_assumed_matrix, env=env_real, state_encoder=None)
        

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
                                                                     t_max=self.horizon)
        
        for tr in expert_trajs_train:
            #print(tr, len(tr))
            
            assert tr.obs[-1] == self.dest
        
        if sampler_over_precalculated_trajs:
            expert_trajs_train = expert_policy_train.obtain_trajectories(n_seeds=self.n_seeds_total,
                                                                     seed=self.seed, stochastic=self.stochastic_expert,
                                                                     repeat_per_seed=self.n_expert_samples_per_seed, 
                                                                     align_funcs_in_policy=profiles,
                                                                     alignments_in_env=profiles,
                                                                     t_max=self.horizon)
            for tr in expert_trajs_train:
                assert tr.obs[-1] == self.dest
            self.vgl_expert_train_sampler = partial(
                random_sampler_among_trajs, expert_trajs_train)
            self.vsi_expert_train_sampler = partial(
                random_sampler_among_trajs, expert_trajs_train)
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
            fragment_length=self.horizon, interactive_imitation_iterations=200, # TODO: The fragments are discarded when they are shorter than this length! Before the cropping, it was 30.
            total_comparisons=7000, initial_comparison_frac=0.15, # 0.1 TODO? 0.08 for ended trajs
            initial_epoch_multiplier=100, transition_oversampling=1,
            random_trajs_proportion=0.8
        ))
        base['vsi'].update(dict(
            max_iter=10000,
            n_seeds_for_sampled_trajectories=2500, # 1000 was too few with full variable length trajectories
            n_sampled_trajs_per_seed=1,
            fragment_length=self.horizon, interactive_imitation_iterations=200, # TODO: The fragments are discarded when they are shorter than this length! Before the cropping, it was 30.
            total_comparisons=7000, initial_comparison_frac=0.15, # 0.1 TODO? 0.08 for ended trajs
            initial_epoch_multiplier=100, transition_oversampling=1,
            random_trajs_proportion=0.8
        ))
        return base


    @property
    def me_config(self):
        return super().me_config

    @property
    def me_train_config(self):
        base = super().me_train_config
        base['vgl'].update(dict(
            max_iter=5))
        base['vsi'].update(dict(
            max_iter=5)) # TODO 200 again... maybe 100 is enough seeing the results..

        return base

    def align_colors(self, align_func): return get_linear_combination_of_colors(
        BASIC_PROFILES, PROFILE_COLORS_VEC, align_func)

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
        self.use_bias = False
        self.basic_layer_classes = [ConvexLinearModule, ConvexAlignmentLayer]
        self.activations = [nn.Identity, nn.Identity]
        self.vgl_targets = BASIC_PROFILES
        self.profile_variety = 4 # If 4 it is 0.67, 0.33...
        self.n_values = 3
        self.negative_grounding_layer = True
        
        self.vsi_optimizer_kwargs = {"lr": 0.2, "weight_decay": 0.0000}
        self.vgl_optimizer_kwargs = {"lr": 0.1, "weight_decay": 0.0000}

        self.policy_approximation_method = PolicyApproximators.MCE_ORIGINAL
        self.approximator_kwargs = {
            'value_iteration_tolerance': 0.0000001, 'iterations': 100000}
        # self.vgl_reference_policy = 'random' # SEE __INIT__!
        # self.vsi_reference_policy = 'random' # SEE __INIT__!

        self.testing_profiles = list(itertools.product(
            [0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]))
        # self.testing_profiles.remove([0.0, 0.0,0.0])
        self.testing_profiles.remove((0.0, 0.0, 0.0))
        self.reward_trainer_kwargs = {
            'epochs': 1,
            'lr': 0.03, # 0.03?
            'batch_size': 32, # 128
        }
