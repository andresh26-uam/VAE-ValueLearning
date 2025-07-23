import argparse
from copy import deepcopy
import os
import pprint
import random
from typing import Dict, Union
import gymnasium
import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
import torch as th
import wandb

from generate_dataset import parse_dtype_torch, parse_learner_class, parse_policy_approximator
from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_custom_reward import MOCustomRewardVector
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.multi_policy.pcn.pcn import PCN, Transition

from envs.firefighters_env_mo import FeatureSelectionFFEnv, FireFightersEnvMO
from src.algorithms.preference_based_vsl_simple import PVSL
from src.dataset_processing.data import VSLPreferenceDataset
from src.dataset_processing.utils import DATASETS_PATH, DEFAULT_SEED, calculate_dataset_save_path
from src.feature_extractors import BaseRewardFeatureExtractor
from src.policies.vsl_policies import RewardVectorFunctionWrapper, ValueSystemLearningPolicyCustomLearner
from src.reward_nets.vsl_reward_functions import ConvexAlignmentLayer, RewardVectorModuleWithKnownRewards, VectorModule, RewardVectorModule, TrainingModes, parse_layer_name
from src.utils import filter_none_args, load_json_config
from train_vsl import parse_cluster_sizes, parse_feature_extractors, parse_optimizer_data
from use_cases.roadworld_env_use_case.network_env import FeaturePreprocess, FeatureSelection

        
        
def parse_args():
    # IMPORTANT: Default Args are specified depending on the environment in config.json

    parser = argparse.ArgumentParser(
        description="This script will generate a total of n_agents * trajectory_pairs of trajectories, and a chain of comparisons between them, per agent type, for the society selected. See the societies.json and algorithm_config.json files")

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument('-dname', '--dataset_name', type=str,
                               default='test_dataset', required=True, help='Dataset name')
    general_group.add_argument('-sname', '--society_name', type=str, default='default',
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')

    general_group.add_argument('-ename', '--experiment_name', type=str,
                               default='test_experiment', required=True, help='Experiment name')

    general_group.add_argument('-sp', '--split_ratio', type=float, default=0.0,
                               help='Test split ratio. If 0.0, no split is done. If 1.0, all data is used for testing.')

    general_group.add_argument('-dtype', '--dtype', type=parse_dtype_torch, default=th.float32, choices=[th.float32, th.float64],
                               help='Society name in the society config file (overrides other defaults here, but not the command line arguments)')

    general_group.add_argument(
        '-s', '--seed', type=int, default=DEFAULT_SEED, required=False, help='Random seed')

    general_group.add_argument('-a', '--algorithm', type=str, choices=[
                               'pc'], default='pc', help='Algorithm to use (preference comparison - pc)')

    general_group.add_argument('-cf', '--config_file', type=str, default='algorithm_config.json',
                               help='Path to JSON general configuration file (overrides other defaults here, but not the command line arguments)')
    general_group.add_argument('-sf', '--society_file', type=str, default='societies.json',
                               help='Path to JSON society configuration file (overrides other defaults here, but not the command line arguments)')

    general_group.add_argument('-e', '--environment', type=str, default='ff', choices=[
                               'rw', 'ffmo', 'vrw', 'mvc'], help='environment (roadworld - rw, firefighters - ff, variablerw - vrw, multi-value car - mvc)')

    general_group.add_argument('-df', '--discount_factor', type=float, default=1.0,
                               help='Discount factor. For some environments, it will be neglected as they need a specific discount factor.')

    alg_group = parser.add_argument_group('Algorithm-specific Parameters')

    alg_group.add_argument('-k', '--k_clusters', type=Union[int, list], default=-1,
                           help="Number of clusters per value (overriging configuration file)")

    debug_params = parser.add_argument_group('Debug Parameters')
    debug_params.add_argument('-db', '--debug_mode', action='store_true',
                              default=False, help='Debug mode')

    env_group = parser.add_argument_group('environment-specific Parameters')

    env_group.add_argument('-rte', '--retrain_experts', action='store_true',
                           default=False, help='Retrain experts (roadworld)')
    env_group.add_argument('-appr', '--is_tabular', action='store_true',
                           default=False, help='Approximate expert (roadworld)')
    env_group.add_argument('-reps', '--reward_epsilon', default=0.0, type=float,
                           help='Distance between the cummulative rewards of each pair of trajectories for them to be considered as equal in the comparisons')

    return parser.parse_args()

agent_kwargs_PCNFF = {
        "scaling_factor": np.array([1, 1, 1.0]),
        "learning_rate": 0.002,
        "batch_size": 64,
        "hidden_dim": 64,
        "project_name": "MORL-Baselines",
        "experiment_name": "PCNFF",
        "log": True,
    }
def main():
    def make_env():
        env = mo_gym.make("FireFightersMO-v0", horizon=50)
        env = MORecordEpisodeStatistics(env, gamma=1.0)
        return env

    env = make_env()

    agent = PCN(env=env, **agent_kwargs_PCNFF)

    agent.train(
        eval_env=make_env(),
        total_timesteps=int(10000),
        ref_point=np.array([-100.0, -100.0]),
        num_er_episodes=30,
        max_buffer_size=200,
        num_model_updates=10,
        max_return=np.array([30.0, 50.0]),
        known_pareto_front=None,
    )
    


def main_minecart():
    def make_env():
        env = mo_gym.make("minecart-deterministic-v0")
        env = MORecordEpisodeStatistics(env, gamma=1.0)
        return env

    env = make_env()

    agent = PCN(
        env,
        scaling_factor=np.array([1, 1, 0.1, 0.1]),
        learning_rate=1e-3,
        batch_size=256,
        project_name="MORL-Baselines",
        experiment_name="PCN",
        log=True,
    )

    agent.train(
        eval_env=make_env(),
        total_timesteps=int(1e5),
        ref_point=np.array([0, 0, -200.0]),
        num_er_episodes=20,
        max_buffer_size=50,
        num_model_updates=50,
        max_return=np.array([1.5, 1.5, -0.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
    )

class PCN_CUSTOM_REWARD(PCN, MOCustomRewardVector):
    def __init__(self, env, scaling_factor, learning_rate = 0.001, gamma = 1, batch_size = 256, hidden_dim = 64, noise = 0.1, project_name = "MORL-Baselines", experiment_name = "PCN", wandb_entity = None, log = True, seed = None, device = "auto", model_class = None, relabel_buffer=True):
        super().__init__(env, scaling_factor, learning_rate, gamma, batch_size, hidden_dim, noise, project_name, experiment_name, wandb_entity, log, seed, device, model_class)
        self.relabel_buffer = relabel_buffer

    def train(self, **kwargs):
        self.max_buffer_size = kwargs.get("max_buffer_size", 100)
        super().train(**kwargs)

    def set_reward_vector(self, reward_vector):
        super().set_reward_vector(reward_vector)
        self.reward_vector = reward_vector
        
        if self.env.has_wrapper_attr("set_reward_vector_function"):
            self.env.get_wrapper_attr("set_reward_vector_function")(reward_vector)
        else:
            self.env = RewardVectorFunctionWrapper(self.env, reward_vector)
        if self.relabel_buffer:
            global_step = 0
            old_replay = deepcopy(self.experience_replay)
            self.experience_replay = []
            if len(self.experience_replay) > 0:
                new_experience_replay = []
                for transitions in old_replay:
                    acc_r = 0
                    new_transitions = []
                    for t, transition in enumerate(transitions):
                        reward = self.reward_vector(
                            transition.observation, transition.action, transition.next_observation, transition.terminal
                        ).detach().cpu().numpy()
                        new_transitions.append(Transition(transition.observation, transition.action, np.float32(reward).copy(), transition.next_observation, transition.terminated))
                        global_step += 1
                    # add episode in-place
                    self._add_episode(transitions, max_size=self.max_buffer_size, step=global_step)
            

    
    
def pvsl():
    parser_args = filter_none_args(parse_args())
    # If a config file is specified, load it and override command line args
    config = load_json_config(parser_args.config_file)
    society_config = load_json_config(parser_args.society_file)

    pprint.pprint(parser_args)
    np.random.seed(parser_args.seed)
    th.manual_seed(parser_args.seed)
    random.seed(parser_args.seed)
    rng_for_algorithms = np.random.default_rng(parser_args.seed)

    environment_data = config[parser_args.environment]
    society_data = society_config[parser_args.environment][parser_args.society_name]
    
    dataset_name = parser_args.dataset_name
    experiment_name = parser_args.experiment_name
    experiment_name = experiment_name #+ '_' + str(parser_args.split_ratio)

    agent_profiles = [tuple(ag['value_system'])
                      for ag in society_data['agents']]
    
    extra_kwargs = {}
    if parser_args.environment == 'ffmo':
        extra_kwargs = {
            'feature_selection': FeatureSelectionFFEnv(environment_data['feature_selection']),
            'initial_state_distribution': environment_data['initial_state_distribution'],
            'horizon': environment_data['horizon']
        }
    if parser_args.environment == 'rwmo' or parser_args.environment == 'vrw':
        raise NotImplementedError(
            "Roadworld and Variable Roadworld environments are not implemented yet in this script")
        extra_kwargs = {'env_kwargs': {
            'feature_selection': FeatureSelection(environment_data['feature_selection']),
            'feature_preprocessing': FeaturePreprocess(environment_data['feature_preprocessing']),
            'horizon': environment_data['horizon']
        }}
        if 'Fixed' in environment_data['name']:
            extra_kwargs['with_destination'] = 64
    alg_config = environment_data['algorithm_config'][parser_args.algorithm]

    environment = MORecordEpisodeStatistics(mo_gym.make(
        environment_data['name'], **extra_kwargs), gamma=alg_config['discount_factor_preferences'])
    environment.reset(seed=parser_args.seed)

    reward_net_features_extractor_class, policy_features_extractor_class, features_extractor_kwargs, policy_features_extractor_kwargs = parse_feature_extractors(
        environment, environment_data, dtype=parser_args.dtype)

    data_reward_net = environment_data['default_reward_net']
    data_reward_net.update(alg_config['reward_net'])

    device = th.device("cuda" if th.cuda.is_available() else "cpu") if (parser_args.device == "auto" or parser_args.device == "cuda") else 'cpu'
    features_extractor = reward_net_features_extractor_class(observation_space=environment.observation_space,
                                                    action_space=environment.action_space,
                                                    use_state=data_reward_net['use_state'], 
                                                    use_action=data_reward_net['use_action'], 
                                                    use_next_state=data_reward_net['use_next_state'], 
                                                    use_done=data_reward_net['use_done'],
                                                    features_extractor_class=reward_net_features_extractor_class,
                                                    features_extractor_kwargs=features_extractor_kwargs,
                                                    dtype=parser_args.dtype,
                                                    device=device)
    reward_net = RewardVectorModule(
        hid_sizes=data_reward_net['hid_sizes'],
        basic_layer_classes=[parse_layer_name(
            l) for l in data_reward_net['basic_layer_classes']],
        activations=[parse_layer_name(l)
                     for l in data_reward_net['activations']],
        #negative_grounding_layer=data_reward_net['negative_grounding_layer'],
        use_bias=data_reward_net['use_bias'],
        clamp_rewards=data_reward_net['clamp_rewards'],
        features_extractor=features_extractor,
    )

    opt_kwargs, opt_class = parse_optimizer_data(environment_data, alg_config)

    dataset_train, dataset_test = retrieve_datasets(environment_data, society_data, dataset_name, rew_epsilon=parser_args.reward_epsilon, split_ratio=parser_args.split_ratio)

    learning_policy_kwargs: Dict = alg_config['learning_policy_kwargs'][alg_config['learning_policy_class']]
    learning_policy_class = alg_config['learning_policy_class']
    epclass, epkwargs = parse_policy_approximator(
            ref_class=learning_policy_class,
            all_agent_groundings_to_save_files=None,
            learner_or_expert= 'learner',
            env_name=environment_data['name'], 
            society_data=society_data, environment_data=environment_data,
            ref_policy_kwargs=learning_policy_kwargs, environment=environment)


    """if parser_args.algorithm == 'pc':
        vsl_algo = PreferenceBasedClusteringTabularMDPVSL(
            env=environment,
            reward_net=reward_net,
            optimizer_cls=opt_class,
            optimizer_kwargs=opt_kwargs,
            discount=environment_data['discount'],
            discount_factor_preferences=alg_config['discount_factor_preferences'],
            dataset=dataset_train,
            training_mode=TrainingModes.SIMULTANEOUS,
            cluster_sizes=parse_cluster_sizes(
                environment_data['K'] if parser_args.k_clusters == -1 else parser_args.k_clusters, n_values=environment_data['n_values']),
            vs_cluster_sizes=

            learn_stochastic_policy=alg_config['learn_stochastic_policy'],
            use_quantified_preference=alg_config['use_quantified_preference'],
            preference_sampling_temperature=0 if alg_config[
                'use_quantified_preference'] else alg_config['preference_sampling_temperature'],
            log_interval=1,
            reward_trainer_kwargs=alg_config['reward_trainer_kwargs'],
            query_schedule=alg_config['query_schedule'],
            vgl_target_align_funcs=environment_data['basic_profiles'],
            #approximator_kwargs=alg_config['approximator_kwargs'], TODO: eliminate this
            #policy_approximator=alg_config['policy_approximation_method'],
            rng=rng_for_algorithms,
            # This is only used for testing purposes
            expert_is_stochastic=society_data['stochastic_expert'],
            loss_class=alg_config['loss_class'],
            loss_kwargs=alg_config['loss_kwargs'],
            assume_variable_horizon=environment_data['assume_variable_horizon'],
            
            learning_policy_class=epclass,
            learning_policy_random_config_kwargs=epkwargs,
            learning_policy_kwargs=learning_policy_kwargs,
            #????,
            debug_mode=parser_args.debug_mode

        )"""
    if parser_args.algorithm == 'pc':
        alg_config['train_kwargs']['experiment_name'] = experiment_name
        
    PVSL(Lmax=environment_data['L'] if isinstance(
                environment_data['L'], int) else None, 
                moagent_class=epclass, moagent_kwargs=epkwargs, 
                alignment_layer_class=ConvexAlignmentLayer,
         discount_factor_preferences=alg_config['discount_factor_preferences'],
         grounding_network=reward_net,
         loss_class=alg_config['loss_class'],
         loss_kwargs=alg_config['loss_kwargs'],
         optim_class=opt_class,
         optim_kwargs=opt_kwargs,)
    ret = PVSL.train(
        dataset=dataset_train,
        environment=environment,
        resume_from=0,
        eval_env=environment,
        **alg_config['train_kwargs'],
    )
    print("DONE", ret)

def retrieve_datasets( environment_data, society_data, dataset_name, rew_epsilon=0.0, split_ratio=0.5):
    try:
        path = os.path.join(
        DATASETS_PATH, calculate_dataset_save_path(dataset_name, environment_data, society_data, epsilon=rew_epsilon))
        dataset_train = VSLPreferenceDataset.load(
            os.path.join(path, "dataset_train.pkl"))
        dataset_test = VSLPreferenceDataset.load(
            os.path.join(path, "dataset_test.pkl"))
        print("LOADING DATASET SPLIT.")
    except FileNotFoundError:
        print("LOADING DATASET FULL. THEN DIVIDE")
        path = os.path.join(
            DATASETS_PATH, calculate_dataset_save_path(dataset_name, environment_data, society_data, epsilon=rew_epsilon))

        dataset = VSLPreferenceDataset.load(os.path.join(path, "dataset.pkl"))

        dataset_test = VSLPreferenceDataset(
            n_values=dataset.n_values, single_agent=False)
        dataset_train = VSLPreferenceDataset(
            n_values=dataset.n_values, single_agent=False)
        for aid, adata in dataset.data_per_agent.items():
            selection = np.arange(int(split_ratio * len(adata)))
            train_selection = np.arange(
                int(split_ratio * len(adata)), len(adata))
            agent_dataset_batch = adata[selection]
            dataset_test.push(fragments=agent_dataset_batch[0], preferences=agent_dataset_batch[1], preferences_with_grounding=agent_dataset_batch[2], agent_ids=[
                aid]*len(selection), agent_data={aid: dataset.agent_data[aid]})
            agent_dataset_batch_t = adata[train_selection]
            dataset_train.push(fragments=agent_dataset_batch_t[0], preferences=agent_dataset_batch_t[1], preferences_with_grounding=agent_dataset_batch_t[2], agent_ids=[
                aid]*len(train_selection), agent_data={aid: dataset.agent_data[aid]})
                
    return dataset_train, dataset_test
    

if __name__ == "__main__":
    #main_minecart()
    pvsl()
