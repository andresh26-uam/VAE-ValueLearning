from functools import partial
import time

from seals import base_envs
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from deep_maxent_irl import SAVED_REWARD_NET_FILE
from src.mce_irl_for_road_network import (
    TrainingModes,
    TrainingSetModes,
    MCEIRL_RoadNetwork
)
import torch

from src.policies import SimplePolicy, ValueIterationPolicy, check_policy_gives_optimal_paths
from src.network_env import DATA_FOLDER, FeaturePreprocess, FeatureSelection, RoadWorldPOMDPStateAsTuple
from src.reward_functions import PositiveBoundedLinearModule, ProfiledRewardFunction

from src.values_and_costs import BASIC_PROFILES
from src.utils.load_data import ini_od_dist
from utils import split_od_train_test



# CUSTOM
PROFILE_VARIETY_TRAIN = 9

log_interval = 10  # interval between training status logs

seed = 260 # random seed for parameter initialization
rng = np.random.default_rng(260)

size = 100  # size of training data [100, 1000, 10000]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""seeding"""

np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

"""environment"""
edge_p = f"{DATA_FOLDER}/edge.txt"
network_p = f"{DATA_FOLDER}/transit.npy"
path_feature_p = f"{DATA_FOLDER}/feature_od.npy"
train_p = f"{DATA_FOLDER}/cross_validation/train_CV%d_size%d.csv" % (0, size)
#test_p = f"{DATA_FOLDER}/cross_validation/test_CV%d.csv" % 0 # Test set is taken as part of the train set directly (of course then that part is not used in training phase)
node_p = f"{DATA_FOLDER}/node.txt"

if __name__ == "__main__":  

    """inialize road environment"""

    PROFILE = (1,0,0)

    od_list, od_dist = ini_od_dist(train_p)
    #env = RoadWorldGymObservationState(network_p, edge_p, node_path=node_p,path_feature_path=path_feature_p, pre_reset=(od_list, od_dist), profile=PROFILE, visualize_example=True)
    env = RoadWorldPOMDPStateAsTuple(network_p, edge_p, node_p, path_feature_p, pre_reset=(od_list, od_dist), profile=PROFILE, visualize_example=True,horizon=100,
                                       one_hot_features=FeatureSelection.NO_ONE_HOT,
                                       use_optimal_reward_per_profile=False,
                                       normalization_or_standarization_or_none=None) # TODO : git push y eso.
    
    start_0, end_0 = [int(a) for a in od_list[0].split('_')]
    print(start_0, end_0)
    policy = ValueIterationPolicy(env)

    feats = env.process_features(torch.tensor([env.get_state_des_transition((107,413),0),]))[0,-3:]
    print(feats)

    def reward_as_negative_costs(s,a,d):
        next_state = env.get_state_des_transition((s,d),a)
        rew = -np.asarray(list(PROFILE)).dot(env.process_features(torch.tensor([next_state,]), feature_selection=FeatureSelection.NO_ONE_HOT, feature_preprocessing=None)[0,-3:].detach().numpy())
        
        return rew
    
    FACTOR = -1
    
    def reward_as_NORMALIZED_negative_costs(s,a,d):
        global FACTOR
        next_state = env.get_state_des_transition((s,d),a)
        rew = -np.asarray(list(PROFILE)).dot(env.process_features(torch.tensor([next_state,]), feature_selection=FeatureSelection.NO_ONE_HOT, feature_preprocessing=FeaturePreprocess.NORMALIZATION)[0,-3:].detach().numpy())
        
        cost = env.cost_model(PROFILE)(next_state)
        if FACTOR == -1:
            FACTOR = cost / rew
            
        assert torch.allclose(cost, rew*FACTOR)
        return rew
    
    expert_paths_per_od_profile = dict()
    for od in env.od_list_int:
        expert_paths_per_od_profile[(od, PROFILE)] = env.shortest_path_edges(profile=PROFILE, to_state=od[1], from_state=od[0], with_length=False, all_alternatives=True)
        print(od, len(expert_paths_per_od_profile[(od, PROFILE)]))
    

    cost_matrix = np.ones((env.state_dim,env.state_dim))
    epolicy: SimplePolicy = SimplePolicy.from_environment_expert(env,profiles=[PROFILE,], custom_cost=cost_matrix)
    trajs = epolicy.sample_trajectories(stochastic=False, repeat_per_od=1, profile=PROFILE)
    print(trajs)
    check_policy_gives_optimal_paths(env, epolicy, profiles=[PROFILE,])

    od_list_int = [od.split('_') for od in od_list]


    """sessions_eco = [policy.generate_session(t_max=50, od=(int(od_try[0]), int(od_try[1])), profile=PROFILE, with_probs=True, formatted_to_file=False, stochastic=False) for od_try in od_list_int]
    # session = states, rewards, actions, traj_probs.


    df_expert = pd.DataFrame([{'states': session[0],'rewards': session[1],'actions': session[2],'traj_probs': session[3], 'ecopref': PROFILE[0], 'secpref': PROFILE[1], 'effpref': PROFILE[2]} for session in sessions_eco])
    
    df_expert.to_csv('expert_session_vi.csv', index=False)"""
    #from imitation.policies.base import FeedForward32policy
   

    