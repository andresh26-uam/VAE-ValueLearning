from copy import deepcopy
import enum
import os
from typing import Iterator
from imitation.rewards import reward_nets
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
import torch as th
import torch.nn.functional as F
from torch import nn
from src.network_env import RoadWorldGym

from stable_baselines3.common import preprocessing
from itertools import chain

from src.values_and_costs import BASIC_PROFILES, VALUE_COSTS_PRE_COMPUTED_FEATURES
from utils import CHECKPOINTS

class ConvexLinearModule(nn.Linear):
    
    def forward(self, input: Tensor) -> Tensor:
        w_normalized = nn.functional.softmax(self.weight, dim=1)
        #assert th.allclose(w_normalized, nn.functional.softmax(self.weight))
        #print(w_normalized)
        output = nn.functional.linear(input, w_normalized)
        assert th.all(output >=0)
        #print(th.where(input<=0.0))
        #print(input[th.where(input<=0.0)])
        assert th.all(input > 0.0)

        return output

class PositiveBoundedLinearModule(nn.Linear):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, data=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.use_bias = bias
        with th.no_grad():
            if data is not None:
                state_dict = self.state_dict()
                state_dict['weight'] = th.log(th.as_tensor([list(data)]).clone())
                
                self.load_state_dict(state_dict)

    def forward(self, input: Tensor) -> Tensor:
        w_bounded, b_bounded = self.get_profile()
        
        if self.use_bias:
            output = nn.functional.linear(input, w_bounded, b_bounded)
        else:
            output = nn.functional.linear(input, w_bounded)
        assert th.all(output < 0.0)
        #print(th.where(input<=0.0))
        #print(input[th.where(input<=0.0)])
        assert th.all(input < 0.0)
        return output
    def get_profile(self):

        w_bounded = nn.functional.softmax(self.weight, dim=1)
        #assert th.allclose(w_bounded, nn.functional.softmax(self.weight))
        b_bounded = 0.0
        if self.use_bias:
            b_bounded = nn.functional.sigmoid(self.bias)
        return w_bounded, b_bounded
    
class TrainingModes(enum.Enum):
    VALUE_SYSTEM_IDENTIFICATION = 'profile_learning'
    VALUE_GROUNDING_LEARNING = 'value_learning'
    SIMULTANEOUS = 'sim_learning'

class ProfiledRewardFunction(reward_nets.RewardNet):

    def set_mode(self, mode):
        self.mode = mode
        
    def reset_learning_profile(self, new_profile: tuple = None, bias=False):
        self.trained_profile_net = PositiveBoundedLinearModule(self.hid_sizes[-1],1, bias=bias, data=new_profile)
        self.use_bias = bias
        

    def reset_value_net(self):
        modules = []
        for i in range(1, len(self.activations)):
            linmod = ConvexLinearModule(self.hid_sizes[i-1], self.hid_sizes[i], bias=False)
            
            if self.activations[i-1] == nn.ReLU or self.activations[i-1] == nn.Identity:
                state_dict = linmod.state_dict()
                state_dict['weight'] = th.rand(*(state_dict['weight'].shape), requires_grad=True)
                    
                state_dict['weight'] = state_dict['weight']/th.sum(state_dict['weight'])
                linmod.load_state_dict(state_dict)
                assert th.all(linmod.state_dict()['weight'] > 0)
            modules.append(linmod)
            modules.append(self.activations[i-1]())
        #modules.append(nn.BatchNorm1d()) ? 
        
        self.values_net = nn.Sequential(*modules,        )

    def __init__(self, environment: RoadWorldGym, use_state=False, use_action=False, use_next_state=True, use_done=True,
                 activations = [nn.Sigmoid, nn.Tanh],
                 hid_sizes = [3,],
                 mode = TrainingModes.VALUE_GROUNDING_LEARNING,
                 reward_bias = 0,
                 **kwargs):
        observation_space = environment.observation_space
        action_space = environment.action_space
        self.cur_profile = environment.last_profile

        super().__init__(observation_space, action_space, True)
        combined_size = 0
        self.reward_bias = reward_bias
        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        self.input_size = combined_size
       
        #self.mlp = networks.build_mlp(**full_build_mlp_kwargs)
            
        self.hid_sizes = deepcopy(hid_sizes)
        self.hid_sizes.insert(0, self.input_size)
        self.final_activation = activations[-1]()
        self.activations = activations
        
        self.reset_value_net()

        self.mode = mode
        self.reset_learning_profile()
        
        #th.tensor((0,0,0), requires_grad=True,device=self.parameters()[0].device)
        
        assert hid_sizes[-1] == 3
    def parameters(self, recurse: bool = True) -> Iterator:
        if self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            return self.values_net.parameters(recurse)
        elif self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            return self.trained_profile_net.parameters(recurse)
        else:
            return chain(self.trained_profile_net.parameters(recurse), self.values_net.parameters(recurse))
    
    def _forward(self, features, profile=(1.0,0.0,0.0)):
        
        """print(x, x.size())"""
        
        """print("VALUES", x, x.size())
        print(self.values_net.parameters())"""

        if self.mode is None or self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            self.values_net.requires_grad_(True)
            self.trained_profile_net.requires_grad_(False)
            x = -self.values_net(features) 
            pt = th.tensor(profile, requires_grad=False, dtype=th.float32)#.reshape((1, len(profile)))
            x = F.linear(x, pt)
            x = self.final_activation(x) + self.reward_bias
            
            return x
        elif self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            self.values_net.requires_grad_(False)
            self.trained_profile_net.requires_grad_(True)
            x = -self.values_net(features)
            
            """with th.no_grad():
                x = -self.values_net(features)"""  
                #x = F.normalize(x, p=float('inf'), dim=0) 

            x = self.trained_profile_net(x)
            x = self.final_activation(x) + self.reward_bias
            return x
        elif self.mode == TrainingModes.SIMULTANEOUS:
            self.values_net.requires_grad_(True)
            self.trained_profile_net.requires_grad_(True)
            x = -self.values_net(features)  
                #x = F.normalize(x, p=float('inf'), dim=0) 
            x = self.trained_profile_net(x)
            x = self.final_activation(x) + self.reward_bias
            return x
        else:
            raise ValueError("Unkown training mode" + str(self.mode))
        
    def set_profile(self, profile: tuple):
        if self.cur_profile != profile:
            self.cur_profile = profile
    
    def value_matrix(self):
        with th.no_grad():
            params = th.tensor(list(p.data for name, p in self.values_net.named_parameters(recurse=True))[-1])
            assert params.dim() == 2
            return nn.functional.softmax(params)

    def get_learned_profile(self, with_bias=False):
        prof_w = tuple(list(self.trained_profile_net.get_profile()[0].squeeze().tolist()))
        prof_b = float(self.trained_profile_net.get_profile()[1])

        assert isinstance(prof_w, tuple)
        assert isinstance(prof_w[0], float) or isinstance(prof_w[0], int) 
        if with_bias:
            return prof_w, prof_b
        else:
            return prof_w
    def forward(self, state: Tensor, action: Tensor, next_state: Tensor, done: Tensor) -> Tensor:
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)
        return self._forward(inputs_concat, profile=self.cur_profile)
    
    def save_checkpoint(self, file="reward_function_checkpoint.pt"):
        th.save(self, os.path.join(CHECKPOINTS, "reward_function_checkpoint.pt"))
        #th.save(self.values_net, os.path.join(CHECKPOINTS, "reward_function_checkpoint.pt"))

    def from_checkpoint(file="reward_function_checkpoint.pt"):
        
        return th.load(os.path.join(CHECKPOINTS, "reward_function_checkpoint.pt"))




def plot_avg_value_matrix(avg_matrix, std_matrix, file='value_matrix_demo.pdf'):
    X, Y = np.meshgrid(np.arange(avg_matrix.shape[1]), np.arange(avg_matrix.shape[0]))

    # Plotting the average matrix as bars
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(avg_matrix.shape[0]):
        for j in range(avg_matrix.shape[1]):
            ax.bar3d(j, i, 0, 1, 1, avg_matrix[i, j], color=plt.cm.viridis(avg_matrix[i, j] / avg_matrix.max()),  alpha=0.5)

    # Overlaying the standard deviation as black segments
    for i in range(avg_matrix.shape[0]):
        for j in range(avg_matrix.shape[1]):
            ax.plot3D([j+0.5, j+0.5], [i+0.5, i+0.5], [avg_matrix[i, j] - std_matrix[i, j], avg_matrix[i, j] + std_matrix[i, j]], color='k')

    ax.set_xlabel('Input real costs')
    ax.set_ylabel('Learned linear combination')
    ax.set_zlabel('Coefficients')
    ax.set_title('Learned correlation between real and learned costs')
    ax.set_xticks(np.arange(avg_matrix.shape[1])+0.5)
    ax.set_yticks(np.arange(avg_matrix.shape[0])+0.5)
    BASIC_VALUES = ['sus', 'sec', 'eff']
    ax.set_xticklabels(BASIC_VALUES)
    ax.set_yticklabels(['sus\'', 'sec\'', 'eff\''])
    
    fig.savefig(file)
    plt.show()
    plt.close()

def squeeze_r(r_output: th.Tensor) -> th.Tensor:
    """Squeeze a reward output tensor down to one dimension, if necessary.

    Args:
         r_output (th.Tensor): output of reward model. Can be either 1D
            ([n_states]) or 2D ([n_states, 1]).

    Returns:
         squeezed reward of shape [n_states].
    """
    if r_output.ndim == 2:
        return th.squeeze(r_output, 1)
    assert r_output.ndim == 1
    return r_output
def cost_model_from_reward_net(reward_net: ProfiledRewardFunction, env: RoadWorldGym, precalculated_rewards_per_pure_pr=None):
                
    def apply(profile, normalization):
        if np.allclose(list(reward_net.get_learned_profile()), list(profile)):
            def call(state_des):
                data = th.tensor(
        [[VALUE_COSTS_PRE_COMPUTED_FEATURES[v](*env.get_separate_features(state_des[0], state_des[1], normalization)) for v in VALUE_COSTS_PRE_COMPUTED_FEATURES.keys()],],
            dtype=reward_net.dtype)
                
                return -squeeze_r(reward_net.forward(None, None, data, th.tensor([state_des[0] == state_des[1], ])))
            return call
        else:
            if precalculated_rewards_per_pure_pr is not None:
                
                def call(state_des):
                    if profile in BASIC_PROFILES:
                        return -precalculated_rewards_per_pure_pr[profile][state_des[0],state_des[1]]
                    
                    final_cost = 0.0
                    for i, pr in enumerate(BASIC_PROFILES):
                        final_cost += profile[i]*precalculated_rewards_per_pure_pr[pr][state_des[0], state_des[1]]
                    return -final_cost
                return call
            else:
                def call(state_des):
                    prev_mode = reward_net.mode
                    reward_net.set_mode(TrainingModes.VALUE_GROUNDING_LEARNING)
                    reward_net.set_profile(profile)
                    data = th.tensor(
        [[VALUE_COSTS_PRE_COMPUTED_FEATURES[v](*env.get_separate_features(state_des[0], state_des[1], normalization)) for v in VALUE_COSTS_PRE_COMPUTED_FEATURES.keys()],],
            dtype=reward_net.dtype)
                    result = squeeze_r(reward_net.forward(None, None, data, th.tensor([state_des[0] == state_des[1], ])))
                    reward_net.set_mode(prev_mode)
                    return -result
                return call

    return apply