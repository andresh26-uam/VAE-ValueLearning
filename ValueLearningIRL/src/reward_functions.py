from copy import deepcopy
import enum
import os
from typing import Iterator
from gymnasium import Space
from imitation.rewards import reward_nets
from torch import Tensor
import torch as th
import torch.nn.functional as F
from torch import nn
from src.network_env import RoadWorldGym

from stable_baselines3.common import preprocessing
from itertools import chain

from utils import CHECKPOINTS

class ConvexLinearModule(nn.Linear):
    
    def forward(self, input: Tensor) -> Tensor:
        w_normalized = nn.functional.softmax(self.weight)
        #print(w_normalized)
        output = nn.functional.linear(input, w_normalized)
        assert th.all(output >=0)
        #print(th.where(input<=0.0))
        #print(input[th.where(input<=0.0)])
        assert th.all(input > 0.0)

        return output

class PositiveBoundedLinearModule(nn.Linear):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.use_bias = bias
    def forward(self, input: Tensor) -> Tensor:
        w_bounded = nn.functional.sigmoid(self.weight)
        if self.use_bias:
            b_bounded = nn.functional.sigmoid(self.bias)
            output = nn.functional.linear(input, w_bounded, b_bounded)
        else:
            output = nn.functional.linear(input, w_bounded)
        assert th.all(output < 0.0)
        #print(th.where(input<=0.0))
        #print(input[th.where(input<=0.0)])
        assert th.all(input < 0.0)
        return output
    def get_profile(self):

        w_bounded = nn.functional.sigmoid(self.weight)
        b_bounded = 0.0
        if self.use_bias:
            b_bounded = nn.functional.sigmoid(self.bias)
        return w_bounded, b_bounded
    
class TrainingModes(enum.Enum):
    PROFILE_LEARNING = 'profile_learning'
    VALUE_LEARNING = 'value_learning'
    SIMULTANEOUS = 'sim_learning'

class ProfiledRewardFunction(reward_nets.RewardNet):

    def set_mode(self, mode):
        self.mode = mode
        
    
    def __init__(self, environment: RoadWorldGym, use_state=False, use_action=False, use_next_state=True, use_done=True,
                 activations = [nn.Sigmoid, nn.Tanh],
                 hid_sizes = [3,],
                 mode = TrainingModes.VALUE_LEARNING,
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

        
       
        #self.mlp = networks.build_mlp(**full_build_mlp_kwargs)
            
        modules = [
        ]
        self.hid_sizes = deepcopy(hid_sizes)
        self.hid_sizes.insert(0, combined_size)
        self.final_activation = activations[-1]()
        for i in range(1, len(activations)):
            linmod = ConvexLinearModule(self.hid_sizes[i-1], self.hid_sizes[i], bias=False)
            
            if activations[i-1] == nn.ReLU or nn.Identity:
                state_dict = linmod.state_dict()
                with th.no_grad():
                    state_dict['weight'] = th.rand(*(state_dict['weight'].shape), requires_grad=True)
                    
                    state_dict['weight'] = state_dict['weight']/th.sum(state_dict['weight'])
                linmod.load_state_dict(state_dict)
                assert th.all(linmod.state_dict()['weight'] > 0)
            modules.append(linmod)
            modules.append(activations[i-1]())
        #modules.append(nn.BatchNorm1d()) ? 
        
        self.values_net = nn.Sequential(*modules,        )
        self.mode = mode
        self.trained_profile_net = PositiveBoundedLinearModule(hid_sizes[-1],1, bias=False) # ?? 
        
        #th.tensor((0,0,0), requires_grad=True,device=self.parameters()[0].device)
        
        assert hid_sizes[-1] == 3
    def parameters(self, recurse: bool = True) -> Iterator:
        if self.mode == TrainingModes.VALUE_LEARNING:
            return self.values_net.parameters(recurse)
        elif self.mode == TrainingModes.PROFILE_LEARNING:
            return self.trained_profile_net.parameters(recurse)
        else:
            return chain(self.trained_profile_net.parameters(recurse), self.values_net.parameters(recurse))
    
    def _forward(self, features, profile=[1,0,0]):
        
        """print(x, x.size())"""
        
        """print("VALUES", x, x.size())
        print(self.values_net.parameters())"""

        if self.mode is None or self.mode == TrainingModes.VALUE_LEARNING:

            x = -self.values_net(features) 
            #x = F.normalize(x, p=float('inf'), dim=0) 
            #assert th.all(th.sum(x, dim=1))
            #assert th.all(x < 0) #and th.all(th.sum(x, dim=1)==1)
            pt = th.tensor(profile, requires_grad=False, dtype=th.float32)#.reshape((1, len(profile)))
            x = F.linear(x, pt)
            x = self.final_activation(x) + self.reward_bias
            """print(x, x.size())
            exit(0)"""
            return x
        elif self.mode == TrainingModes.PROFILE_LEARNING:
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
            x = -self.values_net(features)  
                #x = F.normalize(x, p=float('inf'), dim=0) 
            x = self.trained_profile_net(x)
            x = self.final_activation(x) + self.reward_bias
            return x
        else: 
            print("WTF?")
            print(self.mode)
        
    def set_profile(self, profile: tuple):
        if self.cur_profile != profile:
            self.cur_profile = profile
    
    def value_matrix(self):
        with th.no_grad():
            params = th.tensor(list(p.data for name, p in self.values_net.named_parameters(recurse=True))[-1])
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
