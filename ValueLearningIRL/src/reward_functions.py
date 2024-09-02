from abc import abstractmethod
from copy import deepcopy
import enum
import os
from typing import Iterator
import gymnasium as gym
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

def print_tensor_and_grad_fn(grad_fn, level=0):
            indent = "  " * level
            if grad_fn is None:
                return
            if getattr(grad_fn,'variable',None) is not None:
                print(f"{indent}AccumulateGrad for tensor: {grad_fn.variable}")
            else:
                print(f"{indent}Grad function: {grad_fn}")
                if hasattr(grad_fn, 'next_functions'):
                    for next_fn in grad_fn.next_functions:
                        if next_fn[0] is not None:
                            print_tensor_and_grad_fn(next_fn[0], level + 1)

                          
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

class ConvexTensorModule(nn.Module):
    def __init__(self, size, init_tuple=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size = size
        self.reset(init_tuple)

    def get_tensor_as_tuple(self):
        return tuple(self.forward().detach().cpu().numpy())
    def reset(self, init_tuple=None):
        if init_tuple is None:
            new_profile = np.random.rand(self.size)
            new_profile = new_profile/np.sum(new_profile)
            new_profile = tuple(new_profile)
        self.weights = th.tensor(np.array(new_profile, dtype=np.float64), dtype=th.float32, requires_grad=True)

    def forward(self, x=None):
        return nn.functional.softmax(self.weights, dtype=th.float64)
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return iter([self.weights])

class ProfileLayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, data=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.use_bias = bias
        
    def forward(self, input: Tensor) -> Tensor:
        w_bounded, b_bounded = self.get_profile()
        
        if self.use_bias:
            output = nn.functional.linear(input, w_bounded, b_bounded)
        else:
            output = nn.functional.linear(input, w_bounded)
        
        return output
    def get_profile(self):

        w_bounded = nn.functional.softmax(self.weight, dim=1)
        #assert th.allclose(w_bounded, nn.functional.softmax(self.weight))
        b_bounded = 0.0
        if self.use_bias:
            b_bounded = nn.functional.sigmoid(self.bias)
        return w_bounded, b_bounded
class PositiveBoundedLinearModule(ProfileLayer):
    
    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        assert th.all(output < 0.0)
        #print(th.where(input<=0.0))
        #print(input[th.where(input<=0.0)])
        assert th.all(input < 0.0)
        return output
    
    
class TrainingModes(enum.Enum):
    VALUE_SYSTEM_IDENTIFICATION = 'profile_learning'
    VALUE_GROUNDING_LEARNING = 'value_learning'
    SIMULTANEOUS = 'sim_learning'
    EVAL = 'eval'


class AbstractVSLearningRewardFunction(reward_nets.RewardNet):
    def set_mode(self, mode):
        self.mode = mode
    def set_alignment_function(self, align_func):
        if align_func is None:
            return
        if self.cur_align_func is None or self.cur_align_func != align_func:
            self.cur_align_func = align_func
    @abstractmethod
    def reset_learned_alignment_function(self, new_align_func=None):
        ...

    def set_grounding_function(self, grounding):
        if grounding is None:
            return
        
        if self.cur_value_grounding is None or self.cur_value_grounding != grounding:
            self.cur_value_grounding = grounding
        

    def __init__(self, environment: gym.Env, use_state=False, use_action=False, use_next_state=True, use_done=True, use_one_hot_state_action=False,
                 mode = TrainingModes.VALUE_GROUNDING_LEARNING,
                 **kwargs):
        observation_space = environment.observation_space
        action_space = environment.action_space
        super().__init__(observation_space, action_space, True)

        self.mode = mode
        self.use_action = use_action

        self.use_next_state = use_next_state

        self.use_state = use_state
        self.use_done = use_done

        self.use_one_hot_state_action = use_one_hot_state_action
        self.cur_align_func = None
        self.cur_value_grounding = None

        

    @abstractmethod
    def value_system_layer(self, align_func):    
        pass
    @abstractmethod
    def value_grounding_layer(self, custom_layer):    
        pass


    def _forward(self, features, align_func=None, grounding=None):
        
        self.set_mode(self.mode)
        
        x = self.value_grounding_layer(custom_layer=grounding)(features) 
        """print("VG", x.grad_fn)
        print(align_func)
        print(features.requires_grad)
        print(features[0])
        print(features.shape)
        print_tensor_and_grad_fn(x.grad_fn)
        print_tensor_and_grad_fn(features.grad_fn)
        """
        
        x = self.value_system_layer(align_func)(x)
        """print("VSL", x.grad)
        print(align_func)"""
        return x 
    
    def forward(self, state: Tensor, action: Tensor, next_state: Tensor, done: Tensor) -> Tensor:
        inputs = []

        if self.use_one_hot_state_action:
            inputs.append(th.flatten(state, 1)) # Assume state encodes state-action pairs.
        else:
            if self.use_state:
                inputs.append(th.flatten(state, 1))
            if self.use_action:
                inputs.append(th.flatten(action, 1))
            if self.use_next_state:
                inputs.append(th.flatten(next_state, 1))
            if self.use_done:
                inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)
        
        return self._forward(inputs_concat, align_func=self.cur_align_func, grounding=self.cur_value_grounding) # If mode is not eval, cur_align_func and cur_value_grounding is not used. This depends on the implementation. 
    
    @abstractmethod    
    def get_learned_align_function(self):
        pass
    @abstractmethod
    def get_learned_grounding(self):
        pass

    @abstractmethod
    def copy(self):
        ...
class ProfiledRewardFunction(AbstractVSLearningRewardFunction):

    def set_mode(self, mode):
        super().set_mode(mode)
        if self.mode is None or self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            self.values_net.requires_grad_(True)
            self.trained_profile_net.requires_grad_(False)
        elif self.mode == TrainingModes.EVAL:
            self.values_net.requires_grad_(False)
            self.trained_profile_net.requires_grad_(False)
        elif self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            self.values_net.requires_grad_(False)
            self.trained_profile_net.requires_grad_(True)
        elif self.mode == TrainingModes.SIMULTANEOUS:
            self.values_net.requires_grad_(True)
            self.trained_profile_net.requires_grad_(True)
        
        else:
            raise ValueError("Unkown training mode" + str(self.mode))
    def set_alignment_function(self, align_func):
        if align_func is None:
            return
        if self.cur_align_func is None or self.cur_align_func != align_func:
            self.cur_align_func = align_func
        if self.cur_align_func is not None:
            assert self.hid_sizes[-1] == len(self.cur_align_func)
    
    def reset_learned_alignment_function(self, new_align_func: tuple = None):
        
        self.reset_learning_profile(new_profile=new_align_func)
        
        
    def value_grounding_layer(self, custom_layer = None):
        if custom_layer is None:
            if self.negative_grounding_layer:
                return lambda x: -self.values_net(x)
            else:
                return lambda x: self.values_net(x)
        else:
            def _call(x):
                assert custom_layer.shape == (self.input_size, self.hid_sizes[-1])
                pt = th.tensor(custom_layer, requires_grad=False, dtype=th.float32)#.reshape((1, len(profile)))
                if self.negative_grounding_layer:
                    x = -F.linear(x, pt.T)
                else:
                    x = F.linear(x, pt.T)
                return x
            return _call
    
    def value_system_layer(self, align_func=None):
        if self.mode == TrainingModes.EVAL or self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            if align_func is None:
                align_func = self.cur_align_func
            def _call(x):
                pt = th.tensor(align_func, requires_grad=False, dtype=th.float32)#.reshape((1, len(profile)))
                x = F.linear(x, pt)
                x = self.final_activation(x) + self.reward_bias
                return x
        elif self.mode == TrainingModes.SIMULTANEOUS or self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            def _call(x):
                x = self.trained_profile_net(x)
                x = self.final_activation(x) + self.reward_bias
                return x
        return _call
    
    def reset_learning_profile(self, new_profile: tuple = None, bias=False):
        self.trained_profile_net = self.basic_layer_classes[-1](self.hid_sizes[-1],1, bias=bias)
        with th.no_grad():
            if new_profile is not None and isinstance(self.trained_profile_net, PositiveBoundedLinearModule):
                assert isinstance(new_profile, tuple)
                state_dict = self.trained_profile_net.state_dict()
                state_dict['weight'] = th.log(th.as_tensor([list(new_profile)]).clone())
                
                self.trained_profile_net.load_state_dict(state_dict)

        
        

    def reset_value_net(self):
        modules = []
        for i in range(1, len(self.activations)):
            linmod = self.basic_layer_classes[i-1](self.hid_sizes[i-1], self.hid_sizes[i], bias=False)
            
            if isinstance(linmod, ConvexLinearModule):
                state_dict = linmod.state_dict()
                state_dict['weight'] = th.rand(*(state_dict['weight'].shape), requires_grad=True)
                    
                state_dict['weight'] = state_dict['weight']/th.sum(state_dict['weight'])
                linmod.load_state_dict(state_dict)
                assert th.all(linmod.state_dict()['weight'] > 0)
            modules.append(linmod)
            modules.append(self.activations[i-1]())
        #modules.append(nn.BatchNorm1d()) ? 
        
        self.values_net = nn.Sequential(*modules,        )

    def __init__(self, environment: gym.Env, use_state=False, use_action=False, use_next_state=True, use_done=True, use_one_hot_state_action=False,
                 activations = [nn.Sigmoid, nn.Tanh],
                 hid_sizes = [3,],
                 basic_layer_classes = [ConvexLinearModule, PositiveBoundedLinearModule],
                 mode = TrainingModes.VALUE_GROUNDING_LEARNING,
                 reward_bias = 0,
                 negative_grounding_layer=False,
                 **kwargs):
        
        if isinstance(environment, RoadWorldGym):
            self.cur_align_func = environment.last_profile
            
        observation_space = environment.observation_space
        action_space = environment.action_space

        self.negative_grounding_layer = negative_grounding_layer

        super().__init__(environment=environment, 
                         use_state=use_state, 
                         use_action=use_action, 
                         use_next_state=use_next_state,
                         use_done=use_done,
                         use_one_hot_state_action=use_one_hot_state_action,mode=mode,)
        combined_size = 0

        if self.use_one_hot_state_action:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space) * preprocessing.get_flattened_obs_dim(action_space)
        else:
            if self.use_state:
                combined_size += preprocessing.get_flattened_obs_dim(observation_space)
            if self.use_action:
                combined_size += preprocessing.get_flattened_obs_dim(action_space)
            if self.use_next_state:
                combined_size += preprocessing.get_flattened_obs_dim(observation_space)
            if self.use_done:
                combined_size += 1

        self.input_size = combined_size
    
        self.reward_bias = reward_bias
        #self.mlp = networks.build_mlp(**full_build_mlp_kwargs)
            
        self.hid_sizes = deepcopy(hid_sizes)
        self.hid_sizes.insert(0, self.input_size)
        self.final_activation = activations[-1]()
        self.activations = activations
        self.basic_layer_classes = basic_layer_classes
        
        self.reset_value_net()

        self.reset_learned_alignment_function()
        
        #th.tensor((0,0,0), requires_grad=True,device=self.parameters()[0].device)
        
    def copy(self):
        return deepcopy(self)
    
    def parameters(self, recurse: bool = True) -> Iterator:
        if self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            return self.values_net.parameters(recurse)
        elif self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            return self.trained_profile_net.parameters(recurse)
        else:
            return chain(self.trained_profile_net.parameters(recurse), self.values_net.parameters(recurse))
    
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
        
    def get_learned_align_function(self):
        return self.get_learned_profile()
    def get_learned_grounding(self):
        return self.value_matrix()    
    
    def save_checkpoint(self, file="reward_function_checkpoint.pt"):
        th.save(self, os.path.join(CHECKPOINTS, "reward_function_checkpoint.pt"))
        #th.save(self.values_net, os.path.join(CHECKPOINTS, "reward_function_checkpoint.pt"))

    def from_checkpoint(file="reward_function_checkpoint.pt"):
        return th.load(os.path.join(CHECKPOINTS, "reward_function_checkpoint.pt"))


class DistributionProfiledRewardFunction(ProfiledRewardFunction):
            
    def __init__(self, environment: gym.Env, use_state=False, use_action=False, use_next_state=True, use_done=True, use_one_hot_state_action=False, activations=[nn.Sigmoid,], hid_sizes=[3], basic_layer_classes=[ConvexLinearModule, PositiveBoundedLinearModule], mode=TrainingModes.VALUE_GROUNDING_LEARNING, reward_bias=0, negative_grounding_layer=False, **kwargs):
        super().__init__(environment, use_state, use_action, use_next_state, use_done, use_one_hot_state_action, activations, hid_sizes, basic_layer_classes, mode, reward_bias, negative_grounding_layer, **kwargs)
        self.fixed_align_func_index = None
    def reset_learning_profile(self, new_profile: tuple = None):
        self.trained_profile_net=ConvexTensorModule(size=self.hid_sizes[-1], init_tuple=new_profile)
        self.cur_align_func = self.trained_profile_net.get_tensor_as_tuple()
    
    def fix_alignment_function(self, align_func=None):
        if align_func is None:
            align_func = self.cur_align_func
            self.fixed_align_func_index = np.random.choice(len(align_func), p=align_func)
        else:
            self.fixed_align_func_index = np.argmax(align_func)
    def free_alignment_function(self):
        self.fixed_align_func_index= None

    def get_learned_profile(self, with_bias=False):
        return self.trained_profile_net.get_tensor_as_tuple()
    
    def parameters(self, recurse: bool = True) -> Iterator:
        if self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            return self.values_net.parameters(recurse)
        elif self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            return self.trained_profile_net.parameters(recurse)
        else:
            return chain(self.trained_profile_net.parameters(recurse), self.values_net.parameters(recurse))

    def value_system_layer(self, align_func=None):
        if self.mode == TrainingModes.EVAL or self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            if align_func is None:
                align_func = self.cur_align_func
            def _call(x):
                if self.fixed_align_func_index is not None:
                    selected_index = self.fixed_align_func_index
                else:
                    selected_index = np.random.choice(a=x.size()[-1], p=align_func)
                if len(x.size()) == 2:
                    x = x[:, selected_index]
                    return x
                elif len(x.size()) == 3:
                    x = x[:,:,selected_index]
                x = self.final_activation(x) + self.reward_bias
                return x

        elif self.mode == TrainingModes.SIMULTANEOUS or self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            def _call(x):
                if self.fixed_align_func_index is not None:
                    selected_index = self.fixed_align_func_index
                else:
                    selected_index = np.random.choice(a=x.size()[-1], p=self.trained_profile_net())
                if len(x.size()) == 2:
                        x = x[:, selected_index]
                        return x
                elif len(x.size()) == 3:
                    x = x[:,:,selected_index]
                else:
                    print("X IS SIZE IS WEIRD", x.size())
                    exit(-1)
                x = self.final_activation(x) + self.reward_bias
                return x
        return _call    
    
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
                    reward_net.set_alignment_function(profile)
                    data = th.tensor(
        [[VALUE_COSTS_PRE_COMPUTED_FEATURES[v](*env.get_separate_features(state_des[0], state_des[1], normalization)) for v in VALUE_COSTS_PRE_COMPUTED_FEATURES.keys()],],
            dtype=reward_net.dtype)
                    result = squeeze_r(reward_net.forward(None, None, data, th.tensor([state_des[0] == state_des[1], ])))
                    reward_net.set_mode(prev_mode)
                    return -result
                return call

    return apply