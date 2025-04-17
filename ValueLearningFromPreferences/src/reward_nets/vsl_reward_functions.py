from abc import abstractmethod
from copy import deepcopy
import enum
import os
import time
from typing import Callable, Iterator, Self, Tuple, override
import gymnasium as gym
from imitation.rewards import reward_nets
from matplotlib import pyplot as plt
import numpy as np
import torch as th
from imitation.util import util
from typing import cast

from stable_baselines3.common import preprocessing
from itertools import chain

from envs.tabularVAenv import ValueAlignedEnvironment
from src.feature_extractors import ContextualFeatureExtractorFromVAEnv
from defines import CHECKPOINTS

class TrainingModes(enum.Enum):
    VALUE_SYSTEM_IDENTIFICATION = 'profile_learning'
    VALUE_GROUNDING_LEARNING = 'value_learning'
    SIMULTANEOUS = 'sim_learning'
    EVAL = 'eval'
class ConvexLinearModule(th.nn.Linear):

    def __init__(self, in_features, out_features, bias = False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        state_dict = self.state_dict()
        state_dict['weight'] = th.rand(
            *(state_dict['weight'].shape), requires_grad=True)*10

        state_dict['weight'] = state_dict['weight'] / \
            th.sum(state_dict['weight'])
        
        self.load_state_dict(state_dict)
        assert th.all(self.state_dict()['weight'] > 0)

    def forward(self, input: th.Tensor) -> th.Tensor:
        w_normalized = th.nn.functional.softmax(self.weight, dim=1)
        output = th.nn.functional.linear(input, w_normalized)
        #assert th.all(output >= 0)

        # assert th.all(input > 0.0)

        return output


class ConvexTensorModule(th.nn.Module):
    def __init__(self, size, init_tuple=None, dtype=th.float32, bias=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size = size
        self.dtype= dtype
        self.reset(init_tuple)

    def get_tensor_as_tuple(self):
        return tuple(self.forward().detach().cpu().numpy())

    def reset(self, init_tuple=None):
        new_profile = init_tuple
        if init_tuple is None:
            new_profile = np.random.rand(self.size)
            new_profile = new_profile/np.sum(new_profile)
            new_profile = tuple(new_profile)
        self.weights = th.tensor(
            np.array(new_profile, dtype=np.float64), dtype=self.dtype, requires_grad=True)

    def forward(self, x=None):
        return th.nn.functional.softmax(self.weights, dtype=self.dtype, dim=0)

    def parameters(self, recurse: bool = True) -> Iterator[th.nn.Parameter]:
        return iter([self.weights])


class LinearAlignmentLayer(th.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None, data=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.linear_bias = bias
        with th.no_grad():
            state_dict = self.state_dict()
            random_vector = th.rand_like(state_dict['weight'])
            state_dict['weight'] = th.nn.functional.sigmoid(
                state_dict['weight']) * random_vector

            self.load_state_dict(state_dict)

    def forward(self, input: th.Tensor) -> th.Tensor:
        w_bounded, b_bounded = self.get_alignment_layer()

        if self.linear_bias:
            output = th.nn.functional.linear(input, w_bounded, b_bounded)
        else:
            output = th.nn.functional.linear(input, w_bounded)

        return output

    def get_alignment_layer(self):
        w_bounded = self.weight
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))
        b_bounded = 0.0
        if self.linear_bias:
            b_bounded = self.bias
        return w_bounded, b_bounded


class PositiveLinearAlignmentLayer(LinearAlignmentLayer):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None, data=None):
        super().__init__(in_features, out_features, False, device, dtype, data)

    def get_alignment_layer(self):
        w_bounded = th.nn.functional.softplus(self.weight)
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))
        b_bounded = 0.0
        if self.linear_bias:
            b_bounded = th.nn.functional.sigmoid(self.bias)
        return w_bounded, b_bounded


class ConvexAlignmentLayer(LinearAlignmentLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=th.float16, data=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype, data)

    def get_alignment_layer(self):

        w_bounded = th.nn.functional.softmax(self.weight, dim=1, dtype=self.weight.dtype)
        # assert th.allclose(w_bounded, th.nn.functional.softmax(self.weight))
        b_bounded = 0.0
        if self.linear_bias:
            b_bounded = th.nn.functional.sigmoid(self.bias)
        return w_bounded, b_bounded


class PositiveBoundedLinearModule(ConvexAlignmentLayer):

    def forward(self, input: th.Tensor) -> th.Tensor:
        output = super().forward(input)
        assert th.all(output < 0.0)
        assert th.all(input < 0.0)
        return output


class GroundingEnsemble(th.nn.Module):
    def __init__(self, *args, basic_classes, input_size, hid_sizes, activations, use_bias, debug=False, dtype=th.float32, **kwargs):
        super(GroundingEnsemble, self).__init__()
        
        self.hid_sizes = hid_sizes
        self.num_outputs = self.hid_sizes[-1]
        self.input_size = input_size
        self.basic_layer_classes = basic_classes
        self.desired_dtype=dtype
        self.use_bias = use_bias
        if isinstance(self.use_bias, bool):
            self.use_bias = [self.use_bias]*len(self.hid_sizes)
        self.activations = activations
        self.debug = debug
        # Initialize multiple copies of base network with 1 output each
        self.networks = th.nn.ModuleList([
            self._create_single_output_net() for _ in range(self.num_outputs)
        ])
        

    def _create_single_output_net(self):
        """Creates a single-output version of the original network structure."""
        modules = []
        for i in range(1, len(self.hid_sizes)):
            next_size = self.hid_sizes[i] if i < len(self.hid_sizes)-1 else 1
            linmod = self.basic_layer_classes[i - 1](
                self.hid_sizes[i - 1],  next_size, bias=self.use_bias[i-1],
                dtype=self.desired_dtype
            )

            """# Check if it’s a ConvexLinearModule and initialize weights if so
            if isinstance(linmod, ConvexLinearModule):
                state_dict = linmod.state_dict()
                state_dict['weight'] = th.rand(
                    *(state_dict['weight'].shape), requires_grad=True
                )
                state_dict['weight'] /= th.sum(state_dict['weight'])
                linmod.load_state_dict(state_dict)
                assert th.all(linmod.state_dict()['weight'] > 0)
"""
            modules.append(linmod)
            modules.append(self.activations[i - 1]())
            
        return th.nn.Sequential(*modules)
    def requires_grad_(self, requires_grad = True):
        r = super().requires_grad_(requires_grad)
        for n in self.networks:
            n.requires_grad_(requires_grad)
        self.networks.requires_grad_(requires_grad)
        return r
    def __str__(self):
        
        ret = ""
        for ic in range(self.hid_sizes[-1]):
            ret += (f"{ic}:" + str(self.networks[ic].state_dict()))
        
        
        return (ret + super().__str__())
    def forward(self, x):
        # Run input `x` through each network in self.networks and concatenate outputs
        outputs = [net(x) for net in self.networks]
        outputs =  th.cat(outputs, dim=1)
        if self.debug:
            if th.is_grad_enabled():
                # Define a loss function that only considers the output at `target_output_index`
                target = th.tensor([[1.0]], dtype=self.desired_dtype)  # Example target value for selected output
                profile = [0]*self.hid_sizes[-1]
                profile[0] = 1.0
                loss: th.Tensor = th.nn.MSELoss()(th.nn.functional.linear(outputs, th.tensor(profile, dtype=self.desired_dtype), bias=None), target)

                # Backward pass to calculate gradients
                loss.backward(retain_graph=False)
                
                
                # Check gradients for isolation: verify only target network parameters have gradients
                if __debug__:
                    for idx, network in enumerate(self.networks):
                        #print(f"Gradients for network {idx}:")
                        for param in network.parameters():
                            if param.grad is not None:
                                if idx == 0:
                                    assert param.grad is not None
                                    assert th.all(param.grad == 0)
                                    continue
                                else:
                                    none_grad = param.grad is None
                                    if not none_grad:
                                        
                                        assert th.all(param.grad == 0), f"Network {idx} should not have gradients!"
        
        return outputs
class AbstractVSLRewardFunction(reward_nets.RewardNet):
    def set_mode(self, mode):
        self.mode = mode

    @abstractmethod
    def set_network_for_value(self, value_id, network):
        ...
    
    @abstractmethod
    def get_network_for_value(self, value_id) -> th.nn.Module | Callable:
        ...


    def set_alignment_function(self, align_func=None):
        self.cur_align_func = align_func

    @abstractmethod
    def reset_learned_alignment_function(self, new_align_func=None):
        ...

    @abstractmethod
    def reset_learned_grounding_function(self, new_grounding_func=None):
        ...

    def set_grounding_function(self, grounding):
        if grounding is None:
            return
        self.cur_value_grounding = grounding
    @abstractmethod
    def get_trained_alignment_function_network(self):
        ...
    @abstractmethod
    def set_trained_alignment_function_network(self, nn: th.nn.Module):
        ...

    def set_env(self, env: ValueAlignedEnvironment):
        ...
    def remove_env(self) -> ValueAlignedEnvironment:
        return None

    def __init__(self, environment: gym.Env = None, action_dim=None, state_dim=None,
                 observation_space=None, action_space=None,
                 use_state=False, use_action=False, use_next_state=True,
                 use_done=True, use_one_hot_state_action=False,
                 mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                 features_extractor_class = None,
                 features_extractor_kwargs = None,
                 action_features_extractor_class = None,
                 action_features_extractor_kwargs = None,
                 clamp_rewards=None,
                 dtype=None,
                 **kwargs):
        if environment is None and (observation_space is None or action_space is None):
            raise ValueError(
                "Need to specify observation and action space, no environment or spaces supplied")

        if environment is not None:
            self.observation_space = environment.observation_space
            self.action_space = environment.action_space
        else:
            self.observation_space = observation_space
            self.action_space = action_space

        if environment is not None:
            try:
                self.action_dim = getattr(environment, 'action_dim')
                self.state_dim = getattr(environment, 'state_dim')
            except AttributeError:
                self.state_dim = preprocessing.get_flattened_obs_dim(
                    self.observation_space)
                self.action_dim = preprocessing.get_flattened_obs_dim(
                    self.action_space)
        elif action_dim is not None and state_dim is not None:
            self.action_dim = action_dim
            self.state_dim = state_dim
        else:
            self.state_dim = preprocessing.get_flattened_obs_dim(
                self.observation_space)
            self.action_dim = preprocessing.get_flattened_obs_dim(
                self.action_space)

        super().__init__(self.observation_space, self.action_space, True)

        self.desired_dtype=dtype if dtype is not None else th.float32

        self.mode = mode
        self.use_action = use_action

        self.use_next_state = use_next_state

        self.use_state = use_state
        self.use_done = use_done

        self.use_one_hot_state_action = use_one_hot_state_action
        self.cur_align_func = None
        self.cur_value_grounding = None
        self.clamp_rewards = clamp_rewards

        combined_size = 0

        self.state_size = 0
        self.action_size = 0

        if self.use_one_hot_state_action:
            combined_size += self.state_dim * self.action_dim
        else:
            if self.use_state:
                self.state_size = preprocessing.get_flattened_obs_dim(
                    self.observation_space)
                combined_size += self.state_size

            if self.use_action:
                self.action_size = preprocessing.get_flattened_obs_dim(
                    self.action_space)
                combined_size += self.action_size

            if self.use_next_state:
                self.state_size = preprocessing.get_flattened_obs_dim(
                    self.observation_space)
                combined_size += self.state_size
            if self.use_done:
                combined_size += 1

        self.input_size = combined_size
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        self.action_features_extractor_class = action_features_extractor_class
        self.action_features_extractor_kwargs = action_features_extractor_kwargs
        self.features_extractor = None
        self.action_features_extractor = None
        
                                                                                


    @abstractmethod
    def value_system_layer(self, align_func) -> th.Tensor:
        ...

    @abstractmethod
    def value_grounding_layer(self, custom_layer) -> th.Tensor:
        ...

    def _forward(self, features, align_func=None, grounding=None):

        x = self.value_grounding_layer(custom_layer=grounding)(features)

        x = self.value_system_layer(align_func=align_func)(x)

        return x
    def _forward_all(self, features, align_func=None, grounding=None):

        grounding = self.value_grounding_layer(custom_layer=grounding)(features)

        value_system = self.value_system_layer(align_func=align_func)(grounding)

        return value_system, grounding.t()

    def forward_value_groundings(self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor, info = None):
        inputs_concat = self.construct_input(state, action, next_state, done, info=info)
        return self.value_grounding_layer(custom_layer=self.cur_value_grounding)(inputs_concat)

    def forward(self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor, info=None) -> th.Tensor:
        inputs_concat = self.construct_input(state, action, next_state, done, info=info)
        
        ret = self._forward(
            inputs_concat, align_func=self.cur_align_func, grounding=self.cur_value_grounding)
        if len(ret.shape) > 1 and ret.shape[1] == 1:
            return squeeze_r(ret)
        else:
            return ret
    def forward_all(self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor, info=None) -> th.Tensor:
        inputs_concat = self.construct_input(state, action, next_state, done, info = info)
        
        ret, ret_r = self._forward_all(
            inputs_concat, align_func=self.cur_align_func, grounding=self.cur_value_grounding)
        if len(ret.shape) > 1 and ret.shape[1] == 1:
            return squeeze_r(ret), ret_r
        else:
            return ret, ret_r
        
    def __copy_args__(self, new):
        for k, v in vars(self).items():
            
            try:
                if isinstance(v, dict) and 'env' in v.keys():
                    nv = dict()
                    for a,av in v.items():
                        if a != 'env':
                            nv[a] = deepcopy(av)
                        else:
                            
                            nv[a] = av
                  
                elif isinstance(v, ContextualFeatureExtractorFromVAEnv): 
                    print("This should not happen?")
                    exit(0)
                    nv = None
                else:
                    nv = deepcopy(v)
                """if isinstance(v, th.nn.Module):
                    nv.load_state_dict(v.state_dict())
                if isinstance(v, GroundingEnsemble):
                    for i in range(v.networks):
                        #nv.networks[i] = deepcopy(v.networks[i])
                        nv.networks[i].load_state_dict(v.networks[i].state_dict())"""
                setattr(new, k, nv)
            except Exception as e:
                print(e)
                pass   # non-pickelable stuff wasn't needed
            
        return new

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        if self.use_one_hot_state_action:
            if len(state.shape) > 1 and (state.shape[0] == state.shape[1] and
                                         state.shape[1] == preprocessing.get_flattened_obs_dim(self.observation_space) * preprocessing.get_flattened_obs_dim(self.action_space)):
                # no preprocessing needed
                return super().preprocess(state, action, next_state, done)
            else:
                num_rows = len(state)
                row_length = self.state_dim * self.action_dim
                torch_obs_action_mat = th.zeros(
                    (num_rows, row_length), dtype=self.dtype, device=self.device)
                indices = state * self.action_dim + action
                # Set the positions corresponding to the calculated indices to 1
                torch_obs_action_mat[range(num_rows), indices] = 1.0

                # Retrieve the corresponding rows from the matrix for each index
                state_th = torch_obs_action_mat
                action_th = cast(
                    th.Tensor,
                    preprocessing.preprocess_obs(
                        util.safe_to_tensor(
                            action, dtype=self.dtype).to(self.device),
                        self.action_space,
                        self.normalize_images,
                    ),
                )
                next_state_th = cast(
                    th.Tensor,
                    preprocessing.preprocess_obs(
                        util.safe_to_tensor(
                            next_state, dtype=self.dtype).to(self.device),
                        self.observation_space,
                        self.normalize_images,
                    ),
                )
                done_th = util.safe_to_tensor(done, dtype=self.dtype).to(
                    self.device).to(self.dtype)
                return state_th, action_th, next_state_th, done_th
        else:
            return super().preprocess(state, action, next_state, done)

    def construct_input(self, state, action, next_state, done, info=None):
        inputs = []

        if self.use_one_hot_state_action:
            assert state.shape[1] == preprocessing.get_flattened_obs_dim(
                self.observation_space) * preprocessing.get_flattened_obs_dim(self.action_space)
            # Assume state encodes state-action pairs.
            inputs.append(th.flatten(state, 1))
        else:
            
            if self.use_state:
                if not self.observation_space.contains(state.detach().numpy()[0] if isinstance(state, th.Tensor) else state[0]):
                    if self.features_extractor is None:
                        self.features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
                    self.features_extractor.adapt_info(info)
                    #print("NC", self.features_extractor.env.context)
                    state = self.features_extractor(state)
                inputs.append(th.flatten(state, 1))
            if self.use_action:
                
                if self.action_features_extractor is None:
                    self.action_features_extractor = self.action_features_extractor_class(self.action_space, **self.action_features_extractor_kwargs)
                #self.action_features_extractor.adapt_info(info)
                preprocessed_action = self.action_features_extractor(action)
                inputs.append(preprocessed_action)
            if self.use_next_state:
                if not self.observation_space.contains(next_state.detach().numpy()[0] if isinstance(next_state, th.Tensor) else next_state[0]):
                    if self.features_extractor is None:
                        self.features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
                    self.features_extractor.adapt_info(info)
                    next_state = self.features_extractor(next_state)
                    
                inputs.append(th.flatten(next_state, 1))
            if self.use_done:
                inputs.append(th.reshape(done, [-1, 1]))
        #print("INPUT", inputs)
        inputs_concat = th.cat(inputs, dim=1)
        # If mode is not eval, cur_align_func and cur_value_grounding is not used. This depends on the implementation.
        return inputs_concat

    def get_next_align_func_and_its_probability(self, from_specific_align_func=None):
        used_alignment_func = from_specific_align_func if from_specific_align_func is not None else self.get_learned_align_function()
        probability = 1.0
        return used_alignment_func, probability, None

    @abstractmethod
    def get_learned_align_function(self):
        pass

    @abstractmethod
    def get_learned_grounding(self) -> th.nn.Module:
        pass

    @abstractmethod
    def copy(self) -> Self:
        pass


class LinearVSLRewardFunction(AbstractVSLRewardFunction):
    
    @override
    def set_env(self, env: ValueAlignedEnvironment):
        if env is None:
            return
        if self.features_extractor is not None:
            if hasattr(self.features_extractor, 'env'):
                self.features_extractor.env  = env
        self.features_extractor_kwargs['env']  = env

    @override
    def remove_env(self):
        env = None
        if self.features_extractor is not None:
            if hasattr(self.features_extractor, 'env'):
                env = self.features_extractor.env
                self.features_extractor.env = None
        else:
            env = self.features_extractor_kwargs['env']
            self.features_extractor_kwargs['env']=None
        return env

    def __init__(self, environment: gym.Env = None, action_dim=None, state_dim=None, observation_space=None,
                 action_space=None, use_state=False, use_action=False, use_next_state=True, use_done=True, use_one_hot_state_action=False,
                 activations=[th.nn.Sigmoid, th.nn.Tanh],
                 hid_sizes=[3,],
                 basic_layer_classes=[ConvexLinearModule,
                                      PositiveBoundedLinearModule],
                 mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                 reward_bias=0,
                 use_bias=False,
                 negative_grounding_layer=False,
                 clamp_rewards=None,
                 independent_grounding_layer=True,
                 features_extractor_class=None,
                 features_extractor_kwargs=None,
                 action_features_extractor_class = None,
                 action_features_extractor_kwargs = None,
                 dtype=None,
                 **kwargs):

        if hasattr(environment, 'last_profile'):
            self.cur_align_func = environment.last_profile

        self.negative_grounding_layer = negative_grounding_layer
        if isinstance(use_bias, list):
            self.use_bias = use_bias
            #use bias in the selected layers. Typically you dont use bias in the last layer. 
            
        else:
            # Use bias everywhere.
            self.use_bias = [use_bias]*(len(hid_sizes) + 1)
            
        assert len(self.use_bias) == len(hid_sizes) + 1
        super().__init__(environment=environment, action_dim=action_dim, state_dim=state_dim, observation_space=observation_space,
                         action_space=action_space,
                         use_state=use_state,
                         use_action=use_action,
                         use_next_state=use_next_state,
                         use_done=use_done,
                         clamp_rewards=clamp_rewards,
                         use_one_hot_state_action=use_one_hot_state_action, mode=mode,
                         features_extractor_class=features_extractor_class,
                         features_extractor_kwargs=features_extractor_kwargs,
                         action_features_extractor_class=action_features_extractor_class,
                         action_features_extractor_kwargs=action_features_extractor_kwargs,
                         dtype=dtype)
        
        self.reward_bias = reward_bias
        # self.mlp = networks.build_mlp(**full_build_mlp_kwargs)

        self.hid_sizes = deepcopy(hid_sizes)
        self.hid_sizes.insert(0, self.input_size)

        

        self.final_activation = activations[-1]()
        self.activations = activations
        self.basic_layer_classes = basic_layer_classes
        self.independent_groundings= independent_grounding_layer
        self.reset_learned_grounding_function()

        self.reset_learned_alignment_function()

        

        # th.tensor((0,0,0), requires_grad=True,device=self.parameters()[0].device)

    def set_mode(self, mode):
        super().set_mode(mode)
        if self.mode is None or self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            self.cur_value_grounding = None
            self.values_net.requires_grad_(True)
            self.trained_profile_net.requires_grad_(False)
        elif self.mode == TrainingModes.EVAL:
            self.values_net.requires_grad_(False)
            self.trained_profile_net.requires_grad_(False)
        elif self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            self.cur_align_func = None
            self.values_net.requires_grad_(False)
            self.trained_profile_net.requires_grad_(True)
        elif self.mode == TrainingModes.SIMULTANEOUS:
            self.cur_value_grounding = None
            self.cur_align_func = None
            self.values_net.requires_grad_(True)
            self.trained_profile_net.requires_grad_(True)

        else:
            raise ValueError("Unkown training mode" + str(self.mode))

    def set_alignment_function(self, align_func):
        super().set_alignment_function(align_func)
        if self.cur_align_func is not None:
            assert self.hid_sizes[-1] == len(self.cur_align_func)

    def get_trained_alignment_function_network(self):
        return self.trained_profile_net
    
    def set_trained_alignment_function_network(self, nn: th.nn.Module):
        self.trained_profile_net = nn

    def reset_learned_alignment_function(self, new_align_func: tuple = None):
        self.trained_profile_net: LinearAlignmentLayer = self.basic_layer_classes[-1](
            self.hid_sizes[-1], 1, bias=self.use_bias[-1], dtype=self.desired_dtype)
        with th.no_grad():
            if new_align_func is not None and isinstance(self.trained_profile_net, PositiveBoundedLinearModule):
                assert isinstance(new_align_func, tuple)
                state_dict = self.trained_profile_net.state_dict()
                state_dict['weight'] = th.log(
                    th.as_tensor([list(new_align_func)], dtype=self.desired_dtype).clone())

                self.trained_profile_net.load_state_dict(state_dict)

    def clamp_tensor(self, x):
        if self.clamp_rewards is not None and self.clamp_rewards != False:
            x = th.clamp(x, self.clamp_rewards[0], self.clamp_rewards[1])

        return x

    def value_grounding_layer(self, custom_layer=None):
        if custom_layer is None or self.mode in [TrainingModes.VALUE_GROUNDING_LEARNING, TrainingModes.SIMULTANEOUS]:
            if self.negative_grounding_layer:
                return lambda x: -self.clamp_tensor(self.values_net(x))
            else:
                return lambda x: self.clamp_tensor(self.values_net(x))
        else:
            if isinstance(custom_layer, th.nn.Module) or callable(custom_layer):
                def _call(x):
                    with th.no_grad():
                        if isinstance(custom_layer, th.nn.Module):
                            custom_layer.requires_grad_(False)
                        
                        if self.negative_grounding_layer:
                            x = -custom_layer(x)
                            assert th.all(x <= 0)
                        else:
                            x = custom_layer(x)

                        assert x.shape[-1] == self.hid_sizes[-1]

                        return self.clamp_tensor(x)
            else:
                def _call(x):
                    assert custom_layer.shape == (
                        self.input_size, self.hid_sizes[-1])
                    if isinstance(custom_layer, th.Tensor):
                        pt = custom_layer.clone().detach()
                    else:
                        # .reshape((1, len(profile)))
                        pt = th.tensor(
                            custom_layer, requires_grad=False, dtype=self.dtype)
                    if self.negative_grounding_layer:
                        x = -th.nn.functional.linear(x, pt.T)
                        assert th.all(x <= 0)
                    else:
                        x = th.nn.functional.linear(x, pt.T)
                    return self.clamp_tensor(x)
            return _call

    def value_system_layer(self, align_func=None):
        if self.mode == TrainingModes.EVAL or self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            if align_func is None:
                align_func = self.cur_align_func

            def _call(x):
                # .reshape((1, len(profile)))
                pt = th.tensor(align_func, requires_grad=False,
                               dtype=self.dtype)
                x = th.nn.functional.linear(x, pt)
                x = self.clamp_tensor(
                    self.final_activation(x) + self.reward_bias)
                return x
        elif self.mode == TrainingModes.SIMULTANEOUS or self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            def _call(x):
                x = self.trained_profile_net(x)
                x = self.final_activation(x) + self.reward_bias
                return self.clamp_tensor(x)
        return _call

    def reset_learned_grounding_function(self, new_grounding_func=None):

        if self.independent_groundings:
            self.values_net = GroundingEnsemble(basic_classes=self.basic_layer_classes,input_size=self.input_size,hid_sizes=self.hid_sizes,activations=self.activations,use_bias=self.use_bias, dtype=self.desired_dtype)
        else:
            modules = []
            if new_grounding_func is None:
                for i in range(1, len(self.activations)):
                    linmod = self.basic_layer_classes[i-1](
                        self.hid_sizes[i-1], self.hid_sizes[i], bias=self.use_bias[i], dtype=self.desired_dtype)

                    
                    modules.append(linmod)
                    modules.append(self.activations[i-1]())
                # modules.append(nn.BatchNorm1d()) ?

                self.values_net = th.nn.Sequential(*modules,)
            else:
                self.values_net = new_grounding_func

    def copy(self) -> Self:
        new = self.__class__(env=None, action_dim=self.action_dim, state_dim=self.state_dim, observation_space=self.observation_space, action_space=self.action_space,
                             use_state=self.use_state, use_action=self.use_action, use_next_state=self.use_next_state, use_done=self.use_done, use_one_hot_state_action=self.use_one_hot_state_action,
                             activations=self.activations, hid_sizes=self.hid_sizes[1:], basic_layer_classes=self.basic_layer_classes, mode=self.mode, reward_bias=self.reward_bias,
                             use_bias=self.use_bias, negative_grounding_layer=self.negative_grounding_layer, dtype=self.desired_dtype,
                             
                         features_extractor_kwargs=self.features_extractor_kwargs,
                         features_extractor_class=self.features_extractor_class,
                         action_features_extractor_class=self.action_features_extractor_class,
                         action_features_extractor_kwargs=self.action_features_extractor_kwargs)
        new = self.__copy_args__(new)
        new.cur_align_func = deepcopy(self.cur_align_func)
        new.cur_value_grounding = deepcopy(self.cur_value_grounding)
        new.features_extractor_kwargs['env'] = self.features_extractor_kwargs['env'] # the env should be exactly the same.
        return new

    def parameters(self, recurse: bool = True) -> Iterator:
        if self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            return self.values_net.parameters(recurse)
        elif self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            return self.trained_profile_net.parameters(recurse)
        else:
            return chain(self.trained_profile_net.parameters(recurse), self.values_net.parameters(recurse))
    
    def value_matrix(self):
        with th.no_grad():
            params = th.tensor(
                list(p.data for name, p in self.values_net.named_parameters(recurse=True))[-1], dtype=self.desired_dtype)
            assert params.dim() == 2
            return params

    def get_learned_profile(self, with_bias=False):
        prof_w = tuple(
            list(self.trained_profile_net.get_alignment_layer()[0].squeeze().tolist()))
        prof_b = float(self.trained_profile_net.get_alignment_layer()[1])

        assert isinstance(prof_w, tuple)
        assert isinstance(prof_w[0], float) or isinstance(prof_w[0], int)
        if with_bias:
            return prof_w, prof_b
        else:
            return prof_w

    def get_learned_align_function(self):
        return self.get_learned_profile()

    def get_learned_grounding(self) -> GroundingEnsemble:
        return self.values_net

    def set_network_for_value(self, value_id, network) :
        self.values_net.networks[value_id] = network
    def get_network_for_value(self, value_id) -> th.nn.Module:
        return self.values_net.networks[value_id]
        
    def save_checkpoint(self, file="reward_function_checkpoint.pt"):
        th.save(self, os.path.join(CHECKPOINTS, "reward_function_checkpoint.pt"))
        # th.save(self.values_net, os.path.join(CHECKPOINTS, "reward_function_checkpoint.pt"))

    def from_checkpoint(file="reward_function_checkpoint.pt"):
        return th.load(os.path.join(CHECKPOINTS, "reward_function_checkpoint.pt"))


class ProabilisticProfiledRewardFunction(LinearVSLRewardFunction):

    def __init__(self, environment=None, action_dim=None, state_dim=None, observation_space=None,
                 action_space=None, use_state=False, use_action=False, use_next_state=True, use_done=True, use_one_hot_state_action=False, activations=[th.nn.Sigmoid,], hid_sizes=[3], basic_layer_classes=[ConvexLinearModule, PositiveBoundedLinearModule], mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION, reward_bias=0, negative_grounding_layer=False, use_bias=False, **kwargs):
        super().__init__(environment=environment, action_dim=action_dim, state_dim=state_dim, observation_space=observation_space,
                         action_space=action_space, use_state=use_state, use_action=use_action, use_next_state=use_next_state, use_done=use_done, use_one_hot_state_action=use_one_hot_state_action,
                         activations=activations,
                         hid_sizes=hid_sizes,
                         basic_layer_classes=basic_layer_classes,
                         mode=mode,
                         reward_bias=reward_bias,
                         use_bias=use_bias,
                         negative_grounding_layer=negative_grounding_layer,
                         **kwargs)
        self.fixed_align_func_index = None
        self.reset_learned_alignment_function()

    def reset_learned_alignment_function(self, new_align_func: tuple = None):
        self.trained_profile_net = ConvexTensorModule(
            size=self.hid_sizes[-1], init_tuple=new_align_func, dtype=self.desired_dtype)
        self.cur_align_func = None

    def fix_alignment_function(self, align_func=None, random=True):
        if align_func is None:
            if self.mode == TrainingModes.EVAL or self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
                align_func = self.cur_align_func
            else:
                align_func = self.trained_profile_net().detach().numpy()

        if random:
            self.fixed_align_func_index = np.random.choice(
                len(align_func), p=align_func)
        else:
            self.fixed_align_func_index = np.argmax(
                np.random.permutation(align_func))

    def free_alignment_function(self):
        self.fixed_align_func_index = None

    def get_learned_profile(self, with_bias=False):
        return self.trained_profile_net.get_tensor_as_tuple()


    def parameters(self, recurse: bool = True) -> Iterator:
        if self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            return self.values_net.parameters(recurse)
        elif self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            return self.trained_profile_net.parameters(recurse)
        else:
            return chain(self.trained_profile_net.parameters(recurse), self.values_net.parameters(recurse))

    def get_next_align_func_and_its_probability(self, from_specific_align_func=None):

        if from_specific_align_func is None:
            align_tensor = self.trained_profile_net()
        else:
            if not isinstance(from_specific_align_func, th.Tensor):
                align_tensor = th.as_tensor(from_specific_align_func, dtype=self.desired_dtype, device=self.device).requires_grad_(
                    self.mode != TrainingModes.EVAL)
            else:
                align_tensor = from_specific_align_func

        if self.fixed_align_func_index is not None:
            selected_index = self.fixed_align_func_index
        else:
            selected_index = np.random.choice(
                a=len(align_tensor), p=align_tensor)

        used_alignment_func = np.zeros_like(align_tensor.detach().numpy())
        used_alignment_func[self.fixed_align_func_index] = 1.0
        used_alignment_func = tuple(used_alignment_func)

        probability = align_tensor[selected_index]

        return used_alignment_func, probability, selected_index

    def value_system_layer(self, align_func=None):
        if align_func is None:
            align_func = self.cur_align_func

        if self.mode == TrainingModes.EVAL or self.mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            if align_func is None:
                align_func = self.trained_profile_net().detach()

            def _call(x):
                used_alignment_func, probability, selected_index = self.get_next_align_func_and_its_probability(
                    align_func)
                if len(x.size()) == 2:
                    x = x[:, selected_index]
                    return x
                elif len(x.size()) == 3:
                    x = x[:, :, selected_index]
                x = self.final_activation(x) + self.reward_bias
                return x

        elif self.mode == TrainingModes.SIMULTANEOUS or self.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            def _call(x):
                used_alignment_func, probability, selected_index = self.get_next_align_func_and_its_probability(
                    self.trained_profile_net())
                if len(x.size()) == 2:
                    x = x[:, selected_index]
                    return x
                elif len(x.size()) == 3:
                    x = x[:, :, selected_index]
                else:
                    print("X SIZE IS NOT EXPECTED", x.size())
                    exit(-1)
                x = self.final_activation(x) + self.reward_bias
                return x
        return _call

    def copy(self):
        new = self.__class__(env=None, action_dim=self.action_dim, state_dim=self.state_dim, observation_space=self.observation_space, action_space=self.action_space,
                             use_state=self.use_state, use_action=self.use_action, use_next_state=self.use_next_state, use_done=self.use_done, use_one_hot_state_action=self.use_one_hot_state_action,
                             activations=self.activations, hid_sizes=self.hid_sizes, basic_layer_classes=self.basic_layer_classes, mode=self.mode, reward_bias=self.reward_bias,
                             use_bias=self.use_bias, negative_grounding_layer=self.negative_grounding_layer)
        new = self.__copy_args__(new)
        new.cur_align_func = deepcopy(self.cur_align_func)
        new.cur_value_grounding = deepcopy(self.cur_value_grounding)
        return new


def plot_avg_value_matrix(avg_matrix, std_matrix, file='value_matrix_demo.pdf'):
    X, Y = np.meshgrid(
        np.arange(avg_matrix.shape[1]), np.arange(avg_matrix.shape[0]))

    # Plotting the average matrix as bars
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(avg_matrix.shape[0]):
        for j in range(avg_matrix.shape[1]):
            ax.bar3d(j, i, 0, 1, 1, avg_matrix[i, j], color=plt.cm.viridis(
                avg_matrix[i, j] / avg_matrix.max()),  alpha=0.5)

    # Overlaying the standard deviation as black segments
    for i in range(avg_matrix.shape[0]):
        for j in range(avg_matrix.shape[1]):
            ax.plot3D([j+0.5, j+0.5], [i+0.5, i+0.5], [avg_matrix[i, j] -
                      std_matrix[i, j], avg_matrix[i, j] + std_matrix[i, j]], color='k')

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



def parse_layer_name(layer_name):
        if layer_name == 'nn.Linear':
            return th.nn.Linear
        if layer_name == 'ConvexAlignmentLayer':
            return ConvexAlignmentLayer
        if layer_name == 'nn.LeakyReLU':
            return th.nn.LeakyReLU
        if layer_name == 'nn.Tanh':
            return th.nn.Tanh
        if layer_name == 'nn.Softplus':
            return th.nn.Softplus
        if layer_name == 'nn.Sigmoid':
            return th.nn.Sigmoid
        if layer_name == 'nn.Identity':
            return th.nn.Identity
        if layer_name == 'ConvexLinearModule':
            return ConvexLinearModule
        
        raise ValueError(f'Unknown layer name: {layer_name}')