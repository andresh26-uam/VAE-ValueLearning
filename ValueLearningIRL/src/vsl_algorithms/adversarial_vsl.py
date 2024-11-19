from abc import abstractmethod
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from src.envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment
from src.vsl_algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.vsl_policies import VAlignedDiscreteSpaceActionPolicy, ValueSystemLearningPolicy
from src.vsl_reward_functions import AbstractVSLRewardFunction, LinearVSLRewardFunction, ConvexTensorModule, ProabilisticProfiledRewardFunction, TrainingModes

from imitation.algorithms import base
from imitation.data import types
from imitation.util import logger as imit_logger
from imitation.util import networks

import imitation.algorithms.adversarial.airl as airl
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Type,
    Union,
)
import torch as th


def dict_metrics(**kwargs):
    return dict(kwargs)


class AdversarialVSL(BaseVSLAlgorithm):
    def set_demonstrations(self, demonstrations: Union[Iterable[types.Trajectory], Iterable[types.TransitionMapping], types.TransitionsMinimal]) -> None:
        pass

    def __init__(
        self,
        env: Union[TabularVAMDP, ValueAlignedEnvironment],
        reward_net: AbstractVSLRewardFunction,
        vgl_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        vsi_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        vgl_optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        vsi_optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        discount: float = 1.0,
        log_interval: Optional[int] = 100,
        vgl_expert_policy: Optional[ValueSystemLearningPolicy] = None,
        vsi_expert_policy: Optional[ValueSystemLearningPolicy] = None,

        # A Society or other mechanism might return different alignment functions at different times.
        target_align_func_sampler=lambda *args: args[0],


        vsi_target_align_funcs=[],

        vgl_target_align_funcs=[],
        learn_stochastic_policy=True,

        training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
        # NEW parameters
        learner_class=PPO, # accepts env, and a policy parameter
        learner_kwargs=dict(batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0005,
            gamma=0.95,
            clip_range=0.1,
            vf_coef=0.1,
            n_epochs=5,
        ),
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,

    ) -> None:
        super().__init__(env, reward_net, vgl_optimizer_cls, vsi_optimizer_cls, vgl_optimizer_kwargs, vsi_optimizer_kwargs, discount, log_interval, vgl_expert_policy, vsi_expert_policy, target_align_func_sampler, vsi_target_align_funcs, vgl_target_align_funcs, learn_stochastic_policy, training_mode, custom_logger=custom_logger)
    
    
    def get_metrics(self):
        return {}

    def train_callback(self, t):
        # pass
        return

    def train(self, max_iter: int = 1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None, n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_probabilistic_reward=False, n_reward_reps_if_probabilistic_reward=10) -> np.ndarray:

        self.training_mode = mode

        if use_probabilistic_reward:
            if self.probabilistic_reward_net is None:
                assert isinstance(self.reward_net, LinearVSLRewardFunction)
                self.probabilistic_reward_net = ProabilisticProfiledRewardFunction(
                    environment=self.env, use_action=self.reward_net.use_action, use_done=self.reward_net.use_done, use_next_state=self.reward_net.use_next_state, use_one_hot_state_action=self.reward_net.use_one_hot_state_action, use_state=self.reward_net.use_state,
                    activations=self.reward_net.activations, hid_sizes=self.reward_net.hid_sizes,
                    basic_layer_classes=[*self.reward_net.basic_layer_classes[:len(
                        self.reward_net.basic_layer_classes)-1], ConvexTensorModule],
                    mode=mode, negative_grounding_layer=self.reward_net.negative_grounding_layer, use_bias=self.reward_net.use_bias
                )
                self.probabilistic_reward_net.values_net = self.reward_net.values_net
                self.probabilistic_reward_net.set_alignment_function(
                    self.reward_net.get_learned_align_function())

            self.current_net = self.probabilistic_reward_net
        else:
            self.current_net = self.reward_net

        self.current_net.set_mode(mode)
        if assumed_grounding is not None and mode in [TrainingModes.EVAL, TrainingModes.VALUE_SYSTEM_IDENTIFICATION]:
            self.current_net.set_grounding_function(assumed_grounding)
        if mode in [TrainingModes.VALUE_GROUNDING_LEARNING, TrainingModes.SIMULTANEOUS]:
            self.current_net.reset_learned_grounding_function(
                assumed_grounding)
        target_align_funcs = self.vgl_target_align_funcs if mode == TrainingModes.VALUE_GROUNDING_LEARNING else self.vsi_target_align_funcs
        self.target_align_funcs_to_learned_align_funcs = dict()

        if mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            self.vgl_optimizer = self.vgl_optimizer_cls(
                self.current_net.parameters(), **self.vgl_optimizer_kwargs)
            self.target_align_funcs_to_learned_align_funcs = {
                al: al for al in self.vgl_target_align_funcs}
            with networks.training(self.current_net):
                reward_nets_per_target_align_func = self.train_vgl(
                    max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)

            
        elif mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            reward_nets_per_target_align_func = dict()
            self.target_align_funcs_to_learned_align_funcs = dict()

            for ti, target_align_func in enumerate(target_align_funcs):
                self.current_net.reset_learned_alignment_function()
                self.current_net.set_mode(
                    TrainingModes.VALUE_SYSTEM_IDENTIFICATION)

                if use_probabilistic_reward:
                    self.probabilistic_vsi_optimizer = self.probabilistic_vsi_optimizer_cls(
                        self.current_net.parameters(), **self.probabilistic_vsi_optimizer_kwargs)
                    with networks.training(self.current_net):
                        self.target_align_funcs_to_learned_align_funcs[target_align_func] = self.train_vsl_probabilistic(
                            max_iter=max_iter, n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed,
                            n_reward_reps_if_probabilistic_reward=n_reward_reps_if_probabilistic_reward,
                            target_align_func=target_align_func)

                else:
                    self.vsi_optimizer = self.vsi_optimizer_cls(
                        self.current_net.parameters(), **self.vsi_optimizer_kwargs)
                    with networks.training(self.current_net):
                        self.target_align_funcs_to_learned_align_funcs[target_align_func] = self.train_vsl(
                            max_iter=max_iter, n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories,
                            n_sampled_trajs_per_seed=n_sampled_trajs_per_seed, target_align_func=target_align_func)

                reward_nets_per_target_align_func[target_align_func] = self.current_net.copy(
                )

        else:
            # TODO: Simultaneous learning?

            raise NotImplementedError(
                "Simultaneous learning mode is not implemented")

        # Organizing learned content:

        self.learned_policy_per_va = self.calculate_learned_policies(
            target_align_funcs)

        if mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            return self.target_align_funcs_to_learned_align_funcs, reward_nets_per_target_align_func, self.get_metrics()
        elif mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            return self.current_net.get_learned_grounding(), reward_nets_per_target_align_func, self.get_metrics()
        else:
            return self.current_net.get_learned_grounding(), self.target_align_funcs_to_learned_align_funcs, reward_nets_per_target_align_func, self.get_metrics()

    @abstractmethod
    def test_accuracy_for_align_funcs(self, learned_rewards_nets_per_rep,
                               target_align_funcs_to_learned_align_funcs=None,
                                testing_align_funcs=[]):
        pass
    
    @abstractmethod
    def get_policy_from_reward_per_align_func(self, align_funcs, reward_net=None):
        pass

    def state_action_callable_reward_from_computed_rewards_per_target_align_func(self, rewards_per_target_align_func: Union[Dict, Callable]):
        if isinstance(rewards_per_target_align_func, dict):
            def rewards_per_target_align_func_callable(
                al_f): return rewards_per_target_align_func[al_f]
        else:
            return rewards_per_target_align_func_callable

    @abstractmethod
    def calculate_learned_policies(self, target_align_funcs) -> ValueSystemLearningPolicy:
        ...

    @abstractmethod
    def train_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func) -> Dict[Any, AbstractVSLRewardFunction]:
        ...

    @abstractmethod
    def train_vsl_probabilistic(self, max_iter,
                                n_seeds_for_sampled_trajectories,
                                n_sampled_trajs_per_seed,
                                n_reward_reps_if_probabilistic_reward,
                                target_align_func):
        ...

    @abstractmethod
    def train_vgl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed) -> Dict[Any, AbstractVSLRewardFunction]:
        ...
