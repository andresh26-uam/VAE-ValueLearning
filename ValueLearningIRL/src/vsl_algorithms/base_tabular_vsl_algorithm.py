from typing import Optional

import numpy as np
from src.envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment
from src.vsl_algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.vsl_policies import ValueSystemLearningPolicy
from src.vsl_reward_functions import AbstractVSLRewardFunction, TrainingModes, squeeze_r

from imitation.data import types
from imitation.util import logger as imit_logger
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Type,
    Union,
)
import torch as th


def dict_metrics(**kwargs):
    return dict(kwargs)


class BaseTabularMDPVSLAlgorithm(BaseVSLAlgorithm):
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
        environment_is_stochastic=False,
        training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,

    ) -> None:
        super().__init__(env=env, reward_net=reward_net, vgl_optimizer_cls=vgl_optimizer_cls,
                         vsi_optimizer_cls=vsi_optimizer_cls, vsi_optimizer_kwargs=vsi_optimizer_kwargs,
                         vgl_optimizer_kwargs=vgl_optimizer_kwargs, discount=discount, log_interval=log_interval, vgl_expert_policy=vgl_expert_policy, vsi_expert_policy=vsi_expert_policy,
                         target_align_func_sampler=target_align_func_sampler, vsi_target_align_funcs=vsi_target_align_funcs,
                         vgl_target_align_funcs=vgl_target_align_funcs, training_mode=training_mode, custom_logger=custom_logger, learn_stochastic_policy=learn_stochastic_policy)
        self.rewards_per_target_align_func_callable = None
        self.environment_is_stochastic = environment_is_stochastic
        self.__previous_next_states = None

    def get_metrics(self):
        return dict_metrics(learned_rewards=self.rewards_per_target_align_func_callable)

    def train_callback(self, t):
        # pass
        return

    @property
    def policy(self):
        return self.learned_policy_per_va

    def _resample_next_observations(self):
        n_actions = self.env.action_dim
        n_states = self.env.state_dim
        obs_dim = self.env.obs_dim

        next_obs_mat = np.zeros((n_actions, n_states, obs_dim))
        for a in range(n_actions):
            for s in range(n_states):
                try:
                    ns = np.random.choice(
                        n_states, size=1, p=self.env.transition_matrix[s, a])
                except ValueError:
                    ns = s
                next_obs_mat[a, s, :] = self.env.observation_matrix[ns]

        """next_states = sample_next_states(self.env.transition_matrix)
        next_obs_mat = self.env.observation_matrix[next_states]"""
        torch_next_obs_mat = th.as_tensor(
            next_obs_mat, dtype=self.current_net.dtype, device=self.current_net.device)
        return torch_next_obs_mat

    def _resample_next_states(self):
        n_actions = self.env.action_dim
        n_states = self.env.state_dim

        next_state_mat = np.zeros((n_states, n_actions))
        if not self.environment_is_stochastic:
            if self.__previous_next_states is not None:
                return self.__previous_next_states.clone().detach()

        for a in range(n_actions):
            for s in range(n_states):
                tr = self.env.transition_matrix[s, a]
                if np.allclose(tr, 0.0):
                    ns = s
                else:
                    ns = np.random.choice(n_states, size=1, p=tr)[0]
                next_state_mat[s, a] = ns

        """next_states = sample_next_states(self.env.transition_matrix)
        next_obs_mat = self.env.observation_matrix[next_states]"""
        torch_next_state_mat = th.as_tensor(
            next_state_mat, dtype=th.long).requires_grad_(False).detach()
        self.__previous_next_states = torch_next_state_mat
        return torch_next_state_mat

    def calculation_rew(self, align_func, obs_mat, action_mat=None, obs_action_mat=None, next_state_obs=None, use_probabilistic_reward=False):
        if use_probabilistic_reward:
            self.current_net.fix_alignment_function()

        next_state_observations = None

        if self.current_net.use_next_state:
            if next_state_obs is None:
                next_states = self._resample_next_states()

                next_state_observations = obs_mat[next_states]
            else:
                next_state_observations = next_state_obs

        if self.current_net.use_one_hot_state_action:
            if self.current_net.use_next_state:
                next_state_observations = next_state_observations.view(
                    *obs_action_mat.shape)

            predicted_r = th.reshape(self.current_net(
                obs_action_mat, None, next_state_observations, None), (self.env.state_dim, self.env.action_dim))
        elif self.current_net.use_action or self.current_net.use_next_state:
            if self.current_net.use_action:
                assert action_mat is not None
                assert action_mat.size() == (self.env.action_dim,
                                             obs_mat.shape[0], self.env.action_dim)
            if self.current_net.use_next_state:
                assert next_state_observations is not None

            predicted_r = th.stack([
                squeeze_r(
                    self.current_net(
                        obs_mat,
                        (action_mat[i]
                         if self.current_net.use_action else None),
                        next_state_observations[:, i,
                                                :] if self.current_net.use_next_state else None,
                        None)
                ) for i in range(self.env.action_dim)], dim=1)

        assert predicted_r.size() == (obs_mat.shape[0], self.env.action_dim)

        used_alignment_func, probability, _ = self.current_net.get_next_align_func_and_its_probability(
            align_func)

        if use_probabilistic_reward:
            self.current_net.free_alignment_function()

        state_actions_with_special_reward = self.env.get_state_actions_with_known_reward(
            used_alignment_func)

        if state_actions_with_special_reward is not None:
            predicted_r[state_actions_with_special_reward] = th.as_tensor(
                self.env.reward_matrix_per_align_func(used_alignment_func)[
                    state_actions_with_special_reward],
                dtype=predicted_r.dtype, device=predicted_r.device)

        return predicted_r, used_alignment_func, probability

    def calculate_rewards(self, align_func=None, grounding=None, obs_mat=None, next_state_obs_mat=None, action_mat=None, obs_action_mat=None,
                          reward_mode=TrainingModes.EVAL, recover_previous_config_after_calculation=True,
                          use_probabilistic_reward=False, n_reps_if_probabilistic_reward=10, requires_grad=True):

        if obs_mat is None:
            obs_mat = self.env.observation_matrix
            obs_mat = th.as_tensor(
                obs_mat,
                dtype=self.current_net.dtype,
                device=self.current_net.device,
            )
            obs_mat.requires_grad_(requires_grad)

        if self.current_net.use_one_hot_state_action:
            if obs_action_mat is None:
                obs_action_mat = th.as_tensor(
                    np.identity(self.env.state_dim*self.env.action_dim),
                    dtype=self.current_net.dtype,
                    device=self.current_net.device,
                )
            obs_action_mat.requires_grad_(requires_grad)

        if recover_previous_config_after_calculation:
            previous_rew_mode = self.current_net.mode
            previous_rew_ground = self.current_net.cur_value_grounding
            previous_rew_alignment = self.current_net.cur_align_func

        if requires_grad is False and action_mat is not None:
            action_mat = action_mat.detach()

        self.current_net.set_mode(reward_mode)
        self.current_net.set_grounding_function(grounding)
        self.current_net.set_alignment_function(align_func)

        assert self.current_net.mode == reward_mode

        if use_probabilistic_reward is False:
            predicted_r, used_align_func, _ = self.calculation_rew(
                align_func=align_func, obs_mat=obs_mat, action_mat=action_mat,
                obs_action_mat=obs_action_mat, next_state_obs=next_state_obs_mat,
                use_probabilistic_reward=use_probabilistic_reward)

            predicted_r_np = predicted_r.detach().cpu().numpy()
            ret = predicted_r, predicted_r_np
        else:
            list_of_reward_calculations = []
            align_func_used_in_each_repetition = []
            prob_of_each_repetition = []
            for _ in range(n_reps_if_probabilistic_reward):
                predicted_r, used_align_func, probability = self.calculation_rew(
                    align_func=align_func, obs_mat=obs_mat, action_mat=action_mat,
                    obs_action_mat=obs_action_mat, next_state_obs=next_state_obs_mat,
                    use_probabilistic_reward=use_probabilistic_reward)

                list_of_reward_calculations.append(predicted_r)
                align_func_used_in_each_repetition.append(used_align_func)
                prob_of_each_repetition.append(probability)
            predicted_rs = th.stack(list_of_reward_calculations)
            prob_of_each_repetition_th = th.stack(prob_of_each_repetition)
            predicted_rs_np = predicted_rs.detach().cpu().numpy()

            ret = predicted_rs, predicted_rs_np, align_func_used_in_each_repetition, prob_of_each_repetition_th

        if recover_previous_config_after_calculation:
            self.current_net.set_mode(previous_rew_mode)
            self.current_net.set_grounding_function(previous_rew_ground)
            self.current_net.set_alignment_function(previous_rew_alignment)

        return ret

    def train(self, max_iter: int = 1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None, n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_probabilistic_reward=False, n_reward_reps_if_probabilistic_reward=10) -> np.ndarray:
        obs_mat = self.env.observation_matrix

        self.torch_obs_mat = th.as_tensor(
            obs_mat,
            dtype=self.reward_net.dtype,
            device=self.reward_net.device,
        )
        self.torch_obs_mat.requires_grad_(True)

        self.torch_action_mat = None
        if self.reward_net.use_action:
            actions_one_hot = th.eye(
                self.env.action_space.n, requires_grad=True)
            self.torch_action_mat = th.stack([actions_one_hot[i].repeat(
                obs_mat.shape[0], 1) for i in range(self.env.action_space.n)], dim=0)

        self.torch_obs_action_mat = th.as_tensor(
            np.identity(self.env.state_dim*self.env.action_dim),
            dtype=self.reward_net.dtype,
            device=self.reward_net.device,
        )
        self.torch_obs_action_mat.requires_grad_(True)

        self.rewards_per_target_align_func = None

        self.rewards_per_target_align_func_callable = lambda target: lambda: self.prob_reward_matrix_setter(
            target, use_probabilistic_reward=use_probabilistic_reward, assumed_grounding=assumed_grounding)

        return super().train(max_iter=max_iter, mode=mode, assumed_grounding=assumed_grounding,
                             n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories,
                             use_probabilistic_reward=use_probabilistic_reward, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed, n_reward_reps_if_probabilistic_reward=n_reward_reps_if_probabilistic_reward)

    def prob_reward_matrix_setter(self, target, use_probabilistic_reward, numpy=True, requires_grad=False, assumed_grounding=None):
        if use_probabilistic_reward:
            rewards_th, rewards_np, _, _ = self.calculate_rewards(align_func=self.target_align_funcs_to_learned_align_funcs[target],
                                                                  grounding=assumed_grounding,
                                                                  obs_mat=self.torch_obs_mat,
                                                                  action_mat=self.torch_action_mat,
                                                                  obs_action_mat=self.torch_obs_action_mat,
                                                                  reward_mode=TrainingModes.EVAL, requires_grad=requires_grad,
                                                                  use_probabilistic_reward=True, n_reps_if_probabilistic_reward=1)
            return rewards_th[0] if not numpy else rewards_np[0]
        else:
            if self.rewards_per_target_align_func is None:
                self.rewards_per_target_align_func = dict()
            if target not in self.rewards_per_target_align_func.keys():
                rewards_th, rewards_np = self.calculate_rewards(align_func=self.target_align_funcs_to_learned_align_funcs[target],
                                                                grounding=assumed_grounding,
                                                                obs_mat=self.torch_obs_mat,
                                                                action_mat=self.torch_action_mat,
                                                                obs_action_mat=self.torch_obs_action_mat,
                                                                reward_mode=TrainingModes.EVAL, requires_grad=requires_grad, use_probabilistic_reward=False,
                                                                n_reps_if_probabilistic_reward=1)

                self.rewards_per_target_align_func[target] = self.state_action_reward_from_computed_reward(
                    rewards_th if not numpy else rewards_np)
            return self.rewards_per_target_align_func[target]

    def state_action_reward_from_computed_reward(self, rewards):
        return rewards
