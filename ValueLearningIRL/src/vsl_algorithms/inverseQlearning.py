import gymnasium as gym
from seals import base_envs
from collections import deque
from copy import deepcopy
import datetime
import logging

import os
import pprint
import random
import time
from typing import Optional


import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from src.envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment
from src.vsl_algorithms.base_tabular_vsl_algorithm import BaseTabularMDPVSLAlgorithm, PolicyApproximators
from src.vsl_algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.vsl_policies import VAlignedDictDiscreteStateActionPolicyTabularMDP, ValueSystemLearningPolicy
from src.vsl_reward_functions import AbstractVSLRewardFunction, LinearVSLRewardFunction, TrainingModes, print_tensor_and_grad_fn

from imitation.data import types
from imitation.util import logger as imit_logger


from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Type,
    Union,
)
import torch as th

from src.vsl_policies import LearnerValueSystemLearningPolicy, ValueSystemLearningPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


import dataclasses
from typing import Iterable, Mapping, Optional, Type

import numpy as np
import torch as th

from imitation.data import types
from stable_baselines3.common.torch_layers import FlattenExtractor

from imitation.algorithms import sqil
from imitation.util.util import make_vec_env
import numpy as np
from stable_baselines3 import sac

from stable_baselines3.dqn.policies import QNetwork

from utils import CHECKPOINTS

import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from imitation.data import buffer, rollout, types, wrappers
from imitation.algorithms import base
from imitation.util import util
#


def time_format(sec):
    """
    Args:
        param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs, 2)


def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class IQLPolicy(ValueSystemLearningPolicy):
    def __init__(self, *args, env: ValueAlignedEnvironment, qnetwork_per_al: Dict[tuple, QNetwork] = {}, masked=True, use_checkpoints=True, state_encoder=None, squash_output=False, observation_space=None, action_space=None, **kwargs):
        super().__init__(*args, env=env, use_checkpoints=use_checkpoints, state_encoder=state_encoder,
                         squash_output=squash_output, observation_space=observation_space, action_space=action_space, **kwargs)
        self.masked = masked
        if isinstance(self.env, gym.Wrapper):
            possibly_unwrapped = self.env.unwrapped
        else:
            possibly_unwrapped = self.env
        if isinstance(possibly_unwrapped, base_envs.ResettablePOMDP) or masked:
            self.env = base_envs.ExposePOMDPStateWrapper(self.env)

            self.env_is_tabular = True
        else:
            self.env_is_tabular = False
        assert isinstance(self.env, base_envs.ExposePOMDPStateWrapper)
        self.qnetworks_per_alignment_func = qnetwork_per_al

    def set_qnetwork_for_va(self, alignment, qnetwork):
        self.qnetworks_per_alignment_func[alignment] = qnetwork

    def act(self, state_obs, policy_state=None, exploration=0, stochastic=True, alignment_function=None):

        states = util.safe_to_tensor(state_obs, device=self.device)
        q_values = self.qnetworks_per_alignment_func[alignment_function](
            states.detach()).detach()

        valid_actions = self.env.valid_actions(state_obs, alignment_function)
        if not stochastic:
            action = torch.argmax(q_values).item()
        else:
            # print(th.softmax(q_values, dim=0), np.sum(th.softmax(q_values, dim=0).float().detach().numpy()))
            action = np.random.choice(len(q_values), p=th.softmax(
                q_values, dim=0).float().detach().numpy())

        if action not in valid_actions:
            if not stochastic:
                action = valid_actions[torch.argmax(q_values[valid_actions]).item()]
            else:
                action = np.random.choice(valid_actions, p=th.softmax(
                    q_values[valid_actions], dim=0).float().detach().numpy())

        return action, policy_state


class IQLAgent(base.DemonstrationAlgorithm[types.Transitions]):
    @property
    def policy(self):
        return self.learned_iql_policy  # TODO

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        self._demo_data_loader = base.make_data_loader(
            demonstrations,
            self.batch_size,
        )
        self._endless_expert_iterator = util.endless_iter(
            self._demo_data_loader)
        self.expert_transitions = rollout.flatten_trajectories_with_rew(
            demonstrations)

    def _next_expert_batch(self, as_torch=True) -> Mapping:
        assert self._endless_expert_iterator is not None
        batch = next(self._endless_expert_iterator)
        if as_torch:
            for key, val in batch.items():
                if isinstance(val, np.ndarray):
                    batch[key] = util.safe_to_tensor(val)
        return batch

    def __init__(self, ref_env: ValueAlignedEnvironment,
                 demos,
                 discount_factor,
                 observation_space, action_space,
                 current_net: AbstractVSLRewardFunction,
                 use_ddqn=True,
                 locexp=None,  # TODO ??
                 clip=None,  # TODO ??
                 batch_size=1000,  # TODO ??
                 tau=4,
                 logger=None,
                 iql=None,
                 alignment=None,
                 q_optimizer_kwargs={
                     'lr': 0.01,
                     'weight_decay': 0.0
                 },
                 pred_optimizer_kwargs={
                     'lr': 0.01,
                     'weight_decay': 0.0
                 },
                 rew_optimizer_kwargs={
                     'lr': 0.01,
                     'weight_decay': 0.0
                 },
                 qnetwork_kwargs={
                     'net_arch': [],
                     'activation_fn': torch.nn.Tanh,
                     'features_extractor': FlattenExtractor,
                     'features_extractor_kwargs': {}

                 },
                 pred_network_kwargs={
                     'net_arch': [],
                     'activation_fn': torch.nn.Sigmoid,
                     'features_extractor_class': FlattenExtractor,
                     'features_extractor_kwargs': {}

                 }
                 ):

        qnetwork_kwargs = deepcopy(qnetwork_kwargs)
        pred_network_kwargs = deepcopy(pred_network_kwargs)

        q_feature_extractor = qnetwork_kwargs['features_extractor_class'](
            observation_space, **qnetwork_kwargs['features_extractor_kwargs'])
        pred_feature_extractor = pred_network_kwargs['features_extractor_class'](
            observation_space, **pred_network_kwargs['features_extractor_kwargs'])
        qnetwork_kwargs['features_extractor'] = q_feature_extractor
        pred_network_kwargs['features_extractor'] = pred_feature_extractor
        qnetwork_kwargs.pop('features_extractor_class', None)
        qnetwork_kwargs.pop('features_extractor_kwargs', None)
        pred_network_kwargs.pop('features_extractor_class', None)
        pred_network_kwargs.pop('features_extractor_kwargs', None)

        self.iql = iql
        self.alignment = alignment

        q_features_dim = q_feature_extractor.features_dim  # Necessary.
        pred_features_dim = pred_feature_extractor.features_dim  # Necessary.

        self.exp_name = locexp
        self.logger = logger
        self.clip = clip
        self.device = current_net.device

        """self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        print("Use device {}".format(self.device))"""
        self.double_dqn = use_ddqn
        self.batch_size = batch_size
        self.q_optimizer_kwargs = q_optimizer_kwargs
        self.reward_optimizer_kwargs = rew_optimizer_kwargs
        self.pred_optimizer_kwargs = pred_optimizer_kwargs

        self.tau = tau
        self.gamma = discount_factor

        print(qnetwork_kwargs)

        self.qnetwork_local = QNetwork(observation_space=observation_space, action_space=action_space,
                                       features_dim=q_features_dim, **qnetwork_kwargs).to(self.device)
        self.qnetwork_target = QNetwork(observation_space=observation_space, action_space=action_space,
                                        features_dim=q_features_dim, **qnetwork_kwargs).to(self.device)
        self.optimizer = th.optim.Adam(
            self.qnetwork_local.parameters(), **self.q_optimizer_kwargs)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1)
        self.qshift_network_local = QNetwork(observation_space=observation_space, action_space=action_space,
                                             features_dim=q_features_dim, **qnetwork_kwargs).to(self.device)
        self.qshift_network_target = QNetwork(
            observation_space=observation_space, action_space=action_space, features_dim=q_features_dim, **qnetwork_kwargs).to(self.device)
        self.optimizer_shift = th.optim.Adam(
            self.qshift_network_local.parameters(), **self.q_optimizer_kwargs)
        self.soft_update(self.qshift_network_local,
                         self.qshift_network_target, 1)

        self.action_size = self.qshift_network_local.action_space.n

        self.R_local = current_net
        self.R_target = current_net.copy()
        self.optimizer_r = th.optim.Adam(
            self.R_local.parameters(), **self.reward_optimizer_kwargs)

        self.soft_update(self.R_local, self.R_target, 1)
        self.predicter = QNetwork(observation_space=observation_space, action_space=action_space,
                                  features_dim=pred_features_dim, **pred_network_kwargs).to(self.device)
        self.optimizer_pre = th.optim.Adam(
            self.predicter.parameters(), **self.pred_optimizer_kwargs)
        pathname = "lr_{}_batch_size_{}".format(
            self.q_optimizer_kwargs['lr'], self.batch_size)
        pathname += "_clip_{}".format(self.clip)
        pathname += "_tau_{}".format(self.tau)
        now = datetime.datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        pathname += dt_string
        self.steps = 0
        tensorboard_name = str(locexp) + '/runs/' + pathname
        self.vid_path = str(locexp) + '/vid'
        self.writer = SummaryWriter(tensorboard_name)
        self.average_prediction = deque(maxlen=100)
        self.average_same_action = deque(maxlen=100)
        self.all_actions = []
        for a in range(self.action_size):
            action = torch.Tensor(1) * 0 + a
            self.all_actions.append(action.to(self.device))

        self.learned_iql_policy = IQLPolicy(env=ref_env, qnetwork_per_al={
                                            self.iql.current_target: self.qnetwork_local})
        super().__init__(demonstrations=demos,
                         custom_logger=self.logger, allow_variable_horizon=True)

    def learn(self):
        logging.debug(
            "--------------------------New update-----------------------------------------------")
        expert_batch = self._next_expert_batch(as_torch=True)
        states, next_states, actions, dones = expert_batch['obs'], expert_batch[
            'next_obs'], expert_batch['acts'], expert_batch['dones']
        detached_states = states.detach()
        detached_next_states = next_states.detach()
        self.steps += 1

        # import pdb; pdb.set_trace()
        actions = torch.randint(
            0, self.action_size, (self.batch_size, 1), dtype=torch.int64, device=self.device)
        self.compute_shift_function(
            detached_states, detached_next_states, actions, dones)
        state_visitation_prediction = self.state_action_frq(states, actions)

        self.compute_r_function(detached_states, actions,
                                next_states, state_visitation_prediction)
        self.compute_q_function(
            detached_states, detached_next_states, actions, dones)
        self.soft_update(self.R_local, self.R_target, self.tau)
        self.soft_update(self.qshift_network_local,
                         self.qshift_network_target, self.tau)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return

    def compute_q_function(self, states, next_states, actions, dones):
        """Update value parameters using given batch of experience tuples. """
        actions = actions.type(torch.int64)
        # Get max predicted Q values (for next states) from target model
        if self.double_dqn:
            q_values = self.qnetwork_local(next_states).detach()
            _, best_action = q_values.max(1)
            best_action = best_action.unsqueeze(1)
            max_a_Q_target_next_state = self.qnetwork_target(
                next_states).detach()
            max_a_Q_target_next_state = max_a_Q_target_next_state.gather(
                1, best_action)
        else:
            max_a_Q_target_next_state = self.qnetwork_target(
                next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        # Get expected Q values from local model
        # Compute loss
        # rewards_old = self.R_target(states).detach().gather(1, actions.detach()).squeeze(0)
        self.prev_net = self.iql.current_net
        self.iql.current_net = self.R_target
        rewards, _ = self.iql.calculate_rewards(
            self.alignment,
            grounding=None,
            obs_mat=states,
            action_mat=actions,
            next_state_obs_mat=next_states,
            reward_mode=self.iql.training_mode,
            recover_previous_config_after_calculation=False,
            use_probabilistic_reward=False, requires_grad=True)
        self.iql.current_net = self.prev_net
        rewards = rewards.unsqueeze(1)
        print("R", rewards.shape)
        print("Q", max_a_Q_target_next_state.shape)
        assert rewards.shape == max_a_Q_target_next_state.shape
        # TODO WHY THE DONES PART? TO TEST.
        # * (1 - dones.int())
        y_i_Q = rewards + (self.gamma * max_a_Q_target_next_state)
        Q_local = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_local, y_i_Q.detach())
        # Get max predicted Q values (for next states) from target model
        self.writer.add_scalar('Q_loss', loss, self.steps)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()

    def compute_shift_function(self, states, next_states, actions, dones):
        """Update Q shift parameters using given batch of experience tuples  """
        actions = actions.type(torch.int64)
        with torch.no_grad():
            # Get max predicted Q values (for next states) from target model
            if self.double_dqn:
                q_shift = self.qshift_network_local(next_states)
                max_q, max_actions = q_shift.max(1)
                Q_targets_next = self.qshift_network_target(
                    next_states).gather(1, max_actions.unsqueeze(1))
            else:
                Q_targets_next = self.qshift_network_target(
                    next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
            y_i_Sh = self.gamma * Q_targets_next
            """print("YISH", y_i_Sh, Q_targets_next, self.qshift_network_target(
                    next_states).detach())"""

        # Get expected Q values from local model
        Q_expected = self.qshift_network_local(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(Q_expected, y_i_Sh.detach())
        # Minimize the loss
        self.optimizer_shift.zero_grad()
        loss.backward()
        self.writer.add_scalar('Shift_loss', loss, self.steps)
        self.optimizer_shift.step()

    def compute_r_function(self, states, actions, next_states, state_visitation_prediction, log=True):
        """ compute reward for the state action pair """
        actions = actions.type(torch.int64)
        # sum all other actions
        size = states.shape[0]
        idx = 0
        all_zeros = [1 for i in range(actions.shape[0])]
        zeros = False
        y_shift = self.qshift_network_target(
            states).gather(1, actions).detach()
        log_a_s = self.get_log_action_prob(states, actions).detach()
        n_S_A = log_a_s - y_shift
        mean_other_reward_than_A_minus_n_b_S = torch.empty(
            (size, 1), dtype=torch.float32).to(self.device)
        for i, (a, s, ns) in enumerate(zip(actions, states, next_states)):
            sum_rewards_other_than_a_minus_n_b = 0
            taken_actions = 0  # This is the n-1 denominator
            for b in self.all_actions:
                b = b.type(torch.long).unsqueeze(1)
                log_b_s = self.get_log_action_prob(s.unsqueeze(0), b)
                if log_b_s is None:
                    print("WHY IS NONE?")
                    print(a, b)
                    exit(0)
                if torch.eq(a, b) or log_b_s is None:
                    continue
                taken_actions += 1
                y_shift_b = self.qshift_network_target(
                    s.unsqueeze(0)).detach().gather(1, b).item()
                # print(log_b_s, log_b_s.data.item(), n_S_A.shape)
                # exit(0)
                n_b = log_b_s.data.item() - y_shift_b
                self.prev_net = self.iql.current_net
                self.iql.current_net = self.R_target
                if self.iql.current_net.use_next_state == False:
                    next_state_b = None
                else:
                    next_state_b = util.safe_to_tensor(
                        np.argmax(self.iql.env.transition_matrix[s.item(), b.item()])).unsqueeze(0)
                r_hat, _ = self.iql.calculate_rewards(
                    self.alignment,
                    grounding=None,
                    obs_mat=s.unsqueeze(0),
                    action_mat=b.squeeze(0),
                    next_state_obs_mat=next_state_b,
                    reward_mode=self.iql.training_mode,
                    recover_previous_config_after_calculation=False,
                    use_probabilistic_reward=False, requires_grad=False)
                self.iql.current_net = self.prev_net
                r_hat_old = self.R_target(s.unsqueeze(
                    0), b.squeeze(0), next_state_b, None)

                # print("OLD", r_hat_old)
                # print("NEW", r_hat)
                assert len(r_hat.shape) == 1

                sum_rewards_other_than_a_minus_n_b += (r_hat - n_b)

            if taken_actions == 0:
                print("NO ACTION TAKEN???")
                exit(0)
                all_zeros[idx] = 0
                zeros = True
                mean_other_reward_than_A_minus_n_b_S[idx] = 0.0
            else:
                mean_other_reward_than_A_minus_n_b_S[idx] = (
                    1. / taken_actions) * sum_rewards_other_than_a_minus_n_b
            # print(taken_actions)
            assert taken_actions == self.action_size - 1
            idx += 1
            y_r = n_S_A + mean_other_reward_than_A_minus_n_b_S  # TODO is this all reversed????
        # check if there are zeros (no update for this tuble) remove them from states and
        if zeros:
            mask = torch.BoolTensor(all_zeros)
            states = states[mask]
            actions = actions[mask]
            y_r = y_r[mask]
        # y = self.R_local(states).gather(1, actions)
        self.prev_net = self.iql.current_net
        self.iql.current_net = self.R_local
        y, _ = self.iql.calculate_rewards(self.alignment,
                                          grounding=None,
                                          obs_mat=states,
                                          action_mat=actions,
                                          next_state_obs_mat=next_states,
                                          reward_mode=self.iql.training_mode,
                                          recover_previous_config_after_calculation=False,
                                          use_probabilistic_reward=False, requires_grad=True)

        self.iql.current_net = self.R_local
        if log:
            text = "Action {:.2f} r target {:.2f} =  n_a {:.2f} + n_b {:.2f}  y {:.2f}".format(
                actions[0].item(), y_r[0].item(), n_S_A[0].item(), mean_other_reward_than_A_minus_n_b_S[0].item(), y[0].item())
            logging.debug(text)
        # TODO You  need to reverse this???
        r_loss = F.mse_loss(y, y_r.detach().squeeze(0))
        # Minimize the loss
        print(y[0:10])
        print(y_r[0:10].squeeze(0).detach())
        print("LOSS", r_loss)

        # print_tensor_and_grad_fn(y.grad_fn)

        print(self.optimizer_r.param_groups)

        self.optimizer_r.zero_grad()
        r_loss.backward()
        print("GRAD RLOCAL?")
        for p in self.R_local.parameters():
            print(p.grad)
            assert p.grad is not None  # for type checker
        print("GRAD RTARGET?")
        for p in self.R_target.parameters():
            print(p.grad)
            assert p.grad is None  # for type checker

        # torch.nn.utils.clip_grad_norm_(self.R_local.parameters(), 5)
        self.optimizer_r.step()
        self.iql.current_net = self.R_local
        self.writer.add_scalar('Reward_loss', r_loss, self.steps)

    def get_log_action_prob(self, states, actions):
        """ compute prob for state action pair """
        actions = actions.type(torch.long)
        # check if action prob is zero
        output = self.predicter(states)
        output = F.softmax(output, dim=1)
        action_prob = output.gather(1, actions)
        action_prob = action_prob + torch.finfo(torch.float32).eps
        # check if one action if its to small
        if action_prob.shape[0] == 1:
            if action_prob.cpu().detach().numpy()[0][0] < 1e-4:
                return None
        action_log = torch.log(action_prob)
        action_log = torch.clamp(action_prob, min=self.clip, max=0)
        return action_log

    def state_action_frq(self, states, action):
        """ Train classifer to compute state action freq """
        self.predicter.train()
        output = self.predicter(states)
        output = output.squeeze(0)
        y = action.type(torch.long).squeeze(1)
        soft_p = th.softmax(output, dim=1)
        assert soft_p.shape[0] == y.shape[0]
        th.testing.assert_close(th.sum(soft_p, axis=1),
                                th.ones((len(action), )))

        #loss = th.nn.CrossEntropyLoss()(soft_p, y)
        # TODO Maybe the previous cross entropy formula is not the same as Kalweit's, check that
        
        m = soft_p.shape[0]
        correct_scores = soft_p[torch.arange(m), y]  # Shape: (100,)

        # Compute log-sum-exp for incorrect actions: log ∑_{j ≠ i} exp(ρ(s_i, a_j))
        rho_for_logsumexp = soft_p.clone()  # Clone to avoid modifying the original tensor
        rho_for_logsumexp[torch.arange(m), y] = float('-inf')
        log_sum_exp_incorrect = torch.logsumexp(rho_for_logsumexp, dim=1)  # Shape: (100,)

        # Compute the final loss
        loss = (-correct_scores + log_sum_exp_incorrect).mean()

        #loss_2 = th.mean(-soft_p[states,action] + th.log(th.sum(th.exp(soft_p[states]), dim=1), dim=0))
        # TODO ver esto pero no creo que ayude
        
        self.optimizer_pre.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.predicter.parameters(), 1)
        self.optimizer_pre.step()
        self.writer.add_scalar('Predict_loss', loss, self.steps)
        self.predicter.eval()
        return output

    def test_predicter(self):
        """ Test the classifier """
        self.predicter.eval()
        same_state_predition = 0
        for i in range(len(self.expert_transitions.obs)):

            states = self.expert_transitions.obs[i]
            actions = self.expert_transitions.acts[i]
            states = util.safe_to_tensor(
                states, device=self.device).unsqueeze(0)
            actions = util.safe_to_tensor(actions, device=self.device)
            output = self.predicter(states)
            output = F.softmax(output, dim=1)
            # create one hot encode y from actions
            y = actions.type(torch.long).item()
            p = torch.argmax(output.data).item()
            if y == p:
                same_state_predition += 1
        text = "Same prediction {} of {} ".format(
            same_state_predition, len(self.expert_transitions.obs))
        print(text)
        self.logger.debug(text)

    def soft_update(self, local_model, target_model, tau=0.001):
        print("TAU", tau)
        """
        Soft update model parameters.

        θ_target = τ * θ_local + (1 - τ) * θ_target

        Args:
            local_model (torch.nn.Module): Model whose weights will be copied from.
            target_model (torch.nn.Module): Model whose weights will be updated.
            tau (float): Interpolation parameter for soft update.
        """
        with torch.no_grad():
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.copy_(tau * local_param +
                                   (1.0 - tau) * target_param)

    def load(self, filename):
        self.predicter.load_state_dict(torch.load(filename + "_predicter.pth"))
        self.optimizer_pre.load_state_dict(
            torch.load(filename + "_predicter_optimizer.pth"))
        self.R_local.load_state_dict(torch.load(filename + "_r_net.pth"))
        self.qnetwork_local.load_state_dict(
            torch.load(filename + "_q_net.pth"))
        print("Load models to {}".format(filename))

    def save(self, filename):
        """
        """
        mkdir("", filename)
        torch.save(self.predicter.state_dict(), filename + "_predicter.pth")
        torch.save(self.optimizer_pre.state_dict(),
                   filename + "_predicter_optimizer.pth")
        torch.save(self.qnetwork_local.state_dict(), filename + "_q_net.pth")
        torch.save(self.optimizer.state_dict(),
                   filename + "_q_net_optimizer.pth")
        torch.save(self.R_local.state_dict(), filename + "_r_net.pth")
        torch.save(self.qshift_network_local.state_dict(),
                   filename + "_q_shift_net.pth")
        print("save models to {}".format(filename))

    def test_q_value(self):
        test_elements = len(self.expert_transitions.obs)
        all_diff = 0
        error = True
        used_elements_r = 0
        used_elements_q = 0
        r_error = 0
        q_error = 0
        self.iql.current_net = self.R_local

        for i in np.random.choice(test_elements, size=500, replace=False):
            states = self.expert_transitions.obs[i]
            actions = self.expert_transitions.acts[i]
            next_state = self.expert_transitions.next_obs[i]
            states = util.safe_to_tensor(states, device=self.device)
            actions = util.safe_to_tensor(actions, device=self.device)
            next_state = util.safe_to_tensor(next_state, device=self.device)
            one_hot = torch.Tensor(
                [0 for i in range(self.action_size)], device="cpu")
            one_hot[actions.item()] = 1
            with torch.no_grad():

                r_values, _ = self.iql.calculate_rewards(self.alignment,
                                                         grounding=None,
                                                         obs_mat=th.repeat_interleave(
                                                             states, len(self.all_actions)).float(),
                                                         action_mat=th.eye(
                                                             self.action_size).float(),
                                                         next_state_obs_mat=th.repeat_interleave(
                                                             next_state, len(self.all_actions)).float(),
                                                         reward_mode=self.iql.training_mode,
                                                         recover_previous_config_after_calculation=True,
                                                         use_probabilistic_reward=False, requires_grad=False)

                # r_values = self.R_local(states.detach()).detach()
                q_values = self.qnetwork_local(states.detach()).detach()
                # print("RV", r_values)
                soft_r = F.softmax(r_values, dim=0).to("cpu")
                soft_q = F.softmax(q_values, dim=0).to("cpu")
                actions = actions.type(torch.int64)
                kl_q = F.kl_div(soft_q.log(), one_hot, None, None, 'sum')
                kl_r = F.kl_div(soft_r.log(), one_hot, None, None, 'sum')
                if kl_r == float("inf"):
                    pass
                else:
                    r_error += kl_r
                    used_elements_r += 1
                if kl_q == float("inf"):
                    pass
                else:
                    q_error += kl_q
                    used_elements_q += 1

        average_q_kl = q_error / used_elements_q
        average_r_kl = r_error / used_elements_r
        text = "Kl div of Reward {} of {} elements".format(
            average_q_kl, used_elements_r)
        print(text)
        text = "Kl div of Q_values {} of {} elements".format(
            average_r_kl, used_elements_q)
        print(text)
        self.writer.add_scalar('KL_reward', average_r_kl, self.steps)
        self.writer.add_scalar('KL_q_values', average_q_kl, self.steps)

    def eval_policy(self, ref_env: ValueAlignedEnvironment, record=False, eval_episodes=4):
        run_name = f"{ref_env.class_name()}__{self.exp_name}__{int(time.time())}"

        average_reward = 0
        scores_window = deque(maxlen=100)
        s = 0
        self.learned_iql_policy.set_qnetwork_for_va(
            self.iql.current_target, qnetwork=self.qnetwork_local)
        trajs = self.learned_iql_policy.obtain_trajectories(n_seeds=eval_episodes, seed=0, stochastic=self.iql.learn_stochastic_policy,
                                                            repeat_per_seed=10 if self.iql.learn_stochastic_policy else 1, t_max=self.learned_iql_policy.env.horizon,
                                                            exploration=0, with_reward=True, align_funcs_in_policy=[self.iql.current_target], alignments_in_env=[self.iql.current_target])
        for t in trajs:
            rt = rollout.discounted_sum(t.rews, gamma=self.gamma)
            scores_window.append(rt)
        average_reward = np.mean(scores_window)
        print("LEARNED LOCAL", self.R_local.get_learned_align_function())
        print("LEARNED TARGET", self.R_target.get_learned_align_function())

        print("TARGET", self.iql.current_target)
        print("Eval Episode {}  average Reward {} ".format(
            eval_episodes, average_reward))
        self.writer.add_scalar('Eval_reward', average_reward, self.steps)


class DeepInverseQLearning(BaseVSLAlgorithm):
    def set_demonstrations(self, demonstrations: Union[Iterable[types.Trajectory], Iterable[types.TransitionMapping], types.TransitionsMinimal]) -> None:
        pass

    def __init__(
        self,
        env: Union[TabularVAMDP, ValueAlignedEnvironment],
        reward_net: AbstractVSLRewardFunction,
        vgl_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        vsi_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        vgl_optimizer_kwargs: Optional[Mapping[str, Any]] = dict(
            lr=0.01, weight_decay=0.0
        ),
        vsi_optimizer_kwargs: Optional[Mapping[str, Any]] = dict(
            lr=0.01, weight_decay=0.0
        ),
        discount: float = 1.0,
        log_interval: Optional[int] = 100,
        vgl_expert_policy: Optional[ValueSystemLearningPolicy] = None,
        vsi_expert_policy: Optional[ValueSystemLearningPolicy] = None,
        vgl_expert_sampler=None,
        vsi_expert_sampler=None,
        # A Society or other mechanism might return different alignment functions at different times.
        target_align_func_sampler=lambda *args: args[0],


        vsi_target_align_funcs=[],

        vgl_target_align_funcs=[],
        learn_stochastic_policy=True,

        training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
        # NEW parameters

        qnetwork_kwargs={
            'net_arch': [],
            'activation_fn': torch.nn.Identity,
            'features_extractor': FlattenExtractor,

        },
        pred_network_kwargs=None,

        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,

    ) -> None:
        super().__init__(env, reward_net=reward_net,
                         vgl_optimizer_cls=vgl_optimizer_cls,
                         vsi_optimizer_cls=vsi_optimizer_cls, vgl_optimizer_kwargs=vgl_optimizer_kwargs,
                         vsi_optimizer_kwargs=vsi_optimizer_kwargs, discount=discount,
                         log_interval=log_interval, vgl_expert_policy=vgl_expert_policy,
                         vsi_expert_policy=vsi_expert_policy, target_align_func_sampler=target_align_func_sampler,
                         vsi_target_align_funcs=vsi_target_align_funcs, vgl_target_align_funcs=vgl_target_align_funcs,
                         learn_stochastic_policy=learn_stochastic_policy, training_mode=training_mode,
                         custom_logger=custom_logger)
        self.vgl_sampler = vgl_expert_sampler
        self.vsi_sampler = vsi_expert_sampler
        # learner_kwargs['gamma'] = 0.95

        self.qnetwork_kwargs = qnetwork_kwargs
        self.pred_network_kwargs = deepcopy(
            qnetwork_kwargs) if pred_network_kwargs is None else pred_network_kwargs

        """self.masked_policy = masked_policy"""
        self.current_target = None

        """self.learned_policy_per_va = LearnerValueSystemLearningPolicy(env=self.env, learner_class=self.learner_class, masked=self.masked_policy,
                                                                      policy_class=self.policy_class, learner_kwargs=self.learner_kwargs, policy_kwargs=self.policy_kwargs, state_encoder=lambda obs, info: obs)
        self.expert_learned_policy_per_va = LearnerValueSystemLearningPolicy(env=self.env, learner_class=self.learner_class, masked=self.masked_policy,
                                                                             policy_class=self.policy_class, learner_kwargs=self.learner_kwargs, policy_kwargs=self.policy_kwargs, state_encoder=lambda obs, info: obs)
        """

    def get_metrics(self):
        return {}

    def train_callback(self, t):
        # pass
        return

    def test_accuracy_for_align_funcs(self, learned_rewards_nets_per_rep,
                                      target_align_funcs_to_learned_align_funcs=None,
                                      testing_align_funcs=[]):
        print("TEST_ACC",
              learned_rewards_nets_per_rep)
        self.learned_policy_per_va
        raise NotImplementedError("test_accuracy_for_align_funcs")

        pass

    def get_tabular_policy_from_reward_per_align_func(self, align_funcs, reward_net=None):
        raise NotImplementedError("get_policy_from_reward_per_align_func")
        pass

    """def state_action_callable_reward_from_computed_rewards_per_target_align_func(self, rewards_per_target_align_func: Union[Dict, Callable]):
        if isinstance(rewards_per_target_align_func, dict):
            def rewards_per_target_align_func_callable(
                al_f): return rewards_per_target_align_func[al_f]
        else:
            return rewards_per_target_align_func_callable"""

    def calculate_learned_policies(self, target_align_funcs) -> ValueSystemLearningPolicy:
        return self.learned_policy_per_va

    def train(self, max_iter=1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING,
              assumed_grounding=None, n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_probabilistic_reward=False,
              n_reward_reps_if_probabilistic_reward=10,
              demo_batch_size=512,  # 512
              q_optimizer_kwargs={
                  'lr': 0.01,
                  'weight_decay': 0.0
              },
              pred_optimizer_kwargs={
                  'lr': 0.01,
                  'weight_decay': 0.0
              },
              **kwargs
              ):
        self.demo_batch_size = demo_batch_size
        self.q_optimizer_kwargs = q_optimizer_kwargs
        self.pred_optimizer_kwargs = pred_optimizer_kwargs

        return super().train(max_iter, mode, assumed_grounding, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, use_probabilistic_reward, n_reward_reps_if_probabilistic_reward, **kwargs)

    def train_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func) -> Dict[Any, AbstractVSLRewardFunction]:

        self.current_net.set_mode(
            mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION)

        # target_align_func = (0.0,0.0,1.0)
        # target_align_func = (1.0,0.0,0.0)
        # target_align_func = (0.0,1.0,0.0)
        # 3target_align_func = (0.33, 0.67, 0.0)
        # TODO: TEST AFTER WITH THE REST OF ALIGN FUNCS
        target_align_func = (0.4, 0.6)
        self.current_target = target_align_func

        if self.learn_stochastic_policy:

            demos = self.vsi_sampler(
                [target_align_func], n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)
            print(len(demos), n_sampled_trajs_per_seed,
                  n_seeds_for_sampled_trajectories)
        else:
            demos = self.vsi_sampler(
                [target_align_func], n_seeds_for_sampled_trajectories*n_sampled_trajs_per_seed, 1)
            print(len(demos), 1,
                  n_seeds_for_sampled_trajectories*n_sampled_trajs_per_seed)

        env = self.env
        venv = DummyVecEnv([lambda: env])

        self.current_net.set_alignment_function(target_align_func)
        self.current_net.set_mode(TrainingModes.VALUE_SYSTEM_IDENTIFICATION)

        tau = 0.01

        iql_agent = IQLAgent(
            ref_env=self.env,
            iql=self,
            alignment=self.current_target if self.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING else None,
            demos=demos,
            discount_factor=self.discount,
            observation_space=self.env.state_space,
            action_space=self.env.action_space,
            current_net=self.current_net,
            use_ddqn=False,
            locexp='locexp',
            clip=-10000.0,  # ???
            logger=self.logger,
            batch_size=self.demo_batch_size,
            tau=tau,
            qnetwork_kwargs=self.qnetwork_kwargs,
            pred_network_kwargs=self.pred_network_kwargs,
            q_optimizer_kwargs=self.q_optimizer_kwargs,
            pred_optimizer_kwargs=self.pred_optimizer_kwargs,
            rew_optimizer_kwargs=self.vsi_optimizer_kwargs
        )

        env.reset(seed=26)
        venv.seed(26)
        before_al = deepcopy(self.current_net.get_learned_align_function())
        pprint.pprint(vars(iql_agent))
        print(before_al)
        # exit(0)  # TODO: Seguir aqui.
        # Now you should try to learn as in train_iql.py in #https://github.com/ChrisProgramming2018/IQL_Lunar_Lander/tree/main
        save_models_path = os.path.join(CHECKPOINTS, 'iql/')
        t0 = 0

        for t in range(max_iter):
            text = "Train Predicter {}  \ {}  time {}  \r".format(
                t, max_iter, time_format(time.time() - t0))
            print(text, end='')
            iql_agent.learn()
            if t % int(self.log_interval) == 0:
                print(text)
                print(tuple(
                    [float(f"{a:0.5f}") for a in
                     self.current_net.get_learned_align_function()]))

                iql_agent.save(save_models_path + "/models/{}-".format(t))
                # agent.test_predicter(memory)
                iql_agent.test_q_value()
                iql_agent.eval_policy(ref_env=self.env, eval_episodes=50)

        return self.current_net.get_learned_align_function()

    def train_vsl_probabilistic(self, max_iter,
                                n_seeds_for_sampled_trajectories,
                                n_sampled_trajs_per_seed,
                                n_reward_reps_if_probabilistic_reward,
                                target_align_func):
        pass

    def train_vgl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed) -> Dict[Any, AbstractVSLRewardFunction]:
        pass




class TabularInverseQLearning(BaseTabularMDPVSLAlgorithm):

    def __init__(self, env, reward_net, vgl_optimizer_cls=th.optim.Adam, vsi_optimizer_cls=th.optim.Adam, vgl_optimizer_kwargs={'lr': 0.0001, 'weight_decay': 0.0},
                 vsi_optimizer_kwargs={'lr': 0.0001, 'weight_decay': 0.0}, discount=1.0, log_interval=1, vgl_expert_policy=None, vsi_expert_policy=None,
                 target_align_func_sampler=..., vsi_target_align_funcs=..., vgl_target_align_funcs=...,
                 learn_stochastic_policy=True,
                 environment_is_stochastic=False,
                 vgl_expert_sampler=None,
                 vsi_expert_sampler=None,
                 training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                 stochastic_expert=True,
                 approximator_kwargs=...,
                 policy_approximator=PolicyApproximators.MCE_ORIGINAL,
                 *, custom_logger=None):

        super().__init__(env, reward_net, vgl_optimizer_cls, vsi_optimizer_cls, vgl_optimizer_kwargs, vsi_optimizer_kwargs, discount, log_interval, vgl_expert_policy, vsi_expert_policy, target_align_func_sampler,
                         vsi_target_align_funcs, vgl_target_align_funcs, learn_stochastic_policy, environment_is_stochastic, training_mode, stochastic_expert, approximator_kwargs, policy_approximator, custom_logger=custom_logger)
        self.vgl_sampler = vgl_expert_sampler
        self.vsi_sampler = vsi_expert_sampler

        self.learned_policy_per_va = VAlignedDictDiscreteStateActionPolicyTabularMDP(
            {}, env=self.env, state_encoder=lambda exposed_state, info: exposed_state)  # Starts with random policy
        for al_func in self.vgl_target_align_funcs:
            probability_matrix = np.random.rand(
                self.env.state_dim, self.env.action_dim)
            random_pol = probability_matrix / \
                probability_matrix.sum(axis=1, keepdims=True)
            self.learned_policy_per_va.set_policy_for_va(al_func, random_pol)
        """self.masked_policy = masked_policy"""

    def train(self, max_iter = 100, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None, 
              n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_probabilistic_reward=False, 
              n_reward_reps_if_probabilistic_reward=10, 
              alpha_q=0.01,
              alpha_sh=0.01, 
              optimization_batch_size=None, **kwargs):
        self.alpha_q = alpha_q
        self.alpha_sh = alpha_sh
        self.batch_size = n_seeds_for_sampled_trajectories
        self.rep_per_batch = n_sampled_trajs_per_seed
        self.iterations = max_iter
        self.optimization_batch_size = optimization_batch_size
        return super().train(max_iter, mode, assumed_grounding, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, use_probabilistic_reward, n_reward_reps_if_probabilistic_reward, **kwargs)
    
    def train_global(self, target=None):
        batches = self.vsi_sampler([target], self.batch_size, self.rep_per_batch)
        q, r, boltzman = self._inverse_q_learning(self.torch_obs_mat, nA=self.env.action_dim, gamma=self.discount, trajectories=batches, 
                                            epochs=self.iterations, alpha_r=self.vsi_optimizer_kwargs['lr'], alpha_q=self.alpha_q,alpha_sh=self.alpha_sh,target=target)
        print(q.shape, r.shape, boltzman.shape)
        print(boltzman)
        self.learned_policy_per_va.set_policy_for_va(target, boltzman)
        exit(0)
    def train_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func):
        
        super().train_vsl(max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func)
        for target in self.vsi_target_align_funcs:
            target = (0.6,0.4)
            #target = (0.0,1.0)
            #target = (0.67, 0.33, 0.0)
            self.train_global(target)
        print("finish train")

    def train_vgl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed):
        return super().train_vgl(max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)
    
    def calculate_learned_policies(self, target_align_funcs):
        return self.learned_policy_per_va
    
    def _inverse_q_learning(self, feature_matrix, nA, gamma, trajectories, alpha_r, alpha_q, alpha_sh, epochs, epsilon=1e-8, target=None):
        """
        Implementation of IQL from Deep Inverse Q-learning with Constraints. Gabriel Kalweit, Maria Huegle, Moritz Wehrling and Joschka Boedecker. NeurIPS 2020.
        Arxiv : https://arxiv.org/abs/2008.01712
        """
        nS = feature_matrix.shape[0]
        optimizer = self.vsi_optimizer if self.training_mode==TrainingModes.VALUE_SYSTEM_IDENTIFICATION else self.vgl_optimizer
        
        # initialize tables for reward function, value functions and state-action visitation counter.
        r = np.zeros((nS, nA))
        _, r = self.calculate_rewards(align_func=target if self.training_mode==TrainingModes.VALUE_GROUNDING_LEARNING else None,
                                        obs_mat=self.torch_obs_mat, action_mat=self.torch_action_mat,
                                            obs_action_mat=self.torch_obs_action_mat,
                                            reward_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                            recover_previous_config_after_calculation=False, requires_grad=False)
        
        q = np.zeros((nS, nA))
        q = deepcopy(r)
        q_sh = np.zeros((nS, nA))
        q_sh = deepcopy(r)
        state_action_visitation = np.zeros((nS, nA))

        trans = rollout.flatten_trajectories_with_rew(trajectories)
        self.optimization_batch_size = 1000
        idx  = np.arange( 0, self.optimization_batch_size, step=1)
        sta_known = self.env.get_state_actions_with_known_reward(target)
        real_matrix = self.env.reward_matrix_per_align_func(target)
        r[sta_known] = real_matrix[sta_known]
        for i in range(epochs):
            if i% (len(trans) // self.optimization_batch_size) == 0:
                #random.shuffle(trajectories)
                #state_action_visitation = np.zeros((nS, nA))
                #trans = rollout.flatten_trajectories_with_rew(trajectories)
                pass
            if i % 10 == 0:
                
                print("Epoch %s/%s" % (i+1, epochs))
            if self.optimization_batch_size is None:
                obs_batch, act_batch, next_obs_batch, dones_batch = trans.obs, trans.acts, trans.next_obs, trans.dones
            else:
                assert self.optimization_batch_size <= len(trans)
                idx  = (idx + self.optimization_batch_size) % len(trans)

                #print(idx)
                obs_batch, act_batch, next_obs_batch,dones_batch = trans.obs[idx], trans.acts[idx], trans.next_obs[idx], trans.dones[idx]
            
            for traj in trajectories:
                for j, (s, a, ns) in enumerate(zip(traj.obs[:-1], traj.acts, traj.obs[1:])):
            #
            
            #for (s,a,ns,d) in zip(obs_batch, act_batch, next_obs_batch, dones_batch):
                    state_action_visitation[s][a] += 1
                    other_actions = [oa for oa in self.env.valid_actions(s,align_func=target) if oa != a] # TODO? only valid ones or all?
                    if len(other_actions) == 0:
                        continue

                    
                    #r[sta_known] = real_matrix[sta_known]

                    d =  False #j== len(traj) - 1    # no terminal state


                    # compute shifted q-function.
                    q_sh[s, a] = (1-alpha_sh) * q_sh[s, a] + \
                        alpha_sh * (gamma * (1-d) * np.max(q[ns]))

                    # compute log probabilities.
                    sum_of_state_visitations = np.sum(
                        state_action_visitation[s])
                    log_prob = np.log(
                        (state_action_visitation[s]/sum_of_state_visitations) + epsilon)

                    # compute eta_a and eta_b for Eq. (9).
                    eta_a = log_prob[a] - q_sh[s][a]
                    other_actions = [oa for oa in self.env.valid_actions(s,align_func=target) if oa != a] # TODO? only valid ones or all?
                    eta_b = log_prob[other_actions] - q_sh[s][other_actions]
                    
                    sum_oa = (1/(len(other_actions))) * np.sum(r[s][other_actions] - eta_b)
                    
                    #self.current_net.set_alignment_function(self.current_net.get_learned_align_function())
                    reward_estimation = self.current_net(self.torch_obs_mat[s].unsqueeze(0), self.torch_action_mat[a,s,:] if self.current_net.use_action else None, self.torch_obs_mat[ns].unsqueeze(0), None)
                    #print(reward_estimation)
                    #assert th.allclose(reward_estimation[0][a], reward_estimation[0][0])
                    
                    loss = (reward_estimation - (eta_a + sum_oa))**2

                    #print(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    # TODO: IDEA WEIGHT WITH VISITATION COUNTS 
                    
                    
                        
                    optimizer.step()
                    r[s,a] = reward_estimation.detach().numpy()[0]
                    #r[sta_known] = self.env.reward_matrix_per_align_func(target)[sta_known]
                    #alpha_r = 0.0001
                    # update reward-function.
                    #r[s,a] = (1-alpha_r) * r[s,a] + \
                    #    alpha_r * (eta_a + sum_oa) TODO 
                    r[sta_known] = real_matrix[sta_known]
                    # update value-function.
                    q[s, a] = (1-alpha_q) * q[s, a] + alpha_q * \
                        (r[s, a] + gamma * (1-d) * np.max(q[ns]))
                    
                    #s = ns
            #state_action_visitation = state_action_visitation / (i+1)
            """for ri in range(1):
                #idx  = (idx - self.optimization_batch_size) % len(trans)

                #print(idx)
                obs_batch, act_batch, next_obs_batch,dones_batch = trans.obs[idx], trans.acts[idx], trans.next_obs[idx], trans.dones[idx]
            
                reward_estimation, _ = self.calculate_rewards(align_func=target if self.training_mode==TrainingModes.VALUE_GROUNDING_LEARNING else None,
                                                            obs_mat=self.torch_obs_mat, action_mat=self.torch_action_mat,
                                                                obs_action_mat=self.torch_obs_action_mat,
                                                                reward_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION,
                                                                recover_previous_config_after_calculation=False)
                
                print("Size", reward_estimation[obs_batch, act_batch].size())
                print("total size", reward_estimation.size())
                #loss = F.mse_loss(reward_estimation[obs_batch, act_batch], util.safe_to_tensor(r[obs_batch, act_batch], dtype=th.float32).detach())
                loss = torch.sum((reward_estimation[obs_batch, act_batch] - util.safe_to_tensor(r[obs_batch, act_batch], dtype=th.float32).detach()) ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                # TODO: IDEA WEIGHT WITH VISITATION COUNTS 
                print("GRAD RLOCAL?")
                for p in self.current_net.parameters():
                    print(p.grad)
                    assert p.grad is not None  # for type checker
                    
                optimizer.step()
                
                print("LEARNED", self.current_net.get_learned_align_function())
                print("TARGET", target)
            #alpha_r = 0.01
            r = reward_estimation.detach().numpy()"""
            print("LEARNED", self.current_net.get_learned_align_function())
            print("TARGET", target)
        # compute Boltzmann distribution.
        boltzman_distribution = []
        for s in range(nS):
            boltzman_distribution.append([])
            for a in range(nA):
                boltzman_distribution[-1].append(np.exp(q[s][a]))
        boltzman_distribution = np.array(boltzman_distribution)
        boltzman_distribution /= np.sum(boltzman_distribution,
                                        axis=1).reshape(-1, 1)
        return q, r, boltzman_distribution
    def train_vsl_probabilistic(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, n_reward_reps_if_probabilistic_reward, target_align_func):
        return super().train_vsl_probabilistic(max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, n_reward_reps_if_probabilistic_reward, target_align_func)
    