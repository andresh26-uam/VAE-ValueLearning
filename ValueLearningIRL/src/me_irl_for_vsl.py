"""
Based on https://imitation.readthedocs.io/en/latest/algorithms/mce_irl.html
Adapted for the RoadWorld environment
"""

import enum
from itertools import product
import itertools
import time
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)
from imitation.algorithms import mce_irl
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.special
import torch as th
from seals import base_envs
from imitation.algorithms import base
from imitation.data import types, rollout
from imitation.util import logger as imit_logger
from imitation.util import networks, util

from copy import deepcopy

from src.envs.firefighters_env import FeatureSelection
from src.envs.tabularVAenv import TabularVAPOMDP
from src.reward_functions import AbstractVSLearningRewardFunction, TrainingModes, ProfiledRewardFunction, print_tensor_and_grad_fn, squeeze_r

import pandas as pd

from src.vsl_policies import VAlignedDictSpaceActionPolicy, VAlignedDictDiscreteStateActionPolicyTabularMDP, VAlignedDiscreteSpaceActionPolicy, ValueSystemLearningPolicy

def mce_partition_fh(
    env: base_envs.TabularModelPOMDP,
    *,
    reward: Optional[np.ndarray] = None,
    discount: float = 1.0,
    value_iteration_tolerance = 0.001,
    use_causal_entropy=True,
    deterministic=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Performs the soft Bellman backup for a finite-horizon MDP.

    Calculates V^{soft}, Q^{soft}, and pi using recurrences (9.1), (9.2), and
    (9.3) from Ziebart (2010).

    Args:
        env: a tabular, known-dynamics MDP.
        reward: a reward matrix. Defaults to env.reward_matrix.
        discount: discount rate.

    Returns:
        (V, Q, \pi) corresponding to the soft values, Q-values and MCE policy.
        V is a 2d array, indexed V[t,s]. Q is a 3d array, indexed Q[t,s,a].
        \pi is a 3d array, indexed \pi[t,s,a].

    Raises:
        ValueError: if ``env.horizon`` is None (infinite horizon).
    """
    # shorthand
    horizon = env.horizon
    if horizon is None:
        raise ValueError("Only finite-horizon environments are supported.")
    n_states = env.state_dim
    n_actions = env.action_dim
    T = env.transition_matrix
    if reward is None:
        reward = env.reward_matrix

    
    if len(reward.shape) == 1:
        broad_R = reward[:, None]
    else:
        broad_R = reward
        assert len(reward.shape) == 2

    if use_causal_entropy:
        # Initialization
        # indexed as V[t,s]
        V = np.full((horizon, n_states), -np.inf)
        # indexed as Q[t,s,a]
        Q = np.zeros((horizon, n_states, n_actions))


        # Base case: final timestep
        # final Q(s,a) is just reward
        Q[horizon - 1, :, :] = broad_R
        # V(s) is always normalising constant
        V[horizon - 1, :] = scipy.special.logsumexp(Q[horizon - 1, :, :], axis=1)

        # Recursive case
        for t in reversed(range(horizon - 1)):
            next_values_s_a = T @ V[t + 1, :]
            Q[t, :, :] = broad_R + discount * next_values_s_a
            V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

        pi = np.exp(Q - V[:, :, None])[0] # ?? 
    else:
        # Initialization
        # indexed as V[t,s]
        V = np.full((n_states), -1)
        # indexed as Q[t,s,a]
        Q = broad_R
        err = np.inf
        iterations = 0

        
        while err > value_iteration_tolerance and (iterations < horizon):
            values_prev = V.copy()
            values_prev[env.unwrapped.goal_states] = 0
            next_values_s_a = T @ values_prev
            Q = broad_R + discount * next_values_s_a
            V = np.max(Q, axis=1)
            err = np.max(np.abs(V-values_prev))
            iterations+=1
        pi = scipy.special.softmax(Q - V[:, None], axis=1)
    
    if deterministic:
        max_values = np.max(pi, axis=1, keepdims=True)

        # Create a boolean matrix where True indicates a maximum value
        is_max = (pi == max_values)

        # Count the number of maximum values per row
        num_max_values = np.sum(is_max, axis=1, keepdims=True)

        # Create the final matrix where each maximum value is divided by the number of max values
        pi = is_max / num_max_values

        for i in range(pi.shape[0]):
            assert np.allclose(np.sum(pi[i]), 1)
        


    return V, Q, pi


def mce_occupancy_measures(
    env: base_envs.TabularModelPOMDP,
    *,
    reward: Optional[np.ndarray] = None,
    pi: Optional[np.ndarray] = None,
    discount: float = 1.0,
    value_iteration_tolerance = 0.001,
    deterministic = False,
    use_causal_entropy = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate state visitation frequency Ds for each state s under a given policy pi.

    You can get pi from `mce_partition_fh`.

    Args:
        env: a tabular MDP.
        reward: reward matrix. Defaults is env.reward_matrix.
        pi: policy to simulate. Defaults to soft-optimal policy w.r.t reward
            matrix.
        discount: rate to discount the cumulative occupancy measure D.

    Returns:
        Tuple of ``D`` (ndarray) and ``Dcum`` (ndarray). ``D`` is of shape
        ``(env.horizon, env.n_states)`` and records the probability of being in a
        given state at a given timestep. ``Dcum`` is of shape ``(env.n_states,)``
        and records the expected discounted number of times each state is visited.

    Raises:
        ValueError: if ``env.horizon`` is None (infinite horizon).
    """
    # shorthand
    horizon = env.horizon
    if horizon is None:
        raise ValueError("Only finite-horizon environments are supported.")
    n_states = env.state_dim
    n_actions = env.action_dim
    T = env.transition_matrix
    if reward is None:
        reward = env.reward_matrix
    if pi is None:
        _, _, pi = mce_partition_fh(env, reward=reward, use_causal_entropy=use_causal_entropy, discount=discount,value_iteration_tolerance = value_iteration_tolerance, deterministic=deterministic)

    D = np.zeros((horizon + 1, n_states))
    D[0, :] = env.initial_state_dist
    for t in range(horizon):
        for a in range(n_actions):
            if len(pi.shape) == 3:
                E = D[t] * pi[t, :, a]
            elif len(pi.shape) == 2:
                E = D[t] * pi[:, a]
            else:
                E = D[t] * (pi[:] == a) # TODO test this?
            D[t + 1, :] += E @ T[:, a, :]

    Dcum = rollout.discounted_sum(D, discount)
    assert isinstance(Dcum, np.ndarray)
    return D, Dcum



def get_demo_oms_from_trajectories(trajs: Iterable[types.Trajectory], state_dim, discount =1, groupby='istate_and_al') -> dict:
        demo_om = dict()
        num_demos = dict()
        for traj in trajs:
            
            orig_des_align_func = (traj.infos[0]['init_state'], traj.infos[0]['align_func'])
           
            if orig_des_align_func not in num_demos.keys():
                num_demos[orig_des_align_func] = 0
                demo_om[orig_des_align_func] = np.zeros((state_dim,))

            obs_relevant = traj.obs
            
            np.add.at(demo_om[orig_des_align_func], (
                np.asarray(obs_relevant),), 1)
            
            num_demos[orig_des_align_func] += 1
        for orig_des_align_func, num_demos_is in num_demos.items():
            demo_om[orig_des_align_func] /= num_demos_is

        if groupby == 'istate_and_al':
            return demo_om
        
        elif groupby == 'istate':
            demos_by_is = dict()
            n_demos_per_is = dict()
            for isal, demo_om in demo_om.items():
                istate, al_func = isal
                if istate not in demos_by_is:
                    n_demos_per_is[istate] = 1
                    demos_by_is[istate] = demo_om
                else:
                    n_demos_per_is[istate] += 1
                    demos_by_is[istate] += demo_om
            for istate in demos_by_is.keys():
                demos_by_is[istate] /= n_demos_per_is[istate]
            return demos_by_is
        elif groupby == 'al':
            demos_by_al = dict()
            n_demos_per_al = dict()
            for isal, demo_om in demo_om.items():
                _, al_func = isal
                if al_func not in demos_by_al:
                    n_demos_per_al[al_func] = 1
                    demos_by_al[al_func] = demo_om
                else:
                    n_demos_per_al[al_func] += 1
                    demos_by_al[al_func] += demo_om
            for al_func in demos_by_al.keys():
                demos_by_al[al_func] /= n_demos_per_al[al_func]
            return demos_by_al
        raise ValueError(groupby)


class TrainingSetModes(enum.Enum):
    PROFILED_SOCIETY = 'profiled_society' # Used for align_func learning of a society sampling trajectories according to a probabilty distribution of align_funcs.
    PROFILED_EXPERT = 'cost_model' # Default in Value Learning.

class MaxEntropyIRLForVSL(base.DemonstrationAlgorithm[types.TransitionsMinimal]):
    """
    Based on https://imitation.readthedocs.io/en/latest/algorithms/mce_irl.html
    Adapted for the RoadWorld environment
    """

    def set_reward_net(self, reward_net: AbstractVSLearningRewardFunction):

        self.reward_net = reward_net
        self.optimizer = self.optimizer_cls(self.reward_net.parameters(), **self.optimizer_kwargs)
    def get_reward_net(self):
        return self.reward_net

    def __init__(
        self,
        env: Union[TabularVAPOMDP],
        reward_net: ProfiledRewardFunction,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        discount: float = 1.0,
        mean_vc_diff_eps: float = 1e-3,
        grad_l2_eps: float = 1e-4,
        log_interval: Optional[int] = 100,
        expert_policy: Optional[ValueSystemLearningPolicy] = None,
        expert_trajectories: Optional[List[types.Trajectory]] = None,

        vsi_expert_policy: Optional[ValueSystemLearningPolicy] = None,
        vsi_expert_trajectories: Optional[List[types.Trajectory]] = None,

        vsi_target_align_funcs = 'auto',
        
        training_align_funcs = 'auto',

        demo_om_from_policy = True,
        use_causal_entropy = True,
        
        value_iteration_tolerance = 0.001,

        initial_state_distribution_train = None,
        initial_state_distribution_test = None,
        training_mode = TrainingModes.VALUE_GROUNDING_LEARNING,
        training_set_mode = TrainingSetModes.PROFILED_EXPERT,
        learn_stochastic_policy = True,
        name_method='',
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        
    ) -> None:
        
        self.discount = discount
        self.env = env
        self.use_causal_entropy = use_causal_entropy
        self.value_iteration_tolerance = value_iteration_tolerance
        self.learn_stochastic_policy = learn_stochastic_policy
        
        self.exposed_state_env = base_envs.ExposePOMDPStateWrapper(env)
        self.demo_state_om = dict()
        self.vsi_demo_state_om = dict()
        
        self.expert_trajectories = expert_trajectories # list of expert trajectories just to see different origin destinations and align_funcs
        self.expert_trajectories_per_isal = dict()
        self.expert_trajectories_per_align_func = dict()

        self.vsi_expert_trajectories = vsi_expert_trajectories # list of expert target align_func trajectories
        self.vsi_expert_trajectories_per_isal = dict()
        self.vsi_expert_trajectories_per_align_func = dict()
        
        self.initial_state_distribution_train = initial_state_distribution_train if initial_state_distribution_train is not None else env.initial_state_dist
        self.initial_state_distribution_test= initial_state_distribution_test if initial_state_distribution_test is not None else env.initial_state_dist


        
        for t in self.expert_trajectories:
            al_func = t.infos[0]['align_func']
            isal = (t.infos[0]['init_state'], al_func)
            if isal not in self.expert_trajectories_per_isal.keys():

                self.expert_trajectories_per_isal[isal] = []
            if al_func not in self.expert_trajectories_per_align_func.keys():
                self.expert_trajectories_per_align_func[al_func] = []
                

            self.expert_trajectories_per_isal[isal].append(t)
            self.expert_trajectories_per_align_func[al_func].append(t)
        for t in self.vsi_expert_trajectories:
            al_func = t.infos[0]['align_func']
            isal = (t.infos[0]['init_state'], al_func)
            if isal not in self.vsi_expert_trajectories_per_isal.keys():

                self.vsi_expert_trajectories_per_isal[isal] = []
            if al_func not in self.vsi_expert_trajectories_per_align_func.keys():
                self.vsi_expert_trajectories_per_align_func[al_func] = []
                

            self.vsi_expert_trajectories_per_isal[isal].append(t)
            self.vsi_expert_trajectories_per_align_func[al_func].append(t)
            

        
        if training_align_funcs == 'auto':
            self.training_align_funcs = list(set(tuple(t.infos[0]['align_func']) for t in self.expert_trajectories))
        else:
            self.training_align_funcs = list(set(training_align_funcs))
        if vsi_target_align_funcs == 'auto':
            self.vsi_target_align_funcs = list(set(tuple(t.infos[0]['align_func']) for t in self.vsi_expert_trajectories))
        else:
            self.vsi_target_align_funcs = list(set(vsi_target_align_funcs))

       

        super().__init__(
                demonstrations=None,
                custom_logger=custom_logger,
            )
        self.expert_policy: VAlignedDiscreteSpaceActionPolicy = expert_policy
        self.vsi_expert_policy: VAlignedDiscreteSpaceActionPolicy = vsi_expert_policy


        self.demo_om_from_policy = demo_om_from_policy
        if demo_om_from_policy:
            self._set_demo_oms_from_policy(self.expert_policy)
            self._set_vsi_demo_oms_from_policy(self.vsi_expert_policy)
        else:
            self._set_demo_oms_from_trajectories(self.expert_trajectories)
            self._set_vsi_demo_oms_from_trajectories(self.vsi_expert_trajectories)
        
        
        self.reward_net = reward_net
        self.optimizer_cls = optimizer_cls
        optimizer_kwargs = optimizer_kwargs or {"lr": 1e-2}
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer_cls(reward_net.parameters(), **optimizer_kwargs)
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        

        self.mean_vc_diff_eps = mean_vc_diff_eps
        self.grad_l2_eps = grad_l2_eps
        self.log_interval = log_interval

        self.name_method = name_method
        # Initialize policy to be uniform random. We don't use this for MCE IRL
        # training, but it gives us something to return at all times with `policy`
        # property, similar to other algorithms.
        if self.env.horizon is None:
            raise ValueError("Only finite-horizon environments are supported.")
        #ones = np.ones((self.env.state_dim, self.env.action_dim))
        #uniform_pi = ones / self.env.action_dim

        # TODO: que hacer aqui, usar el EXPOSEPOMDP state wrapper quiza...
        
        self.learned_policy_per_va = VAlignedDictDiscreteStateActionPolicyTabularMDP({},env=self.env, state_encoder=lambda exposed_state, info: exposed_state) # Starts with random policy
        #self.vi_policy.train(0.001, verbose=True, stochastic=True, custom_reward_function=lambda s,a,d: self.env.reward_matrix[self.env.netconfig[s][a]]) # alpha stands for error tolerance in value_iteration
        for al_func in self.training_align_funcs:
            probability_matrix = np.random.rand(self.env.state_dim, self.env.action_dim)
            random_pol = probability_matrix / probability_matrix.sum(axis=1, keepdims=True)
            self.learned_policy_per_va.set_policy_for_va(al_func, random_pol)

        

        self.training_mode = training_mode
        self.training_set_mode = training_set_mode

        self.reward_net.set_mode(self.training_mode)
    
    def calculate_rewards(self, align_func=None, grounding=None, obs_mat=None, action_mat = None, obs_action_mat = None, reward_mode=TrainingModes.EVAL, recover_previous_mode_after_calculation = True):
        # TODO: support for rewards that take into account actions and next states.
        if obs_mat is None:
            obs_mat = self.exposed_state_env.observation_matrix
            obs_mat = th.as_tensor(
                obs_mat,
                dtype=self.reward_net.dtype,
                device=self.reward_net.device,
            )

        if self.reward_net.use_one_hot_state_action:
            if obs_action_mat is None:
                obs_action_mat = th.as_tensor(
                        np.identity(self.env.state_dim*self.env.action_dim),
                        dtype=self.reward_net.dtype,
                        device=self.reward_net.device,
                    )
        
        if recover_previous_mode_after_calculation:
            previous_rew_mode = self.reward_net.mode

        self.reward_net.set_mode(reward_mode)
        self.reward_net.set_alignment_function(align_func)
        self.reward_net.set_grounding_function(grounding)

        
        if self.reward_net.use_action is False:
            predicted_r = squeeze_r(self.reward_net(obs_mat, action_mat, None, None))
            assert predicted_r.shape == (obs_mat.shape[0],)
            
        else:
            assert action_mat is not None
            assert action_mat.size() == (self.env.action_space.n, obs_mat.shape[0], self.env.action_space.n)
            #vecs = [self.reward_net(obs_mat, action_mat[i], None, None) for i in range(self.env.action_space.n)]
            #print("USING ONE HOT?", self.reward_net.use_one_hot_state_action)
            if self.reward_net.use_one_hot_state_action:
                predicted_r = th.reshape(self.reward_net(obs_action_mat, None, None, None), (self.env.state_dim,self.env.action_dim))
            else:
                predicted_r = th.stack([squeeze_r(self.reward_net(obs_mat, action_mat, None, None)) for i in range(self.env.action_space.n)], dim=1)
            
            assert predicted_r.shape == (obs_mat.shape[0],action_mat.shape[0])
            assert predicted_r.grad_fn is not None

                
        predicted_r_np = predicted_r.detach().cpu().numpy()
        if recover_previous_mode_after_calculation:
            self.reward_net.set_mode(previous_rew_mode)

        return predicted_r, predicted_r_np
    def mce_partition_fh_per_align_func(self, align_func, reward_matrix=None, action_mat=None, obs_action_mat=None, reward_mode=TrainingModes.VALUE_GROUNDING_LEARNING):
        # TODO: mce partition function considers only tabular rewards per state, not per (state, action, next state).
        if reward_matrix is None:
            reward_matrix_torch, reward_matrix = self.calculate_rewards(align_func=align_func, 
                                                   obs_mat=self.env.observation_matrix, 
                                                   action_mat=action_mat, 
                                                   obs_action_mat=obs_action_mat,
                                                   reward_mode=reward_mode)
        
        V,Q, pi = mce_partition_fh(env=self.exposed_state_env, reward=reward_matrix, discount=self.discount, use_causal_entropy=self.use_causal_entropy, 
                                   value_iteration_tolerance = self.value_iteration_tolerance,
                                   deterministic=not self.learn_stochastic_policy)
        
        return pi
            
       
    
    def mce_occupancy_measures(
            self,
        reward_matrix: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if pi is None:
            V,Q, pi = mce_partition_fh(self.exposed_state_env, reward=reward_matrix, discount=self.discount, use_causal_entropy=self.use_causal_entropy, 
                                       value_iteration_tolerance = self.value_iteration_tolerance,
                                       deterministic=not self.learn_stochastic_policy)
        
        D, Dcums = mce_occupancy_measures(env=self.exposed_state_env, pi=pi, 
                                          discount=self.discount, 
                                          use_causal_entropy=self.use_causal_entropy, 
                                          value_iteration_tolerance = self.value_iteration_tolerance,
                                          deterministic=not self.learn_stochastic_policy)
        return D, Dcums, pi
    from seals.diagnostics.cliff_world import CliffWorldEnv

    def train(self, max_iter: int = 1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None) -> np.ndarray:
        """Runs MCE IRL for Value System Learning.

        Args:
            max_iter: The maximum number of iterations to train for. May terminate
                earlier if `self.linf_eps` or `self.grad_l2_eps` thresholds are reached.
            mode: TrainingMode that determines whether performing Value grounding learning or Value System Learning.
            assumed_grounding: Specific grounding for Value System Learning. If None, the cur_grounding property of the reward net is used.
            
        Returns:
            State occupancy measure for the final reward function. `self.reward_net`
            and `self.optimizer` will be updated in-place during optimisation.
        """
        # use the same device and dtype as the rmodel parameters
        obs_mat = self.env.observation_matrix
        
        torch_obs_mat = th.as_tensor(
            obs_mat,
            dtype=self.reward_net.dtype,
            device=self.reward_net.device,
        )
        torch_obs_mat.requires_grad_(True)
        torch_action_mat = None
        if self.reward_net.use_action:
            actions_one_hot = th.eye(self.env.action_space.n, requires_grad=True)
            torch_action_mat = th.stack([actions_one_hot[i].repeat(obs_mat.shape[0],1) for i in range(self.env.action_space.n)], dim=0)
        
        torch_obs_action_mat = th.as_tensor(
                        np.identity(self.env.state_dim*self.env.action_dim),
                        dtype=self.reward_net.dtype,
                        device=self.reward_net.device,
                    )
        torch_obs_action_mat.requires_grad_(True)

        self.reward_net.set_mode(mode)
        if assumed_grounding is not None and mode in [TrainingModes.EVAL, TrainingModes.VALUE_SYSTEM_IDENTIFICATION]:
                self.reward_net.set_grounding_function(assumed_grounding)
        
        if mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            
            
            
            self.optimizer = self.optimizer_cls(self.reward_net.parameters(), **self.optimizer_kwargs)
            linf_delta_per_align_func =  {al : [] for al in self.training_align_funcs}
            grad_norm_per_align_func = {al : [] for al in self.training_align_funcs}
            rewards_per_align_func = {al : [] for al in self.training_align_funcs}
            with networks.training(self.reward_net):
                for t in range(max_iter):
                    for align_func in self.training_align_funcs:
                        assert self.demo_state_om[align_func] is not None
                        assert self.demo_state_om[align_func].shape == (len(obs_mat),)
                        # switch to training mode (affects dropout, normalization)
                        
                        old_reward, visitations, old_pi, loss = self._train_step(torch_obs_mat, align_func=align_func, action_mat=torch_action_mat)
                        reward_th, reward = self.calculate_rewards(align_func, 
                                                                   obs_mat=torch_obs_mat, 
                                                                   action_mat=torch_action_mat,
                                                                   obs_action_mat=torch_obs_action_mat,
                                                                   reward_mode=TrainingModes.EVAL)
                        rewards_per_align_func[align_func] = reward
                        
                        # these are just for termination conditions & debug logging
                        grads = []
                        for p in self.reward_net.parameters():
                            assert p.grad is not None  # for type checker
                            grads.append(p.grad)
                        grad_norm = util.tensor_iter_norm(grads).item()
                        abs_diff_in_visitation_counts = np.abs(self.demo_state_om[align_func] - visitations)
                        linf_delta = np.max(abs_diff_in_visitation_counts)
                        avg_linf_delta = np.mean(abs_diff_in_visitation_counts)
                        norm_reward = np.linalg.norm(reward)
                        max_error_state = np.argmax(abs_diff_in_visitation_counts)
                        
                        if self.log_interval is not None and 0 == (t % self.log_interval):
                            params = self.reward_net.parameters()
                            weight_norm = util.tensor_iter_norm(params).item()
                            self.logger.record("iteration", t)
                            self.logger.record("Align_func", align_func)
                            self.logger.record("linf_delta", linf_delta)
                            self.logger.record("weight_norm", weight_norm)
                            self.logger.record("grad_norm", grad_norm)
                            self.logger.record("loss", loss)
                            self.logger.record("avg_linf_delta", avg_linf_delta)
                            self.logger.record("norm reward", norm_reward)
                            self.logger.record("state_worse", max_error_state)
                            self.logger.record("state_worse_rew", reward[max_error_state])
                            self.logger.record("state_worse_visit", visitations[max_error_state])
                            self.logger.record("state_worse_worig", self.demo_state_om[align_func][max_error_state])
                            self.logger.dump(t)
                        linf_delta_per_align_func[align_func].append(linf_delta)
                        grad_norm_per_align_func[align_func].append(grad_norm)
                    last_max_vc_diff = max([lvlist[-1] for lvlist in linf_delta_per_align_func.values()]) 
                    last_max_grad_norm = max([grlist[-1] for grlist in grad_norm_per_align_func.values()]) 
                    
                    if last_max_vc_diff <= self.mean_vc_diff_eps or last_max_grad_norm <= self.grad_l2_eps:
                        break
            for align_func in self.training_align_funcs:
                pi = self.mce_partition_fh_per_align_func(align_func, 
                                                                reward_matrix=rewards_per_align_func[align_func], 
                                                                action_mat=torch_action_mat, 
                                                                obs_action_mat=torch_obs_action_mat,
                                                                reward_mode=TrainingModes.EVAL)
                self.learned_policy_per_va.set_policy_for_va(align_func, pi)

            rewards_per_align_func_callable = self._state_action_callable_reward_from_computed_rewards_per_align_func(rewards_per_align_func)

            return self.reward_net.get_learned_grounding(), rewards_per_align_func_callable
        elif mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            # TODO: Value System Identification...
            

            linf_delta_per_align_func =  {al : [] for al in self.vsi_target_align_funcs}
            grad_norm_per_align_func = {al : [] for al in self.vsi_target_align_funcs}
            rewards_per_target_align_func = {al : [] for al in self.vsi_target_align_funcs}

            target_align_funcs_to_learned_align_funcs = dict()
            
            for target_align_func in self.vsi_target_align_funcs:
                self.reward_net.reset_learned_alignment_function()
                self.reward_net.set_mode(TrainingModes.VALUE_SYSTEM_IDENTIFICATION)
                self.optimizer = self.optimizer_cls(self.reward_net.parameters(), **self.optimizer_kwargs)

                learned_al_function = self.reward_net.get_learned_align_function()

                with networks.training(self.reward_net):
                    
                    for t in range(max_iter):
                        predicted_r_np, visitations, old_pi, loss = self._train_step_vsi(torch_obs_mat, target_align_func=target_align_func, action_mat=torch_action_mat,obs_action_mat=torch_obs_action_mat)
                        learned_al_function = self.reward_net.get_learned_align_function()

                        target_align_funcs_to_learned_align_funcs[target_align_func] = learned_al_function

                        reward_th, reward = self.calculate_rewards(learned_al_function, 
                                                                   obs_mat=torch_obs_mat, 
                                                                   action_mat=torch_action_mat,
                                                                   obs_action_mat=torch_obs_action_mat,
                                                                   reward_mode=TrainingModes.EVAL)
                        rewards_per_target_align_func[target_align_func] = reward

                        grads = []
                        for p in self.reward_net.parameters():
                            assert p.grad is not None  # for type checker
                            grads.append(p.grad)
                        grad_norm = util.tensor_iter_norm(grads).item()
                        abs_diff_in_visitation_counts = np.abs(self.vsi_demo_state_om[target_align_func] - visitations)

                        linf_delta = np.max(abs_diff_in_visitation_counts)
                        avg_linf_delta = np.mean(abs_diff_in_visitation_counts)
                        norm_reward = np.linalg.norm(reward)
                        max_error_state = np.argmax(abs_diff_in_visitation_counts)
                        
                        if self.log_interval is not None and 0 == (t % self.log_interval):
                            params = self.reward_net.parameters()
                            weight_norm = util.tensor_iter_norm(params).item()
                            self.logger.record("iteration", t)
                            self.logger.record("Target align_func", target_align_func)
                            self.logger.record("Learned align_func", learned_al_function)
                            self.logger.record("linf_delta", linf_delta)
                            self.logger.record("weight_norm", weight_norm)
                            self.logger.record("grad_norm", grad_norm)
                            self.logger.record("loss", loss)
                            self.logger.record("avg_linf_delta", avg_linf_delta)
                            self.logger.record("norm reward", norm_reward)
                            self.logger.record("state_worse", max_error_state)
                            self.logger.record("state_worse_rew", reward[max_error_state])
                            self.logger.record("state_worse_visit", visitations[max_error_state])
                            self.logger.record("state_worse_worig", self.vsi_demo_state_om[target_align_func][max_error_state])
                            self.logger.dump(t)
                        linf_delta_per_align_func[target_align_func].append(linf_delta)
                        grad_norm_per_align_func[target_align_func].append(grad_norm)
                        
                        if linf_delta <= self.mean_vc_diff_eps or grad_norm <= self.grad_l2_eps:
                            break
                        
                    pi = self.mce_partition_fh_per_align_func(learned_al_function, 
                                                                reward_matrix=rewards_per_target_align_func[target_align_func], 
                                                                action_mat=torch_action_mat, 
                                                                obs_action_mat=torch_obs_action_mat,
                                                                reward_mode=TrainingModes.EVAL)
                        
                        
                    #self.learned_policy_per_va.set_policy_for_va(target_align_func, pi)
                    self.learned_policy_per_va.set_policy_for_va(learned_al_function, pi)

            rewards_per_target_align_func_callable = self._state_action_callable_reward_from_computed_rewards_per_align_func(rewards_per_target_align_func)

            return target_align_funcs_to_learned_align_funcs, rewards_per_target_align_func_callable
        
    def _state_action_callable_reward_from_computed_rewards_per_align_func(self, rewards_per_target_align_func: Dict):
        if self.reward_net.use_action:
                rewards_per_target_align_func_callable  = lambda al_f: rewards_per_target_align_func[al_f]
        else:
            print("NOT USING ACTION")
            for al_f in rewards_per_target_align_func.keys():
                rewards_per_state_action = np.zeros((self.env.state_dim, self.env.action_dim))
                for s in range(self.env.state_dim):
                    for a in range(self.env.action_dim):
                            ns_probs = self.env.transition_matrix[s,a,:]
                            rewards_per_state_action[s,a] = np.dot(ns_probs, rewards_per_target_align_func[al_f])
                            
            rewards_per_target_align_func_callable = lambda al_f: rewards_per_target_align_func[al_f]
        return rewards_per_target_align_func_callable  

    def _train_step_vsi(self, obs_mat: th.Tensor, target_align_func, action_mat: th.Tensor=None, obs_action_mat: th.Tensor=None) -> Tuple[np.ndarray, np.ndarray]:
        self.optimizer.zero_grad()
        assert self.reward_net.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION
        predicted_r, predicted_r_np = self.calculate_rewards(align_func=None, obs_mat=obs_mat, action_mat=action_mat, 
                                                             obs_action_mat=obs_action_mat,
                                                             reward_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION, 
                                                             recover_previous_mode_after_calculation=False)
        
        if self.reward_net.use_action is False:
            
            _, visitations, prev_pi = self.mce_occupancy_measures(
                reward_matrix=predicted_r_np,
                
            )
            # Forward/back/step (grads are zeroed at the top).
            # weights_th(s) = \pi(s) - D(s)
            weights_th = th.as_tensor(
                visitations - self.vsi_demo_state_om[target_align_func],
                dtype=self.reward_net.dtype,
                device=self.reward_net.device,
            )
            # The "loss" is then:
            #   E_\pi[r_\theta(S)] - E_D[r_\theta(S)]
            loss = th.dot(weights_th, predicted_r)
            # This gives the required gradient:
            #   E_\pi[\nabla r_\theta(S)] - E_D[\nabla r_\theta(S)].
            print("LOSS using states only", loss)
        else: # Use action in the reward.
            
            _, visitations, prev_pi = self.mce_occupancy_measures(
                reward_matrix=predicted_r_np,
            )
            # Forward/back/step (grads are zeroed at the top).
            # weights_th(s) = \pi(s) - D(s)
            next_state_prob = th.as_tensor(self.env.transition_matrix, dtype=self.reward_net.dtype,
                device=self.reward_net.device)
            
            weights_th = th.as_tensor(
                visitations - self.vsi_demo_state_om[target_align_func],
                dtype=self.reward_net.dtype,
                device=self.reward_net.device,
            )
            
            
            #loss = th.sum(th.vstack([th.matmul(predicted_r.t(), th.mul(next_state_prob[:,:,k],weights_th[k])) for k in range(self.env.state_dim)]))
            # Expand dimensions of `weights_th` to align with `next_state_prob`
            
            loss_matrix = predicted_r.unsqueeze(2) * (next_state_prob * weights_th.unsqueeze(0).unsqueeze(0))  # Shape: (N, M, K)
            loss = loss_matrix.sum()
            # loss = Sum_s,a,s'[R(s,a)*P(s,a,s')*weights_th(s')]
        
        loss.backward(retain_graph=False)
        
        self.optimizer.step()

        return predicted_r_np, visitations, prev_pi, loss
    def _train_step(self, obs_mat: th.Tensor, align_func: Any, action_mat: th.Tensor=None, obs_action_mat: th.Tensor=None, ) -> Tuple[np.ndarray, np.ndarray]:
        self.optimizer.zero_grad()

        # get reward predicted for each state by current model, & compute
        # expected # of times each state is visited by soft-optimal policy
        # w.r.t that reward function
        # TODO(adam): support not just state-only reward?
        self.reward_net.set_alignment_function(align_func)

        
        predicted_r, predicted_r_np = self.calculate_rewards(align_func=align_func, obs_mat=obs_mat, action_mat=action_mat, 
                                                             obs_action_mat=obs_action_mat,
                                                             reward_mode=TrainingModes.VALUE_GROUNDING_LEARNING, 
                                                             recover_previous_mode_after_calculation=False)
        

        if self.reward_net.use_action is False:
            
            _, visitations, prev_pi = self.mce_occupancy_measures(
                reward_matrix=predicted_r_np,
                
            )
            # Forward/back/step (grads are zeroed at the top).
            # weights_th(s) = \pi(s) - D(s)
            weights_th = th.as_tensor(
                visitations - self.demo_state_om[align_func],
                dtype=self.reward_net.dtype,
                device=self.reward_net.device,
            )
            # The "loss" is then:
            #   E_\pi[r_\theta(S)] - E_D[r_\theta(S)]
            loss = th.dot(weights_th, predicted_r)
            # This gives the required gradient:
            #   E_\pi[\nabla r_\theta(S)] - E_D[\nabla r_\theta(S)].
            print("LOSS using states only", loss)
        else: # Use action in the reward.
            
            _, visitations, prev_pi = self.mce_occupancy_measures(
                reward_matrix=predicted_r_np,
            )
            # Forward/back/step (grads are zeroed at the top).
            # weights_th(s) = \pi(s) - D(s)
            next_state_prob = th.as_tensor(self.env.transition_matrix, dtype=self.reward_net.dtype,
                device=self.reward_net.device)
            
            weights_th = th.as_tensor(
                visitations - self.demo_state_om[align_func],
                dtype=self.reward_net.dtype,
                device=self.reward_net.device,
            )
            
            
            #loss = th.sum(th.vstack([th.matmul(predicted_r.t(), th.mul(next_state_prob[:,:,k],weights_th[k])) for k in range(self.env.state_dim)]))
            # Expand dimensions of `weights_th` to align with `next_state_prob`
            
            loss_matrix = predicted_r.unsqueeze(2) * (next_state_prob * weights_th.unsqueeze(0).unsqueeze(0))  # Shape: (N, M, K)
            loss = loss_matrix.sum()
            # loss = Sum_s,a,s'[R(s,a)*P(s,a,s')*weights_th(s')]
            
        loss.backward()
        self.optimizer.step()

        return predicted_r_np, visitations, prev_pi, loss


    @property
    def policy(self):
        return self.learned_policy_per_va
    
    def _set_demo_oms_from_policy(self, policy: VAlignedDiscreteSpaceActionPolicy)  -> None:
        
        for al in self.training_align_funcs:
            _, self.demo_state_om[al], _ = self.mce_occupancy_measures(pi=policy.policy_per_va(al))
    
    def _set_vsi_demo_oms_from_policy(self, policy: VAlignedDiscreteSpaceActionPolicy) -> None:
        for al in self.vsi_target_align_funcs:
            _, self.vsi_demo_state_om[al], _ = self.mce_occupancy_measures(pi=policy.policy_per_va(al))

    def _set_demo_oms_from_trajectories(self, trajs: Iterable[types.Trajectory])  -> None:
        num_demos_per_al = dict()
        
        
        for traj in trajs:
            al = traj.infos[0]['align_func']
            if al not in self.demo_state_om:
                num_demos_per_al[al] = 0
                self.demo_state_om[al] = np.zeros((self.env.state_dim,))
    
            cum_discount = 1.0
            for obs in types.assert_not_dictobs(traj.obs):
                self.demo_state_om[al][obs] += cum_discount
                cum_discount *= self.discount
            num_demos_per_al[al] += 1
        for al in self.demo_state_om.keys():
            self.demo_state_om[al] /= num_demos_per_al[al]
    def _set_vsi_demo_oms_from_trajectories(self, trajs: Iterable[types.Trajectory])  -> None:
        num_demos_per_al = dict()
        
        
        for traj in trajs:
            al = traj.infos[0]['align_func']
            if al not in self.vsi_demo_state_om:
                num_demos_per_al[al] = 0
                self.vsi_demo_state_om[al] = np.zeros((self.env.state_dim,))
    
            cum_discount = 1.0
            for obs in types.assert_not_dictobs(traj.obs):
                self.vsi_demo_state_om[al][obs] += cum_discount
                cum_discount *= self.discount
            num_demos_per_al[al] += 1
        for al in self.vsi_demo_state_om.keys():
            self.vsi_demo_state_om[al] /= num_demos_per_al[al]

        #self.demo_state_om = get_demo_oms_from_trajectories(trajs, state_dim = self.env.state_dim, discount=self.discount, groupby='al')

    def set_demonstrations(self, demonstrations: Union[Iterable[types.Trajectory],Iterable[types.TransitionMapping], types.TransitionsMinimal]) -> None:
        pass