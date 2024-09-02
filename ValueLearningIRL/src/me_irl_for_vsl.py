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
from src.reward_functions import AbstractVSLearningRewardFunction, DistributionProfiledRewardFunction, TrainingModes, ProfiledRewardFunction, print_tensor_and_grad_fn, squeeze_r

import pandas as pd

from src.vsl_policies import VAlignedDictSpaceActionPolicy, VAlignedDictDiscreteStateActionPolicyTabularMDP, VAlignedDiscreteSpaceActionPolicy, ValueSystemLearningPolicy


def mce_partition_fh(
    env: base_envs.TabularModelPOMDP,
    *,
    reward: Optional[np.ndarray] = None,
    discount: float = 1.0,
    value_iteration_tolerance = 0.001,
    policy_approximator='mce_original',
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

    if policy_approximator == 'mce_original':
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
    elif policy_approximator == 'value_iteration':
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

    else:
        pi = policy_approximator(env, reward, discount)

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
    policy_approximator = 'mce_original',
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
        _, _, pi = mce_partition_fh(env, reward=reward, policy_approximator=policy_approximator, discount=discount,value_iteration_tolerance = value_iteration_tolerance, deterministic=deterministic)

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

def get_demo_oms_from_trajectories(trajs: Iterable[types.Trajectory], state_dim, discount):
    num_demos_per_al = dict()
    demo_state_om = dict()

    for traj in trajs:
        al = traj.infos[0]['align_func']
        if al not in demo_state_om:
            num_demos_per_al[al] = 0
            demo_state_om[al] = np.zeros((state_dim,))

        cum_discount = 1.0
        for obs in types.assert_not_dictobs(traj.obs):
            demo_state_om[al][obs] += cum_discount
            cum_discount *= discount
        num_demos_per_al[al] += 1
    for al in demo_state_om.keys():
        demo_state_om[al] /= num_demos_per_al[al]
    return demo_state_om


def get_demo_oms_from_trajectories_grouped(trajs: Iterable[types.Trajectory], state_dim, discount =1, groupby='istate_and_al') -> dict:
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

    def set_distributional_net(self, distributional_net: DistributionProfiledRewardFunction):
        self.distributional_reward_net = distributional_net
        
    def get_reward_net(self):
        return self.reward_net
    def get_distributional_net(self):
        return self.distributional_reward_net
    
    def get_current_reward_net(self):
        return self.current_net
    
    def __init__(
        self,
        env: Union[TabularVAPOMDP],
        reward_net: ProfiledRewardFunction,
        vgl_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        vsi_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        vgl_optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        vsi_optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        discount: float = 1.0,
        vc_diff_epsilon: float = 1e-3,
        gradient_norm_epsilon: float = 1e-4,
        log_interval: Optional[int] = 100,
        vgl_expert_policy: Optional[ValueSystemLearningPolicy] = None,
        vsi_expert_policy: Optional[ValueSystemLearningPolicy] = None,
        vgl_expert_sampler = None,
        vsi_expert_sampler = None,
        target_align_func_sampler = lambda *args: args[0], # A Society or other mechanism might return different alignment functions at different times.


        vsi_target_align_funcs = [],
        
        vgl_target_align_funcs = [],

        demo_om_from_policy = True,
        policy_approximator = 'mce_original',
        
        value_iteration_tolerance = 0.00001,

        initial_state_distribution_train = None,
        initial_state_distribution_test = None,
        training_mode = TrainingModes.VALUE_GROUNDING_LEARNING,
        learn_stochastic_policy = True,
        name_method='',
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        
    ) -> None:
        
        self.discount = discount
        self.env = env
        self.policy_approximator = policy_approximator
        self.value_iteration_tolerance = value_iteration_tolerance
        self.learn_stochastic_policy = learn_stochastic_policy
        
        self.exposed_state_env = base_envs.ExposePOMDPStateWrapper(env)
        self.vgl_demo_state_om = dict()
        self.vsi_demo_state_om = dict()
        
        self.vgl_expert_sampler = vgl_expert_sampler # list of expert trajectories just to see different origin destinations and align_funcs
        self.vgl_expert_trajectories_per_isal = dict()
        self.vgl_expert_trajectories_per_align_func = dict()

        self.vsi_expert_sampler = vsi_expert_sampler # list of expert target align_func trajectories
        self.vsi_expert_trajectories_per_isal = dict()
        self.vsi_expert_trajectories_per_align_func = dict()
        
        self.initial_state_distribution_train = initial_state_distribution_train if initial_state_distribution_train is not None else env.initial_state_dist
        self.initial_state_distribution_test= initial_state_distribution_test if initial_state_distribution_test is not None else env.initial_state_dist

        """
        
        for t in self.vgl_expert_trajectories:
            al_func = t.infos[0]['align_func']
            isal = (t.infos[0]['init_state'], al_func)
            if isal not in self.vgl_expert_trajectories_per_isal.keys():

                self.vgl_expert_trajectories_per_isal[isal] = []
            if al_func not in self.vgl_expert_trajectories_per_align_func.keys():
                self.vgl_expert_trajectories_per_align_func[al_func] = []
                

            self.vgl_expert_trajectories_per_isal[isal].append(t)
            self.vgl_expert_trajectories_per_align_func[al_func].append(t)
        for t in self.vsi_expert_trajectories:
            al_func = t.infos[0]['align_func']
            isal = (t.infos[0]['init_state'], al_func)
            if isal not in self.vsi_expert_trajectories_per_isal.keys():

                self.vsi_expert_trajectories_per_isal[isal] = []
            if al_func not in self.vsi_expert_trajectories_per_align_func.keys():
                self.vsi_expert_trajectories_per_align_func[al_func] = []
                

            self.vsi_expert_trajectories_per_isal[isal].append(t)
            self.vsi_expert_trajectories_per_align_func[al_func].append(t)
            """

        self.vgl_target_align_funcs = vgl_target_align_funcs
        self.vsi_target_align_funcs = vsi_target_align_funcs

        """if vgl_target_align_funcs == 'auto':
            self.vgr_target_align_funcs = list(set(tuple(t.infos[0]['align_func']) for t in self.vsi_expert_sampler()))
        else:
            self.vgr_target_align_funcs = vgl_target_align_funcs
        if vsi_target_align_funcs == 'auto':
            self.vsi_target_align_funcs = list(set(tuple(t.infos[0]['align_func']) for t in self.vsi_expert_trajectories))
        else:
            self.vsi_target_align_funcs = vsi_target_align_funcs"""

       

        super().__init__(
                demonstrations=None,
                custom_logger=custom_logger,
            )
        self.vgl_expert_policy: VAlignedDiscreteSpaceActionPolicy = vgl_expert_policy
        self.vsi_expert_policy: VAlignedDiscreteSpaceActionPolicy = vsi_expert_policy


        self.demo_om_from_policy = demo_om_from_policy
        self.target_align_function_sampler = target_align_func_sampler
        if demo_om_from_policy:
            self._set_vgl_demo_oms_from_policy(self.vgl_expert_policy)
            self._set_vsi_demo_oms_from_policy(self.vsi_expert_policy)
            
        
        self.reward_net = reward_net
        self.distributional_reward_net = None
        self.current_net = reward_net
        
        self.vgl_optimizer_cls = vgl_optimizer_cls
        self.vsi_optimizer_cls = vsi_optimizer_cls

        vgl_optimizer_kwargs = vgl_optimizer_kwargs or {"lr": 1e-1}
        vsi_optimizer_kwargs = vsi_optimizer_kwargs or {"lr": 2e-1}
        self.vsi_optimizer_kwargs = vsi_optimizer_kwargs
        self.vgl_optimizer_kwargs = vgl_optimizer_kwargs

        self.distributional_vsi_optimizer_cls = vsi_optimizer_cls
        self.distributional_vsi_optimizer_kwargs = vsi_optimizer_kwargs
        

        self.vc_diff_epsilon = vc_diff_epsilon
        self.gradient_norm_epsilon = gradient_norm_epsilon
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
        for al_func in self.vgl_target_align_funcs:
            probability_matrix = np.random.rand(self.env.state_dim, self.env.action_dim)
            random_pol = probability_matrix / probability_matrix.sum(axis=1, keepdims=True)
            self.learned_policy_per_va.set_policy_for_va(al_func, random_pol)

        

        self.training_mode = training_mode

        self.reward_net.set_mode(self.training_mode)
    
    def calculate_rewards(self, align_func=None, grounding=None, obs_mat=None, action_mat = None, obs_action_mat = None, 
                          reward_mode=TrainingModes.EVAL, recover_previous_mode_after_calculation = True, 
                          use_distributional_reward=False, n_reps_if_distributional_reward=10, requires_grad=True):
        # TODO: support for rewards that take into account actions and next states.

        reward_net = self.current_net
        if obs_mat is None:
            obs_mat = self.exposed_state_env.observation_matrix
            obs_mat = th.as_tensor(
                obs_mat,
                dtype=reward_net.dtype,
                device=reward_net.device,
            )
            obs_mat.requires_grad_(requires_grad)

        if reward_net.use_one_hot_state_action:
            if obs_action_mat is None:
                obs_action_mat = th.as_tensor(
                        np.identity(self.env.state_dim*self.env.action_dim),
                        dtype=reward_net.dtype,
                        device=reward_net.device,
                    )
            obs_action_mat.requires_grad_(requires_grad)
        
        if recover_previous_mode_after_calculation:
            previous_rew_mode = reward_net.mode

        
        obs_mat.requires_grad_(requires_grad)
        if requires_grad is False:
            action_mat = action_mat.detach()
        reward_net.set_mode(reward_mode)
        
        reward_net.set_grounding_function(grounding)

        def calculation(align_func):
            reward_net.set_alignment_function(align_func)
            if use_distributional_reward:
                reward_net.fix_alignment_function()

            if reward_net.use_action is False:
                predicted_r = squeeze_r(reward_net(obs_mat, action_mat, None, None))
                assert predicted_r.shape == (obs_mat.shape[0],)
                
            else:
                assert action_mat is not None
                assert action_mat.size() == (self.env.action_space.n, obs_mat.shape[0], self.env.action_space.n)
                #vecs = [self.reward_net(obs_mat, action_mat[i], None, None) for i in range(self.env.action_space.n)]
                #print("USING ONE HOT?", self.reward_net.use_one_hot_state_action)
                if reward_net.use_one_hot_state_action:
                    predicted_r = th.reshape(reward_net(obs_action_mat, None, None, None), (self.env.state_dim,self.env.action_dim))
                else:
                    
                    predicted_r = th.stack([squeeze_r(reward_net(obs_mat, action_mat, None, None)) for i in range(self.env.action_space.n)], dim=1)
                    
                assert predicted_r.shape == (obs_mat.shape[0],action_mat.shape[0])
                if reward_mode != TrainingModes.EVAL and requires_grad:
                    assert predicted_r.grad_fn is not None
            
            if use_distributional_reward:
                used_alignment_func = np.zeros_like(reward_net.get_learned_align_function())
                used_alignment_func[reward_net.fixed_align_func_index] = 1.0
                used_alignment_func = tuple(used_alignment_func)
                
                probability = reward_net.trained_profile_net()[reward_net.fixed_align_func_index]
                reward_net.free_alignment_function()
            else:
                used_alignment_func = reward_net.get_learned_align_function()
                probability = 1.0

            return predicted_r, used_alignment_func, probability
        
        
        if recover_previous_mode_after_calculation:
            reward_net.set_mode(previous_rew_mode)
        
        if use_distributional_reward is False:
            predicted_r, used_align_func, _ = calculation(align_func)
            predicted_r_np = predicted_r.detach().cpu().numpy()
            return predicted_r, predicted_r_np
        else:
            list_of_reward_calculations = []
            align_func_used_in_each_repetition = []
            prob_of_each_repetition = []
            for rep in range(n_reps_if_distributional_reward):
                predicted_r, used_align_func, probability = calculation(align_func) 
                list_of_reward_calculations.append(predicted_r)
                align_func_used_in_each_repetition.append(used_align_func)
                prob_of_each_repetition.append(probability)
                
            predicted_rs = th.stack(list_of_reward_calculations)
            prob_of_each_repetition_th = th.stack(prob_of_each_repetition)
            predicted_rs_np = predicted_rs.detach().cpu().numpy()

            return predicted_rs, predicted_rs_np, align_func_used_in_each_repetition, prob_of_each_repetition_th
                

    def mce_partition_fh_per_align_func(self, align_func, reward_matrix=None, action_mat=None, obs_action_mat=None, reward_mode=TrainingModes.VALUE_GROUNDING_LEARNING):
        # TODO: mce partition function considers only tabular rewards per state, not per (state, action, next state).
        if reward_matrix is None:
            reward_matrix_torch, reward_matrix = self.calculate_rewards(align_func=align_func, 
                                                   obs_mat=self.env.observation_matrix, 
                                                   action_mat=action_mat, 
                                                   obs_action_mat=obs_action_mat,
                                                   reward_mode=reward_mode)
        
        V,Q, pi = mce_partition_fh(env=self.exposed_state_env, reward=reward_matrix, discount=self.discount, 
                                   policy_approximator=self.policy_approximator, 
                                   value_iteration_tolerance = self.value_iteration_tolerance,
                                   deterministic=not self.learn_stochastic_policy)
        
        return pi
            
       
    
    def mce_occupancy_measures(
            self,
        reward_matrix: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if pi is None:
            V,Q, pi = mce_partition_fh(self.exposed_state_env, reward=reward_matrix, discount=self.discount, policy_approximator=self.policy_approximator, 
                                       value_iteration_tolerance = self.value_iteration_tolerance,
                                       deterministic=not self.learn_stochastic_policy)
            
        D, Dcums = mce_occupancy_measures(env=self.exposed_state_env, pi=pi, 
                                          discount=self.discount, 
                                          policy_approximator=self.policy_approximator, 
                                          value_iteration_tolerance = self.value_iteration_tolerance,
                                          deterministic=not self.learn_stochastic_policy)
        return D, Dcums, pi
    from seals.diagnostics.cliff_world import CliffWorldEnv

    def train(self, max_iter: int = 1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None, n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_distributional_reward=False, n_reward_reps_if_distributional_reward=10) -> np.ndarray:
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
        self.training_mode = mode

        if use_distributional_reward:
            if self.distributional_reward_net is None:
                self.distributional_reward_net = DistributionProfiledRewardFunction(
                    environment=self.env, use_action=self.reward_net.use_action, use_done=self.reward_net.use_done, use_next_state=self.reward_net.use_next_state, use_one_hot_state_action=self.reward_net.use_one_hot_state_action, use_state=self.reward_net.use_state, 
                    activations=self.reward_net.activations, hid_sizes=self.reward_net.hid_sizes, basic_layer_classes=self.reward_net.basic_layer_classes, mode=mode,
                )
                self.distributional_reward_net.values_net = self.reward_net.values_net
                self.distributional_reward_net.set_alignment_function(self.reward_net.get_learned_align_function())

            self.current_net = self.distributional_reward_net
        
        torch_obs_mat = th.as_tensor(
            obs_mat,
            dtype=self.current_net.dtype,
            device=self.current_net.device,
        )
        torch_obs_mat.requires_grad_(True)
        
        
        torch_action_mat = None
        if self.current_net.use_action:
            actions_one_hot = th.eye(self.env.action_space.n, requires_grad=True)
            torch_action_mat = th.stack([actions_one_hot[i].repeat(obs_mat.shape[0],1) for i in range(self.env.action_space.n)], dim=0)
        
        
            
        torch_obs_action_mat = th.as_tensor(
                        np.identity(self.env.state_dim*self.env.action_dim),
                        dtype=self.current_net.dtype,
                        device=self.current_net.device,
                    )
        torch_obs_action_mat.requires_grad_(True)

        self.current_net.set_mode(mode)
        if assumed_grounding is not None and mode in [TrainingModes.EVAL, TrainingModes.VALUE_SYSTEM_IDENTIFICATION]:
                self.current_net.set_grounding_function(assumed_grounding)
        
        target_align_funcs = self.vgl_target_align_funcs if mode == TrainingModes.VALUE_GROUNDING_LEARNING else self.vsi_target_align_funcs 
        linf_delta_per_align_func =  {al : [] for al in target_align_funcs}
        grad_norm_per_align_func = {al : [] for al in target_align_funcs}
        rewards_per_target_align_func = {al : [] for al in target_align_funcs}
        
        if mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            self.vgl_optimizer = self.vgl_optimizer_cls(self.current_net.parameters(), **self.vgl_optimizer_kwargs)
            
            with networks.training(self.current_net):
                for t in range(max_iter):
                    for align_func in target_align_funcs:
                        old_reward, visitations, old_pi, loss, reward = self._train_step_vgl(torch_obs_mat, 
                                                                                             align_func=align_func, 
                                                                                             action_mat=torch_action_mat, 
                                                                                             obs_action_mat=torch_obs_action_mat,
                                                                                             n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed)
                        rewards_per_target_align_func[align_func] = reward
                        
                        grad_norm, abs_diff_in_vc = self.train_statistics(t, self.vgl_demo_state_om[align_func], visitations, loss, reward, align_func)
                        linf_delta_per_align_func[align_func].append(np.max(abs_diff_in_vc))
                        grad_norm_per_align_func[align_func].append(grad_norm)

                    last_max_vc_diff = max([lvlist[-1] for lvlist in linf_delta_per_align_func.values()]) 
                    last_max_grad_norm = max([grlist[-1] for grlist in grad_norm_per_align_func.values()]) 
                    
                    if last_max_vc_diff <= self.vc_diff_epsilon or last_max_grad_norm <= self.gradient_norm_epsilon:
                        self.print_statistics(t, visitations, loss, reward, align_func, None, grad_norm, abs_diff_in_vc)
                        break


        elif mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            reward_nets_per_target_align_func = dict()
            target_align_funcs_to_learned_align_funcs = dict()
            
            if use_distributional_reward:
                # TODO: SEGUIR AQUI. 
                
                
                for target_align_func in target_align_funcs:
                    self.distributional_reward_net.reset_learned_alignment_function()
                    self.distributional_reward_net.set_mode(TrainingModes.VALUE_SYSTEM_IDENTIFICATION)
                    self.distributional_vsi_optimizer = self.distributional_vsi_optimizer_cls(self.distributional_reward_net.parameters(), **self.distributional_vsi_optimizer_kwargs)
                    average_losses = 0.0
                    for t in range(max_iter):
                        predicted_rs_np, visitations, old_pi, loss, reward, learned_al_function, average_losses = self._train_vsi_distribution_reward(torch_obs_mat, target_align_func, 
                                                            action_mat=torch_action_mat,obs_action_mat=torch_obs_action_mat, 
                                                            n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories,
                                                            n_reward_reps=n_reward_reps_if_distributional_reward,
                                                            average_losses=average_losses, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed)
                        rewards_per_target_align_func[target_align_func] = reward
                        target_align_funcs_to_learned_align_funcs[target_align_func] = learned_al_function

                        grad_norm, abs_diff_in_vc = self.train_statistics(t, self.vsi_demo_state_om[target_align_func], visitations, loss, reward, target_align_func, learned_al_function)
                        linf_delta_per_align_func[target_align_func].append(np.max(abs_diff_in_vc))
                        grad_norm_per_align_func[target_align_func].append(grad_norm)

                        if linf_delta_per_align_func[target_align_func][-1] <= self.vc_diff_epsilon or grad_norm <= self.gradient_norm_epsilon:
                            self.print_statistics(t, visitations, loss, reward, target_align_func, learned_al_function, grad_norm, abs_diff_in_vc)
                            break
                        
                    reward_nets_per_target_align_func[target_align_func] = self.distributional_reward_net.copy()

            else:
                
                for target_align_func in target_align_funcs:
                    self.current_net.reset_learned_alignment_function()
                    self.current_net.set_mode(TrainingModes.VALUE_SYSTEM_IDENTIFICATION)
                    self.vsi_optimizer = self.vsi_optimizer_cls(self.current_net.parameters(), **self.vsi_optimizer_kwargs)
                    with networks.training(self.current_net):
                        
                        for t in range(max_iter):
                            predicted_r_np, visitations, old_pi, loss, reward, learned_al_function = self._train_step_vsi(torch_obs_mat, 
                                                                                                                          target_align_func=target_align_func, 
                                                                                                                          action_mat=torch_action_mat,
                                                                                                                          obs_action_mat=torch_obs_action_mat, 
                                                                                                                          n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories, 
                                                                                                                          n_sampled_trajs_per_seed=n_sampled_trajs_per_seed)
                            rewards_per_target_align_func[target_align_func] = reward
                            target_align_funcs_to_learned_align_funcs[target_align_func] = learned_al_function

                            grad_norm, abs_diff_in_vc = self.train_statistics(t, self.vsi_demo_state_om[target_align_func], visitations, loss, reward, target_align_func, learned_al_function)
                            linf_delta_per_align_func[target_align_func].append(np.max(abs_diff_in_vc))
                            grad_norm_per_align_func[target_align_func].append(grad_norm)

                            if linf_delta_per_align_func[target_align_func][-1] <= self.vc_diff_epsilon or grad_norm <= self.gradient_norm_epsilon:
                                self.print_statistics(t, visitations, loss, reward, target_align_func, learned_al_function, grad_norm, abs_diff_in_vc)
                                break
                        
                        reward_nets_per_target_align_func[target_align_func] = self.current_net.copy()
        else:
            # TODO: Simultaneous learning?
            
            raise NotImplementedError("Simultaneous learning mode is not implemented")
            
        # Organizing learned content:
        for target_align_func in target_align_funcs:
            learned_al_function = target_align_func if mode == TrainingModes.VALUE_GROUNDING_LEARNING else target_align_funcs_to_learned_align_funcs[target_align_func]
            pi = self.mce_partition_fh_per_align_func(learned_al_function, 
                                                                reward_matrix=rewards_per_target_align_func[target_align_func], 
                                                                action_mat=torch_action_mat, 
                                                                obs_action_mat=torch_obs_action_mat,
                                                                reward_mode=TrainingModes.EVAL)
            #self.learned_policy_per_va.set_policy_for_va(target_align_func, pi)
            self.learned_policy_per_va.set_policy_for_va(learned_al_function, pi)
            
        rewards_per_target_align_func_callable = self._state_action_callable_reward_from_computed_rewards_per_target_align_func(rewards_per_target_align_func)

        if mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            return target_align_funcs_to_learned_align_funcs, rewards_per_target_align_func_callable, reward_nets_per_target_align_func, linf_delta_per_align_func, grad_norm_per_align_func
        elif mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            return self.current_net.get_learned_grounding(), rewards_per_target_align_func_callable, self.current_net, linf_delta_per_align_func, grad_norm_per_align_func
        else:
            return self.current_net.get_learned_grounding(), target_align_funcs_to_learned_align_funcs, rewards_per_target_align_func_callable, reward_nets_per_target_align_func, linf_delta_per_align_func, grad_norm_per_align_func

    
    def train_statistics(self, t, expert_demo_om, visitations, loss, reward, target_align_func, learned_al_function=None):
        grads = []
        for p in self.current_net.parameters():
            assert p.grad is not None  # for type checker
            grads.append(p.grad)
        grad_norm = util.tensor_iter_norm(grads).item()
        abs_diff_in_visitation_counts = np.abs(expert_demo_om - visitations)
        if self.log_interval is not None and 0 == (t % self.log_interval):
            self.print_statistics(t, visitations, loss, reward, target_align_func, learned_al_function, grad_norm, abs_diff_in_visitation_counts)
        return grad_norm,abs_diff_in_visitation_counts

    def print_statistics(self, t, visitations, loss, reward, target_align_func, learned_al_function, grad_norm, abs_diff_in_visitation_counts):
        avg_linf_delta = np.mean(abs_diff_in_visitation_counts)
        norm_reward = np.linalg.norm(reward)
        max_error_state = np.argmax(abs_diff_in_visitation_counts)
        params = self.current_net.parameters()
        weight_norm = util.tensor_iter_norm(params).item()
        self.logger.record("iteration", t)
        self.logger.record("Target align_func", target_align_func)
        if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            self.logger.record("Learned align_func", learned_al_function)
        self.logger.record("linf_delta", np.max(abs_diff_in_visitation_counts))
        self.logger.record("weight_norm", weight_norm)
        self.logger.record("grad_norm", grad_norm)
        self.logger.record("loss", loss)
        self.logger.record("avg_linf_delta", avg_linf_delta)
        self.logger.record("norm reward", norm_reward)
        self.logger.record("state_worse", max_error_state)
        self.logger.record("state_worse_visit", visitations[max_error_state])
        self.logger.record("state_worse_worig", self.vsi_demo_state_om[target_align_func][max_error_state])
        self.logger.dump(t)
        
    def _state_action_callable_reward_from_computed_rewards_per_target_align_func(self, rewards_per_target_align_func: Dict):
        if self.current_net.use_action:
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
    
    def get_expert_demo_om(self, seed_target_align_func, n_seeds_for_sampled_trajectories=30, n_sampled_trajs_per_seed=3):
        
        target_align_func = self.target_align_function_sampler(seed_target_align_func)
        
        if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:

            if self.demo_om_from_policy:
                if target_align_func not in self.vsi_demo_state_om.keys():
                    self.vsi_demo_state_om[target_align_func] =self.mce_occupancy_measures(pi=self.vsi_expert_policy.policy_per_va(target_align_func))
                
            else:
                trajs = self.vsi_expert_sampler([target_align_func], n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)
                self.vsi_demo_state_om[target_align_func] = get_demo_oms_from_trajectories(trajs, state_dim = self.env.state_dim, discount=self.discount)[target_align_func]
            return self.vsi_demo_state_om[target_align_func]
        else:   
            if self.demo_om_from_policy:    
                if target_align_func not in self.vgl_demo_state_om.keys():
                    self.vgl_demo_state_om[target_align_func] =self.mce_occupancy_measures(pi=self.vgl_expert_policy.policy_per_va(target_align_func))
            else:
                trajs = self.vgl_expert_sampler([target_align_func], n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)
                self.vgl_demo_state_om[target_align_func] = get_demo_oms_from_trajectories(trajs, state_dim = self.env.state_dim, discount=self.discount)[target_align_func]
            return self.vgl_demo_state_om[target_align_func]

    
    def _train_vsi_distribution_reward(self, obs_mat, target_align_func, beta=0.9, action_mat=None, obs_action_mat=None, n_reward_reps=10, n_seeds_for_sampled_trajectories=30, n_sampled_trajs_per_seed=3, average_losses=0):
        self.distributional_vsi_optimizer.zero_grad()
        learned_al_function = self.distributional_reward_net.get_learned_align_function()
        #print("PREVIOUS ONE", learned_al_function)
        predicted_rs, predicted_rs_np, align_func_used_in_each_repetition, prob_of_each_repetition = self.calculate_rewards(align_func=None, obs_mat=obs_mat, action_mat=action_mat, 
                                                             obs_action_mat=obs_action_mat,
                                                             reward_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION, 
                                                             recover_previous_mode_after_calculation=False, use_distributional_reward=True, 
                                                             n_reps_if_distributional_reward=n_reward_reps,requires_grad=False)
        
        
        policy_per_target = dict()
        for i, align_func_i in enumerate(align_func_used_in_each_repetition):
            if align_func_i not in policy_per_target.keys():
                prev_pi = self.mce_partition_fh_per_align_func(align_func_i, reward_matrix=predicted_rs_np[i], reward_mode=self.current_net.mode)
                policy_per_target[align_func_i] = VAlignedDictDiscreteStateActionPolicyTabularMDP(policy_per_va_dict={align_func_i: prev_pi}, env=self.env)
        
        visitations_per_repetition = [[]]*n_reward_reps
        demo_om_per_repetition = [[]]*n_reward_reps
        om_per_align_func = dict()
        n_seeds = n_seeds_for_sampled_trajectories//n_reward_reps

        for i, align_func_i in enumerate(align_func_used_in_each_repetition):

            demo_om_per_repetition[i] = self.get_expert_demo_om(target_align_func, n_seeds_for_sampled_trajectories=n_seeds, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed)
            if self.demo_om_from_policy:
                if align_func_i not in om_per_align_func.keys():

                    _, visitations_per_repetition[i], prev_pi = self.mce_occupancy_measures(
                        reward_matrix=predicted_rs_np[i],
                    )
                    om_per_align_func[align_func_i] = visitations_per_repetition[i]
                else:
                    visitations_per_repetition[i] = om_per_align_func[align_func_i]
            else:
                
                trajs = policy_per_target[align_func_i].obtain_trajectories(
                        n_seeds=n_seeds, repeat_per_seed=n_sampled_trajs_per_seed, stochastic=self.learn_stochastic_policy, with_alignfunctions=[align_func_i], t_max=self.env.horizon, exploration=0
                    )
                visitations_per_repetition[i] = get_demo_oms_from_trajectories(trajs=trajs, state_dim=self.env.state_dim, discount=self.discount)[align_func_i]
        """visitations = np.mean(visitations_per_repetition, axis=0) # mean or sum(?)
        demo_visitations = np.mean([self.get_expert_demo_om_for_target_align_func(target_align_func, n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories//n_reward_reps) for _ in range(n_reward_reps)])

        weights_th = th.as_tensor(
            visitations - demo_visitations,
            dtype=self.distributional_reward_net.dtype,
            device=self.distributional_reward_net.device,
            )"""
        losses_per_r = [[]]*n_reward_reps
        for i in range(n_reward_reps):
            with th.no_grad():
                weights_th = th.as_tensor(
                    visitations_per_repetition[i] - demo_om_per_repetition[i],
                    dtype=self.distributional_reward_net.dtype,
                    device=self.distributional_reward_net.device,
                    )
            if self.distributional_reward_net.use_action is False:
                #losses_per_r[i] = th.dot(weights_th, predicted_rs[i])
                losses_per_r[i] = weights_th
            else: # Use action in the reward.
                next_state_prob = th.as_tensor(self.env.transition_matrix, dtype=self.distributional_reward_net.dtype,
                    device=self.distributional_reward_net.device)
                #loss = th.sum(th.vstack([th.matmul(predicted_r.t(), th.mul(next_state_prob[:,:,k],weights_th[k])) for k in range(self.env.state_dim)]))
                # Expand dimensions of `weights_th` to align with `next_state_prob`
                #loss_matrix = predicted_rs[i].unsqueeze(2) * (next_state_prob * weights_th.unsqueeze(0).unsqueeze(0))  # Shape: (N, M, K)
                loss_matrix = (next_state_prob * weights_th.unsqueeze(0).unsqueeze(0))  # Shape: (N, M, K)
                losses_per_r[i] = loss_matrix.sum()
                # loss = Sum_s,a,s'[R(s,a)*P(s,a,s')*weights_th(s')]
                

        #real_loss = th.mean(th.stack([th.mul(-1/(1+th.abs(losses_per_r[i]))+average_losses,prob_of_each_repetition[i]) for i in range(n_reward_reps)]))
        real_loss = th.mean(th.stack([th.mul(th.abs(losses_per_r[i]),prob_of_each_repetition[i]) for i in range(n_reward_reps)]))
        #print(average_losses)
            #print_tensor_and_grad_fn(real_loss.grad_fn)
            #print_tensor_and_grad_fn(prob_of_each_repetition[0].grad_fn)
            #print(real_loss.requires_grad)
            #print(prob_of_each_repetition[0].requires_grad)
        real_loss.backward()
        #average_losses = (1-beta)*np.mean([-(1/(1+th.abs(losses_per_r[i]))).detach().numpy() for i in range(len(losses_per_r))]) + beta*average_losses
        average_losses = (1-beta)*np.mean([th.abs(losses_per_r[i]).detach().numpy() for i in range(len(losses_per_r))]) + beta*average_losses

        
        self.distributional_vsi_optimizer.step()

        
        with th.no_grad():
            learned_al_function = self.distributional_reward_net.get_learned_align_function()
            #print("LEARNED ALIG", learned_al_function)
            _, new_rewards, align_func_used_in_each_repetition, _ = self.calculate_rewards(learned_al_function, 
                                                                    obs_mat=obs_mat, 
                                                                    action_mat=action_mat,
                                                                    obs_action_mat=obs_action_mat,
                                                                    reward_mode=TrainingModes.EVAL, use_distributional_reward=True,
                                                                    n_reps_if_distributional_reward=n_reward_reps)
            avg_new_reward = np.mean(new_rewards, axis=0)
        visitations = np.mean(visitations_per_repetition, axis=0) # mean or sum(?)
        demo_om_per_repetition = np.mean(demo_om_per_repetition, axis=0)
        self.vsi_demo_state_om[target_align_func] = demo_om_per_repetition

        return predicted_rs_np, visitations, prev_pi, real_loss, avg_new_reward, learned_al_function, average_losses

    def mce_vsl_loss_calculation(self, target_align_func, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, predicted_r, predicted_r_np):

        if self.demo_om_from_policy:
            _, visitations, prev_pi = self.mce_occupancy_measures(
                    reward_matrix=predicted_r_np,
                )
            
        else:
            prev_pi = self.mce_partition_fh_per_align_func(target_align_func, reward_matrix=predicted_r_np, reward_mode=self.current_net.mode)
            policy = VAlignedDictDiscreteStateActionPolicyTabularMDP(policy_per_va_dict={target_align_func: prev_pi}, env=self.env)
                
            trajs = policy.obtain_trajectories(
                        n_seeds=n_seeds_for_sampled_trajectories, repeat_per_seed=n_sampled_trajs_per_seed, stochastic=self.learn_stochastic_policy, with_alignfunctions=[target_align_func], t_max=self.env.horizon, exploration=0
                    )
            visitations = get_demo_oms_from_trajectories(trajs=trajs, state_dim=self.env.state_dim, discount=self.discount)[target_align_func]
        # Forward/back/step (grads are zeroed at the top).
        # weights_th(s) = \pi(s) - D(s)
        
        weights_th = th.as_tensor(
            visitations - self.get_expert_demo_om(target_align_func, n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed=n_sampled_trajs_per_seed),
            dtype=self.current_net.dtype,
            device=self.current_net.device,
            )
        if self.current_net.use_action is False:
            # The "loss" is then:
            #   E_\pi[r_\theta(S)] - E_D[r_\theta(S)]
            loss = th.dot(weights_th, predicted_r)
            # This gives the required gradient:
            #   E_\pi[\nabla r_\theta(S)] - E_D[\nabla r_\theta(S)].
            print("LOSS using states only", loss)
        else: # Use action in the reward.
            # Forward/back/step (grads are zeroed at the top).
            # weights_th(s) = \pi(s) - D(s)
            next_state_prob = th.as_tensor(self.env.transition_matrix, dtype=self.current_net.dtype,
                device=self.current_net.device)
            #loss = th.sum(th.vstack([th.matmul(predicted_r.t(), th.mul(next_state_prob[:,:,k],weights_th[k])) for k in range(self.env.state_dim)]))
            # Expand dimensions of `weights_th` to align with `next_state_prob`
            loss_matrix = predicted_r.unsqueeze(2) * (next_state_prob * weights_th.unsqueeze(0).unsqueeze(0))  # Shape: (N, M, K)
            loss = loss_matrix.sum()
            # loss = Sum_s,a,s'[R(s,a)*P(s,a,s')*weights_th(s')]
        
        loss.backward()
        return visitations,prev_pi,loss
    
    def _train_step_vsi(self, obs_mat: th.Tensor, target_align_func, action_mat: th.Tensor=None, obs_action_mat: th.Tensor=None, n_seeds_for_sampled_trajectories=30, n_sampled_trajs_per_seed=3) -> Tuple[np.ndarray, np.ndarray]:
        
        assert self.current_net.mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION
        self.vsi_optimizer.zero_grad()
        predicted_r, predicted_r_np = self.calculate_rewards(align_func=None, obs_mat=obs_mat, action_mat=action_mat, 
                                                             obs_action_mat=obs_action_mat,
                                                             reward_mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION, 
                                                             recover_previous_mode_after_calculation=False)
        
        visitations, prev_pi, loss = self.mce_vsl_loss_calculation(target_align_func, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, predicted_r, predicted_r_np)
        self.vsi_optimizer.step()

        learned_al_function = self.current_net.get_learned_align_function()
        _, new_reward = self.calculate_rewards(learned_al_function, 
                                                                   obs_mat=obs_mat, 
                                                                   action_mat=action_mat,
                                                                   obs_action_mat=obs_action_mat,
                                                                   reward_mode=TrainingModes.EVAL)
        
        return predicted_r_np, visitations, prev_pi, loss, new_reward, learned_al_function
    
    def _train_step_vgl(self, obs_mat: th.Tensor, align_func: Any, action_mat: th.Tensor=None, obs_action_mat: th.Tensor=None, n_seeds_for_sampled_trajectories=30, n_sampled_trajs_per_seed=3) -> Tuple[np.ndarray, np.ndarray]:
        assert self.current_net.mode == TrainingModes.VALUE_GROUNDING_LEARNING
        self.vgl_optimizer.zero_grad()
        self.current_net.set_alignment_function(align_func)
        predicted_r, predicted_r_np = self.calculate_rewards(align_func=align_func, obs_mat=obs_mat, action_mat=action_mat, 
                                                             obs_action_mat=obs_action_mat,
                                                             reward_mode=TrainingModes.VALUE_GROUNDING_LEARNING, 
                                                             recover_previous_mode_after_calculation=False)
        visitations, prev_pi, loss = self.mce_vsl_loss_calculation(align_func, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, predicted_r, predicted_r_np)
        self.vgl_optimizer.step()
        
        _, new_reward = self.calculate_rewards(align_func, 
                                               obs_mat=obs_mat, 
                                                action_mat=action_mat,
                                                obs_action_mat=obs_action_mat,
                                                reward_mode=TrainingModes.EVAL)
        
        return predicted_r_np, visitations, prev_pi, loss, new_reward


    @property
    def policy(self):
        return self.learned_policy_per_va
    
    def _set_vgl_demo_oms_from_policy(self, policy: VAlignedDiscreteSpaceActionPolicy)  -> None:
        
        for al in self.vgl_target_align_funcs:
            _, self.vgl_demo_state_om[al], _ = self.mce_occupancy_measures(pi=policy.policy_per_va(al))
    
    def _set_vsi_demo_oms_from_policy(self, policy: VAlignedDiscreteSpaceActionPolicy) -> None:
        for al in self.vsi_target_align_funcs:
            _, self.vsi_demo_state_om[al], _ = self.mce_occupancy_measures(pi=policy.policy_per_va(al))

    def _set_vgl_demo_oms_from_trajectories(self, trajs: Iterable[types.Trajectory])  -> None:
        self.vgl_demo_state_om = get_demo_oms_from_trajectories(trajs, state_dim=self.env.state_dim, discount=self.discount)
    def _set_vsi_demo_oms_from_trajectories(self, trajs: Iterable[types.Trajectory])  -> None:
        self.vsi_demo_state_om = get_demo_oms_from_trajectories(trajs, state_dim = self.env.state_dim, discount=self.discount)

    def set_demonstrations(self, demonstrations: Union[Iterable[types.Trajectory],Iterable[types.TransitionMapping], types.TransitionsMinimal]) -> None:
        pass