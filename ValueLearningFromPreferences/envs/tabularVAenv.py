
from abc import abstractmethod
import abc
from typing import Any, Callable, Dict, Generic, Iterable, Optional, SupportsFloat, Tuple, TypeVar, Union
import imitation
import imitation.util
import imitation.util.sacred
import torch
from typing_extensions import override
from numpy import ndarray
from seals import base_envs
import numpy as np

import gymnasium as gym


def encrypt_state(state, original_state_space):

    new_number = 0
    total_states = 1

    for i in range(original_state_space.shape[0]):
        new_number += state[i]*total_states
        total_states *= original_state_space[i].n

    return int(new_number)


def translate_state(state, original_state_space):

    new_state = np.zeros(original_state_space.shape[0], dtype=np.int64)

    for i in range(len(new_state)):
        new_modulo = original_state_space[i].n
        new_state[i] = state % new_modulo

        state -= new_state[i]
        state /= new_modulo

    return new_state



class ValueAlignedEnvironment(gym.Env):

    def __init__(self, n_values: int, horizon: int = None, done_when_horizon_is_met: bool = False,  trunc_when_horizon_is_met: bool = True):
        super().__init__()
        self.horizon = horizon
        self.done_when_horizon_is_met = done_when_horizon_is_met
        self.trunc_when_horizon_is_met = trunc_when_horizon_is_met
        self._cur_align_func = None
        self.current_assumed_grounding = None
        self.n_values = n_values
        self.basic_profiles = [tuple([float(ti) for ti in t]) for t in np.eye(self.n_values)]
        # self.action_space = self.env.action_space
        # self.observation_space = self.observation_space

        # self.metadata = self.env.metadata
        # self.reward_range = self.env.reward_range

        # self.spec = self.env.spec

    def align_func_yielder(self, a, ns=None, prev_align_func=None, info=None):
        return self._cur_align_func

    
    def get_reward_per_value(self, vindex, obs=None, action=None, next_obs=None, info=None, custom_grounding=None) -> SupportsFloat:
        return self.get_reward_per_align_func(align_func=self.basic_profiles[vindex], obs=obs, action=action, next_obs=next_obs, info=info, custom_grounding=custom_grounding)
    @abstractmethod
    def get_reward_per_align_func(self, align_func, obs=None, action=None, next_obs=None, info=None, custom_grounding=None) -> SupportsFloat:
        ...

    def set_align_func(self, align_func):
        self._cur_align_func = align_func

    def get_align_func(self):
        return self._cur_align_func
    
    @abstractmethod
    def real_reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        ...

    @abstractmethod
    def real_step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        ...
           
    def reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        s, info = self.real_reset(seed=seed, options=options)
        info['align_func'] = self.get_align_func()
        self.prev_observation, self.prev_info = s, info
        self.time = 0
        return s, info
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        ns, original_rew, done, trunc, info =  self.real_step(action)
        self.set_align_func(self.align_func_yielder(
            action, ns=ns, prev_align_func=self.get_align_func(), info=self.prev_info))
        info['align_func'] = self.get_align_func()  
              
        r = self.get_reward_per_align_func(
            self.get_align_func(), self.prev_observation, action, ns, info, custom_grounding=self.assumed_grounding)
        self.time += 1
        if self.horizon is not None and self.time >= self.horizon:
            trunc = self.trunc_when_horizon_is_met or trunc
            done = self.done_when_horizon_is_met or done
        self.prev_observation = ns
        self.prev_info = info
        return ns, r, done, trunc, info

       
    @abstractmethod
    def calculate_assumed_grounding(self, **kwargs) -> tuple[Union[torch.nn.Module, np.ndarray], np.ndarray]:
        ...

    @property
    def assumed_grounding(self):
        return self.current_assumed_grounding
    
DiscreteSpaceInt = np.int64
from seals import util

# Note: we redefine the type vars from gymnasium.core here, because pytype does not
# recognize them as valid type vars if we import them from gymnasium.core.

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

class TabularVAMDP(ValueAlignedEnvironment):

    """Base class for tabular environments with known dynamics.

    This is the general class that also allows subclassing for creating
    MDP (where observation == state) or POMDP (where observation != state).
    """

    transition_matrix: np.ndarray
    reward_matrix: np.ndarray

    state_space: gym.spaces.Discrete

    state_space: gym.spaces.Space[DiscreteSpaceInt]

    _cur_state: Optional[DiscreteSpaceInt]
    _n_actions_taken: Optional[int]


    def initial_state(self) -> DiscreteSpaceInt:
        """Samples from the initial state distribution."""
        return DiscreteSpaceInt(
            util.sample_distribution(
                self.initial_state_dist,
                random=self.np_random,
            ),
        )

    def transition(
        self,
        state: DiscreteSpaceInt,
        action: DiscreteSpaceInt,
    ) -> DiscreteSpaceInt:
        """Samples from transition distribution."""
        return DiscreteSpaceInt(
            util.sample_distribution(
                self.transition_matrix[state, action],
                random=self.np_random,
            ),
        )

    def reward(
        self,
        state: DiscreteSpaceInt,
        action: DiscreteSpaceInt,
        new_state: DiscreteSpaceInt,
    ) -> float:
        """Computes reward for a given transition."""
        inputs = (state, action, new_state)[: len(self.reward_matrix.shape)]
        return self.reward_matrix[inputs]

    def terminal(self, state: DiscreteSpaceInt, n_actions_taken: int) -> bool:
        """Checks if state is terminal."""
        del state
        return self.horizon is not None and n_actions_taken >= self.horizon

    @property
    def feature_matrix(self):
        """Matrix mapping states to feature vectors."""
        # Construct lazily to save memory in algorithms that don't need features.
        if self._feature_matrix is None:
            n_states = self.state_space.n
            self._feature_matrix = np.eye(n_states)
        return self._feature_matrix

    @property
    def state_dim(self):
        """Number of states in this MDP (int)."""
        return self.transition_matrix.shape[0]

    @property
    def action_dim(self) -> int:
        """Number of action vectors (int)."""
        return self.transition_matrix.shape[1]
    
    @property
    def obs_dim(self) -> int:
        """Size of observation vectors for this MDP."""
        return self.observation_matrix.shape[1]

    @property
    def obs_dtype(self) -> np.dtype:
        """Data type of observation vectors (e.g. np.float32)."""
        return self.observation_matrix.dtype

    @property
    def n_actions_taken(self) -> int:
        """Number of steps taken so far."""
        assert self._n_actions_taken is not None
        return self._n_actions_taken

    @property
    def state(self) -> DiscreteSpaceInt:
        """Current state."""
        assert self._cur_state is not None
        return self._cur_state

    @state.setter
    def state(self, state: DiscreteSpaceInt):
        """Set the current state."""
        if state not in self.state_space:
            raise ValueError(f"{state} not in {self.state_space}")
        self._cur_state = state
    
    def base_reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the episode and return initial observation."""
        if options is not None:
            raise NotImplementedError("Options not supported.")

        self.state = self.initial_state()
        self._n_actions_taken = 0
        obs = self.obs_from_state(self.state)
        info: Dict[str, Any] = dict()
        return obs, info
    
    def base_step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Transition state using given action."""
        if self._cur_state is None or self._n_actions_taken is None:
            raise RuntimeError("Need to call reset() before first step()")
        if action not in self.action_space:
            raise ValueError(f"{action} not in {self.action_space}")

        old_state = self.state
        self.state = self.transition(self.state, action)
        obs = self.obs_from_state(self.state)
        assert obs in self.observation_space
        reward = self.reward(old_state, action, self.state)
        self._n_actions_taken += 1
        terminated = self.terminal(self.state, self.n_actions_taken)
        truncated = False

        infos = {"old_state": old_state, "new_state": self._cur_state}
        return obs, reward, terminated, truncated, infos
    
    def __init__(self, n_values: int, transition_matrix: ndarray, observation_matrix: ndarray,
                 reward_matrix_per_va: Callable[[Any], ndarray],
                 default_reward_matrix: ndarray, initial_state_dist: ndarray = None,
                 horizon: int = None, done_when_horizon_is_met: bool = False,  trunc_when_horizon_is_met: bool = True):
        """self._real_env = base_envs.TabularModelPOMDP(transition_matrix=transition_matrix, observation_matrix=observation_matrix,
                                                     reward_matrix=default_reward_matrix, horizon=horizon, initial_state_dist=initial_state_dist)
        """

        
        self._cur_state = None
        self._n_actions_taken = None
        
        # The following matrices should conform to the shapes below:

        # transition matrix: n_states x n_actions x n_states
        n_states = transition_matrix.shape[0]
        if n_states != transition_matrix.shape[2]:
            raise ValueError(
                "Malformed transition_matrix:\n"
                f"transition_matrix.shape: {transition_matrix.shape}\n"
                f"{n_states} != {transition_matrix.shape[2]}",
            )

        # reward matrix: n_states x n_actions x n_states
        #   OR n_states x n_actions
        #   OR n_states
        if default_reward_matrix.shape != transition_matrix.shape[: len(default_reward_matrix.shape)]:
            raise ValueError(
                "transition_matrix and reward_matrix are not compatible:\n"
                f"transition_matrix.shape: {transition_matrix.shape}\n"
                f"reward_matrix.shape: {default_reward_matrix.shape}",
            )

        # initial state dist: n_states
        if initial_state_dist is None:
            initial_state_dist = util.one_hot_encoding(0, n_states)
        if initial_state_dist.ndim != 1:
            raise ValueError(
                "initial_state_dist has multiple dimensions:\n"
                f"{initial_state_dist.ndim} != 1",
            )
        if initial_state_dist.shape[0] != n_states:
            raise ValueError(
                "transition_matrix and initial_state_dist are not compatible:\n"
                f"number of states = {n_states}\n"
                f"len(initial_state_dist) = {len(initial_state_dist)}",
            )
        self.observation_matrix = observation_matrix
        
        self.transition_matrix = transition_matrix
        self.reward_matrix = default_reward_matrix
        self._feature_matrix = None
        self.horizon = horizon
        self.initial_state_dist = initial_state_dist

        self.state_space = gym.spaces.Discrete(self.state_dim)
        self.action_space = gym.spaces.Discrete(self.action_dim)

        # observation matrix: n_states x n_observations
        if observation_matrix.shape[0] != self.state_dim:
            raise ValueError(
                "transition_matrix and observation_matrix are not compatible:\n"
                f"transition_matrix.shape[0]: {self.state_dim}\n"
                f"observation_matrix.shape[0]: {observation_matrix.shape[0]}",
            )

        min_val: float
        max_val: float
        try:
            dtype_iinfo = np.iinfo(self.obs_dtype)
            min_val, max_val = dtype_iinfo.min, dtype_iinfo.max
        except ValueError:
            min_val = -np.inf
            max_val = np.inf
        self.observation_space = gym.spaces.Box(
            low=min_val,
            high=max_val,
            shape=(self.obs_dim,),
            dtype=self.obs_dtype,  # type: ignore
        )

        self.reward_matrix_per_align_func = reward_matrix_per_va
        self.done_when_horizon_is_met = done_when_horizon_is_met
        self.trunc_when_horizon_is_met = trunc_when_horizon_is_met

        self.set_initial_state_distribution(initial_state_dist)
        super().__init__( n_values=n_values, horizon=horizon, done_when_horizon_is_met=done_when_horizon_is_met,
                         trunc_when_horizon_is_met=trunc_when_horizon_is_met)
        self.env: base_envs.TabularModelPOMDP
        
        #self.current_assumed_grounding = self.calculate_assumed_grounding()

    def valid_actions(self, state, align_func=None):
        return np.arange(self.reward_matrix.shape[1])

    def get_reward_per_align_func(self, align_func, obs=None, action=None, next_obs=None, info=None, custom_grounding=None):
        return self.reward_matrix_per_align_func(align_func, custom_grounding=custom_grounding)[info['state'] if obs is None or isinstance(obs, Iterable) else int(obs), action]

    def get_state_actions_with_known_reward(self, align_func):
        return None

    def set_initial_state_distribution(self, dist):
        self.initial_state_dist = dist

    @property
    @abstractmethod
    def goal_states(self):
        ...

    @override
    def real_reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        s, i = self.base_reset(seed=seed, options=options)
        i['state'] = self.state
        i['next_state'] = self.state      

        self.prev_observation = i['state']
        return s, i
    
    @override
    def real_step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        prev_state = self.state
        ns, r, d, t, i = self.base_step(action)
        # d = d or self.env.state in self.goal_states
        
        i['state'] = prev_state
        i['next_state'] = self.state

        self.prev_observation = i['state']
        # d = True if d is True else self.state in self.goal_states
        return ns, r, d, t, i

    @property
    def invalid_states(self):
        return None

    
    def obs_from_state(self, state: np.int64) -> np.ndarray[Any, np.dtype]:
        return self.observation_matrix[state]
    


class ContextualEnv(ValueAlignedEnvironment):

    @property
    @abstractmethod
    def context(self):
        return None
    @abstractmethod
    def contextualize(self, context: Any):
        ...
    def reset(self, *, seed = None, options = None):
        s,i = super().reset(seed=seed, options=options)
        i['context'] = self.context
        return s, i
    def step(self, action):
        ns, r, d,t,i = super().step(action)
        i['context'] = self.context
        return ns, r, d,t,i

    