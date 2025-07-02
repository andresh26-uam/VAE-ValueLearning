
from copy import deepcopy
from ctypes import sizeof
import enum
import os
import random
import sys
from typing import Any, Callable, Optional, Self, SupportsFloat, Union, override
import numpy as np
from gymnasium import spaces
import torch
from envs.firefighters_env import calculate_s_trans_ONE_HOT_FEATURES
from use_cases.multivalue_car_use_case import ADS_Environment
from use_cases.multivalue_car_use_case.VI import value_iteration

from gymnasium import spaces

from envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment, encrypt_state
from use_cases.multivalue_car_use_case.VI import generate_model
import pathlib

ADS_Environment.Environment(seed=-1, obstacles=0)

class MVFS(enum.Enum):

    ONE_HOT_FEATURES = 'features_one_hot'  # Each feature is one-hot encoded
    # Use the original observations of the form [0:]
    ORIGINAL_OBSERVATIONS = 'observations'
    # Use State encrption, but one hot encoded
    ONE_HOT_OBSERVATIONS = 'observations_one_hot'
    # TODO: Ordinal Features may remain as 0, 1, 2 but categorical ones are one hot encoded for now.
    ORDINAL_AND_ONE_HOT_FEATURES = 'ordinal_and_one_hot'
    DEFAULT = None
def retrieve_policy_from_path(path, env):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Policy file {path} does not exist.")
            
            policy_ndarray = np.load(path, allow_pickle=True)
            return lambda state: MultiValuedCarEnv.policy_functional(state, env, policy_ndarray)
class MultiValuedCarEnv(ValueAlignedEnvironment):
    def policy_functional(obs, env: Self, policy_ndarray):
            prev_state = env.get_state()
            if len(obs) != 3:
                env.set_state(obs=obs)
            else:
                env.set_state(state_env=obs)
            state_env = env.real_env.get_state()
            env.set_state(state=prev_state)
            
            chosen_act = int(policy_ndarray[*state_env])
            #print("CA", chosen_act)
            probs = np.zeros(env.real_env.n_actions, dtype=np.float32)
            probs[chosen_act] = 1.0
            return probs


    """
    A simplified two-objective MDP environment for an urban high-rise fire scenario.
    Objectives: Professionalism and Proximity
    """
    metadata = {'render.modes': ['human']}
    basic_profiles = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    
    def transition(self, state, action):
        if self.observation_space.contains(state):
            self.set_state(obs=state)
        else:
            self.set_state(state=state)
        next_state = self.step(action)[0]
        return next_state

    def compute_policy(env, reward: Callable[[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]], np.ndarray[float]]=None, discount=1.0, weights=None, use_expert_grounding=False, **kwargs):
        env.reset()
        env.real_env.reset('hard')
        def to_reward_aux (agent, action=None):
            r =reward(next_state=env.obs_from_state(env.get_state()))
        
            assert r.shape == (1, env.n_values) or r.shape == (env.n_values, ), f"Reward shape {r} ({r.shape}) does not match expected shape {(1, env.n_values)} or {(env.n_values, )}"
            env.real_env.internal_damage = False if env.real_env.internal_damage == True else env.real_env.internal_damage
            env.real_env.external_damage = 0 if env.real_env.external_damage > 0 else env.real_env.external_damage
            
            return r.reshape((env.n_values,))
        if weights is None:
            raise ValueError("Weights must be provided for value iteration.")
        # Determine the directory of this file
        base_dir = pathlib.Path(__file__).parent
        if isinstance(weights[0], str):
            weights_pure = np.array([float(w) for w in weights[1]], dtype=np.float32)
        else:
            weights_pure = np.array(weights, dtype=np.float32)

        if use_expert_grounding:
            policy_dir = base_dir / "_mvc_expert_vi_policies"
        else:
            policy_dir = base_dir / "_mvc_learned_vi_policies"

        policy_dir.mkdir(exist_ok=True)

        # Create a unique filename based on weights and discount
        weights_str = str(weights_pure)#"_".join([f"{w:.4f}" for w in weights])
        policy_filename = f"policy_w_{weights_str}_d_{discount:.4f}.npy"
        #print(policy_filename)
        policy_path = policy_dir / policy_filename
        #print(policy_path)
        if policy_path.exists():
            policy_ndarray = np.load(policy_path)
            # Dummy v and q, since only policy is loaded; you may want to save/load these too if needed
            v = None
            q = None
        else:
            
            if use_expert_grounding:
                policy_ndarray, v, q = value_iteration(env.real_env, weights_pure, lex=None, discount_factor=discount, theta=1)
            else:
                prev_tor = env.real_env.to_reward
                env.real_env.to_reward = to_reward_aux
                policy_ndarray, v, q = value_iteration(env.real_env, weights_pure, lex=None, discount_factor=discount, theta=1)
                env.real_env.to_reward = prev_tor
            np.save(policy_path, policy_ndarray)
            print(f"Policy saved to {policy_path}")

        
        
        return (lambda state: v[*state]), (lambda state: q[*state]), lambda state: MultiValuedCarEnv.policy_functional(state, env, policy_ndarray), policy_path, np.save, retrieve_policy_from_path
    def render(self):
        return self.real_env.render()
    
    def __init__(self, env_name='MultiValuedCarEnv-v0', feature_selection=MVFS.ONE_HOT_FEATURES, horizon = None, done_when_horizon_is_met = False, trunc_when_horizon_is_met = True):
        
        self.init__kwargs = locals()
        self.init__kwargs.pop('self', None) 
        self.init__kwargs.pop('__class__', None)
        super().__init__(env_name=env_name, n_values=3, horizon=horizon, 
                         done_when_horizon_is_met=done_when_horizon_is_met, 
                         trunc_when_horizon_is_met=trunc_when_horizon_is_met)

        self.real_env = ADS_Environment.Environment(seed=-1, obstacles=9)
        #self.real_env.render()
        
        self.n_values = 3

        print("Env states: ", self.real_env.states_agent_left, self.real_env.states_agent_right, self.real_env.states_agent_right)
        self.states_with_goal_left = self.real_env.states_agent_left + [self.real_env.translate(self.real_env.agent_left_goal),self.real_env.translate(self.real_env.agent_right_goal),]
        self.states_with_goal_right = self.real_env.states_agent_right + [self.real_env.translate(self.real_env.agent_right_goal),self.real_env.translate(self.real_env.agent_left_goal)]
        
        
        # State 0 is injury, state 1 is external damages. You need this to predict next state.?
        self.state_sizes = (len(self.states_with_goal_left), len(self.states_with_goal_right), len(self.states_with_goal_right))
        
        self.state_space = spaces.Tuple([spaces.Discrete(1 ), spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float32), spaces.MultiDiscrete(
                nvec=self.state_sizes)])
    
        if feature_selection == MVFS.ORIGINAL_OBSERVATIONS:
            self.observation_space = self.state_space
        elif feature_selection == MVFS.ONE_HOT_OBSERVATIONS:
            self.observation_space = spaces.MultiBinary(
                n=self.n_states)
        elif feature_selection == MVFS.ONE_HOT_FEATURES:
            self.observation_space = spaces.flatten_space(spaces.Tuple([spaces.Discrete(1 ), spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float32), spaces.MultiBinary(
                n=sum([nv for nv in self.state_sizes]))]))
        else:
            self.observation_space = self.state_space


        # No, states as indexes, observations in one-hot encoding.
        self.action_space = spaces.Discrete(self.real_env.n_actions)
        self.feature_selection = feature_selection
        # self.action_dim = self.real_env.action_space.n
        #         self.state_dim = self.real_env.n_states
        self.basic_profiles = MultiValuedCarEnv.basic_profiles
        self.obs_size = sum([nv for nv in self.state_sizes]) + 2 # 2 for internal and external damages
        self._av_actions = list(range(self.action_space.n))
        print("Obs size", self.obs_size)
        print("State sizes", self.state_sizes)
        print("State space", self.state_space)
   
    def get_state_actions_with_known_reward(self, align_func):
        return None
    
    def get_state(self, internal_damage=None, external_damage=None) -> Any:
        if internal_damage is None or external_damage is None:
            state = [self.real_env.internal_damage, self.real_env.external_damage]
        else:
            state = [internal_damage, external_damage]
        state_env = self.real_env.get_state()
        state_env_index = [self.states_with_goal_left.index(int(state_env[0])), self.states_with_goal_right.index(int(state_env[1])), 
        self.states_with_goal_right.index(int(state_env[2]))]
        
        state.extend(state_env_index)
        return state
    def obs_from_state(self, state: Any) -> Any:
        """
        Converts the state to an observation based on the feature selection.
        """
        if self.feature_selection == MVFS.ORIGINAL_OBSERVATIONS:
            return state
        elif self.feature_selection == MVFS.ONE_HOT_OBSERVATIONS:
            raise NotImplementedError("One-hot observations not implemented for this environment.")
        elif self.feature_selection == MVFS.ONE_HOT_FEATURES:
            #print("State in obs_from_state", state)
            obs = np.zeros((self.obs_size,), dtype=np.float32)
            #print("State in obs_from_state", state)
            obs[0:2] = state[0:2]
            obs[2+state[2]] = 1.0
            obs[2+self.state_sizes[0]+state[3]] = 1.0
            obs[2+self.state_sizes[1]+self.state_sizes[0]+state[4]] = 1.0
            assert state[2] == list(obs[2:2+self.state_sizes[0]]).index(1.0)
            assert state[3] == list(obs[2+self.state_sizes[0]: 2+self.state_sizes[0]+self.state_sizes[1]]).index(1.0)
            assert state[4] == list(obs[2+self.state_sizes[0]+self.state_sizes[1]:]).index(1.0)
            return obs
        else:
            raise ValueError(f"Feature selection not registered {self.feature_selection}")
    @override
    def real_reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        self.real_env.reset('hard')
        state = self.get_state()
        obs = self.obs_from_state(state)
        return obs, {}
    @override
    def real_step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        s_in_real_env = self.real_env.translate_state(self.real_env.get_state())
        actions2 = ADS_Environment.Environment.pedestrian_move_map[s_in_real_env[1][0]][s_in_real_env[1][1]]
        actions3 = ADS_Environment.Environment.pedestrian_move_map[s_in_real_env[2][0]][s_in_real_env[2][1]]
        r2 = random.choice(actions2)
        r3 = random.choice(actions3)
        
        s,r_orig,done = self.real_env.step([action, r2, r3])
        obs = self.obs_from_state(self.get_state(internal_damage=s[0], external_damage=s[1]))
        #print("State in real env", s, "getted state", self.get_state(), "OBS", obs, "Action", action, "Reward", r_orig, "Done", done)
        
        return obs, r_orig, done[0], False, {}
    
    def step(self, action):
        ns, original_rew, done, trunc, info =  self.real_step(action)
        

        self.set_align_func(self.align_func_yielder(
            action, ns=ns, prev_align_func=self.get_align_func(), info=self.prev_info))
        info['align_func'] = self.get_align_func()  
        if self.horizon is not None and self.time >= self.horizon:
            trunc = self.trunc_when_horizon_is_met or trunc
            done = self.done_when_horizon_is_met or done
        cg = self.get_grounding_func()
        pure_al_func = info['align_func'][1] if isinstance(info['align_func'][0], str) else info['align_func']
        if cg is not None:
            #print("Custom grounding", cg)
            r = self.get_reward_per_align_func(
                align_func=pure_al_func, state=self.prev_observation, action=action, next_state=ns, 
                done=done, info=info, custom_grounding=cg)
        else:
            r = np.asarray(original_rew).dot(np.asarray(pure_al_func))
        
        """if original_rew[1] != 0.0 or original_rew[2] != 0.0:
            print("Original rew", original_rew, "Action", action, "State", self.get_state(), "Next state", ns)
            print("Reward", r, "Align func", info['align_func'])"""
            
            #exit(0)
        self.time += 1
        
        self.prev_observation = ns
        self.prev_info = info
        return ns, r, done, trunc, info
    
    def get_reward_per_align_func(self, align_func, state=None, action=None, 
                                  next_state=None, info=None, done=None, custom_grounding=None) -> SupportsFloat:
        #assert custom_grounding is None, "Custom grounding not implemented for this environment."
        if custom_grounding is None:
            gr = self._grounding(obs=state,action=action, next_obs=next_state, done=done)
        else:
            if next_state is not None:
                gr = custom_grounding(None, None, next_state, None)
            else:
                gr = custom_grounding(state, action, next_state, None) 
        if isinstance(align_func[0], str):
            align_func = [float(a) for a in align_func[1]]
            
        return sum([align_func[i]*gr[i] for i in range(len(align_func))])
    
    def valid_actions(self, state_obs, alignment_function=None) -> np.ndarray:
        #return self._av_actions
        prev_state = self.get_state()
        self.set_state(obs=state_obs)
        state_env = self.real_env.get_state()
        ret = [a for a in self._av_actions if a != 6 or state_env[0] < 2*self.real_env.map_width]
        self.set_state(state=prev_state)
        return ret
    def set_state(self, state: Any = None, obs: Any = None, state_env=None) -> None:
        """
        Sets the state of the real environment based on the provided state.
        """
        # Assuming state is a list with the first two elements as internal and external damage
        if state is None and state_env is None:
            assert obs is not None, "Either state or obs must be provided"
            if self.feature_selection == MVFS.ORIGINAL_OBSERVATIONS:
                state = obs
            elif self.feature_selection == MVFS.ONE_HOT_OBSERVATIONS:
                raise NotImplementedError("One-hot observations not implemented for this environment.")         
            elif self.feature_selection == MVFS.ONE_HOT_FEATURES:
                state = [obs[0], obs[1]] + list(np.where(obs[2:])[0])
                state[3] -= self.state_sizes[0]
                state[4] -= self.state_sizes[0] + self.state_sizes[1]
                assert state[2] == list(obs[2:2+self.state_sizes[0]]).index(1.0)
                assert state[3] == list(obs[2+self.state_sizes[0]: 2+self.state_sizes[0]+self.state_sizes[1]]).index(1.0)
                assert state[4] == list(obs[2+self.state_sizes[0]+self.state_sizes[1]:]).index(1.0)
        
        if state_env is None:
            assert state is not None, "State cannot be None if state_env is None"
            state_env = self.translate_state_to_env(state)
        else:
            state_env = state_env

        self.real_env.easy_reset(*self.real_env.translate_state(state_env) )
        self.real_env.internal_damage = state[0]
        self.real_env.external_damage = state[1]

    def translate_state_to_env(self, state):
        return [self.states_with_goal_left[state[2]], self.states_with_goal_right[state[3]], self.states_with_goal_right[state[4]]]
    
    def _grounding(self, state=None, obs=None,action=None,next_state=None,next_obs=None,done=None):
        #print("IE (should be False 0 always...)", self.real_env.internal_damage, self.real_env.external_damage)
        #print("But next_state should have the true things")
        prev_state, idam, edam = self.real_env.get_state(),self.real_env.internal_damage, self.real_env.external_damage

        if next_state is None and next_obs is None:
            self.set_state(state, obs)
            _,r,_,_,_ = self.real_step(action)
        else:
            self.set_state(next_state, obs=next_obs)
            r = self.real_env.to_reward(self.real_env.agents[0], action=None)

        self.real_env.easy_reset(*self.real_env.translate_state(prev_state))
        self.real_env.internal_damage = idam
        self.real_env.external_damage = edam 
        return r
    
    
    def default_grounding(self, state, action, next_state, done): 
        return self._grounding(obs=state, action=action, next_obs=next_state, done=done)
    @override
    def calculate_assumed_grounding(self, **kwargs) -> tuple[Union[torch.nn.Module, np.ndarray], np.ndarray]:
        # TODO: support different grounding functions?
        return self.default_grounding

    
if __name__ == "__main__":
    env = MultiValuedCarEnv()
    print(env.observation_space)
    print(env.action_space)
    print(env.state_space)
    #print(env.n_states)
    #print(env.initial_state_dist)
    #print(env.reward_matrix_per_va_dict)
    #print(env._goal_states)
    #print(env._invalid_states)
    #print(env.transition_matrix.shape)
    #print(env.observation_matrix.shape)