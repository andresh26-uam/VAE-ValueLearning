from abc import abstractmethod
import datetime
import signal
import sys
from typing import Any, Callable, Dict, List, Tuple, Union
import gymnasium as gym
import numpy as np


from imitation.data.types import Trajectory

from numpy._typing import NDArray
from stable_baselines3.common.type_aliases import PyTorchObs
import torch

from stable_baselines3.common.policies import BasePolicy

from src.envs.tabularVAenv import TabularVAPOMDP

class ValueSystemLearningPolicy(BasePolicy):

    def __init__(self, *args, env: gym.Env, use_checkpoints=True, state_encoder=None, squash_output: bool = False, observation_space = None, action_space = None, **kwargs):
        if observation_space is None:

            
            self.observation_space = env.observation_space
        
        if action_space is None:
            self.action_space = env.action_space
        super().__init__(*args, squash_output=squash_output, observation_space = self.observation_space, action_space = self.action_space, **kwargs)
        self.env = env
        self.default_alignment = None
        self.default_exploration = 0
        self.state_encoder = state_encoder if state_encoder is not None else (lambda st, info: st)
        

        if use_checkpoints:
            def _save_checkpoint_and_exit(sig, frame):
                print("Ctrl-C pressed. Saving checkpoint for policy: " + self.__class__.__name__ + " " + self._get_name())
                self._save_checkpoint()
                sys.exit(0)
            signal.signal(signal.SIGINT, _save_checkpoint_and_exit)


    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        raise NotImplementedError("Predict not implemented for this VSL policy subclass: " + self.__class__.__name__)
    @abstractmethod
    def act(self, state_obs, policy_state=None, exploration=0, stochastic=True, alignment_function=None):
        pass

    def fit_for_alignment_function(self, alignment_function):
        pass

    def reset(self, seed=None, state=None):
        return None

    def obtain_trajectory(self, alignment_function=None, seed=32, options: Union[None, Dict] = None,t_max=None, stochastic=False, exploration=0, only_states=False) -> Union[List, Trajectory]:
        state_obs, info = self.env.reset(seed = seed, options=options) if options is not None else self.env.reset(seed=seed) 
        policy_state = self.reset(seed=seed)
        init_state = self.state_encoder(state_obs, info)

        terminated = False
        truncated = False

        if getattr(self.env, 'horizon', None):
            if t_max is not None:
                t_max = min(t_max, self.env.horizon)
            else:
                t_max = self.env.horizon
        #reward_function = lambda s,a,d: 1 # for fast computation

        if only_states:
            path = []
            path.append(state_obs)
            t = 0
            while not (terminated or truncated) and (t_max is None or (t < t_max)):
                
                action, policy_state = self.act(state_obs, policy_state=policy_state,exploration=0, stochastic=stochastic, alignment_function=alignment_function)
                #print(self.environ.cur_state, action, index)
                state_obs, rew, terminated, truncated, info = self.env.step(action)
                
                path.append(state_obs)
            
                t+=1
            return path
        else:
            obs = [state_obs,]
            acts = []
            infos = []
            terminal = []
            #edge_path.append(self.environ.cur_state)
            t = 0
            while not (terminated or truncated) and (t_max is None or (t < t_max)):
                
                action, policy_state = self.act(state_obs, policy_state = policy_state, exploration=0, stochastic=stochastic, alignment_function=alignment_function)
                
                state_obs, rew, terminated, truncated, info = self.env.step(action)
                obs.append(state_obs)
                terminal.append(terminated)
                #state_des = self.environ.get_edge_to_edge_state(obs)

                acts.append(action)
                info['align_func'] = alignment_function
                info['rewards'] = rew
                info['init_state'] = init_state
                infos.append(info)
                t+=1
                if t_max is not None and t > t_max:
                    break
            
            return Trajectory(obs=obs, acts=acts, infos = infos, terminal=terminal)

    def obtain_trajectories(self, n_seeds=100, seed=32, options: Union[None, List, Dict] = None, stochastic=True, repeat_per_seed = 1, with_alignfunctions=[None,], t_max=None, exploration=0) -> List[Trajectory]:
        trajs = []
        
        for si in range(n_seeds):
            
            for r in range(repeat_per_seed):
                for af in with_alignfunctions:
                    trajs.append(
                        self.obtain_trajectory(af, seed=seed if si == 0 else None, exploration=exploration, options=options[si] if isinstance(options, list) else options, t_max=t_max, stochastic=stochastic, only_states=False))
        return trajs
    
    def _save_checkpoint(self, save_last=True):
        self.save("checkpoints/" + self._get_name() + "_" + str(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')))


    
    """def sb3_policy_call(self, observations: np.ndarray, state = None, dones: np.ndarray = None):
        if isinstance(observations, dict):
            actions = []
            n_samples = observations[list(observations.keys())[0]].shape[0]
            obs_items = observations.items()
            for i in range(n_samples):
                actions.append(self.act(dict({k: v[i] for k,v in obs_items})))
            return np.array(actions), state
        else:
            print("sb3 trying", observations)
            return np.array([[self.act(obs) for obs in obs_per_env] for obs_per_env in observations]), state
    """
class VAlignedDiscreteSpaceActionPolicy(ValueSystemLearningPolicy):
    def __init__(self, policy_per_va: Callable[[Any], np.ndarray], env: gym.Env, state_encoder=None, *args, **kwargs):
        super().__init__(*args, env = env, use_checkpoints=True, state_encoder=state_encoder, **kwargs)
        self.policy_per_va = policy_per_va
        self.env: TabularVAPOMDP = base_envs.ExposePOMDPStateWrapper(env)
    
    def reset(self, seed=None, state=None):
        policy_state = state if state is not None else 0
        return policy_state


    def act(self, state_obs: int, policy_state=None, exploration=0, stochastic=True, alignment_function=None):
        policy = self.policy_per_va(alignment_function)
        if len(policy.shape) == 2:
            probs = self.policy_per_va(alignment_function)[state_obs,:]
        elif len(policy.shape) == 3:
            assert policy_state is not None
            probs = self.policy_per_va(alignment_function)[policy_state,state_obs,:]
        else:
            assert len(policy.shape) == 1
            probs = np.array([self.policy_per_va(alignment_function)[state_obs] ,])
                             
        if np.random.rand() < exploration:
            action = self.action_space.sample()
            probs = np.ones_like(probs)/probs.shape[0]
        else: 
            if stochastic:
                action = np.random.choice(np.arange(len(probs)), p=probs)
            else:
                
                max_prob = np.max(probs)
                max_q = np.where(probs == max_prob)[0]
                action = np.random.choice(max_q)
        policy_state += 1
        return action, policy_state 
    


class VAlignedDictSpaceActionPolicy(VAlignedDiscreteSpaceActionPolicy):
   
    def _callable_from_dict(self):
        return lambda x: self.policy_per_va_dict[x]
    
    def __init__(self, policy_per_va_dict: Dict[Tuple, np.ndarray], env: gym.Env, state_encoder=None, *args, **kwargs):
        
        self.policy_per_va_dict = policy_per_va_dict
        policy_per_va = self._callable_from_dict()
        super().__init__(policy_per_va=policy_per_va, env=env, state_encoder=state_encoder, *args, **kwargs)
        

    def set_policy_for_va(self, va, policy: np.ndarray):
        self.policy_per_va_dict[va] = policy


from seals import base_envs

class VAlignedDictDiscreteStateActionPolicyTabularMDP(VAlignedDictSpaceActionPolicy):
    def __init__(self, policy_per_va_dict: Dict[Tuple, NDArray], env: TabularVAPOMDP, state_encoder=None, *args, **kwargs):
        
        super().__init__(policy_per_va_dict=policy_per_va_dict, env=env, state_encoder=state_encoder, *args, **kwargs)
    
    
    
    
    