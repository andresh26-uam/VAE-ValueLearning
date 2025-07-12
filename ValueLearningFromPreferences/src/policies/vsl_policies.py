from abc import abstractmethod
from copy import deepcopy
import pickle
import dill
from stable_baselines3.ppo import MlpPolicy
from sb3_contrib.common.wrappers import ActionMasker
from minari.serialization import serialize_space, deserialize_space
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.ppo_mask.policies import MlpPolicy as MASKEDMlpPolicy
import json
import os
from seals import base_envs
import signal
import sys
from typing import Any, Callable, Dict, List, Tuple, Union, override
import gymnasium as gym
import numpy as np


from imitation.data.types import Trajectory, TrajectoryWithRew

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import PyTorchObs
import torch

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment
from src.algorithms.utils import PolicyApproximators, concentrate_on_max_policy, mce_partition_fh
from src.dataset_processing.data import TrajectoryWithValueSystemRews
from defines import CHECKPOINTS

from src.utils import NpEncoder, deconvert, deserialize_policy_kwargs, serialize_lambda, deserialize_lambda, import_from_string, serialize_policy_kwargs


class ValueSystemLearningPolicy():

    def __init__(self, env: ValueAlignedEnvironment, use_checkpoints=True, state_encoder=None, observation_space=None, action_space=None, **kwargs):
        
        if observation_space is None:

            self.observation_space = env.observation_space
        else:
            self.observation_space = observation_space
        if action_space is None:
            self.action_space = env.action_space
        else:
            self.action_space = action_space

        self.use_checkpoints = use_checkpoints
        """super().__init__(squash_output=squash_output,
                         observation_space=self.observation_space, action_space=self.action_space, **kwargs)
        """
        self.env = env
        self.default_alignment = None
        self.default_exploration = 0
        self.state_encoder = state_encoder if state_encoder is not None else (
            lambda st, info: st)

        if use_checkpoints:
            def _save_checkpoint_and_exit(sig, frame):
                print("Ctrl-C pressed. Saving checkpoint (if needed) for policy: " +
                      self.__class__.__name__ + " " + self._get_name())
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
        raise NotImplementedError(
            "Predict not implemented for this VSL policy subclass: " + self.__class__.__name__)

    @abstractmethod
    def act(self, state_obs, policy_state=None, exploration=0, stochastic=True, alignment_function=None):
        pass
    @abstractmethod
    def save(self, path='learner_dummy') -> str:
        pass
    @abstractmethod
    def load(ref_env, path='learner_dummy'):
        pass
    def get_learner_for_alignment_function(self, alignment_function):
        pass

    def reset(self, seed=None, state=None):
        return None

    def obtain_observation(self, next_state_obs):
        return next_state_obs

    def obtain_trajectory(self, alignment_func_in_policy=None, seed=32, options=None, t_max=None, stochastic=False, exploration=0, only_states=False, with_reward=False, with_grounding=False, alignment_func_in_env=None,
                          recover_previous_alignment_func_in_env=True, end_trajectories_when_ended=False, reward_dtype=None, agent_name='none') -> Trajectory:

        if alignment_func_in_env is None:
            alignment_func_in_env = alignment_func_in_policy
        if recover_previous_alignment_func_in_env:
            prev_al_env = self.env.get_align_func()
        
        self.env.set_align_func(alignment_func_in_env)

        state_obs, info = self.env.reset(
            seed=seed, options=options) if options is not None else self.env.reset(seed=seed)

        obs_in_state = self.state_encoder(self.obtain_observation(state_obs), info=info)

        policy_state = self.reset(seed=seed)
        init_state = self.state_encoder(state_obs, info)

        
        info['align_func'] = alignment_func_in_policy
        info['init_state'] = init_state
        info['ended'] = False

        terminated = False
        truncated = False

        if getattr(self.env, 'horizon', None):
            if t_max is not None:
                t_max = min(t_max, self.env.horizon)
            else:
                t_max = self.env.horizon
        # reward_function = lambda s,a,d: 1 # for fast computation

        if only_states:
            path = []
            path.append(obs_in_state)
            t = 0
            while not (terminated or truncated) and (t_max is None or (t < t_max)):

                action, policy_state = self.act(self.state_encoder(state_obs, info), policy_state=policy_state, exploration=exploration,
                                                stochastic=stochastic, alignment_function=alignment_func_in_policy)
                state_obs, rew, terminated, truncated, info_next = self.env.step(
                    action)
                obs_in_state = self.state_encoder(self.obtain_observation(state_obs), info=info_next)
                path.append(obs_in_state)

                t += 1
            return path
        else:
            obs = [obs_in_state,]
            rews = []
            v_rews = [[] for _ in range(self.env.n_values)]
            acts = []
            infos = []
            # edge_path.append(self.environ.cur_state)
            t = 0
            while not ((terminated or truncated) and end_trajectories_when_ended) and (t_max is None or (t < t_max)):
                #print("oh?", self.env.get_align_func(), alignment_func_in_env, alignment_func_in_policy)
                action, policy_state = self.act(self.state_encoder(state_obs, info), policy_state=policy_state, exploration=exploration,
                                                stochastic=stochastic, alignment_function=alignment_func_in_policy)
                
                next_state_obs, rew, terminated, truncated, info_next = self.env.step(
                    action)
                next_obs_in_state = self.state_encoder(self.obtain_observation(next_state_obs), info_next)
                
                obs.append(next_obs_in_state)
                # state_des = self.environ.get_edge_to_edge_state(obs)

                acts.append(action)
                info_next['align_func'] = alignment_func_in_policy
                info_next['init_state'] = init_state
                info_next['ended'] = terminated or truncated
                infos.append(info_next)
                if with_reward:
                    reward_should_be = self.env.get_reward_per_align_func(self.env.get_align_func(
                    ), obs_in_state, action, next_state=next_obs_in_state, done=terminated, info=info_next, custom_grounding=self.env.get_grounding_func()) # custom grounding is set from before
                    
                    assert self.env.get_align_func() == alignment_func_in_env
                    assert reward_should_be == rew, "Reward mismatch: expected {}, got {}".format(
                        reward_should_be, rew)
                    #assert np.allclose(info_next['state'], state_obs)
                    rews.append(rew)
                if with_grounding:
                    for value_index in range(self.env.n_values):
                        v_rews[value_index].append(self.env.get_reward_per_value(value_index, obs_in_state, action, next_state=next_obs_in_state, info=info_next, 
                                                                                 custom_grounding=self.env.get_grounding_func()))
                state_obs = next_state_obs
                info = info_next
                obs_in_state = next_obs_in_state
                t += 1
                if (t_max is not None and t > t_max) or (end_trajectories_when_ended and info['ended']):
                    break
            acts = np.asarray(acts)
            infos = np.asarray(infos)
            rews = np.asarray(rews, dtype=reward_dtype)
            v_rews = np.asarray(v_rews, dtype=reward_dtype)
            obs = np.asarray(obs)
            if recover_previous_alignment_func_in_env and with_reward:
                self.env.set_align_func(prev_al_env)

            if with_reward and not with_grounding:
                return TrajectoryWithRew(obs=obs, acts=acts, infos=infos, terminal=terminated, rews=rews)
            if with_reward and with_grounding:
                return TrajectoryWithValueSystemRews(obs=obs, acts=acts, infos=infos, rews=rews, terminal=terminated, n_vals=self.env.n_values, v_rews=v_rews, agent=agent_name)
            else:
                return Trajectory(obs=obs, acts=acts, infos=infos, terminal=terminated)

    def obtain_trajectories(self, n_seeds=100, seed=32,
                            options: Union[None, List, Dict] = None, stochastic=True, repeat_per_seed=1, align_funcs_in_policy=[None,], t_max=None,
                            exploration=0, with_reward=False, alignments_in_env=[None,],
                            end_trajectories_when_ended=True,
                            with_grounding=False,reward_dtype=None, agent_name='None',
                            from_initial_states=None) -> List[Trajectory]:
        trajs = []
        if len(alignments_in_env) != len(align_funcs_in_policy):
            alignments_in_env = align_funcs_in_policy
        trajs_sus_sus = []
        trajs_eff_sus = []
        trajs_sus_eff = []
        trajs_eff_eff = []
        has_initial_state_dist = False
        if isinstance(self.env, gym.Wrapper):
            possibly_unwrapped = self.env.unwrapped
        else:
            possibly_unwrapped = self.env
        if isinstance(possibly_unwrapped, TabularVAMDP):
            has_initial_state_dist = True
            prev_init_state_dist = possibly_unwrapped.initial_state_dist

            base_dist = np.zeros_like(prev_init_state_dist)
        for si in range(n_seeds):
            if has_initial_state_dist and from_initial_states is not None:
                assert repeat_per_seed == 1
                base_dist[from_initial_states[si]] = 1.0
                possibly_unwrapped.set_initial_state_distribution(base_dist)
            for af, af_in_env in zip(align_funcs_in_policy, alignments_in_env):
                for r in range(repeat_per_seed):
                    traj = self.obtain_trajectory(af,
                                                  seed=seed*n_seeds+si,
                                                  exploration=exploration,
                                                  end_trajectories_when_ended=end_trajectories_when_ended,
                                                  reward_dtype=reward_dtype,
                                                  options=options[si] if isinstance(options, list) else options, t_max=t_max, stochastic=stochastic, only_states=False,
                                                  with_reward=with_reward, with_grounding=with_grounding, alignment_func_in_env=af_in_env, agent_name=agent_name)
                    trajs.append(
                        traj
                    )
                    if __debug__:
                        if (not stochastic and exploration == 0.0) and not self.env.is_stochastic:
                            if af == (1.0, 0.0, 0.0):
                                traj_w = self.obtain_trajectory(af,
                                                                seed=seed*n_seeds+si,
                                                                exploration=exploration,
                                                                end_trajectories_when_ended=end_trajectories_when_ended,
                                                                options=options[si] if isinstance(options, list) else options, t_max=t_max,
                                                                stochastic=stochastic, only_states=False,
                                                                reward_dtype=reward_dtype,
                                                                with_reward=True, with_grounding=with_grounding, alignment_func_in_env=(1.0, 0.0, 0.0), agent_name=agent_name)

                                trajs_sus_sus.append(traj_w)
                                assert np.all(traj_w.obs == traj.obs)

                                traj_w2 = self.obtain_trajectory(af,
                                                                 seed=seed*n_seeds+si,
                                                                 exploration=exploration,
                                                                 end_trajectories_when_ended=end_trajectories_when_ended,
                                                                 options=options[si] if isinstance(options, list) else options, t_max=t_max,
                                                                 stochastic=stochastic, only_states=False,
                                                                 reward_dtype=reward_dtype,
                                                                 with_reward=True, with_grounding=with_grounding, alignment_func_in_env=(0.0, 0.0, 1.0), agent_name=agent_name)

                                trajs_sus_eff.append(traj_w2)
                                # print(traj.obs, traj_w2.obs, seed, n_seeds, si)
                                assert np.all(traj_w2.obs == traj.obs)
                            elif af == (0.0, 0.0, 1.0):
                                traj_w = self.obtain_trajectory(af,
                                                                seed=seed*n_seeds+si,
                                                                exploration=exploration,
                                                                end_trajectories_when_ended=end_trajectories_when_ended,
                                                                options=options[si] if isinstance(options, list) else options, t_max=t_max,
                                                                stochastic=stochastic, only_states=False,
                                                                reward_dtype=reward_dtype,
                                                                with_reward=True, with_grounding=with_grounding, alignment_func_in_env=(1.0, 0.0, 0.0), agent_name=agent_name)
                                trajs_eff_sus.append(traj_w)
                                # print(traj.obs, traj_w.obs, seed, n_seeds, si)
                                assert np.all(traj_w.obs == traj.obs)

                                traj_w2 = self.obtain_trajectory(af,
                                                                 seed=seed*n_seeds+si,
                                                                 exploration=exploration,
                                                                 end_trajectories_when_ended=end_trajectories_when_ended,
                                                                 options=options[si] if isinstance(options, list) else options, t_max=t_max,
                                                                 stochastic=stochastic, only_states=False,
                                                                 reward_dtype=reward_dtype,
                                                                 with_reward=True, with_grounding=with_grounding, alignment_func_in_env=(0.0, 0.0, 1.0), agent_name=agent_name)

                                trajs_eff_eff.append(traj_w2)
                                # print(traj.obs, traj_w2.obs, seed, n_seeds, si)
                                assert np.all(traj_w2.obs == traj.obs)
            if has_initial_state_dist and from_initial_states is not None:
                base_dist[from_initial_states[si]] = 0.0
            # testing in roadworld...
            if __debug__ and (not stochastic and len(af) == 3 and exploration == 0.0 ) and not  self.env.is_stochastic:
                for t, t2 in zip(trajs_sus_sus, trajs_eff_sus):
                    # print(self.policy_per_va((1.0,0.0,0.0))[t.obs[1]])
                    # print(self.policy_per_va((0.0,0.0,1.0))[t.obs[1]])
                    assert np.all(t.obs[0] == t2.obs[0]) and np.all(
                        t.obs[-1] == t2.obs[-1])

                    print("SUS routes, measuring SUS", [
                        t.infos[ko]['old_state'] for ko in range(len(t.infos))])

                    print("EFF routes, measuring SUS", [
                        t2.infos[ko]['old_state'] for ko in range(len(t2.infos))])
                    print(np.sum(t.rews), " vs ", np.sum(t2.rews))

                    assert np.sum(t.rews) >= np.sum(t2.rews)
                    print("CHECK!!!")

                for t, t2 in zip(trajs_eff_eff, trajs_sus_eff):
                    # print(self.policy_per_va((1.0,0.0,0.0))[t.obs[1]])
                    # print(self.policy_per_va((0.0,0.0,1.0))[t.obs[1]])
                    # print(t.obs, t2.obs)
                    assert np.sum(t.rews) >= np.sum(t2.rews)
                    print("CHECKED EFF!!!")
                    assert np.all(t.obs[0] == t2.obs[0]) and np.all(
                        t.obs[-1] == t2.obs[-1])

        if has_initial_state_dist and from_initial_states is not None:
            self.env.set_initial_state_distribution(prev_init_state_dist)
        return trajs

    """def _save_checkpoint(self, save_last=True):
        self.save("checkpoints/" + self._get_name() + "_" +
                  str(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')))"""

    def calculate_value_grounding_expectancy(self, value_grounding: Callable[[Any, Any, Any, bool, int|str], Any], 
                                             policy_align_func, 
                                             n_seeds=100, n_rep_per_seed=10, 
                                             exploration=0, stochastic=True, 
                                             seed=26,options=None, 
                                             initial_state_distribution=None):
        """if align_function_sampler is not None:
            trajs = []
            for al_rep in range(n_align_func_samples):
                al_fun = align_function_sampler(target_align_func)
                trajs.extend(self.obtain_trajectories(n_seeds=n_seeds, seed=seed, options=options, stochastic=stochastic,
                             repeat_per_seed=n_rep_per_seed, align_funcs_in_policy=[al_fun,], t_max=self.env.horizon, exploration=exploration))
        else:
            
            """
        if initial_state_distribution is not None:
            self.env.set_initial_state_distribution(initial_state_distribution)
        trajs = self.obtain_trajectories(n_seeds=n_seeds, seed=seed, options=options, stochastic=stochastic,
                                             repeat_per_seed=n_rep_per_seed, 
                                             align_funcs_in_policy=[policy_align_func,],
                                             #alignments_in_env=[env_align_func,],
                                             t_max=self.env.horizon, exploration=exploration,
                                              with_reward=True, with_grounding=True,end_trajectories_when_ended=True)
        
        precalc = None
        
        if initial_state_distribution is not None: 
        
            precalc = np.zeros((*(value_grounding(vi=0).shape),self.env.n_values), dtype=np.float64)
            
            for vi in range(self.env.n_values):
                precalc[:, :, vi] = value_grounding(vi=vi)
       
        ntrajs = float(len(trajs))
        real_ground = np.zeros((self.env.n_values,), dtype=np.float64)
        expected_gr = np.zeros((self.env.n_values,), dtype=np.float64)
        for t in trajs:
            #cur_t_gr = np.zeros((precalc.shape[-1],), dtype=np.float64)
            for i, (to,ta, tn) in enumerate(zip(t.obs[:-1], t.acts, t.obs[1:])):
                #print(t.vs_rews[i], t.v_rews[:, i], cur_t_gr)
                
                if precalc is not None:
                    expected_gr += (precalc[to,ta,:] )/len(t)
                else:
                    expected_gr += (np.reshape(value_grounding(to, ta, tn, None, vi='vg'), (self.env.n_values, ))) / len(t) # TODO terminal?
                
                if i == len(t.obs) - 2:
                    pass
                    #print("C1",(precalc[to,ta,:] ), to, ta)
                    #print("R1",t.v_rews[:,-1])
            #print("C", expected_gr)   
            real_ground += (np.mean(t.v_rews , axis=1)  )
            
            #print("R", real_ground)
        return expected_gr/len(trajs), real_ground/ len(trajs), trajs

    def get_environ(self, alignment_function):
        return self.env
    
    @abstractmethod
    def learn(self, alignment_function, grounding_function, reward=None, discount=1.0, stochastic=True, **kwargs):
        raise NotImplementedError("Learning not implemented for this VSL policy subclass: " + self.__class__.__name__)

class ContextualValueSystemLearningPolicy(ValueSystemLearningPolicy):
    def __init__(self, *args, env, use_checkpoints=True, state_encoder=None, observation_space=None, action_space=None, **kwargs):
        super().__init__(*args, env=env, use_checkpoints=use_checkpoints, state_encoder=state_encoder, observation_space=observation_space, action_space=action_space, **kwargs)
        self.context = None
    
    def reset(self, seed=None, state=None):
        self.context = self.update_context()
        return super().reset(seed, state)
    
    @abstractmethod
    def update_context(self):
        return self.context
    
    
def action_mask(valid_actions, action_shape, *va_args):
    valid_actions = valid_actions(*va_args)

    # Step 2: Create a zeroed-out array of the same shape as probs
    mask = np.zeros(shape=action_shape, dtype=np.bool_)
    mask[valid_actions] = True

    return mask

class MaskedPolicySimple(MlpPolicy):
    def __init__(self, *args, mask_method=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mask_method = mask_method

    def get_distribution(self, obs):
        action_masks = self._mask_method(obs.squeeze())
        dist = super().get_distribution(obs)
        dist.distribution.probs[...,action_masks == False] = 0.0
        return dist
    
class LearnerValueSystemLearningPolicy(ValueSystemLearningPolicy,BasePolicy):
    def learn(self, alignment_function, grounding_function=None, reward=None, discount=1.0, stochastic=True,
              total_timesteps=1000, callback=None, log_interval=1, tb_log_name='', reset_num_timesteps=False, progress_bar=True, **kwargs):
        learner = self.get_learner_for_alignment_function(alignment_function)
        self.env: ValueAlignedEnvironment
        prev_align_func = self.env.get_align_func()
        prev_grounding_function = self.env.get_grounding_func()

        self.env.set_align_func(alignment_function)
        self.env.set_grounding_func(grounding_function)

        learner.gamma = discount
        learner = learner.learn(total_timesteps=total_timesteps, 
                      log_interval=log_interval,
                      tb_log_name=f"{tb_log_name}_{total_timesteps}_{alignment_function}", callback=callback, reset_num_timesteps=reset_num_timesteps, progress_bar=progress_bar)
        self.learner_per_align_func[alignment_function] = learner
        self._alignment_func_in_policy = None
        self._sampling_learner = None

        self.env.set_align_func(prev_align_func)
        self.env.set_grounding_func(prev_grounding_function)

    def save(self, path='learner_dummy'):
        save_folder = os.path.join(CHECKPOINTS, path)
        print("SAVING LEARNER VSL POLICY TO ", save_folder)
        os.makedirs(save_folder, exist_ok=True)

        

        # Serialize policy_kwargs with special handling for classes
        serialized_policy_kwargs = serialize_policy_kwargs(self.policy_kwargs)

        # Save learners
        for alignment, learner in self.learner_per_align_func.items():
            learner_path = os.path.join(save_folder, f'alignment_{alignment}')
            learner.policy_kwargs.pop('mask_method', None)
            learner.save(learner_path)
        # Save initialization parameters (excluding env)
        init_params = {
            'learner_class': self.learner_class.__module__ + "." + self.learner_class.__name__ if type(self.learner_class) != str else self.learner_class,
            'learner_kwargs': self.learner_kwargs,
            'policy_class': self.policy_class.__module__ + "." + self.policy_class.__name__ if type(self.policy_class) != str else self.policy_class,
            'policy_kwargs': serialized_policy_kwargs,
            'masked': self.masked,
            'use_checkpoints': self.use_checkpoints,
            'state_encoder': None if self.state_encoder is None else serialize_lambda(self.state_encoder),
            'observation_space': None if self.observation_space is None else serialize_space(self.observation_space),
            'action_space': None if self.action_space is None else serialize_space(self.action_space)
        }

        with open(os.path.join(save_folder, 'init_params.json'), 'w') as f:
            json.dump(init_params, f, default=NpEncoder().default)
        print("SAVED LEARNER VSL POLICY TO ", save_folder)

    def load(ref_env, path='learner_dummy'):

        save_folder = os.path.join(CHECKPOINTS, path)
        print("LOADING LEARNER VSL POLICY FROM ", save_folder)
        if not os.path.exists(save_folder):
            raise FileNotFoundError(
                f"Save folder {save_folder} does not exist.")

        # Load initialization parameters
        with open(os.path.join(save_folder, 'init_params.json'), 'r') as f:
            init_params = json.load(f, object_hook=deconvert)

        # Reconstruct complex objects
        init_params['learner_class'] = import_from_string(
            init_params['learner_class'])
        init_params['policy_class'] = import_from_string(
            init_params['policy_class'])
        init_params['policy_kwargs'] = deserialize_policy_kwargs(
            init_params['policy_kwargs'])
        init_params['state_encoder'] = None if init_params['state_encoder'] is None else deserialize_lambda(
            init_params['state_encoder'])
        init_params['observation_space'] = None if init_params['observation_space'] is None else deserialize_space(
            init_params['observation_space'])
        init_params['action_space'] = None if init_params['action_space'] is None else deserialize_space(
            init_params['action_space'])
        init_params['env'] = ref_env  # Add environment back

        # Create a new instance with loaded parameters
        new_instance = LearnerValueSystemLearningPolicy(**init_params)
        print("PKWAGRS", new_instance.policy_kwargs)
        # Load learners
        prev_ob_space = ref_env.observation_space
        for file_name in os.listdir(save_folder):
            if file_name.startswith("alignment_"):
                alignment = eval(file_name[len("alignment_"):])
                assert isinstance(alignment, tuple),  "Alignment function must be a tuple"
                # TODO: aclarar si usar (ag, tuple) o solo tuple
                #assert len(alignment) == ref_env.n_values, f"Alignment function must be a tuple of length n_values ({ref_env.n_values}), {alignment} given"
                #assert isinstance(alignment[0], float), "Alignment function must be a tuple of floats"

                learner_path = os.path.join(save_folder, file_name)
                #ref_env.observation_space = ref_env.state_space
                pkwargs_new = deepcopy(new_instance.policy_kwargs)
                print(new_instance.policy_class)
                if issubclass(new_instance.policy_class, MaskedPolicySimple):
                    pkwargs_new['mask_method'] = None
                    pkwargs_new.pop('mask_method')
                    
                print("PKWAGRS_new", pkwargs_new, init_params['learner_class'])    
                new_instance.learner_per_align_func[alignment] = init_params['learner_class'].load(
                    learner_path, env=ref_env)
                
                if issubclass(new_instance.policy_class, MaskedPolicySimple):
                    
                    
                    pkwargs_new['mask_method'] = lambda obs: action_mask(
                        ref_env.valid_actions, (new_instance.env.action_space.n,), obs, alignment)
                new_instance.learner_per_align_func[alignment].policy_kwargs = pkwargs_new
                if issubclass(new_instance.policy_class, MaskedPolicySimple): 
                    new_instance.learner_per_align_func[alignment].policy._mask_method = pkwargs_new['mask_method']
        ref_env.observation_space = prev_ob_space
        print("LOADED LEARNER VSL POLICY FROM ", save_folder)
        return new_instance

    def __init__(self, *args, env: ValueAlignedEnvironment, learner_class: BaseAlgorithm = PPO, learner_kwargs={'learning_rate': 0.1, }, policy_class=MlpPolicy, policy_kwargs={}, masked=False, use_checkpoints=True, state_encoder=None,observation_space=None, action_space=None,  **kwargs):
        super().__init__(*args, env=env, use_checkpoints=use_checkpoints, state_encoder=state_encoder, observation_space=observation_space, action_space=action_space, **kwargs)
        self.learner_per_align_func: Dict[str, PPO] = dict()
        self.learner_class = learner_class
        self.learner_kwargs = learner_kwargs
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs
        
        self.masked = masked
        if isinstance(self.env, gym.Wrapper):
            possibly_unwrapped = self.env.unwrapped
        else:
            possibly_unwrapped = self.env
        if isinstance(possibly_unwrapped, base_envs.ResettablePOMDP) or masked:
            self.env = base_envs.ExposePOMDPStateWrapper(self.env)

            self.env_is_tabular = True
            assert isinstance(self.env, base_envs.ExposePOMDPStateWrapper)
        else:
            self.env_is_tabular = False
        
        if self.masked:
            self.learner_class = MaskablePPO

        self.policy_class = policy_class if not self.masked else MASKEDMlpPolicy
        self._sampling_learner = None
        self._alignment_func_in_policy = None

    def get_environ(self, alignment_function):
        self.env.set_align_func(alignment_function)
        if self.masked:
            
            environ = ActionMasker(env=self.env, action_mask_fn=lambda env: action_mask(
                env.valid_actions, (env.action_space.n,), env.prev_info['next_state'], alignment_function))
        else:
            environ = self.env
        
        return environ

    def get_learner_for_alignment_function(self, alignment_function):

        environ = self.get_environ(alignment_function)

        if alignment_function not in self.learner_per_align_func.keys():
            self.learner_per_align_func[alignment_function] = self.create_algo(
                environ=environ, alignment_function=alignment_function)

        learner: PPO = self.learner_per_align_func[alignment_function]

        if self.masked:
            assert isinstance(learner.policy, MASKEDMlpPolicy)
        # learner.save("dummy_save.zip")
        return learner

    def obtain_trajectory(self, alignment_func_in_policy=None, seed=32, **kwargs):
        if alignment_func_in_policy != self._alignment_func_in_policy or self._sampling_learner is None or self._alignment_func_in_policy is None:
            self._sampling_learner, self._alignment_func_in_policy = self.get_learner_for_alignment_function(
                alignment_function=alignment_func_in_policy), alignment_func_in_policy
        # print("KWARGS? see exploration", kwargs)
        traj = super().obtain_trajectory(alignment_func_in_policy=alignment_func_in_policy, seed=seed, **kwargs)

        return traj

    def create_algo(self, environ=None, alignment_function=None):
        if environ is None:
            environ = self.get_environ(alignment_function)
        pkwargs_new = deepcopy(self.policy_kwargs)
        if issubclass(self.policy_class, MaskedPolicySimple):
            pkwargs_new['mask_method'] = lambda obs: action_mask(
                self.env.valid_actions, (self.env.action_space.n,), obs, alignment_function)
            
        ret = self.learner_class(policy=self.policy_class, env=environ, policy_kwargs=pkwargs_new, **self.learner_kwargs)
        
        return ret

    def obtain_trajectories(self, **kwargs):
        if self.env_is_tabular:
            self._act_prob_cache = dict()
        return super().obtain_trajectories(**kwargs)

    def act(self, state_obs, policy_state=None, exploration=0, stochastic=True, alignment_function=None):
        a, ns, prob = self.act_and_obtain_action_distribution(
            state_obs=state_obs, policy_state=policy_state, exploration=exploration, stochastic=stochastic, alignment_function=alignment_function)
        return a, ns

    def act_and_obtain_action_distribution(self, state_obs, policy_state=None, exploration=0, stochastic=True, alignment_function=None):
        # pf_total = time.perf_counter()
        if self._sampling_learner is None:
            learner: BaseAlgorithm = self.get_learner_for_alignment_function(
                alignment_function)
        else:
            learner = self._sampling_learner

        if self.masked:
            menv = learner.get_env()
            action_masks = menv.env_method('action_masks')

        valid_actions = self.env.valid_actions(state_obs, alignment_function)

        if np.random.rand() > exploration:
            act_prob = None

            if self.masked:
                assert isinstance(learner.policy, MASKEDMlpPolicy)
                if self.env_is_tabular:
                    act_prob = learner.policy.get_distribution(
                        learner.policy.obs_to_tensor(state_obs)[0], action_masks=action_masks)
                    # self._act_prob_cache[state_obs] = act_prob
                    a = int(act_prob.get_actions(deterministic=not stochastic))

                    next_policy_state = None
                else:
                    a, next_policy_state = learner.policy.predict(
                        state_obs, state=policy_state, deterministic=not stochastic, action_masks=action_masks)

            else:
                act_prob = learner.policy.get_distribution(
                        learner.policy.obs_to_tensor(state_obs)[0])
                if self.env_is_tabular:
                    
                    a = int(act_prob.get_actions(deterministic=not stochastic))
                    next_policy_state = None
                else:
                    
                    a, next_policy_state = learner.policy.predict(
                        state_obs, state=policy_state, deterministic=not stochastic)

            base_distribution = act_prob.distribution.probs[0]
            # TODO... Maybe this is wrong when backpropagation? Not used now
            valid_distribution = torch.zeros_like(base_distribution)
            valid_distribution[valid_actions] = base_distribution[valid_actions] / \
                torch.sum(base_distribution[valid_actions])
            assert isinstance(a, int) or len(a.shape) == 0
            assert len(valid_distribution) == self.action_space.n
            if __debug__ and self.masked:

                indices = np.where(action_masks[0] == True)[0]
                np.testing.assert_equal(
                    len(np.setdiff1d(indices, valid_actions)), 0)

            if self.masked:
                assert a in valid_actions
            elif a not in valid_actions:
                if not stochastic:
                    max_prob = torch.max(base_distribution[valid_actions])
                    max_q = torch.where(
                        base_distribution[valid_actions] == max_prob)[0]
                    action_index = np.random.choice(max_q.detach().numpy())
                    a = valid_actions[action_index]
                else:
                    a = np.random.choice(
                        len(valid_distribution), p=valid_distribution.detach().numpy())

        else:
            #assert True is False  # This should never ever occur
            valid_distribution = None
            if len(valid_actions) == 0:
                a = int(menv.action_space.sample())
            else:
                a = int(np.random.choice(valid_actions))

            assert isinstance(a, int)
            next_policy_state = None if policy_state is None else policy_state + \
                1  # (Not used)
        # pf_total_finish = time.perf_counter()
        assert a in valid_actions
        return a, next_policy_state, valid_distribution


class VAlignedDiscretePolicy(ValueSystemLearningPolicy):
    def __init__(self, policy_per_va: Callable[[Any], Union[np.ndarray, Callable[[Any],np.ndarray]]], env: gym.Env, state_encoder=None, expose_state=True, assume_env_produce_state = True, *args, **kwargs):

        super().__init__(*args, env=env,
                         state_encoder=state_encoder, **kwargs)
        self.policy_per_va = policy_per_va
        self.assume_env_produce_state = assume_env_produce_state
        if expose_state:
            self.env: TabularVAMDP = base_envs.ExposePOMDPStateWrapper(env)
            
        self.expose_state = expose_state
    def reset(self, seed=None, state=None):
        policy_state = state if state is not None else 0
        return policy_state

    def obtain_observation(self, next_state_obs):
        if self.expose_state is False and self.assume_env_produce_state is True:
            obs_in_state = self.env.obs_from_state(next_state_obs)
        else:
            obs_in_state = next_state_obs
        return obs_in_state
    
    def act(self, state_obs: Any, policy_state=None, exploration=0, stochastic=True, alignment_function=None):
        policy = self.policy_per_va(alignment_function)
        probs = self.get_policy_probs(policy=policy, state_obs=state_obs, policy_state=policy_state)
        do_explore = False
        if np.random.rand() < exploration:
            do_explore = True
            probs = np.ones_like(probs)/probs.shape[0]

        valid_actions = self.env.valid_actions(state_obs, alignment_function)

        # Step 2: Create a zeroed-out array of the same shape as probs
        filtered_probs = np.zeros_like(probs)

        # Step 3: Set the probabilities of valid actions
        filtered_probs[valid_actions] = probs[valid_actions]

        # Step 4: Normalize the valid probabilities so they sum to 1
        if filtered_probs.sum() > 0:
            filtered_probs /= filtered_probs.sum()
        else:
            # Handle edge case where no actions are valid
            # or some default uniform distribution
            filtered_probs[:] = 1.0 / len(filtered_probs)

        # Now `filtered_probs` has the same shape as `probs`
        probs = filtered_probs

        assert np.allclose([np.sum(probs),], [1.0,])

        if stochastic or do_explore:
            action = np.random.choice(np.arange(len(probs)), p=probs)
        else:
            max_prob = np.max(probs)
            max_q = np.where(probs == max_prob)[0]
            action = np.random.choice(max_q)
        policy_state += 1
        return action, policy_state
    def get_policy_probs(self, policy, state_obs, policy_state=None):
        
        return policy(state_obs)
        
        
        raise NotImplementedError(
            "This method should be implemented in subclasses to return the policy probabilities for the given state observation.")

class VAlignedDiscreteDictPolicy(VAlignedDiscretePolicy):
    def save(self, path='learner_dummy'):
        save_folder = os.path.join(CHECKPOINTS, path)
        os.makedirs(save_folder, exist_ok=True)
        
        with open(os.path.join(save_folder, 'p.pkl'), 'wb') as f:
            dill.dump( self.policy_per_va_dict, f)
        with open(os.path.join(save_folder, 'state_encoder.pkl'), 'wb') as f:
            dill.dump(self.state_encoder, f)
        print("SAVED VAlignedDictSpaceActionPolicy to ", f)

        init_params = {
            'expose_state': self.expose_state,
            'assume_env_produce_state': self.assume_env_produce_state,
            #'state_encoder': None if self.state_encoder is None else serialize_lambda(self.state_encoder),
            'observation_space': None if self.observation_space is None else serialize_space(self.observation_space),
            'action_space': None if self.action_space is None else serialize_space(self.action_space)   
        }
        with open(os.path.join(save_folder, 'init_params.json'), 'wb') as f:
            dill.dump(init_params, f)
        return save_folder

    def __init__(self, policy_per_va_dict, env, state_encoder=None, expose_state=True, assume_env_produce_state=True, *args, **kwargs):
        self.policy_per_va_dict = policy_per_va_dict

        policy_per_va = lambda x:  self.policy_per_va_dict[x]
        
        super().__init__(policy_per_va, env, state_encoder, expose_state, assume_env_produce_state, *args, **kwargs)
        

    def set_policy_for_va(self, va, policy: Any):
        self.policy_per_va_dict[va] = policy

    

    

    def calculate_value_grounding_expectancy_precise(self, value_grounding: Callable[[np.ndarray,np.ndarray], float], policy_align_func, n_seeds=100, n_rep_per_seed=10, exploration=0, stochastic=True, t_max=None, seed=None, p_state=None, env_seed=None, options=None, initial_state_distribution=None):

        
        if initial_state_distribution is None:
            initial_state_distribution = np.ones(
                (self.env.state_dim))/self.env.state_dim
        initial_state_dist = initial_state_distribution
        try:
            pi = self.policy_per_va(policy_align_func)
        except KeyError:
            pi = self.policy_per_va(policy_align_func[1])
        
        self.reset(seed=seed, state=p_state)
        
        self.env.reset(seed=env_seed, options=options)


        state_dist = initial_state_dist
        accumulated_feature_expectations = 0
        accumulated_feature_expectations_real = 0

        precalc = np.zeros((*(value_grounding(vi=0).shape),self.env.n_values), dtype=np.float64)
        print("POLICY IN PREC", pi[150:200])
        reward_matric_precalc = np.zeros((*(value_grounding(vi=0).shape),self.env.n_values), dtype=np.float64)
        for vi in range(self.env.n_values):
            precalc[:, :, vi] = value_grounding(vi=vi)
            reward_matric_precalc[:,:,vi] = self.env.reward_matrix_per_align_func(self.env.basic_profiles[vi]) 
        for t in range(self.env.horizon):

            pol_t = pi if len(pi.shape) == 2 else pi[t]
            if not stochastic:
                pol_t = concentrate_on_max_policy(pol_t, distribute_probability_on_max_prob_actions=False, valid_action_getter=lambda s: self.env.valid_actions(s, None))
            state_action_prob = np.multiply(pol_t, state_dist[:, np.newaxis])
            features_time_t = np.sum(
                precalc * state_action_prob[:, :, np.newaxis], axis=(0, 1))/self.env.state_dim/self.env.action_dim
            
            features_time_t_real = np.sum(
                reward_matric_precalc * state_action_prob[:, :, np.newaxis], axis=(0, 1))/self.env.state_dim/self.env.action_dim

            if t == 0:
                accumulated_feature_expectations = features_time_t
                accumulated_feature_expectations_real = features_time_t_real
            else:
                accumulated_feature_expectations += features_time_t
                accumulated_feature_expectations_real += features_time_t_real
            # /self.env.state_dim
            state_dist = np.sum(self.env.transition_matrix *
                                state_action_prob[:, :, np.newaxis], axis=(0, 1))
            assert np.allclose(np.sum(state_dist), 1.0)
        return accumulated_feature_expectations, accumulated_feature_expectations_real

    

    
        # self.env = env

    
    
    
    
        


class VAlignedDictSpaceActionPolicy(VAlignedDiscreteDictPolicy):
    def get_policy_probs(self, policy, state_obs, policy_state=None):
        
        if len(policy.shape) == 2:
            probs = policy[state_obs, :]
        elif len(policy.shape) == 3:
            assert policy_state is not None
            probs = policy[
                policy_state, state_obs, :]
        else:
            assert len(policy.shape) == 1
            probs = np.array(
                [policy[state_obs],])
                
        return probs
    
    

    
    def load(ref_env, path='learner_dummy'):
        route = os.path.join(CHECKPOINTS, path, 'p.pkl')
        if not os.path.exists(route):
            raise FileNotFoundError(
                f"Save file {route} does not exist.")
        print("LOADING VAlignedDictSpaceActionPolicy from ", route)
        with open(route, 'rb') as f:
            policy_per_va_dict = dill.load(f)
        print("LOADED VAlignedDictSpaceActionPolicy from ", route)
        with open(os.path.join(CHECKPOINTS, path, 'init_params.json'), 'rb') as f:
            init_params = dill.load(f)
        with open(os.path.join(CHECKPOINTS, path, 'state_encoder.pkl'), 'rb') as f:
            state_encoder = dill.load(f)
        return VAlignedDictSpaceActionPolicy(policy_per_va_dict=policy_per_va_dict, env=ref_env, state_encoder=state_encoder, **init_params)
                
    def learn(self, alignment_function=None, grounding_function=None, reward=None, discount=1.0, stochastic=True, policy_approximation_method=None, **kwargs):
        
        #print("REWARD", reward)
        if reward is None:
            reward_matrix = self.env.reward_matrix_per_align_func(alignment_function, custom_grounding=grounding_function)
            #print("1")
        else:
            reward_matrix = reward()  
            #print("2")
        #print("REWARD_MATRIX", reward_matrix)
        if isinstance(policy_approximation_method, str):
            policy_approximation_method = PolicyApproximators(policy_approximation_method)
        _,_,pi = mce_partition_fh(self.env, reward=reward_matrix, discount=discount, approximator_kwargs=kwargs['approximator_kwargs'], deterministic=not stochastic, policy_approximator=policy_approximation_method)
        self.set_policy_for_va(alignment_function, pi)

class ContextualVAlignedDictSpaceActionPolicy(ContextualValueSystemLearningPolicy, VAlignedDictSpaceActionPolicy):
    
    def __init__(self, *args, env, contextual_reward_matrix: Callable[[Any, Any], np.ndarray], contextual_policy_estimation_kwargs = dict(
            discount=1.0,
            approximator_kwargs={'value_iteration_tolerance': 0.000001, 'iterations': 5000},
            policy_approximator = PolicyApproximators.MCE_ORIGINAL,
            deterministic=True
    ), use_checkpoints=True, state_encoder=None, observation_space=None, action_space=None,  **kwargs):
        super().__init__(*args, env=env, use_checkpoints=use_checkpoints, state_encoder=state_encoder,
                         observation_space=observation_space, action_space=action_space, **kwargs)
        self.contextual_reward_matrix = contextual_reward_matrix # new context, old context -> np.ndarray
        self.contextual_policy_estimation_kwargs = contextual_policy_estimation_kwargs
        self.contextual_policy_estimation_kwargs['horizon'] = self.env.horizon

        self.contextual_policies = dict()
    
    def update_context(self):
        for va in self.policy_per_va_dict.keys():
            new_context = self.get_environ(self.env.get_align_func()).context
            if self.context != new_context:
                if self.context != new_context:
                    if new_context not in self.contextual_policies.keys():
                        self.contextual_policies[new_context] =  dict()
                    if va not in self.contextual_policies[new_context].keys():
                        _,_, pi_va = mce_partition_fh(self.env, reward=self.contextual_reward_matrix(new_context, self.context, align_func=va), **self.contextual_policy_estimation_kwargs)
                        # This is Roadworld testing, remove after no need.
                        np.testing.assert_allclose(self.contextual_reward_matrix(new_context, self.context, align_func=va)[new_context], 0.0)
                        
                        self.contextual_policies[new_context][va] = pi_va
                    self.set_policy_for_va(va, self.contextual_policies[new_context][va])
                    self.context = new_context
        return self.context
    
class ValueSystemLearningPolicyCustomLearner(VAlignedDiscreteDictPolicy):

    def __init__(self, *args, policy_per_va_dict, env, learner_method: Callable[[...], Any], saved_methods_per_va_dict=None, state_encoder=None, expose_state=True, assume_env_produce_state=False, **kwargs):
        super().__init__(*args, policy_per_va_dict=policy_per_va_dict, env=env, state_encoder=state_encoder, expose_state=expose_state, assume_env_produce_state=assume_env_produce_state, **kwargs)
        self.learner_method = learner_method
        self.saved_policies_per_va_dict = dict() if saved_methods_per_va_dict is None else saved_methods_per_va_dict
    def learn(self, alignment_function=None, grounding_function=None, reward=None, discount=1, stochastic=True, **kwargs):
        
        v,q,policy, save_file, save_m, load_m = self.learner_method(environment=self.env, alignment_function=alignment_function, grounding_function=grounding_function, reward=reward, discount=discount, stochastic=stochastic, **kwargs)
        self.saved_policies_per_va_dict[alignment_function] = save_file, save_m, load_m
        self.set_policy_for_va(alignment_function, policy)
    
    def save(self, path='learner_dummy'):
        save_folder = os.path.join(CHECKPOINTS, path)
        os.makedirs(save_folder, exist_ok=True)
        
        with open(os.path.join(save_folder, 'save_methods.pkl'), 'wb') as f:
            dill.dump( self.saved_policies_per_va_dict, f)
        with open(os.path.join(save_folder, 'state_encoder.pkl'), 'wb') as f:
            dill.dump(self.state_encoder, f)
        print("SAVED VAlignedDictSpaceActionPolicy to ", f)

        init_params = {
            'expose_state': self.expose_state,
            'assume_env_produce_state': self.assume_env_produce_state,
            #'state_encoder': None if self.state_encoder is None else serialize_lambda(self.state_encoder),
            'observation_space': None if self.observation_space is None else serialize_space(self.observation_space),
            'action_space': None if self.action_space is None else serialize_space(self.action_space)   
        }
        with open(os.path.join(save_folder, 'init_params.json'), 'wb') as f:
            dill.dump(init_params, f)
    
        learner_save_file = os.path.join(save_folder, 'learner.pkl')
        print("SAVING learner to ", learner_save_file)
        with open(learner_save_file, 'wb') as f:
            dill.dump(self.learner_method, f)
        print("SAVED learner to ", learner_save_file)   
        return save_folder

    def load(ref_env, path='learner_dummy'):
        route = os.path.join(CHECKPOINTS, path, 'save_methods.pkl')
        if not os.path.exists(route):
            raise FileNotFoundError(
                f"Save file {route} does not exist.")
        print("LOADING VAlignedDictSpaceActionPolicy from ", route)
        with open(route, 'rb') as f:
            saved_methods_per_va_dict = dill.load(f)
        policy_per_va_dict = dict()
        for va, (policy_file, save_m, load_m) in saved_methods_per_va_dict.items():
            policy_per_va_dict[va] = load_m(policy_file, ref_env)
        print("LOADED VAlignedDictSpaceActionPolicy from ", route)
        with open(os.path.join(CHECKPOINTS, path, 'init_params.json'), 'rb') as f:
            init_params = dill.load(f)
        with open(os.path.join(CHECKPOINTS, path, 'state_encoder.pkl'), 'rb') as f:
            state_encoder = dill.load(f)
        with open(os.path.join(CHECKPOINTS, path, 'learner.pkl'), 'rb') as f:
            learner = dill.load(f)
        return ValueSystemLearningPolicyCustomLearner(policy_per_va_dict=policy_per_va_dict, saved_methods_per_va_dict=saved_methods_per_va_dict, env=ref_env, learner_method=learner, state_encoder=state_encoder, **init_params)
                

def profile_sampler_in_society(align_func_as_basic_profile_probs):
    index_ = np.random.choice(
        a=len(align_func_as_basic_profile_probs), p=align_func_as_basic_profile_probs)
    target_align_func = [0.0]*len(align_func_as_basic_profile_probs)
    target_align_func[index_] = 1.0
    target_align_func = tuple(target_align_func)
    return target_align_func


def random_sampler_among_trajs(trajs, align_funcs, n_seeds, n_trajs_per_seed, replace=True):

    all_trajs = []
    size = n_seeds*n_trajs_per_seed
    for al in align_funcs:
        trajs_from_al = [
            traj for traj in trajs if traj.infos[0]['align_func'] == al]
        all_trajs.extend(np.random.choice(
            trajs_from_al, replace=replace if len(trajs_from_al) >= size else True, size=size))
    return all_trajs


def sampler_from_policy(policy: ValueSystemLearningPolicy, align_funcs, n_seeds, n_trajs_per_seed, stochastic, horizon, with_reward=True):
    return policy.obtain_trajectories(n_seeds=n_seeds, stochastic=stochastic, repeat_per_seed=n_trajs_per_seed, align_funcs_in_policy=align_funcs, t_max=horizon, with_reward=with_reward, alignments_in_env=align_funcs)


def profiled_society_traj_sampler_from_policy(policy: ValueSystemLearningPolicy, align_funcs, n_seeds, n_trajs_per_seed, stochastic, horizon):
    trajs = []
    for al in align_funcs:
        for rep in range(n_seeds):
            target_align_func = profile_sampler_in_society(al)

            trajs.extend(policy.obtain_trajectories(n_seeds=1, stochastic=stochastic,
                                                    repeat_per_seed=n_trajs_per_seed, align_funcs_in_policy=[target_align_func], t_max=horizon))

    return trajs
