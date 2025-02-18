from stable_baselines3.ppo import MlpPolicy
from sb3_contrib.common.wrappers import ActionMasker
from minari.serialization import serialize_space, deserialize_space
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.ppo_mask.policies import MlpPolicy as MASKEDMlpPolicy
import json
import os
from seals import base_envs
from abc import abstractmethod
import datetime
import signal
import sys
from typing import Any, Callable, Dict, List, Tuple, Union
import gymnasium as gym
import numpy as np


from imitation.data.types import Trajectory, TrajectoryWithRew

from numpy._typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import PyTorchObs
import torch

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment
from utils import CHECKPOINTS, NpEncoder, deconvert, deserialize_policy_kwargs, serialize_lambda, deserialize_lambda, import_from_string, serialize_policy_kwargs


class ValueSystemLearningPolicy(BasePolicy):

    def __init__(self, *args, env: ValueAlignedEnvironment, use_checkpoints=True, state_encoder=None, squash_output: bool = False, observation_space=None, action_space=None, **kwargs):
        if observation_space is None:

            self.observation_space = env.observation_space
        else:
            self.observation_space = observation_space
        if action_space is None:
            self.action_space = env.action_space
        else:
            self.action_space = action_space

        self.use_checkpoints = use_checkpoints
        super().__init__(*args, squash_output=squash_output,
                         observation_space=self.observation_space, action_space=self.action_space, **kwargs)
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

    def get_learner_for_alignment_function(self, alignment_function):
        pass

    def reset(self, seed=None, state=None):
        return None

    def obtain_observation(self, next_state_obs):
        return next_state_obs

    def obtain_trajectory(self, alignment_func_in_policy=None, seed=32, options=None, t_max=None, stochastic=False, exploration=0, only_states=False, with_reward=False, alignment_func_in_env=None,
                          recover_previous_alignment_func_in_env=True, end_trajectories_when_ended=False, custom_grounding=None) -> Trajectory:

        if alignment_func_in_env is None:
            alignment_func_in_env = alignment_func_in_policy
        if recover_previous_alignment_func_in_env:
            prev_al_env = self.env.get_align_func()
        
        self.env.set_align_func(alignment_func_in_env)

        state_obs, info = self.env.reset(
            seed=seed, options=options) if options is not None else self.env.reset(seed=seed)

        obs_in_state = self.obtain_observation(state_obs)

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
                obs_in_state = self.obtain_observation(state_obs)
                path.append(obs_in_state)

                t += 1
            return path
        else:
            obs = [obs_in_state,]
            rews = []
            acts = []
            infos = []
            # edge_path.append(self.environ.cur_state)
            t = 0
            while not ((terminated or truncated) and end_trajectories_when_ended) and (t_max is None or (t < t_max)):

                action, policy_state = self.act(self.state_encoder(state_obs, info), policy_state=policy_state, exploration=exploration,
                                                stochastic=stochastic, alignment_function=alignment_func_in_policy)

                next_state_obs, rew, terminated, truncated, info_next = self.env.step(
                    action)
                next_obs_in_state = self.obtain_observation(next_state_obs)
                obs.append(next_obs_in_state)
                # state_des = self.environ.get_edge_to_edge_state(obs)

                acts.append(action)
                info_next['align_func'] = alignment_func_in_policy
                info_next['init_state'] = init_state
                info_next['ended'] = terminated or truncated
                infos.append(info_next)
                if with_reward:
                    reward_should_be = self.env.get_reward_per_align_func(self.env.get_align_func(
                    ), obs_in_state, action, next_obs=next_obs_in_state, info=info_next, custom_grounding=self.env.current_assumed_grounding) # custom grounding is set from before
                    assert self.env.get_align_func() == alignment_func_in_env
                    assert reward_should_be == rew
                    assert np.allclose(info_next['state'], state_obs)
                    rews.append(rew)

                state_obs = next_state_obs
                info = info_next
                obs_in_state = next_obs_in_state
                t += 1
                if (t_max is not None and t > t_max) or (end_trajectories_when_ended and info['ended']):
                    break
            acts = np.asarray(acts)
            infos = np.asarray(infos)
            rews = np.asarray(rews)
            obs = np.asarray(obs)
            if recover_previous_alignment_func_in_env and with_reward:
                self.env.set_align_func(prev_al_env)

            if with_reward:
                return TrajectoryWithRew(obs=obs, acts=acts, infos=infos, terminal=terminated, rews=rews)
            else:
                return Trajectory(obs=obs, acts=acts, infos=infos, terminal=terminated)

    def obtain_trajectories(self, n_seeds=100, seed=32,
                            options: Union[None, List, Dict] = None, stochastic=True, repeat_per_seed=1, align_funcs_in_policy=[None,], t_max=None,
                            exploration=0, with_reward=False, alignments_in_env=[None,],
                            end_trajectories_when_ended=True,
                            from_initial_states=None) -> List[Trajectory]:
        trajs = []
        if len(alignments_in_env) != len(align_funcs_in_policy):
            alignments_in_env = align_funcs_in_policy
        trajs_sus_sus = []
        trajs_eff_sus = []
        trajs_sus_eff = []
        trajs_eff_eff = []
        if isinstance(self.env, gym.Wrapper):
            possibly_unwrapped = self.env.unwrapped
        else:
            possibly_unwrapped = self.env
        if isinstance(possibly_unwrapped, base_envs.ResettablePOMDP):
            has_initial_state_dist = True
            prev_init_state_dist = possibly_unwrapped.initial_state_dist

            base_dist = np.zeros_like(prev_init_state_dist)
        for si in range(n_seeds):
            if has_initial_state_dist and from_initial_states is not None:
                assert repeat_per_seed == 1
                base_dist[from_initial_states[si]] = 1.0
                self.env.set_initial_state_distribution(base_dist)
            for af, af_in_env in zip(align_funcs_in_policy, alignments_in_env):
                for r in range(repeat_per_seed):

                    traj = self.obtain_trajectory(af,
                                                  seed=seed*n_seeds+si,
                                                  exploration=exploration,
                                                  end_trajectories_when_ended=end_trajectories_when_ended,
                                                  options=options[si] if isinstance(options, list) else options, t_max=t_max, stochastic=stochastic, only_states=False,
                                                  with_reward=with_reward, alignment_func_in_env=af_in_env)
                    trajs.append(
                        traj
                    )
                    if __debug__:
                        if not stochastic and exploration == 0.0:
                            if af == (1.0, 0.0, 0.0):
                                traj_w = self.obtain_trajectory(af,
                                                                seed=seed*n_seeds+si,
                                                                exploration=exploration,
                                                                end_trajectories_when_ended=end_trajectories_when_ended,
                                                                options=options[si] if isinstance(options, list) else options, t_max=t_max,
                                                                stochastic=stochastic, only_states=False,
                                                                with_reward=True, alignment_func_in_env=(1.0, 0.0, 0.0))

                                trajs_sus_sus.append(traj_w)
                                assert np.all(traj_w.obs == traj.obs)

                                traj_w2 = self.obtain_trajectory(af,
                                                                 seed=seed*n_seeds+si,
                                                                 exploration=exploration,
                                                                 end_trajectories_when_ended=end_trajectories_when_ended,
                                                                 options=options[si] if isinstance(options, list) else options, t_max=t_max,
                                                                 stochastic=stochastic, only_states=False,
                                                                 with_reward=True, alignment_func_in_env=(0.0, 0.0, 1.0))

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
                                                                with_reward=True, alignment_func_in_env=(1.0, 0.0, 0.0))
                                trajs_eff_sus.append(traj_w)
                                # print(traj.obs, traj_w.obs, seed, n_seeds, si)
                                assert np.all(traj_w.obs == traj.obs)

                                traj_w2 = self.obtain_trajectory(af,
                                                                 seed=seed*n_seeds+si,
                                                                 exploration=exploration,
                                                                 end_trajectories_when_ended=end_trajectories_when_ended,
                                                                 options=options[si] if isinstance(options, list) else options, t_max=t_max,
                                                                 stochastic=stochastic, only_states=False,
                                                                 with_reward=True, alignment_func_in_env=(0.0, 0.0, 1.0))

                                trajs_eff_eff.append(traj_w2)
                                # print(traj.obs, traj_w2.obs, seed, n_seeds, si)
                                assert np.all(traj_w2.obs == traj.obs)
            if has_initial_state_dist and from_initial_states is not None:
                base_dist[from_initial_states[si]] = 0.0
        # testing in roadworld...
        if __debug__ and (not stochastic and len(af) == 3 and exploration == 0.0):
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

    def _save_checkpoint(self, save_last=True):
        self.save("checkpoints/" + self._get_name() + "_" +
                  str(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')))

    def calculate_value_grounding_expectancy(self, value_grounding: Callable[[np.ndarray], np.ndarray], target_align_func, align_function_sampler=None, n_align_func_samples=1, n_seeds=100, n_rep_per_seed=10, exploration=0, stochastic=True, t_max=None, seed=None, p_state=None, env_seed=None, options=None, initial_state_distribution=None):
        if align_function_sampler is not None:
            trajs = []
            for al_rep in range(n_align_func_samples):
                al_fun = align_function_sampler(target_align_func)
                trajs.extend(self.obtain_trajectories(n_seeds=n_seeds, seed=seed, options=options, stochastic=stochastic,
                             repeat_per_seed=n_rep_per_seed, align_funcs_in_policy=[al_fun,], t_max=t_max, exploration=exploration))
        else:
            trajs = self.obtain_trajectories(n_seeds=n_seeds, seed=seed, options=options, stochastic=stochastic,
                                             repeat_per_seed=n_rep_per_seed, align_funcs_in_policy=[al_fun,], t_max=t_max, exploration=exploration)
        expected_gr = None
        for t in trajs:
            cur_t_gr = None
            for to in t.obs:
                if cur_t_gr is None:
                    cur_t_gr = value_grounding(to)
                else:
                    cur_t_gr += value_grounding(to)
            cur_t_gr /= len(t)
            expected_gr += cur_t_gr
        expected_gr /= len(trajs)
        return expected_gr

    def get_environ(self, alignment_function):
        return self.env


def action_mask(valid_actions, action_shape, *va_args):
    valid_actions = valid_actions(*va_args)

    # Step 2: Create a zeroed-out array of the same shape as probs
    mask = np.zeros(shape=action_shape, dtype=np.bool_)
    mask[valid_actions] = True

    return mask


class LearnerValueSystemLearningPolicy(ValueSystemLearningPolicy):
    def learn(self, alignment_function, total_timesteps, callback=None, log_interval=1, tb_log_name='', reset_num_timesteps=False, progress_bar=True):
        learner = self.get_learner_for_alignment_function(alignment_function)
        learner.learn(total_timesteps=total_timesteps, tb_log_name=f'{tb_log_name}_{total_timesteps}_{
                      alignment_function}', callback=callback, reset_num_timesteps=reset_num_timesteps, progress_bar=progress_bar)
        self.learner_per_align_func[alignment_function] = learner

    def save(self, path='learner_dummy'):
        save_folder = os.path.join(CHECKPOINTS, path)
        print("SAVING LEARNER VSL POLICY TO ", save_folder)
        os.makedirs(save_folder, exist_ok=True)

        # Save learners
        for alignment, learner in self.learner_per_align_func.items():
            learner_path = os.path.join(save_folder, f'alignment_{alignment}')
            learner.save(learner_path)

        # Serialize policy_kwargs with special handling for classes
        serialized_policy_kwargs = serialize_policy_kwargs(self.policy_kwargs)

        # Save initialization parameters (excluding env)
        init_params = {
            'learner_class': self.learner_class.__module__ + "." + self.learner_class.__name__ if type(self.learner_class) != str else self.learner_class,
            'learner_kwargs': self.learner_kwargs,
            'policy_class': self.policy_class.__module__ + "." + self.policy_class.__name__ if type(self.policy_class) != str else self.policy_class,
            'policy_kwargs': serialized_policy_kwargs,
            'masked': self.masked,
            'use_checkpoints': self.use_checkpoints,
            'state_encoder': None if self.state_encoder is None else serialize_lambda(self.state_encoder),
            'squash_output': self.squash_output,
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

        # Load learners
        prev_ob_space = ref_env.observation_space
        for file_name in os.listdir(save_folder):
            if file_name.startswith("alignment_"):
                alignment = file_name[len("alignment_"):]
                learner_path = os.path.join(save_folder, file_name)
                ref_env.observation_space = ref_env.state_space

                new_instance.learner_per_align_func[alignment] = init_params['learner_class'].load(
                    learner_path, env=ref_env)
        ref_env.observation_space = prev_ob_space
        print("LOADED LEARNER VSL POLICY FROM ", save_folder)
        return new_instance

    def __init__(self, *args, env, learner_class: BaseAlgorithm = PPO, learner_kwargs={'learning_rate': 0.1, }, policy_class=MlpPolicy, policy_kwargs={}, masked=False, use_checkpoints=True, state_encoder=None, squash_output=False, observation_space=None, action_space=None,  **kwargs):
        super().__init__(*args, env=env, use_checkpoints=use_checkpoints, state_encoder=state_encoder,
                         squash_output=squash_output, observation_space=observation_space, action_space=action_space, **kwargs)
        self.learner_per_align_func: Dict[str, PPO] = dict()
        self.learner_class = learner_class
        self.learner_kwargs = learner_kwargs
        self.policy_kwargs = policy_kwargs
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
        if self.masked:
            self.learner_class = MaskablePPO

        self.policy_class = policy_class if not self.masked else MASKEDMlpPolicy
        self._sampling_learner = None
        self._alignment_func_in_policy = None

    def get_environ(self, alignment_function):
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

    def obtain_trajectory(self, alignment_func_in_policy=None, seed=32, options=None, t_max=None, stochastic=False, exploration=0, only_states=False, with_reward=False, alignment_func_in_env=None, recover_previous_alignment_func_in_env=True, end_trajectories_when_ended=False):
        if alignment_func_in_policy != self._alignment_func_in_policy or self._sampling_learner is None or self._alignment_func_in_policy is None:
            self._sampling_learner, self._alignment_func_in_policy = self.get_learner_for_alignment_function(
                alignment_function=alignment_func_in_policy), alignment_func_in_policy
        traj = super().obtain_trajectory(alignment_func_in_policy, seed, options, t_max, stochastic, exploration, only_states,
                                         with_reward, alignment_func_in_env, recover_previous_alignment_func_in_env, end_trajectories_when_ended)

        return traj

    def create_algo(self, environ=None, alignment_function=None):
        if environ is None:
            environ = self.get_environ(alignment_function)
        return self.learner_class(policy=self.policy_class, env=environ, policy_kwargs=self.policy_kwargs, **self.learner_kwargs)

    def obtain_trajectories(self, n_seeds=100, seed=32, options=None, stochastic=True, repeat_per_seed=1, align_funcs_in_policy=[None], t_max=None, exploration=0, with_reward=False, alignments_in_env=[None], end_trajectories_when_ended=True, from_initial_states=None):
        if self.env_is_tabular:
            self._act_prob_cache = dict()
        return super().obtain_trajectories(n_seeds=n_seeds, seed=seed, options=options, stochastic=stochastic, repeat_per_seed=repeat_per_seed, align_funcs_in_policy=align_funcs_in_policy, t_max=t_max, exploration=exploration, with_reward=with_reward, alignments_in_env=alignments_in_env, end_trajectories_when_ended=end_trajectories_when_ended, from_initial_states=from_initial_states)

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
                if self.env_is_tabular:
                    act_prob = learner.policy.get_distribution(
                        learner.policy.obs_to_tensor(state_obs)[0])
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
            assert isinstance(a, int)
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
            assert True is False  # This should never ever occur
            if len(valid_actions) == 0:
                a = menv.action_space.sample()
            else:
                a = np.random.choice(valid_actions)
            assert isinstance(a, int)
            next_policy_state = None if policy_state is None else policy_state + \
                1  # (Not used)
        # pf_total_finish = time.perf_counter()
        assert a in valid_actions
        assert valid_distribution[a] > 0
        return a, next_policy_state, valid_distribution


class VAlignedDiscreteSpaceActionPolicy(ValueSystemLearningPolicy):
    def __init__(self, policy_per_va: Callable[[Any], np.ndarray], env: gym.Env, state_encoder=None, expose_state=True, *args, **kwargs):

        super().__init__(*args, env=env, use_checkpoints=True,
                         state_encoder=state_encoder, **kwargs)
        self.policy_per_va = policy_per_va

        self.env: TabularVAMDP = base_envs.ExposePOMDPStateWrapper(env)
        self.expose_state = expose_state
        # self.env = env

    def reset(self, seed=None, state=None):
        policy_state = state if state is not None else 0
        return policy_state

    def obtain_observation(self, next_state_obs):
        if self.expose_state is False:
            obs_in_state = self.env.obs_from_state(next_state_obs)
        else:
            obs_in_state = next_state_obs
        return obs_in_state

    def calculate_value_grounding_expectancy(self, value_grounding: np.ndarray, align_function, n_seeds=100, n_rep_per_seed=10, exploration=0, stochastic=True, t_max=None, seed=None, p_state=None, env_seed=None, options=None, initial_state_distribution=None):

        pi = self.policy_per_va(align_function)
        self.reset(seed=seed, state=p_state)
        self.env.reset(seed=env_seed, options=options)

        if initial_state_distribution is None:
            initial_state_distribution = np.ones(
                (self.env.state_dim))/self.env.state_dim
        initial_state_dist = initial_state_distribution

        state_dist = initial_state_dist
        accumulated_feature_expectations = 0

        assert (value_grounding.shape[0], value_grounding.shape[1]) == (
            self.env.state_dim, self.env.action_dim)

        for t in range(self.env.horizon):

            pol_t = pi if len(pi.shape) == 2 else pi[t]
            state_action_prob = np.multiply(pol_t, state_dist[:, np.newaxis])

            features_time_t = np.sum(
                value_grounding * state_action_prob[:, :, np.newaxis], axis=(0, 1))/self.env.state_dim

            if t == 0:
                accumulated_feature_expectations = features_time_t
            else:
                accumulated_feature_expectations += features_time_t
            # /self.env.state_dim
            state_dist = np.sum(self.env.transition_matrix *
                                state_action_prob[:, :, np.newaxis], axis=(0, 1))
            assert np.allclose(np.sum(state_dist), 1.0)
        return accumulated_feature_expectations

    def act(self, state_obs: int, policy_state=None, exploration=0, stochastic=True, alignment_function=None):
        policy = self.policy_per_va(alignment_function)
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


class VAlignedDictSpaceActionPolicy(VAlignedDiscreteSpaceActionPolicy):

    def _callable_from_dict(self):
        return lambda x: self.policy_per_va_dict[x]

    def __init__(self, policy_per_va_dict: Dict[Tuple, np.ndarray], env: gym.Env, state_encoder=None, expose_state=True, *args, **kwargs):

        self.policy_per_va_dict = policy_per_va_dict
        policy_per_va = self._callable_from_dict()
        super().__init__(policy_per_va=policy_per_va, env=env,
                         state_encoder=state_encoder, expose_state=expose_state, *args, **kwargs)

    def set_policy_for_va(self, va, policy: np.ndarray):
        self.policy_per_va_dict[va] = policy


class VAlignedDictDiscreteStateActionPolicyTabularMDP(VAlignedDictSpaceActionPolicy):
    def __init__(self, policy_per_va_dict: Dict[Tuple, NDArray], env: TabularVAMDP, state_encoder=None, expose_state=True, *args, **kwargs):

        super().__init__(policy_per_va_dict=policy_per_va_dict, env=env,
                         state_encoder=state_encoder, expose_state=expose_state, *args, **kwargs)


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
