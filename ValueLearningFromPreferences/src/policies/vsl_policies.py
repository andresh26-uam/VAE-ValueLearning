from seals import base_envs
from abc import abstractmethod
import datetime
import signal
import sys
from typing import Any, Callable, Dict, List, Tuple, Union
import gymnasium as gym
import numpy as np


from imitation.data.types import Trajectory, TrajectoryWithRew

from stable_baselines3.common.type_aliases import PyTorchObs
import torch

from stable_baselines3.common.policies import BasePolicy
from envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment
from src.algorithms.utils import PolicyApproximators, mce_partition_fh
from src.dataset_processing.data import TrajectoryWithValueSystemRews



class ValueSystemLearningPolicy(BasePolicy):

    def __init__(self, env: ValueAlignedEnvironment, use_checkpoints=True, state_encoder=None, squash_output: bool = False, observation_space=None, action_space=None, **kwargs):
        if observation_space is None:

            self.observation_space = env.observation_space
        else:
            self.observation_space = observation_space
        if action_space is None:
            self.action_space = env.action_space
        else:
            self.action_space = action_space

        self.use_checkpoints = use_checkpoints
        super().__init__(squash_output=squash_output,
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

    def obtain_trajectory(self, alignment_func_in_policy=None, seed=32, options=None, t_max=None, stochastic=False, exploration=0, only_states=False, with_reward=False, with_grounding=False, alignment_func_in_env=None,
                          recover_previous_alignment_func_in_env=True, end_trajectories_when_ended=False, reward_dtype=None, agent_name='none') -> Trajectory:

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
            v_rews = [[] for _ in range(self.env.n_values)]
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
                        # custom grounding is set from before
                    ), obs_in_state, action, next_obs=next_obs_in_state, info=info_next, custom_grounding=self.env.current_assumed_grounding)
                    assert self.env.get_align_func() == alignment_func_in_env
                    assert reward_should_be == rew
                    assert np.allclose(info_next['state'], state_obs)
                    rews.append(rew)
                if with_grounding:
                    for value_index in range(self.env.n_values):
                        v_rews[value_index].append(self.env.get_reward_per_value(
                            value_index, obs_in_state, action, next_obs=next_obs_in_state, info=info_next, custom_grounding=self.env.current_assumed_grounding))
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
                            with_grounding=False, reward_dtype=None, agent_name='None',
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
                        if not stochastic and exploration == 0.0:
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


class ContextualValueSystemLearningPolicy(ValueSystemLearningPolicy):
    def __init__(self, *args, env, use_checkpoints=True, state_encoder=None, squash_output=False, observation_space=None, action_space=None, **kwargs):
        super().__init__(*args, env=env, use_checkpoints=use_checkpoints, state_encoder=state_encoder,
                         squash_output=squash_output, observation_space=observation_space, action_space=action_space, **kwargs)
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

class VAlignedDiscreteSpaceActionPolicy(ValueSystemLearningPolicy):
    def __init__(self, policy_per_va: Callable[[Any], np.ndarray], env: gym.Env, state_encoder=None, expose_state=True, *args, **kwargs):

        super().__init__(*args, env=env,
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

    def __init__(self,  *args, policy_per_va_dict: Dict[Tuple, np.ndarray], env: gym.Env, state_encoder=None, expose_state=True, **kwargs):

        self.policy_per_va_dict = policy_per_va_dict
        policy_per_va = self._callable_from_dict()
        super().__init__(policy_per_va=policy_per_va, env=env,
                         state_encoder=state_encoder, expose_state=expose_state, *args, **kwargs)

    def set_policy_for_va(self, va, policy: np.ndarray):
        self.policy_per_va_dict[va] = policy


class ContextualVAlignedDictSpaceActionPolicy(ContextualValueSystemLearningPolicy, VAlignedDictSpaceActionPolicy):

    def __init__(self, *args, env, contextual_reward_matrix: Callable[[Any, Any], np.ndarray], contextual_policy_estimation_kwargs=dict(
            discount=1.0,
            approximator_kwargs={
                'value_iteration_tolerance': 0.000001, 'iterations': 5000},
            policy_approximator=PolicyApproximators.MCE_ORIGINAL,
            deterministic=True
    ), use_checkpoints=True, state_encoder=None, squash_output=False, observation_space=None, action_space=None,  **kwargs):
        super().__init__(*args, env=env, use_checkpoints=use_checkpoints, state_encoder=state_encoder,
                         squash_output=squash_output, observation_space=observation_space, action_space=action_space, **kwargs)
        # new context, old context -> np.ndarray
        self.contextual_reward_matrix = contextual_reward_matrix
        self.contextual_policy_estimation_kwargs = contextual_policy_estimation_kwargs
        self.contextual_policy_estimation_kwargs['horizon'] = self.env.horizon

        self.contextual_policies = dict()

    def update_context(self):
        for va in self.policy_per_va_dict.keys():
            new_context = self.get_environ(self.env.get_align_func()).context
            if self.context != new_context:
                if self.context != new_context:
                    if new_context not in self.contextual_policies.keys():
                        self.contextual_policies[new_context] = dict()
                    if va not in self.contextual_policies[new_context].keys():
                        _, _, pi_va = mce_partition_fh(self.env, reward=self.contextual_reward_matrix(
                            new_context, self.context, align_func=va), **self.contextual_policy_estimation_kwargs)
                        # This is Roadworld testing, remove after no need.
                        np.testing.assert_allclose(self.contextual_reward_matrix(
                            new_context, self.context, align_func=va)[new_context], 0.0)

                        self.contextual_policies[new_context][va] = pi_va
                    self.set_policy_for_va(
                        va, self.contextual_policies[new_context][va])
                    self.context = new_context
        return self.context


