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
from stable_baselines3.common.type_aliases import PyTorchObs
import torch

from stable_baselines3.common.policies import BasePolicy

from src.envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment


class ValueSystemLearningPolicy(BasePolicy):

    def __init__(self, *args, env: ValueAlignedEnvironment, use_checkpoints=True, state_encoder=None, squash_output: bool = False, observation_space=None, action_space=None, **kwargs):
        if observation_space is None:

            self.observation_space = env.observation_space

        if action_space is None:
            self.action_space = env.action_space
        super().__init__(*args, squash_output=squash_output,
                         observation_space=self.observation_space, action_space=self.action_space, **kwargs)
        self.env = env
        self.default_alignment = None
        self.default_exploration = 0
        self.state_encoder = state_encoder if state_encoder is not None else (
            lambda st, info: st)

        if use_checkpoints:
            def _save_checkpoint_and_exit(sig, frame):
                print("Ctrl-C pressed. Saving checkpoint for policy: " +
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

    def fit_for_alignment_function(self, alignment_function):
        pass

    def reset(self, seed=None, state=None):
        return None

    def obtain_trajectory(self, alignment_function=None, seed=32, options: Union[None, Dict] = None, t_max=None, stochastic=False, exploration=0, only_states=False, with_reward=False, alignment_func_in_env=None, recover_previous_alignment_func_in_env=True) -> Union[List, Trajectory]:
        state_obs, info = self.env.reset(
            seed=seed, options=options) if options is not None else self.env.reset(seed=seed)

        if with_reward:
            if alignment_func_in_env is None:
                alignment_func_in_env = alignment_function
            if recover_previous_alignment_func_in_env:
                prev_al_env = self.env.cur_align_func
            self.env.cur_align_func = alignment_func_in_env

        policy_state = self.reset(seed=seed)
        init_state = self.state_encoder(state_obs, info)
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
            path.append(state_obs)
            t = 0
            while not (terminated or truncated) and (t_max is None or (t < t_max)):

                action, policy_state = self.act(state_obs, policy_state=policy_state, exploration=exploration,
                                                stochastic=stochastic, alignment_function=alignment_function)
                # print(self.environ.cur_state, action, index)
                state_obs, rew, terminated, truncated, info = self.env.step(
                    action)

                path.append(state_obs)

                t += 1
            return path
        else:
            obs = [state_obs,]
            rews = []
            acts = []
            infos = []
            # edge_path.append(self.environ.cur_state)
            t = 0
            while not (terminated or truncated) and (t_max is None or (t < t_max)):
                action, policy_state = self.act(state_obs, policy_state=policy_state, exploration=exploration,
                                                stochastic=stochastic, alignment_function=alignment_function)
                state_obs, rew, terminated, truncated, info = self.env.step(
                    action)

                obs.append(state_obs)
                # state_des = self.environ.get_edge_to_edge_state(obs)

                acts.append(action)
                info['align_func'] = alignment_function
                info['init_state'] = init_state
                info['ended'] = terminated or truncated
                infos.append(info)
                if with_reward:
                    rews.append(rew)
                t += 1
                if t_max is not None and t > t_max:
                    break
            acts = np.asarray(acts)
            infos = np.asarray(infos)
            rews = np.asarray(rews)
            if recover_previous_alignment_func_in_env:
                self.env.cur_align_func = prev_al_env

            if with_reward:
                return TrajectoryWithRew(obs=obs, acts=acts, infos=infos, terminal=terminated, rews=rews)
            else:
                return Trajectory(obs=obs, acts=acts, infos=infos, terminal=terminated)

    def obtain_trajectories(self, n_seeds=100, seed=32,
                            options: Union[None, List, Dict] = None, stochastic=True,
                            custom_discount=None, repeat_per_seed=1, with_alignfunctions=[None,], t_max=None,
                            exploration=0, with_reward=False, alignments_in_env=[None,],
                            use_observations=False, end_trajectories_when_ended=True) -> List[Trajectory]:
        trajs = []
        if len(alignments_in_env) != len(with_alignfunctions):
            alignments_in_env = with_alignfunctions
        trajs_sus_sus = []
        trajs_eff_sus = []
        trajs_sus_eff = []
        trajs_eff_eff = []
        for si in range(n_seeds):

            for r in range(repeat_per_seed):
                for af, af_in_env in zip(with_alignfunctions, alignments_in_env):
                    traj = self.obtain_trajectory(af,
                                               seed=seed*n_seeds+si,
                                               exploration=exploration,
                                               custom_discount=custom_discount,
                                               end_trajectories_when_ended=end_trajectories_when_ended,
                                               options=options[si] if isinstance(options, list) else options, t_max=t_max, stochastic=stochastic, only_states=False, 
                                               with_reward=with_reward, alignment_func_in_env=af_in_env)
                    trajs.append(
                        traj
                    )
                    if not stochastic:
                        if af == (1.0,0.0,0.0):
                            traj_w = self.obtain_trajectory(af,
                                                seed=seed*n_seeds+si,
                                                exploration=exploration,
                                                custom_discount=custom_discount,
                                                end_trajectories_when_ended=end_trajectories_when_ended,
                                                options=options[si] if isinstance(options, list) else options, t_max=t_max, 
                                                stochastic=stochastic, only_states=False, 
                                                with_reward=True, alignment_func_in_env=(1.0,0.0,0.0))
                        
                            trajs_sus_sus.append(traj_w)
                            print(traj.obs, traj_w.obs, seed, n_seeds, si)
                            assert np.all(traj_w.obs == traj.obs)

                            traj_w2 = self.obtain_trajectory(af,
                                                seed=seed*n_seeds+si,
                                                exploration=exploration,
                                                custom_discount=custom_discount,
                                                end_trajectories_when_ended=end_trajectories_when_ended,
                                                options=options[si] if isinstance(options, list) else options, t_max=t_max, 
                                                stochastic=stochastic, only_states=False, 
                                                with_reward=True, alignment_func_in_env=(0.0,0.0,1.0))
                        
                            trajs_sus_eff.append(traj_w2)
                            #print(traj.obs, traj_w2.obs, seed, n_seeds, si)
                            assert np.all(traj_w2.obs == traj.obs)
                        elif af == (0.0,0.0,1.0):
                            traj_w = self.obtain_trajectory(af,
                                                seed=seed*n_seeds+si,
                                                exploration=exploration,
                                                custom_discount=custom_discount,
                                                end_trajectories_when_ended=end_trajectories_when_ended,
                                                options=options[si] if isinstance(options, list) else options, t_max=t_max, 
                                                stochastic=stochastic, only_states=False, 
                                                with_reward=True, alignment_func_in_env=(1.0,0.0,0.0))
                            trajs_eff_sus.append(traj_w)
                            #print(traj.obs, traj_w.obs, seed, n_seeds, si)
                            assert np.all(traj_w.obs == traj.obs)

                            traj_w2 = self.obtain_trajectory(af,
                                                seed=seed*n_seeds+si,
                                                exploration=exploration,
                                                custom_discount=custom_discount,
                                                end_trajectories_when_ended=end_trajectories_when_ended,
                                                options=options[si] if isinstance(options, list) else options, t_max=t_max, 
                                                stochastic=stochastic, only_states=False, 
                                                with_reward=True, alignment_func_in_env=(0.0,0.0,1.0))
                        
                            trajs_eff_eff.append(traj_w2)
                            #print(traj.obs, traj_w2.obs, seed, n_seeds, si)
                            assert np.all(traj_w2.obs == traj.obs)
        if not stochastic and len(af) == 3: # testing in roadworld...
            for t,t2 in zip(trajs_sus_sus, trajs_eff_sus):
                #print(self.policy_per_va((1.0,0.0,0.0))[t.obs[1]])
                #print(self.policy_per_va((0.0,0.0,1.0))[t.obs[1]])
                assert np.sum(t.rews) >= np.sum(t2.rews)
                print("CHECK!!!")
                assert t.obs[0] == t2.obs[0] and t.obs[-1] == t2.obs[-1]

            for t,t2 in zip(trajs_eff_eff, trajs_sus_eff):
                #print(self.policy_per_va((1.0,0.0,0.0))[t.obs[1]])
                #print(self.policy_per_va((0.0,0.0,1.0))[t.obs[1]])
                print(t.obs, t2.obs)
                assert np.sum(t.rews) >= np.sum(t2.rews)
                print("CHECK EFF!!!")
                assert t.obs[0] == t2.obs[0] and t.obs[-1] == t2.obs[-1]

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
                             repeat_per_seed=n_rep_per_seed, with_alignfunctions=[al_fun,], t_max=t_max, exploration=exploration))
        else:
            trajs = self.obtain_trajectories(n_seeds=n_seeds, seed=seed, options=options, stochastic=stochastic,
                                             repeat_per_seed=n_rep_per_seed, with_alignfunctions=[al_fun,], t_max=t_max, exploration=exploration)
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

    def obtain_trajectory(self, alignment_function=None, seed=32, options=None, t_max=None, stochastic=False, exploration=0, only_states=False, with_reward=False, alignment_func_in_env=None,
                          custom_discount=None, recover_previous_alignment_func_in_env=True, end_trajectories_when_ended=False) -> Trajectory:
        state_obs, info = self.env.reset(
            seed=seed, options=options) if options is not None else self.env.reset(seed=seed)
        if self.expose_state is False:
            obs_in_state = self.env.obs_from_state(state_obs)
        else:
            obs_in_state = state_obs

        if with_reward:
            if alignment_func_in_env is None:
                alignment_func_in_env = alignment_function
            if recover_previous_alignment_func_in_env:
                prev_al_env = self.env.cur_align_func
            self.env.cur_align_func = alignment_func_in_env
        policy_state = self.reset(seed=seed)
        init_state = self.state_encoder(state_obs, info)
        if alignment_func_in_env is not None:
            r_matrix = self.env.reward_matrix_per_align_func(
                alignment_func_in_env)

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

                action, policy_state = self.act(state_obs, policy_state=policy_state, exploration=exploration,
                                                stochastic=stochastic, alignment_function=alignment_function)
                # print(self.environ.cur_state, action, index)
                state_obs, rew, terminated, truncated, info = self.env.step(
                    action)
                if self.expose_state is False:
                    obs_in_state = self.env.obs_from_state(state_obs)
                else:
                    obs_in_state = state_obs
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
                action, policy_state = self.act(state_obs, policy_state=policy_state, exploration=exploration,
                                                stochastic=stochastic, alignment_function=alignment_function)
                next_state_obs, rew, terminated, truncated, info = self.env.step(
                    action)
                if self.expose_state is False:
                    obs_in_state = self.env.obs_from_state(next_state_obs)
                else:
                    obs_in_state = next_state_obs
                obs.append(obs_in_state)
                # state_des = self.environ.get_edge_to_edge_state(obs)

                acts.append(action)
                info['align_func'] = alignment_function
                info['init_state'] = init_state
                info['ended'] = terminated or truncated
                infos.append(info)
                if with_reward:
                    rews.append(r_matrix[state_obs, action])

                state_obs = next_state_obs
                t += 1
                if (t_max is not None and t > t_max) or (end_trajectories_when_ended and info['ended']):
                    break
            acts = np.asarray(acts)
            infos = np.asarray(infos)
            rews = np.asarray(rews)
            obs = np.asarray(obs)
            if recover_previous_alignment_func_in_env and with_reward:
                self.env.cur_align_func = prev_al_env

            if with_reward:
                return TrajectoryWithRew(obs=obs, acts=acts, infos=infos, terminal=terminated, rews=rews)
            else:
                return Trajectory(obs=obs, acts=acts, infos=infos, terminal=terminated)

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
            filtered_probs[:] = 1.0 / len(filtered_probs)  # or some default uniform distribution

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


def profiled_society_sampler(align_func_as_basic_profile_probs):
    index_ = np.random.choice(
        a=len(align_func_as_basic_profile_probs), p=align_func_as_basic_profile_probs)
    target_align_func = [0.0]*len(align_func_as_basic_profile_probs)
    target_align_func[index_] = 1.0
    target_align_func = tuple(target_align_func)
    return target_align_func


def random_sampler_among_trajs(trajs, align_funcs, n_seeds, n_trajs_per_seed):

    all_trajs = []
    for al in align_funcs:
        all_trajs.extend(np.random.choice(
            [traj for traj in trajs if traj.infos[0]['align_func'] == al], replace=True, size=n_seeds*n_trajs_per_seed))
    return all_trajs


def sampler_from_policy(policy: ValueSystemLearningPolicy, align_funcs, n_seeds, n_trajs_per_seed, stochastic, horizon):
    return policy.obtain_trajectories(n_seeds=n_seeds, stochastic=stochastic, repeat_per_seed=n_trajs_per_seed, with_alignfunctions=align_funcs, t_max=horizon)


def profiled_society_traj_sampler_from_policy(policy: ValueSystemLearningPolicy, align_funcs, n_seeds, n_trajs_per_seed, stochastic, horizon):
    trajs = []
    for al in align_funcs:
        for rep in range(n_seeds):
            target_align_func = profiled_society_sampler(al)

            trajs.extend(policy.obtain_trajectories(n_seeds=1, stochastic=stochastic,
                                                    repeat_per_seed=n_trajs_per_seed, with_alignfunctions=[target_align_func], t_max=horizon))

    return trajs
