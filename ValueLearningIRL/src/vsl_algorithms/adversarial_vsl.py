import dataclasses
from typing import Iterator, Optional

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from src.envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment
from src.vsl_algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.vsl_policies import ValueSystemLearningPolicy, VAlignedDictDiscreteStateActionPolicyTabularMDP
from src.vsl_reward_functions import AbstractVSLRewardFunction, TrainingModes


from imitation.algorithms.adversarial.common import compute_train_stats


from gymnasium import spaces

import torch as th

from src.vsl_policies import LearnerValueSystemLearningPolicy, ValueSystemLearningPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm

from typing import Iterable, Iterator, Mapping, Optional, Type
from typing import (
    Any,
    Dict,
    Union,
)

import numpy as np
import torch as th
from torch.nn import functional as F
from imitation.rewards import reward_nets
from imitation.algorithms import base
from imitation.data import rollout, types, wrappers
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import logger as imit_logger
import imitation.algorithms.adversarial.common as common

from stable_baselines3.common.torch_layers import FlattenExtractor


class AIRLforVSL(common.AdversarialTrainer):
    """Adversarial Inverse Reinforcement Learning (`AIRL`_).

    .. _AIRL: https://arxiv.org/abs/1710.11248
    """

    def _make_disc_train_batches(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> Iterator[Mapping[str, th.Tensor]]:
        """Adapted from the imitation package.
        """
        batch_size = self.demo_batch_size

        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first.",
                )

            gen_samples_dataclass = self._gen_replay_buffer.sample(batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples_dataclass)

        if not (len(gen_samples["obs"]) == len(expert_samples["obs"]) == batch_size):
            raise ValueError(
                "Need to have exactly `demo_batch_size` number of expert and "
                "generator samples, each. "
                f"(n_gen={len(gen_samples['obs'])} "
                f"n_expert={len(expert_samples['obs'])} "
                f"demo_batch_size={batch_size})",
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        assert batch_size == len(expert_samples["acts"])
        assert batch_size == len(expert_samples["next_obs"])
        assert batch_size == len(gen_samples["acts"])
        assert batch_size == len(gen_samples["next_obs"])

        for start in range(0, batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            # take minibatch slice (this creates views so no memory issues)
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
            gen_batch = {k: v[start:end] for k, v in gen_samples.items()}

            # Concatenate rollouts, and label each row as expert or generator.

            obs = np.concatenate([expert_batch["obs"], gen_batch["obs"]])

            # states = np.concatenate([np.asarray([einfo['state'] for einfo in expert_batch["infos"]]), np.asarray([ginfo['state'] for ginfo in gen_batch["infos"]])])
            # next_states = np.concatenate([np.asarray([einfo['next_state'] for einfo in expert_batch["infos"]]), np.asarray([ginfo['next_state'] for ginfo in gen_batch["infos"]])])
            acts = np.concatenate([expert_batch["acts"], gen_batch["acts"]])

            next_obs = np.concatenate(
                [expert_batch["next_obs"], gen_batch["next_obs"]])
            dones = np.concatenate([expert_batch["dones"], gen_batch["dones"]])
            # notice that the labels use the convention that expert samples are
            # labelled with 1 and generator samples with 0.
            # TODO: THIS might be a problem in some environments... Or the reason behind the learner not learning correctly
            # GENERATOR SAMPLES IF THEY COINCIDE WITH THE EXPERT... THE DICRIMINATOR CANNOT GIVE BOTH 0 and 1 TO THOSE.
            # YOU CANNOT ASSUME ALL GENERATOR TARGETS ARE 0. PERHAPS REMOVE ENTROPY OR SOMETHING LIKE THAT.
            # EQ. 22 of https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9849003
            labels_expert_is_one = np.concatenate(
                [
                    np.ones(len(expert_batch['obs']), dtype=int),
                    np.zeros(len(gen_batch['obs']), dtype=int),
                ],
            )
            # Calculate generator-policy log probabilities.
            with th.no_grad():
                obs_th = th.as_tensor(obs, device=self.gen_algo.device)
                acts_th = th.as_tensor(acts, device=self.gen_algo.device)
                # states_th = th.as_tensor(states, device=self.gen_algo.device)
                # next_states_th = th.as_tensor(next_states, device=self.gen_algo.device)
                log_policy_act_prob = self._get_log_policy_act_prob(
                    obs_th, acts_th)
                if log_policy_act_prob is not None:
                    assert len(log_policy_act_prob) == 2 * \
                        self.demo_minibatch_size
                    log_policy_act_prob = log_policy_act_prob.reshape(
                        (2 * self.demo_minibatch_size,),
                    )
                del obs_th, acts_th  # unneeded

            obs_th, acts_th, next_obs_th, dones_th = self.reward_train.preprocess(
                obs,
                acts,
                next_obs,
                dones,
            )
            batch_dict = {
                "obs": obs_th,
                "action": acts_th,
                "next_obs": next_obs_th,
                "done": dones_th,
                "labels_expert_is_one": self._torchify_array(labels_expert_is_one),
                "log_policy_act_prob": log_policy_act_prob,
            }

            yield batch_dict

    def _reward_function_for_policy_training(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, dones: np.ndarray, ground_truth=False):
        """Proxy for the calculate_rewards method, that overrides the reward call to put constraints, or other special treatment of the reward model
        Only works now in the tabular environments.
        """
        with th.no_grad():
            obs = th.as_tensor(obs, dtype=self.advsl.current_net.dtype,
                               device=self.advsl.current_net.device).long()
            action = th.as_tensor(action, dtype=self.advsl.current_net.dtype,
                                  device=self.advsl.current_net.device).long()
            next_obs = th.as_tensor(
                next_obs, dtype=self.advsl.current_net.dtype, device=self.advsl.current_net.device).long()
            # TODO: possibly add support for the dones attribute...
            reward_real = self.advsl.env.reward_matrix_per_align_func(
                self.advsl.current_target)[obs, action]

            if not ground_truth:
                reward, reward_np = self.advsl.calculate_rewards(self.alignment,
                                                                 grounding=None,
                                                                 obs_mat=obs,
                                                                 action_mat=action,
                                                                 next_state_obs_mat=next_obs,
                                                                 # obs_action_mat TODO maybe...
                                                                 reward_mode=self.advsl.training_mode,
                                                                 recover_previous_config_after_calculation=True,
                                                                 use_probabilistic_reward=False, requires_grad=False)

            else:
                reward_np = reward_real
        return reward_np

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: VecEnv,
        reward_net: AbstractVSLRewardFunction,
        adversarialVSL: BaseVSLAlgorithm,
        gen_algo: BaseAlgorithm = None,
        expert_algo: BaseAlgorithm = None,
        alignment=None,
        stochastic_gen=True,
        **kwargs,
    ):
        """ 
        Based on the imitation version, without assertions on specific classes that made the code not work.
        """
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )
        self.stochastic_gen = stochastic_gen
        self.advsl: AdversarialVSL = adversarialVSL
        self.alignment = alignment

        self.venv_buffering = wrappers.BufferingWrapper(self.venv)

        self.venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
            self.venv_buffering,
            reward_fn=self._reward_function_for_policy_training)
        self.gen_callback = self.venv_wrapped.make_log_callback()
        self.venv_train = self.venv_wrapped

        self.gen_algo.set_env(self.venv_train)
        self.gen_algo.set_logger(self.logger)

        self.ground_truth_gen_algo = expert_algo
        self.ground_truth_env = reward_wrapper.RewardVecEnvWrapper(
            self.venv_buffering,
            reward_fn=lambda o, a, n, d: self._reward_function_for_policy_training(o, a, n, d, ground_truth=True))
        self.last_train_stats = {}

    def logits_expert_is_high(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        next_obs: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        if log_policy_act_prob is None:
            raise TypeError(
                "Non-None `log_policy_act_prob` is required for this method.",
            )

        reward_output_train, _ = self.advsl.calculate_rewards(self.alignment,
                                                              grounding=None,
                                                              obs_mat=obs,
                                                              action_mat=action,
                                                              next_state_obs_mat=next_obs,
                                                              reward_mode=self.advsl.training_mode,
                                                              recover_previous_config_after_calculation=False,
                                                              use_probabilistic_reward=False, requires_grad=True)

        return reward_output_train - log_policy_act_prob

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        """Returns the unshaped version of reward network used for testing."""
        reward_net = self._reward_net
        # Recursively return the base network of the wrapped reward net
        while isinstance(reward_net, reward_nets.RewardNetWrapper):
            reward_net = reward_net.base
        return reward_net

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        """
        Adapted from imitation package
        """
        with self.logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0

            # compute loss
            self._disc_opt.zero_grad()

            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )

            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    obs=batch["obs"],
                    action=batch["action"],
                    next_obs=batch["next_obs"],
                    done=batch["done"],
                    log_policy_act_prob=batch["log_policy_act_prob"],
                )

                loss = F.binary_cross_entropy_with_logits(
                    disc_logits,
                    batch["labels_expert_is_one"].float(),
                )

                # Renormalise the loss to be averaged over the whole
                # batch size instead of the minibatch size.
                assert len(batch["obs"]) == 2 * self.demo_minibatch_size
                loss *= self.demo_minibatch_size / self.demo_batch_size
                loss.backward()

            # do gradient step
            self._disc_opt.step()
            self._disc_step += 1

            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    batch["labels_expert_is_one"],
                    loss,
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram(
                    "disc_logits", disc_logits.detach())
        self.last_train_stats = train_stats
        return train_stats

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ) -> None:
        """Adapted from imitation package
        """
        if total_timesteps is None:
            total_timesteps = self.gen_train_timesteps
        if learn_kwargs is None:
            learn_kwargs = {}

        with self.logger.accumulate_means("gen"):
            self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.gen_callback,
                **learn_kwargs,
            )
            self._global_step += 1

        gen_trajs, ep_lens = self.venv_buffering.pop_trajectories()
        self._check_fixed_horizon(ep_lens)
        gen_samples = rollout.flatten_trajectories_with_rew(gen_trajs)
        self._gen_replay_buffer.store(gen_samples)


class AdversarialVSL(BaseVSLAlgorithm):
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
        stochastic_expert=True,
        environment_is_stochastic=False,
        training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
        # custom parameters:
        learner_class=PPO,  # accepts env, and a policy parameter
        learner_kwargs=dict(batch_size=32,
                            n_steps=1024,
                            ent_coef=0.4,
                            learning_rate=0.001,  # 0.003
                            gamma=1.0,
                            gae_lambda=0.999,
                            clip_range=1.0,
                            vf_coef=0.8,
                            n_epochs=10,
                            normalize_advantage=True,
                            tensorboard_log="./ppo_tensorboard_expert/"
                            ),

        masked_policy=True,
        policy_class=MlpPolicy,
        policy_kwargs=dict(
            features_extractor_class=FlattenExtractor,
            #  net arch is input to output linearly,
            net_arch=dict(pi=[], vf=[]),

        ),

        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,

    ) -> None:
        super().__init__(env, reward_net=reward_net,
                         vgl_optimizer_cls=vgl_optimizer_cls,
                         stochastic_expert=stochastic_expert,
                         environment_is_stochastic=environment_is_stochastic,
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

        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs
        self.learner_class = learner_class
        self.learner_kwargs = learner_kwargs

        self.masked_policy = masked_policy
        self.current_target = None

        self.learned_policy_per_va = LearnerValueSystemLearningPolicy(env=self.env, learner_class=self.learner_class, masked=self.masked_policy,
                                                                      policy_class=self.policy_class, learner_kwargs=self.learner_kwargs, policy_kwargs=self.policy_kwargs, state_encoder=lambda obs, info: obs)

    
    def get_metrics(self):
        metrics = super().get_metrics()
        metrics.update({'accuracy': self.last_accuracies_per_align_func})
        return metrics

    def train_callback(self, t, namespace='mean/'):
        self.trainer.logger.record(namespace+"Target", self.current_target)
        if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION or self.training_mode == TrainingModes.SIMULTANEOUS:
            self.trainer.logger.record(namespace+"Learned", tuple(
                [float(f"{a:0.5f}") for a in self.current_net.get_learned_align_function()]))
        elif self.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING or self.training_mode == TrainingModes.SIMULTANEOUS:
            for i in range(len(self.current_target)):
                self.trainer.logger.record(namespace+f"Learned NN params for value {i}", [p.detach().numpy(
                ) for p in list(self.current_net.get_learned_grounding().networks[i].parameters())])

        self.last_accuracies_per_align_func[self.current_target].append(
            self.trainer.last_train_stats['disc_acc'])

    def calculate_learned_policies(self, target_align_funcs) -> ValueSystemLearningPolicy:
        return self.learned_policy_per_va

    def train(self, max_iter=1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING,
              assumed_grounding=None, n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_probabilistic_reward=False, n_reward_reps_if_probabilistic_reward=10,
              demo_batch_size=512,  # 512
              gen_replay_buffer_capacity=2048,  #  2048 for RW
              n_disc_updates_per_round=2,
              **kwargs
              ):
        self.demo_batch_size = demo_batch_size
        self.gen_replay_buffer_capacity = gen_replay_buffer_capacity
        self.n_disc_updates_per_round = n_disc_updates_per_round
        self.last_accuracies_per_align_func = {al: [] for al in (
            self.vsi_target_align_funcs if mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION else self.vgl_target_align_funcs)}

        return super().train(max_iter, mode, assumed_grounding, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, use_probabilistic_reward, n_reward_reps_if_probabilistic_reward, **kwargs)

    def train_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func) -> Dict[Any, AbstractVSLRewardFunction]:

        learning_algo: PPO = self.learned_policy_per_va.get_learner_for_alignment_function(
            target_align_func)

        demos = self.vsi_sampler(
            [target_align_func], n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)

        env = self.learned_policy_per_va.get_environ(target_align_func)
        venv = DummyVecEnv([lambda: env])

        self.current_net.set_mode(
            mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION)

        self.train_global(max_iter=max_iter, demos=demos, venv=venv,
                          learning_algo=learning_algo, target_align_func=target_align_func)

        return self.current_net.get_learned_align_function()

    def train_vgl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed) -> Dict[Any, AbstractVSLRewardFunction]:
        reward_nets_per_target_align_func = dict()
        for target_align_func in self.vgl_target_align_funcs:
            self.current_net.set_alignment_function(target_align_func)
            self.current_net.set_mode(
                mode=TrainingModes.VALUE_GROUNDING_LEARNING)

            learning_algo: PPO = self.learned_policy_per_va.get_learner_for_alignment_function(
                target_align_func)

            demos = self.vgl_sampler(
                [target_align_func], n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)

            env = self.learned_policy_per_va.get_environ(target_align_func)
            venv = DummyVecEnv([lambda: env])

            self.train_global(max_iter=max_iter, demos=demos, venv=venv,
                              learning_algo=learning_algo, target_align_func=target_align_func)
            reward_nets_per_target_align_func[target_align_func] = self.current_net.copy(
            )
        return reward_nets_per_target_align_func

    def train_global(self, max_iter, demos, venv, learning_algo, target_align_func):
        self.current_target = target_align_func

        learning_algo.set_env(env=venv)

        self.trainer = AIRLforVSL(
            demonstrations=demos,
            demo_batch_size=self.demo_batch_size,
            gen_replay_buffer_capacity=self.gen_replay_buffer_capacity,
            n_disc_updates_per_round=self.n_disc_updates_per_round,
            venv=venv,
            gen_algo=learning_algo,
            reward_net=self.current_net,
            allow_variable_horizon=True,
            adversarialVSL=self,
            alignment=None if self.training_mode != TrainingModes.VALUE_GROUNDING_LEARNING else self.current_target,
            disc_opt_cls=self.vsi_optimizer_cls,
            disc_opt_kwargs=self.vsi_optimizer_kwargs,
            stochastic_gen=self.learn_stochastic_policy
        )

        # Train for 2_000_000 steps to match expert.
        self.trainer.train(max_iter, callback=self.train_callback)

    def train_vsl_probabilistic(self, max_iter,
                                n_seeds_for_sampled_trajectories,
                                n_sampled_trajs_per_seed,
                                n_reward_reps_if_probabilistic_reward,
                                target_align_func):
        raise NotImplementedError("probabilistic vsl not implemented")

    def get_tabular_policy_from_reward_per_align_func(self, align_funcs, reward_net_per_al: Dict[tuple, AbstractVSLRewardFunction], expert=False, random=False,
                                                      use_probabilistic_reward=False,
                                                      state_encoder=None, expose_state=True, precise_deterministic=False):
        if not isinstance(self.env.state_space, spaces.Discrete) and isinstance(self.env.action_space, spaces.Discrete):
            raise NotImplementedError(
                "not implemented on non discrete state-action spaces...")

        reward_matrix_per_al = dict()
        profile_to_assumed_matrix = {}
        if random:
            profile_to_assumed_matrix = {pr: np.ones(
                (self.env.state_dim, self.env.action_dim))/self.env.action_dim for pr in align_funcs}
            # TODO: random only in feasible states...
        else:
            if not expert:
                reward = self.state_action_callable_reward_from_reward_net_per_target_align_func(
                    self.current_net, targets=align_funcs)

            prev_net = self.current_net 
            for w in align_funcs:
                if expert:
                    reward_w = self.env.reward_matrix_per_align_func(w)
                else:
                    reward_w = reward(w)
                    self.current_net = reward_net_per_al[w]

                    if use_probabilistic_reward:
                        raise NotImplementedError(
                            "Probabilistic reward is yet to be tested")

                if precise_deterministic and expert:
                    policy_matrix = np.load(
                        f'roadworld_env_use_case/expert_policy_{w}.npy')

                elif expert:
                    policy_matrix = self.vsi_expert_policy.policy_per_va(w)
                else:
                    policy_matrix = np.ones(
                        (self.env.state_dim, self.env.action_dim))/self.env.action_dim
                    with th.no_grad():
                        for s in range(self.env.state_dim):
                            policy_matrix[s] = self.learned_policy_per_va.act_and_obtain_action_distribution(s,
                                                                                                             stochastic=self.learn_stochastic_policy, alignment_function=w)[2].detach().numpy()

                profile_to_assumed_matrix[w] = policy_matrix
                reward_matrix_per_al[w] = reward_w
            self.current_net = prev_net
        policy = VAlignedDictDiscreteStateActionPolicyTabularMDP(
            policy_per_va_dict=profile_to_assumed_matrix, env=self.env, state_encoder=state_encoder, expose_state=expose_state)
        return policy, reward_matrix_per_al
