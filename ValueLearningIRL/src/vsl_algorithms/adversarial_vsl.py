from copy import deepcopy
import dataclasses
import itertools
import pprint
from typing import Iterator, Optional

import imitation.algorithms
import imitation.algorithms.adversarial
import imitation.algorithms.adversarial.common
from imitation.algorithms.adversarial.gail import RewardNetFromDiscriminatorLogit
import imitation.algorithms.base
import imitation.data
import imitation.util
import imitation.util.util
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from src.envs.tabularVAenv import TabularVAMDP, ValueAlignedEnvironment
from src.vsl_algorithms.base_tabular_vsl_algorithm import BaseTabularMDPVSLAlgorithm, PolicyApproximators, mce_partition_fh
from src.vsl_algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.vsl_policies import ValueSystemLearningPolicy, VAlignedDictDiscreteStateActionPolicyTabularMDP
from src.vsl_reward_functions import AbstractVSLRewardFunction, TrainingModes

from imitation.algorithms import base
from imitation.data import types
from imitation.util import logger as imit_logger
from imitation.util import networks

import imitation.algorithms.adversarial.common as common

from imitation.algorithms.adversarial.common import compute_train_stats

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Type,
    Union,
)

from gymnasium import spaces

import torch as th

from src.vsl_policies import LearnerValueSystemLearningPolicy, ValueSystemLearningPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from imitation.rewards import reward_nets


import dataclasses
from typing import Callable, Iterable, Iterator, Mapping, Optional, Type

import numpy as np
import torch as th
import tqdm
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.data import rollout, types, wrappers
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import networks
from stable_baselines3.common.torch_layers import FlattenExtractor


def compute_train_stats_tabular(
    disc_logits_expert_is_high: th.Tensor,
    labels_expert_is_over12: th.Tensor,
    labels_expert_is_one: th.Tensor,
    disc_loss: th.Tensor,
) -> Mapping[str, float]:
    """Train statistics for GAIL/AIRL discriminator.

    Args:
        disc_logits_expert_is_high: discriminator logits produced by
            `AdversarialTrainer.logits_expert_is_high`.
        labels_expert_is_one: integer labels describing whether logit was for an
            expert (0) or generator (1) sample.
        disc_loss: final discriminator loss.

    Returns:
        A mapping from statistic names to float values.
    """
    with th.no_grad():
        # Logits of the discriminator output; >0 for expert samples, <0 for generator.
        bin_is_generated_pred = disc_logits_expert_is_high <= 0

        # Binary label, so 1 is for expert, 0 is for generator.
        bin_is_generated_true = labels_expert_is_over12 <= 0.5
        bin_is_expert_true = labels_expert_is_over12 > 0.5
        int_is_generated_pred = bin_is_generated_pred.long()
        int_is_generated_true = bin_is_generated_true.long()

        int_is_expert_true = bin_is_expert_true.long()
        n_generated = float(th.sum(int_is_generated_true))
        n_expert = float(th.sum(int_is_expert_true))
        n_labels = float(len(labels_expert_is_one))

        pct_expert = n_expert / \
            float(n_labels) if n_labels > 0 else float("NaN")
        n_expert_pred = int(n_labels - th.sum(int_is_generated_pred))
        if n_labels > 0:
            pct_expert_pred = n_expert_pred / float(n_labels)
        else:
            pct_expert_pred = float("NaN")
        correct_vec = th.eq(bin_is_generated_pred, bin_is_generated_true)
        acc = th.mean(correct_vec.float())

        _n_pred_expert = th.sum(th.logical_and(
            bin_is_expert_true, correct_vec))
        if n_expert < 1:
            expert_acc = float("NaN")
        else:
            # float() is defensive, since we cannot divide Torch tensors by
            # Python ints
            expert_acc = _n_pred_expert.item() / float(n_expert)

        _n_pred_gen = th.sum(th.logical_and(
            bin_is_generated_true, correct_vec))
        _n_gen_or_1 = max(1, n_generated)
        generated_acc = _n_pred_gen / float(_n_gen_or_1)

        label_dist = th.distributions.Bernoulli(
            logits=disc_logits_expert_is_high)
        entropy = th.mean(label_dist.entropy())

    return {
        "disc_loss": float(th.mean(disc_loss)),
        "disc_acc": float(acc),
        # accuracy on just expert examples
        "disc_acc_expert": float(expert_acc),
        # accuracy on just generated examples
        "disc_acc_gen": float(generated_acc),
        # entropy of the predicted label distribution, averaged equally across
        # both classes (if this drops then disc is very good or has given up)
        "disc_entropy": float(entropy),
        # true number of expert demos and predicted number of expert demos
        "disc_proportion_expert_true": float(pct_expert),
        "disc_proportion_expert_pred": float(pct_expert_pred),
        "n_expert": float(n_expert),
        "n_generated": float(n_generated),
    }


def dict_metrics(**kwargs):
    return dict(kwargs)


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
        """Build and return training minibatches for the next discriminator update.

        Args:
            gen_samples: Same as in `train_disc`.
            expert_samples: Same as in `train_disc`.

        Yields:
            The training minibatch: state, action, next state, dones, labels
            and policy log-probabilities.

        Raises:
            RuntimeError: Empty generator replay buffer.
            ValueError: `gen_samples` or `expert_samples` batch size is
                different from `self.demo_batch_size`.
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
            """print("GS", self._gen_replay_buffer.size(), "out of", self._gen_replay_buffer.capacity)
            """
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
            # TODO: THIS IS A BIG PROBLEM.
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

    def _get_log_policy_act_prob(
        self,
        obs_th: th.Tensor,
        acts_th: th.Tensor,
        # obs_th should already be the state in POMDP tabular environments
        states_th: th.Tensor = None
    ) -> Optional[th.Tensor]:
        ret = None
        if states_th is None:
            ret = super()._get_log_policy_act_prob(obs_th, acts_th)

        if ret is None:
            _, ret, _ = self.policy.evaluate_actions(
                obs_th,
                acts_th,
            )

        return ret

    def reward_function_for_policy_training(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, dones: np.ndarray, ground_truth=False):
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
                # print(reward_real)
                # print(reward_np)
                # assert reward_real.shape == reward_np.shape

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
            reward_fn=self.reward_function_for_policy_training)
        self.gen_callback = self.venv_wrapped.make_log_callback()
        self.venv_train = self.venv_wrapped

        self.gen_algo.set_env(self.venv_train)
        self.gen_algo.set_logger(self.logger)

        self.ground_truth_gen_algo = expert_algo
        self.ground_truth_env = reward_wrapper.RewardVecEnvWrapper(
            self.venv_buffering,
            reward_fn=lambda o, a, n, d: self.reward_function_for_policy_training(o, a, n, d, ground_truth=True))
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
        # reward_output_train = self._reward_net(state, action, next_state, done)
        """print("MODELO")
        print(action)
        print(obs)
        print(reward_output_train)
        exit(0)"""
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
        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.demo_batch_size` samples. If this argument is not provided, then
                `self.demo_batch_size` expert samples from `self.demo_data_loader` are
                used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.demo_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.

        Returns:
            Statistics for discriminator (e.g. loss, accuracy).
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
        """Trains the generator to maximize the discriminator loss.

        After the end of training populates the generator replay buffer (used in
        discriminator training) with `self.disc_batch_size` transitions.

        Args:
            total_timesteps: The number of transitions to sample from
                `self.venv_train` during training. By default,
                `self.gen_train_timesteps`.
            learn_kwargs: kwargs for the Stable Baselines `RLModel.learn()`
                method.
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
        stochastic_expert = True,
        environment_is_stochastic = False,
        training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
        # NEW parameters
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
        
    def get_metrics(self): # TODO: Implement all these things in general... El next state es un problema grave. Quizá no interesa. 
        # TODO. Realmente es evaluate policy y punto en plan... Coger un montón de trayectorias/transitions y hacer la media?
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
                self.trainer.logger.record(namespace+f"Learned NN params for value {i}", [p.detach().numpy() for p in list(self.current_net.get_learned_grounding().networks[i].parameters())])
        
        """self.trainer.logger.record(
            "mean/Learned Bias", self.current_net.get_learned_profile(with_bias=True)[1]),"""
        """airl_trainer.logger.record(
            "mean/Learned OTHER??", list(self.current_net.get_learned_grounding().parameters())),
        )
        """
        self.last_accuracies_per_align_func[self.current_target].append(self.trainer.last_train_stats['disc_acc'])
    

    """def state_action_callable_reward_from_computed_rewards_per_target_align_func(self, rewards_per_target_align_func: Union[Dict, Callable]):
        if isinstance(rewards_per_target_align_func, dict):
            def rewards_per_target_align_func_callable(
                al_f): return rewards_per_target_align_func[al_f]
        else:
            return rewards_per_target_align_func_callable"""

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

        
        # target_align_func = (0.0,0.0,1.0)
        # target_align_func = (1.0,0.0,0.0)
        # target_align_func = (0.0,1.0,0.0)
        # target_align_func = (1.0, 0.0, 0.0)
        # target_align_func = (0.8, 0.2) # TODO: TEST AFTER WITH THE REST OF ALIGN FUNCS
        
        learning_algo: PPO = self.learned_policy_per_va.get_learner_for_alignment_function(
            target_align_func)
        
        demos = self.vsi_sampler(
            [target_align_func], n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)

        env = self.learned_policy_per_va.get_environ(target_align_func)
        venv = DummyVecEnv([lambda: env])
        #self.current_net.reset_learned_alignment_function()
        self.current_net.set_mode(
            mode=TrainingModes.VALUE_SYSTEM_IDENTIFICATION)

        self.train_global(max_iter=max_iter, demos=demos, env=env, venv=venv, 
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
            
            self.train_global(max_iter=max_iter, demos=demos, env=env, venv=venv, 
                              learning_algo=learning_algo, target_align_func=target_align_func)
            reward_nets_per_target_align_func[target_align_func] = self.current_net.copy()
        return reward_nets_per_target_align_func
    def train_global(self, max_iter, demos, env, venv, learning_algo, target_align_func):
        self.current_target = target_align_func

        learning_algo.set_env(env=venv)

        # ROAD WORLD:
        airl_trainer = AIRLforVSL(
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
        self.trainer = airl_trainer

        env.reset(seed=26)
        venv.seed(26)
        """TODO debugging...
        before_al = deepcopy(self.current_net.get_learned_align_function())

        learner_rewards_before_training, _ = evaluate_policy(
            airl_trainer.gen_algo, venv, 1000, return_episode_rewards=True,
        )
        learner_rewards_before_training_st, _ = evaluate_policy(
            airl_trainer.gen_algo, venv, 1000, return_episode_rewards=True,
            deterministic=False,
        )
        trajs_before = self.learned_policy_per_va.obtain_trajectories(
            n_seeds=n_seeds_for_sampled_trajectories, repeat_per_seed=n_sampled_trajs_per_seed, seed=26, stochastic=False, align_funcs_in_policy=[target_align_func],
            end_trajectories_when_ended=True, with_reward=True,
        )"""
        # prev_eval = airl_trainer.policy.evaluate_actions(th.tensor([204]*2, dtype=th.long), actions=th.tensor([2,0], dtype=th.long))[1]
        
        self.trainer.train(max_iter, callback=self.train_callback)  # Train for 2_000_000 steps to match expert.
        #env.reset(seed=self.seed)
        #venv.seed(26)
        """TODO: this is for debugging...
        learner_rewards_after_training, _ = evaluate_policy(
            airl_trainer.gen_algo, venv, 1000, return_episode_rewards=True,
        )
        learner_rewards_after_training_st, _ = evaluate_policy(
            airl_trainer.gen_algo, venv, 1000, return_episode_rewards=True,
            deterministic=False,
        )

        trajs = self.learned_policy_per_va.obtain_trajectories(
            n_seeds=n_seeds_for_sampled_trajectories, repeat_per_seed=n_sampled_trajs_per_seed, seed=26, stochastic=False, 
            align_funcs_in_policy=[target_align_func],
            end_trajectories_when_ended=True, with_reward=True,
        )
        expert_trajs = self.vsi_expert_policy.obtain_trajectories(
            n_seeds=n_seeds_for_sampled_trajectories, repeat_per_seed=n_sampled_trajs_per_seed, seed=26, stochastic=False, 
            align_funcs_in_policy=[target_align_func],
            end_trajectories_when_ended=True, with_reward=True,
        )
        print("TRAJS BEFORE", [t.obs for t in trajs_before[0:5]
              if t.obs[0] in [tt.obs[0] for tt in demos]])
        print("TRAJS LEARN", [t.obs for t in trajs[0:5]
              if t.obs[0] in [tt.obs[0] for tt in demos]])
        print("TRAJS REAL", [t.obs for t in expert_trajs[0:5]
              if t.obs[0] in [tt.obs[0] for tt in demos]])

        print(learner_rewards_after_training)
        print(learner_rewards_before_training)
        print("TARGET:", target_align_func)
        print("AFTER", self.current_net.get_learned_align_function())
        print("BEFore", before_al)

        print("mean reward after training:", np.mean(
            learner_rewards_after_training))
        print("mean reward before training:", np.mean(
            learner_rewards_before_training))

        print("mean reward stochastic after training:",
              np.mean(learner_rewards_after_training_st))
        print("mean reward stochastic before training:",
              np.mean(learner_rewards_before_training_st))"""

        

    def train_vsl_probabilistic(self, max_iter,
                                n_seeds_for_sampled_trajectories,
                                n_sampled_trajs_per_seed,
                                n_reward_reps_if_probabilistic_reward,
                                target_align_func):
        raise NotImplementedError("probabilistic vsl not implemented")

    
    


    def get_tabular_policy_from_reward_per_align_func(self, align_funcs, reward_net_per_al: Dict[tuple, AbstractVSLRewardFunction], expert=False, random=False, use_custom_grounding=None, 
                                              target_to_learned =None, use_probabilistic_reward=False, n_reps_if_probabilistic_reward=10,
                                              state_encoder = None, expose_state=True, precise_deterministic=False):
        if not isinstance(self.env.state_space, spaces.Discrete) and isinstance(self.env.action_space, spaces.Discrete):
            raise NotImplementedError("not implemented on non discrete state-action spaces...")
        
        
        reward_matrix_per_al = dict()
        profile_to_assumed_matrix = {}
        if random:
            profile_to_assumed_matrix = {pr: np.ones(
                (self.env.state_dim, self.env.action_dim))/self.env.action_dim for pr in align_funcs}
            # TODO: random only in feasible states...
        else:
            if not expert:
                reward = self.state_action_callable_reward_from_reward_net_per_target_align_func(self.current_net, targets = align_funcs)
                    
            prev_net = self.current_net # TODO SEG
            for w in align_funcs: 
                learned_w_or_real_w = w
                if expert:
                    deterministic = not self.stochastic_expert
                    reward_w = self.env.reward_matrix_per_align_func(w)
                else:
                    deterministic = not self.learn_stochastic_policy
                    if use_custom_grounding is not None:
                        assumed_grounding = use_custom_grounding
                    else:
                        assumed_grounding = reward_net_per_al[w].get_learned_grounding()
                    reward_w = reward(w)
                    self.current_net = reward_net_per_al[w]
                    if target_to_learned is not None and w in target_to_learned.keys():
                        learned_w_or_real_w = target_to_learned[w]
                        
                    else:
                        learned_w_or_real_w = w
                    if use_probabilistic_reward:
                        raise NotImplementedError("Probabilistic reward is yet to be tested")
                 
                if precise_deterministic and expert:
                    policy_matrix = np.load(f'roadworld_env_use_case/expert_policy_{w}.npy')
                    
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
        policy = VAlignedDictDiscreteStateActionPolicyTabularMDP(policy_per_va_dict = profile_to_assumed_matrix, env = self.env, state_encoder=state_encoder, expose_state=expose_state)
        return policy, reward_matrix_per_al

class TabularAdversarialVSL(BaseTabularMDPVSLAlgorithm, AdversarialVSL):

    def __init__(self, env, reward_net, vgl_optimizer_cls=th.optim.Adam, vsi_optimizer_cls=th.optim.Adam,
                 vgl_optimizer_kwargs=None, vsi_optimizer_kwargs=None, discount=1, log_interval=100,
                 vgl_expert_sampler=None,
                 vsi_expert_sampler=None,
                 vgl_expert_policy=None, vsi_expert_policy=None, target_align_func_sampler=...,
                 vsi_target_align_funcs=..., vgl_target_align_funcs=..., learn_stochastic_policy=True,
                 environment_is_stochastic=False, training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                 policy_kwargs=None, policy_class=None, learner_kwargs=None, learner_class=None, masked_policy=None,
                 stochastic_expert=True, approximator_kwargs=..., policy_approximator=PolicyApproximators.MCE_ORIGINAL, *, custom_logger=None):
        super().__init__(env, reward_net=reward_net,
                         vgl_optimizer_cls=vgl_optimizer_cls,
                         vsi_optimizer_cls=vsi_optimizer_cls, vgl_optimizer_kwargs=vgl_optimizer_kwargs,
                         vsi_optimizer_kwargs=vsi_optimizer_kwargs, discount=discount,
                         log_interval=log_interval, vgl_expert_policy=vgl_expert_policy,
                         vsi_expert_policy=vsi_expert_policy, target_align_func_sampler=target_align_func_sampler,
                         vsi_target_align_funcs=vsi_target_align_funcs, vgl_target_align_funcs=vgl_target_align_funcs,
                         learn_stochastic_policy=learn_stochastic_policy, training_mode=training_mode,
                         stochastic_expert=stochastic_expert,
                         custom_logger=custom_logger, policy_approximator=policy_approximator, approximator_kwargs=approximator_kwargs, environment_is_stochastic=environment_is_stochastic,
                         )
        self.vsi_target_align_funcs = [self.vsi_target_align_funcs[-5]]
        self.vgl_sampler = vgl_expert_sampler
        self.vsi_sampler = vsi_expert_sampler
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs
        self.learner_class = learner_class
        self.learner_kwargs = learner_kwargs
        self.masked_policy = masked_policy
        self.current_target = None
        self.learned_policy_per_va = VAlignedDictDiscreteStateActionPolicyTabularMDP(
            {}, env=self.env, state_encoder=lambda exposed_state, info: exposed_state)  # Starts with random policy
        for al_func in itertools.chain(self.vgl_target_align_funcs, self.vsi_target_align_funcs):
            probability_matrix = np.random.rand(
                self.env.state_dim, self.env.action_dim)
            random_pol = probability_matrix / \
                probability_matrix.sum(axis=1, keepdims=True)
            self.learned_policy_per_va.set_policy_for_va(al_func, random_pol)

    def train_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func):
        # super().train_vsl(max_iter=max_iter,n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories,n_sampled_trajs_per_seed=n_sampled_trajs_per_seed,target_align_func=target_align_func)
        demos = self.vsi_sampler(
            [target_align_func], n_seeds_for_sampled_trajectories*10, n_sampled_trajs_per_seed)

        print(len(demos), n_seeds_for_sampled_trajectories,
              n_sampled_trajs_per_seed)

        self.current_target = target_align_func

        env = self.learned_policy_per_va.get_environ(target_align_func)
        venv = DummyVecEnv([lambda: env])

        airl_trainer = TabularAIRLforVSL(
            demonstrations=demos,
            # demo batch size is not used, it is number of trajs to sample and this.
            demo_batch_size=1000,
            sampled_trajs_per_round=1,
            n_disc_updates_per_round=1,
            venv=venv,
            reward_net=self.current_net,
            allow_variable_horizon=True,
            adversarialVSL=self,
            alignment=None,
            disc_opt_cls=self.vsi_optimizer_cls,
            disc_opt_kwargs=self.vsi_optimizer_kwargs,
            stochastic_gen=self.learn_stochastic_policy
        )

        airl_trainer.train(1000000, callback=lambda t: (airl_trainer.logger.record("mean/Target", target_align_func), airl_trainer.logger.record("mean/Learned",
                           tuple([float(f"{a:0.5f}") for a in self.current_net.get_learned_align_function()]))))  # Train for 2_000_000 steps to match expert.

    def train(self, max_iter=1000, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None, n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_probabilistic_reward=False, n_reward_reps_if_probabilistic_reward=10,
              demo_batch_size=512,  # 512
              gen_replay_buffer_capacity=2048,  #  2048 for RW
              n_disc_updates_per_round=2,
              **kwargs
              ):
        self.demo_batch_size = demo_batch_size
        self.gen_replay_buffer_capacity = gen_replay_buffer_capacity
        self.n_disc_updates_per_round = n_disc_updates_per_round

        return super().train(max_iter, mode, assumed_grounding, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, use_probabilistic_reward, n_reward_reps_if_probabilistic_reward, **kwargs)


class TabularAIRLforVSL(AIRLforVSL):

    def __init__(self, *, demonstrations, demo_batch_size, sampled_trajs_per_round, venv, reward_net=None, adversarialVSL: TabularAdversarialVSL = None, alignment=None, stochastic_gen=True, **kwargs):
        dummy_algo = adversarialVSL.learner_class(
            policy=adversarialVSL.policy_class, env=venv, policy_kwargs=adversarialVSL.policy_kwargs, **adversarialVSL.learner_kwargs)
        super().__init__(demonstrations=demonstrations, demo_batch_size=demo_batch_size, venv=venv, gen_algo=dummy_algo, expert_algo=dummy_algo,
                         reward_net=reward_net, adversarialVSL=adversarialVSL, alignment=alignment, stochastic_gen=stochastic_gen, **kwargs)
        self.advsl: TabularAdversarialVSL
        self.reward_cache = None
        self.sampled_trajs_per_round = sampled_trajs_per_round
        self.reward_cache_th = None

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ) -> None:
        """Trains the generator to maximize the discriminator loss.

        After the end of training populates the generator replay buffer (used in
        discriminator training) with `self.disc_batch_size` transitions.

        Args:
            total_timesteps: The number of transitions to sample from
                `self.venv_train` during training. By default,
                `self.gen_train_timesteps`.
            learn_kwargs: kwargs for the Stable Baselines `RLModel.learn()`
                method.
        """
        if total_timesteps is None:
            total_timesteps = self.gen_train_timesteps
        if learn_kwargs is None:
            learn_kwargs = {}
        self.reward_cache = None
        with self.logger.accumulate_means("gen"):

            reward, reward_np = self.advsl.calculate_rewards(align_func=self.alignment,
                                                             grounding=None,
                                                             obs_mat=self.advsl.torch_obs_mat,
                                                             action_mat=self.advsl.torch_action_mat,
                                                             obs_action_mat=self.advsl.torch_obs_action_mat,
                                                             # obs_action_mat TODO maybe...
                                                             reward_mode=self.advsl.training_mode,
                                                             recover_previous_config_after_calculation=True,
                                                             use_probabilistic_reward=False, requires_grad=False)
            self.reward_cache = reward_np
            _, _, policy_matrix = mce_partition_fh(self.venv.envs[0],
                                                   reward=reward_np,
                                                   discount=self.advsl.discount,
                                                   horizon=self.advsl.env.horizon,
                                                   approximator_kwargs=self.advsl.approximator_kwargs,
                                                   policy_approximator=self.advsl.policy_approximator,
                                                   deterministic=not self.advsl.learn_stochastic_policy)
            self.advsl.learned_policy_per_va.set_policy_for_va(
                self.advsl.current_target, policy_matrix)
            self._global_step += 1
    
    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        self.reward_cache_th = None
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
                exp_visitation_counts, gen_visitation_counts = self.calculate_visitation_counts(
                    batch)

                labels = 1/2.0 + ((exp_visitation_counts - gen_visitation_counts)/(
                    exp_visitation_counts + gen_visitation_counts + 1e-16)).float()/2.0
                # labels[i] = 0 if exp_visitation_matrix[i] is 0, ~ 1/2 if exp_visitation_matrix[i] is equal to gen_visitation_matrix[i],
                # and 1 if gen_visitation_matrix[i] is 0. When both are 0, 0/epsilon in the denominator, which yields 1/2.

                loss = F.binary_cross_entropy_with_logits(
                    disc_logits,
                    # TODO ? the variable "labels" should work better?
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
                train_stats = compute_train_stats_tabular(
                    disc_logits,
                    labels_expert_is_over12=labels,
                    labels_expert_is_one=batch["labels_expert_is_one"],
                    disc_loss=loss,
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram(
                    "disc_logits", disc_logits.detach())

        return train_stats

    def calculate_visitation_counts(self, batch):
        assert np.array_equal(batch['obs'].long().float(), batch['obs'])
        all_obs = batch['obs'].long()
        gen_obs = all_obs[batch["labels_expert_is_one"] == 0]
        expert_obs = all_obs[batch["labels_expert_is_one"] == 1]
        unique_vals, gen_counts = th.unique(gen_obs, return_counts=True)
        unique_vals_exp, exp_counts = th.unique(expert_obs, return_counts=True)

        gen_visitation_matrix = th.zeros(
            (self.advsl.env.state_dim,), dtype=th.long)
        gen_visitation_matrix[unique_vals] = gen_counts
        exp_visitation_matrix = th.zeros_like(
            gen_visitation_matrix, dtype=th.long)
        exp_visitation_matrix[unique_vals_exp] = exp_counts

        assert self.venv.envs[0].initial_state_dist[self.venv.reset()[0]] > 0

        return exp_visitation_matrix[all_obs], gen_visitation_matrix[all_obs]

    def _next_expert_batch(self):
        # expert_batch  =  super()._next_expert_batch()
        expert_trajs = self.advsl.vsi_sampler(
            [self.advsl.current_target], self.sampled_trajs_per_round, 5)
        initial_states = [t.obs[0] for t in expert_trajs]
        exp_samples = rollout.flatten_trajectories_with_rew(expert_trajs)
        expert_all = types.dataclass_quick_asdict(exp_samples)

        if len(expert_all['obs']) < self.demo_batch_size:
            # sample with replace if not enough data
            ind = np.random.randint(
                len(expert_all['obs']), size=self.demo_batch_size)

        else:
            # sample no replace if enough data.
            ind = np.random.permutation(self.demo_batch_size)
        return {k: buffer[ind] for k, buffer in expert_all.items()}, initial_states

    def _next_gen_batch(self, expert_initial_states=None):
        if expert_initial_states is not None:
            # np.array(expert_batch['obs']).flatten()
            initial_states_of_expert = expert_initial_states

        trajectories_from_initial_states = self.advsl.learned_policy_per_va.obtain_trajectories(
            n_seeds=len(initial_states_of_expert), seed=np.random.randint(0, 100000),
            stochastic=self.stochastic_gen, repeat_per_seed=1,
            from_initial_states=initial_states_of_expert,
            align_funcs_in_policy=[self.advsl.current_target],
            t_max=1,  # self.venv.envs[0].horizon,
            with_reward=True, alignments_in_env=[self.advsl.current_target])

        gen_samples = rollout.flatten_trajectories_with_rew(
            trajectories_from_initial_states)
        gen_samples_dic = types.dataclass_quick_asdict(gen_samples)

        nobs = len(gen_samples_dic['obs'])
        if nobs < self.demo_batch_size:
            # sample with replace if not enough data
            ind = np.random.randint(nobs, size=self.demo_batch_size)

        else:
            # sample no replace if enough data.
            ind = np.random.permutation(self.demo_batch_size)
        return {k: buffer[ind] for k, buffer in gen_samples_dic.items()}

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Alternates between training the generator and discriminator.

        Every "round" consists of a call to `train_gen(self.gen_train_timesteps)`,
        a call to `train_disc`, and finally a call to `callback(round)`.

        Training ends once an additional "round" would cause the number of transitions
        sampled from the environment to exceed `total_timesteps`.

        Args:
            total_timesteps: An upper bound on the number of transitions to sample
                from the environment during training.
            callback: A function called at the end of every round which takes in a
                single argument, the round number. Round numbers are in
                `range(total_timesteps // self.gen_train_timesteps)`.
        """
        n_rounds = total_timesteps // self.gen_train_timesteps
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        self._gen_replay_buffer_per_initial_state = dict()

        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.train_gen(self.gen_train_timesteps)
            for _ in range(self.n_disc_updates_per_round):
                with networks.training(self.reward_train):
                    expert_samples, expert_initial_states = self._next_expert_batch()
                    gen_samples = self._next_gen_batch(expert_initial_states)
                    # switch to training mode (affects dropout, normalization)
                    self.train_disc(expert_samples=expert_samples,
                                    gen_samples=gen_samples)
            if callback:
                callback(r)
            self.logger.dump(self._global_step)

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
        if action.shape[-1] != 1:
            with th.no_grad():
                action = th.argmax(action, dim=1)
        if self.reward_cache_th is None:
            self.reward_cache_th, _ = self.advsl.calculate_rewards(align_func=self.alignment,
                                                                   grounding=None,
                                                                   obs_mat=self.advsl.torch_obs_mat,
                                                                   action_mat=self.advsl.torch_action_mat,
                                                                   obs_action_mat=self.advsl.torch_obs_action_mat,
                                                                   reward_mode=self.advsl.training_mode,
                                                                   recover_previous_config_after_calculation=False,
                                                                   use_probabilistic_reward=False, requires_grad=True)
        """rew,_ = self.advsl.calculate_rewards(align_func=self.alignment,
                                                                            grounding=None,
                                                                            obs_mat=self.advsl.torch_obs_mat,
                                                                            action_mat=self.advsl.torch_action_mat,
                                                                            obs_action_mat=self.advsl.torch_obs_action_mat,
                                                                            reward_mode=self.advsl.training_mode,
                                                                      recover_previous_config_after_calculation=False,
                                                                      use_probabilistic_reward=False, requires_grad=True)
        """
        reward_output_train = self.reward_cache_th[obs.long(), action.long()]

        assert not th.any(log_policy_act_prob.isinf())
        # reward_output_train = self._reward_net(state, action, next_state, done)

        return reward_output_train - log_policy_act_prob

    def reward_function_for_policy_training(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, dones: np.ndarray, ground_truth=False):
        with th.no_grad():
            obs = th.as_tensor(obs, dtype=self.advsl.current_net.dtype,
                               device=self.advsl.current_net.device).long()
            action = th.as_tensor(action, dtype=self.advsl.current_net.dtype,
                                  device=self.advsl.current_net.device).long()
            next_obs = th.as_tensor(
                next_obs, dtype=self.advsl.current_net.dtype, device=self.advsl.current_net.device).long()
            # TODO: possibly use the dones attribute...

            if not ground_truth:
                if self.reward_cache is None:
                    _, self.reward_cache = self.advsl.calculate_rewards(self.alignment,
                                                                        grounding=None,
                                                                        obs_mat=self.advsl.torch_obs_mat,
                                                                        action_mat=self.advsl.torch_action_mat,
                                                                        obs_action_mat=self.advsl.torch_obs_action_mat,
                                                                        reward_mode=self.advsl.training_mode,
                                                                        recover_previous_config_after_calculation=True,
                                                                        use_probabilistic_reward=False, requires_grad=False)

                reward_np = self.reward_cache[obs, action]

            else:
                reward_real = self.advsl.env.reward_matrix_per_align_func(
                    self.advsl.current_target)[obs, action]
                reward_np = reward_real
        return reward_np

    def _get_log_policy_act_prob(self, obs_th, acts_th):
        policy_acts_probs = th.as_tensor(self.advsl.learned_policy_per_va.policy_per_va(self.advsl.current_target),
                                         dtype=self.advsl.current_net.dtype,
                                         device=self.advsl.current_net.device)
        log_policy_act_prob_th = th.log(
            th.clamp(policy_acts_probs[obs_th, acts_th], 0.001, 0.999))
        assert np.all(th.exp(log_policy_act_prob_th).detach().numpy() <= 1.0)
        assert np.all(th.exp(log_policy_act_prob_th).detach().numpy() >= 0.0)
        return log_policy_act_prob_th


class GAILforVSL(common.AdversarialTrainer):

    def _make_disc_train_batches(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> Iterator[Mapping[str, th.Tensor]]:
        """Build and return training minibatches for the next discriminator update.

        Args:
            gen_samples: Same as in `train_disc`.
            expert_samples: Same as in `train_disc`.

        Yields:
            The training minibatch: state, action, next state, dones, labels
            and policy log-probabilities.

        Raises:
            RuntimeError: Empty generator replay buffer.
            ValueError: `gen_samples` or `expert_samples` batch size is
                different from `self.demo_batch_size`.
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
            """print("GS", self._gen_replay_buffer.size(), "out of", self._gen_replay_buffer.capacity)
            print(len(gen_samples_dataclass))
            print(gen_samples_dataclass)
            exit(0)"""
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
            """print("EB", expert_batch["acts"])
            print(gen_batch["acts"])
            print(gen_batch["acts"][0])
            print(expert_batch["acts"][0])"""

            obs = np.concatenate([expert_batch["obs"], gen_batch["obs"]])

            # states = np.concatenate([np.asarray([einfo['state'] for einfo in expert_batch["infos"]]), np.asarray([ginfo['state'] for ginfo in gen_batch["infos"]])])
            # next_states = np.concatenate([np.asarray([einfo['next_state'] for einfo in expert_batch["infos"]]), np.asarray([ginfo['next_state'] for ginfo in gen_batch["infos"]])])
            acts = np.concatenate([expert_batch["acts"], gen_batch["acts"]])

            next_obs = np.concatenate(
                [expert_batch["next_obs"], gen_batch["next_obs"]])
            dones = np.concatenate([expert_batch["dones"], gen_batch["dones"]])
            # notice that the labels use the convention that expert samples are
            # labelled with 1 and generator samples with 0.
            # TODO: THIS IS A BIG PROBLEM.
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

    def _get_log_policy_act_prob(
        self,
        obs_th: th.Tensor,
        acts_th: th.Tensor,
        # obs_th should already be the state in POMDP tabular environments
        states_th: th.Tensor = None
    ) -> Optional[th.Tensor]:
        ret = None
        if states_th is None:
            ret = super()._get_log_policy_act_prob(obs_th, acts_th)

        if ret is None:
            _, ret, _ = self.policy.evaluate_actions(
                obs_th,
                acts_th,
            )

        return ret

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

        reward_net = reward_net.to(gen_algo.device)
        self._processed_reward = RewardNetFromDiscriminatorLogit(reward_net)

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
            reward_fn=self.reward_function_for_policy_training)
        self.gen_callback = self.venv_wrapped.make_log_callback()
        self.venv_train = self.venv_wrapped

        self.gen_algo.set_env(self.venv_train)
        self.gen_algo.set_logger(self.logger)

        self.ground_truth_gen_algo = expert_algo
        self.ground_truth_env = reward_wrapper.RewardVecEnvWrapper(
            self.venv_buffering,
            reward_fn=lambda o, a, n, d: self.reward_function_for_policy_training(o, a, n, d, ground_truth=True))

        """ 
        Based on the imitation version, without assertions on specific classes that made the code not work.
        """

        # Process it to produce output suitable for RL training
        # Applies a -log(sigmoid(-logits)) to the logits (see class for explanation)
        # Raw self._reward_net is discriminator logits

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
        # TODO: reward calculation should use calculate_rewards...# TODO: SEGUIR AQUI

        reward_output_train, _ = self.advsl.calculate_rewards(self.alignment,
                                                              grounding=None,
                                                              obs_mat=obs,
                                                              action_mat=action,
                                                              next_state_obs_mat=next_obs,
                                                              reward_mode=self.advsl.training_mode,
                                                              recover_previous_config_after_calculation=False,
                                                              use_probabilistic_reward=False, requires_grad=True)

        # reward_output_train = -F.logsigmoid(-reward_output_train) # TODO?
        # reward_output_train = self._reward_net(state, action, next_state, done)

        return reward_output_train  # - log_policy_act_prob

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._processed_reward

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        return self._processed_reward

    def reward_function_for_policy_training(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, dones: np.ndarray, ground_truth=False):
        with th.no_grad():
            obs = th.as_tensor(obs, dtype=self.advsl.current_net.dtype,
                               device=self.advsl.current_net.device).long()
            action = th.as_tensor(action, dtype=self.advsl.current_net.dtype,
                                  device=self.advsl.current_net.device).long()
            next_obs = th.as_tensor(
                next_obs, dtype=self.advsl.current_net.dtype, device=self.advsl.current_net.device).long()
            # TODO: possibly use the dones attribute...
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

                # print(reward_real)
                # reward = -F.logsigmoid(-reward)
                # reward_np = reward.detach().numpy()
                # print(reward_np)
                # assert reward_real.shape == reward_np.shape

            else:
                reward_np = reward_real
        return reward_np

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.demo_batch_size` samples. If this argument is not provided, then
                `self.demo_batch_size` expert samples from `self.demo_data_loader` are
                used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.demo_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.

        Returns:
            Statistics for discriminator (e.g. loss, accuracy).
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

        return train_stats

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ) -> None:
        """Trains the generator to maximize the discriminator loss.

        After the end of training populates the generator replay buffer (used in
        discriminator training) with `self.disc_batch_size` transitions.

        Args:
            total_timesteps: The number of transitions to sample from
                `self.venv_train` during training. By default,
                `self.gen_train_timesteps`.
            learn_kwargs: kwargs for the Stable Baselines `RLModel.learn()`
                method.
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
