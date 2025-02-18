import dataclasses
import enum
from typing import List, Sequence
import numpy as np
from src.algorithms.base_vsl_algorithm import BaseVSLAlgorithm
import torch as th

from src.algorithms.preference_based_irl import BasicRewardTrainerVSL, PreferenceModelTabularVSL, RandomFragmenterVariableHorizon, SupportedFragmenters
from src.algorithms.utils import PolicyApproximators, mce_partition_fh
from src.data import TrajectoryWithValueSystemRews, TrajectoryWithValueSystemRewsPair, VSLPreferenceDataset
from src.policies.vsl_policies import VAlignedDictSpaceActionPolicy, ValueSystemLearningPolicy
from src.reward_nets.vsl_reward_functions import TrainingModes

from imitation.data import rollout, types
from imitation.data.types import (
    TrajectoryPair,
    TrajectoryWithRew,
    Transitions,
    TrajectoryWithRewPair,
)
from imitation.algorithms import preference_comparisons

class PrefLossClasses(enum.Enum):
    CROSS_ENTROPY = 'cross_entropy'
    CROSS_ENTROPY_MODIFIED = 'cross_entropy_modified'

def _trajectory_pair_includes_vs(fragment_pair: TrajectoryWithValueSystemRewsPair) -> bool:
    """Return true if and only if both fragments in the pair include rewards."""
    frag1, frag2 = fragment_pair
    return isinstance(frag1, TrajectoryWithValueSystemRews) and isinstance(frag2, TrajectoryWithValueSystemRews)




class PreferenceBasedClusteringTabularMDPVSL(BaseVSLAlgorithm):
    def train_vsl_probabilistic(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, n_reward_reps_if_probabilistic_reward, reward_nets_per_target_align_func, target_align_func):
        raise NotImplementedError(
            "Probabilistic reward is not clear in this context...")

    def __init__(self, env,
                 reward_net,
                 optimizer_cls=th.optim.Adam,
                 optimizer_kwargs=None,
                 discount=1,
                 log_interval=100,
                 dataset: VSLPreferenceDataset = None,
                 target_align_func_sampler=...,
                 vsi_target_align_funcs=...,
                 vgl_target_align_funcs=...,
                 training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                 rng=np.random.default_rng(0),
                 discount_factor_preferences=None,
                 use_quantified_preference=False,
                 query_schedule="hyperbolic",
                 learn_stochastic_policy=True,
                 expert_is_stochastic=True,
                 approximator_kwargs={},
                 policy_approximator=PolicyApproximators.MCE_ORIGINAL,
                 # 0 for deterministic preference sampling, 1 for totally random according to softmax probabilities
                 preference_sampling_temperature=0,
                 reward_trainer_kwargs={
                     'epochs': 5, 'lr': 0.05, 'regularizer_factory': None, 'batch_size': 32, 'minibatch_size': None, },
                loss_class=preference_comparisons.CrossEntropyRewardLoss, loss_kwargs={},
                active_fragmenter_on=SupportedFragmenters.RANDOM_FRAGMENTER,
                 *, custom_logger=None):
        super().__init__(env=env, reward_net=reward_net, vgl_optimizer_cls=optimizer_cls, policy_approximator=policy_approximator, approximator_kwargs=approximator_kwargs, vsi_optimizer_cls=vsi_optimizer_cls,
                         vgl_optimizer_kwargs=optimizer_kwargs, vsi_optimizer_kwargs=optimizer_kwargs, discount=discount,
                         log_interval=log_interval, vgl_expert_policy=None, vsi_expert_policy=None,
                         target_align_func_sampler=target_align_func_sampler, vsi_target_align_funcs=vsi_target_align_funcs,
                         vgl_target_align_funcs=vgl_target_align_funcs, training_mode=training_mode, custom_logger=custom_logger, learn_stochastic_policy=learn_stochastic_policy,stochastic_expert=expert_is_stochastic)

        self.rng = rng
        if discount_factor_preferences is None:
            self.discount_factor_preferences = discount
        else:
            self.discount_factor_preferences = discount_factor_preferences

        self.sample = not use_quantified_preference

        if self.vgl_reference_policy == 'random':
            self.vgl_reference_policy = VAlignedDictSpaceActionPolicy(env=self.env, policy_per_va_dict={pr: np.ones(
                (self.env.state_dim, self.env.action_dim))/self.env.action_dim for pr in self.vgl_target_align_funcs}, expose_state=True)
        if self.vsi_reference_policy == 'random':
            self.vsi_reference_policy = VAlignedDictSpaceActionPolicy(env=self.env, policy_per_va_dict={pr: np.ones(
                (self.env.state_dim, self.env.action_dim))/self.env.action_dim for pr in self.vsi_target_align_funcs}, expose_state=True)


        self.dataset = dataset

        self.reward_trainer_kwargs = reward_trainer_kwargs
        self.temperature = preference_sampling_temperature
        self.query_schedule = query_schedule
        self.loss_class = loss_class
        self.loss_kwargs = loss_kwargs
        self.active_fragmenter_on = active_fragmenter_on
        for k in SupportedFragmenters:
            if active_fragmenter_on == k or active_fragmenter_on == k.value:
                self.active_fragmenter_on = k
        assert self.active_fragmenter_on in [f for f in SupportedFragmenters]

      


    @property
    def logger(self):
        return self.pref_comparisons.logger

    def train(self, max_iter=5000, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None, n_seeds_for_sampled_trajectories=None, n_sampled_trajs_per_seed=1, use_probabilistic_reward=False, n_reward_reps_if_probabilistic_reward=10,
              resample_trajectories_if_not_already_sampled=True, 
              new_rng=None, fragment_length='horizon', 
              interactive_imitation_iterations=100, 
              total_comparisons=500, initial_epoch_multiplier=10, 
              transition_oversampling=5, initial_comparison_frac=0.2,
              random_trajs_proportion=0.0, **kwargs):

        self.resample_trajectories_if_not_already_sampled = resample_trajectories_if_not_already_sampled
        self.initial_comparison_frac = initial_comparison_frac
        self.random_trajs_proportion = random_trajs_proportion
        
        if new_rng is not None:
            self.rng = new_rng
        if fragment_length == 'horizon':
            self.fragment_length = self.env.horizon
        else:
            self.fragment_length = fragment_length
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.transition_oversampling = transition_oversampling
        self.total_comparisons = total_comparisons
        self.interactive_imitation_iterations = interactive_imitation_iterations
        self.gatherer = preference_comparisons.SyntheticGatherer(rng=self.rng,
                                                                 discount_factor=self.discount_factor_preferences,
                                                                 sample=self.sample, temperature=self.temperature)

        if self.active_fragmenter_on == SupportedFragmenters.RANDOM_FRAGMENTER:
            self.fragmenter = RandomFragmenterVariableHorizon(
                warning_threshold=1,
                rng=self.rng
            )
            pass # It will be initialized in _train_global.
        self.last_accuracies_per_align_func = {al: [] for al in (
            self.vsi_target_align_funcs if mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION else self.vgl_target_align_funcs)}

        return super().train(max_iter=max_iter,
                             mode=mode, assumed_grounding=assumed_grounding,
                             n_seeds_for_sampled_trajectories=n_seeds_for_sampled_trajectories,
                             n_sampled_trajs_per_seed=n_sampled_trajs_per_seed,
                             use_probabilistic_reward=use_probabilistic_reward,
                             n_reward_reps_if_probabilistic_reward=n_reward_reps_if_probabilistic_reward,
                             **kwargs)

    def train_callback(self, t):
        self.last_accuracies_per_align_func[self.current_target].append(
            self.preference_model.get_last_accuracy())
        if t % self.log_interval == 0:
            self.logger.record("iteration", t)
            self.logger.record("Target align_func", self.current_target)
            if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
                self.logger.record("Learned align_func: ", tuple(
                    [float("{0:.3f}".format(v)) for v in self.current_net.get_learned_align_function()]))
            else:
                self.logger.record("Learned grounding: ",
                                   self.current_net.get_learned_grounding())

    def train_vgl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed):
        
        reward_net_per_target = dict()
            
        for vi, target_align_func in enumerate(reversed(self.vgl_target_align_funcs)):
            self.current_net.set_alignment_function(target_align_func)
            self._train_global(max_iter, target_align_func, batch_size=n_seeds_for_sampled_trajectories*n_sampled_trajs_per_seed)
            
        for target_align_func in self.vgl_target_align_funcs:
            reward_net_per_target[target_align_func] = self.current_net.copy()
        return reward_net_per_target
        

    def _train_global(self, max_iter, target_align_func, batch_size, partition=1, starting_t=0):
        self.current_target = target_align_func
        if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            alignment = None
        else:
            alignment = self.current_target
        #idxs = list(range(len(reference_trajs_per_profile[target_align_func])))
        #chosen_trajs = np.random.choice(idxs, size=len(reference_trajs_per_profile[target_align_func])//partition)
        
        
        """if self.active_fragmenter_on != SupportedFragmenters.RANDOM_FRAGMENTER:
            if SupportedFragmenters.CONNECTED_FRAGMENTER == self.active_fragmenter_on:
                self.fragmenter = ConnectedFragmenter(
                    warning_threshold=1,
                    rng=self.rng,
                )
            else:
                self.fragmenter = ActiveSelectionFragmenterVSL(
                    preference_model=self.preference_model,
                    base_fragmenter=ConnectedFragmenter(
                    warning_threshold=1,
                    rng=self.rng,
                ),
            fragment_sample_factor=0.5,
            uncertainty_on=self.active_fragmenter_on.value

            )"""
        self.preference_model = PreferenceModelTabularVSL(
            model=self.current_net,
            algorithm=self,
            noise_prob=0,
            discount_factor=self.discount_factor_preferences,
            threshold=50,
            alingment=alignment
        )
        reward_trainer = BasicRewardTrainerVSL(
            preference_model=self.preference_model,
            loss=self.loss_class(**self.loss_kwargs),
            rng=self.rng,
            batch_size=batch_size,
            **self.reward_trainer_kwargs
        )

        reward_trainer.train(dataset=self.dataset)
        # TODO: Dataset is there. SEGUIR?

        metric = self.pref_comparisons.train(
            max_iter//partition, total_comparisons=self.total_comparisons//partition, callback=lambda t: self.train_callback(t+starting_t))
        self.last_accuracies_per_align_func[self.current_target].append(
            metric['reward_accuracy'])

    def train_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func):
        
        self._train_global(max_iter, target_align_func, batch_size=n_seeds_for_sampled_trajectories*n_sampled_trajs_per_seed)

        return self.current_net.get_learned_align_function()

    def calculate_learned_policies(self, target_align_funcs) -> ValueSystemLearningPolicy:
        self.learned_policy_per_va = VAlignedDictSpaceActionPolicy(
            {}, self.env, state_encoder=None, expose_state=True)
        for target_align_func in target_align_funcs:
            learned_al_function = target_align_func if self.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING else self.target_align_funcs_to_learned_align_funcs[
                target_align_func]
            _, _, pi = mce_partition_fh(self.env, reward=self.rewards_per_target_align_func_callable(target_align_func)(),
                                        discount=self.discount, deterministic=not self.learn_stochastic_policy)

            # self.learned_policy_per_va.set_policy_for_va(target_align_func, pi)
            self.learned_policy_per_va.set_policy_for_va(
                learned_al_function, pi)
        return self.learned_policy_per_va

    def get_metrics(self):
        metrics = super().get_metrics()
        metrics.update({'accuracy': self.last_accuracies_per_align_func})
        return metrics