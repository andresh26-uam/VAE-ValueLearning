from copy import deepcopy
import enum
from math import ceil, floor
from random import shuffle
from typing import Dict, List, NoReturn, Sequence
from imitation.data.types import TrajectoryPair

from src.vsl_algorithms.base_tabular_vsl_algorithm import BaseTabularMDPVSLAlgorithm, PolicyApproximators
from src.vsl_algorithms.me_irl_for_vsl import mce_partition_fh
from src.vsl_policies import VAlignedDictSpaceActionPolicy, ValueSystemLearningPolicy
from src.vsl_reward_functions import AbstractVSLRewardFunction, TrainingModes
import torch as th

from typing import (
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import torch as th
from imitation.data import rollout, types
from imitation.data.types import (
    TrajectoryPair,
    TrajectoryWithRew,
    Transitions,
    TrajectoryWithRewPair,
)
from imitation.algorithms import preference_comparisons


def _trajectory_pair_includes_reward(fragment_pair: TrajectoryPair) -> bool:
    """Return true if and only if both fragments in the pair include rewards."""
    frag1, frag2 = fragment_pair
    return isinstance(frag1, TrajectoryWithRew) and isinstance(frag2, TrajectoryWithRew)



class ConnectedFragmenter(preference_comparisons.RandomFragmenter):
    """Sample fragments of trajectories with the rule of pairs being connected
    """

    def __init__(
        self,
        rng: np.random.Generator,
        warning_threshold: int = 10,
        custom_logger = None,
        discard_short_trajectories = False,
    ) -> None:
        super().__init__(rng, warning_threshold, custom_logger)
        self.nexus = None
        self.discard_short_trajectories = discard_short_trajectories
    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        

        fragments: List[TrajectoryWithRew] = []

        prev_num_trajectories = len(trajectories)
        # filter out all trajectories that are too short
        if self.discard_short_trajectories:
            trajectories = [traj for traj in trajectories if len(traj) >= fragment_length]
            if len(trajectories) == 0:
                raise ValueError(
                    "No trajectories are long enough for the desired fragment length "
                    f"of {fragment_length}.",
                )
            num_discarded = prev_num_trajectories - len(trajectories)
            if num_discarded:
                self.logger.log(
                    f"Discarded {num_discarded} out of {prev_num_trajectories} "
                    "trajectories because they are shorter than the desired length "
                    f"of {fragment_length}.",
                )

        weights = [len(traj) for traj in trajectories]

        # number of transitions that will be contained in the fragments
        num_transitions = 2 * num_pairs * fragment_length
        if sum(weights) < num_transitions:
            self.logger.warn(
                "Fewer transitions available than needed for desired number "
                "of fragment pairs. Some transitions will appear multiple times.",
            )
        elif (
            self.warning_threshold
            and sum(weights) < self.warning_threshold * num_transitions
        ):
            # If the number of available transitions is not much larger
            # than the number of requires ones, we already give a warning.
            # But only if self.warning_threshold is non-zero.
            self.logger.warn(
                f"Samples will contain {num_transitions} transitions in total "
                f"and only {sum(weights)} are available. "
                f"Because we sample with replacement, a significant number "
                "of transitions are likely to appear multiple times.",
            )

        # we need two fragments for each comparison
        
        for _ in range( num_pairs//2+1):
            # NumPy's annotation here is overly-conservative, but this works at runtime
            traj = self.rng.choice(
                trajectories,  # type: ignore[arg-type]
                p=np.array(weights) / sum(weights),
            )
            n = len(traj)
            f_length = min(fragment_length, n)
            start = self.rng.integers(0, n - f_length, endpoint=True)
            end = start + f_length
            terminal = (end == n) and traj.terminal
            fragment = TrajectoryWithRew(
                obs=traj.obs[start : end + 1],
                acts=traj.acts[start:end],
                infos=traj.infos[start:end] if traj.infos is not None else None,
                rews=traj.rews[start:end],
                terminal=terminal,
            )
            
            fragments.append(fragment)
        # fragments is currently a list of single fragments. We want to pair up
        # fragments to get a list of (fragment1, fragment2) tuples. To do so,
        # we create a single iterator of the list and zip it with itself:
        if self.nexus is None:
            self.nexus = fragments[0]

        fragment_chain = list(zip(fragments, fragments[1:])) + list(zip([self.nexus]*len(fragments[:-1]), fragments[:-1]))
        # a chain is formed, and joined back to a previous point.
        #assert len(fragment_chain) == num_pairs, f'{len(fragment_chain)} vs {num_pairs}'
        self.nexus = fragments[-1]
        return fragment_chain 
class RandomFragmenterVariableHorizon(preference_comparisons.RandomFragmenter):
    def __init__(
        self,
        rng: np.random.Generator,
        warning_threshold: int = 10,
        custom_logger = None,
        discard_short_trajectories = False,
    ) -> None:
        super().__init__(rng, warning_threshold, custom_logger)
        self.discard_short_trajectories = discard_short_trajectories

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        fragments: List[TrajectoryWithRew] = []

        prev_num_trajectories = len(trajectories)
        # filter out all trajectories that are too short
        if self.discard_short_trajectories:
            trajectories = [traj for traj in trajectories if len(traj) >= fragment_length]
            if len(trajectories) == 0:
                raise ValueError(
                    "No trajectories are long enough for the desired fragment length "
                    f"of {fragment_length}.",
                )
            num_discarded = prev_num_trajectories - len(trajectories)
            if num_discarded:
                self.logger.log(
                    f"Discarded {num_discarded} out of {prev_num_trajectories} "
                    "trajectories because they are shorter than the desired length "
                    f"of {fragment_length}.",
                )

        weights = [len(traj) for traj in trajectories]

        # number of transitions that will be contained in the fragments
        num_transitions = 2 * num_pairs * fragment_length
        if sum(weights) < num_transitions:
            self.logger.warn(
                "Fewer transitions available than needed for desired number "
                "of fragment pairs. Some transitions will appear multiple times.",
            )
        elif (
            self.warning_threshold
            and sum(weights) < self.warning_threshold * num_transitions
        ):
            # If the number of available transitions is not much larger
            # than the number of requires ones, we already give a warning.
            # But only if self.warning_threshold is non-zero.
            self.logger.warn(
                f"Samples will contain {num_transitions} transitions in total "
                f"and only {sum(weights)} are available. "
                f"Because we sample with replacement, a significant number "
                "of transitions are likely to appear multiple times.",
            )


        # we need two fragments for each comparison
        for _ in range(2 * num_pairs):
            traj = self.rng.choice(
                trajectories,  # type: ignore[arg-type]
                p=np.array(weights) / sum(weights),
            )
            n = len(traj)
            f_length = min(fragment_length, n)
            start = self.rng.integers(0, n - f_length, endpoint=True)
            end = start + f_length
            terminal = (end == n) and traj.terminal
            fragment = TrajectoryWithRew(
                obs=traj.obs[start : end + 1],
                acts=traj.acts[start:end],
                infos=traj.infos[start:end] if traj.infos is not None else None,
                rews=traj.rews[start:end],
                terminal=terminal,
            )
            
            fragments.append(fragment)
        # fragments is currently a list of single fragments. We want to pair up
        # fragments to get a list of (fragment1, fragment2) tuples. To do so,
        # we create a single iterator of the list and zip it with itself:
        iterator = iter(fragments)
        return list(zip(iterator, iterator))
class ActiveSelectionFragmenterVSL(preference_comparisons.Fragmenter):
    def __init__(
        self,
        preference_model: preference_comparisons.PreferenceModel,
        base_fragmenter: preference_comparisons.Fragmenter,
        fragment_sample_factor: float,
        uncertainty_on: str = "logit",
        custom_logger = None,
    ) -> None:
        """Initialize the active selection fragmenter.

        Args:
            preference_model: an ensemble model that predicts the
                preference of the first fragment over the other.
            base_fragmenter: fragmenter instance to get
                fragment pairs from trajectories
            fragment_sample_factor: the factor of the number of
                fragment pairs to sample from the base_fragmenter
            uncertainty_on: the variable to calculate the variance on.
                Can be logit|probability|label.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: Preference model not wrapped over an ensemble of networks.
        """
        super().__init__(custom_logger=custom_logger)
        self.preference_model = preference_model
        self.base_fragmenter = base_fragmenter
        self.fragment_sample_factor = fragment_sample_factor
        self._uncertainty_on = uncertainty_on
        if not (uncertainty_on in ["logit", "probability", "label"]):
            self.raise_uncertainty_on_not_supported()

    @property
    def uncertainty_on(self) -> str:
        return self._uncertainty_on
    

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,

        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        # sample a large number (self.fragment_sample_factor*num_pairs)
        # of fragments from all the trajectories
        fragments_to_sample = int(self.fragment_sample_factor * num_pairs)
        fragment_pairs = self.base_fragmenter(
            trajectories=trajectories,
            fragment_length=fragment_length,
            num_pairs=fragments_to_sample,
        )
        var_estimates = np.zeros(len(fragment_pairs))
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            with th.no_grad():
                rews1 = self.preference_model.rewards(trans1)
                rews2 = self.preference_model.rewards(trans2)
            var_estimate = self.variance_estimate(rews1, rews2)
            var_estimates[i] = var_estimate
        fragment_idxs = np.argsort(var_estimates)[::-1]  # sort in descending order
        # return fragment pairs that have the highest uncertainty
        return [fragment_pairs[idx] for idx in fragment_idxs[:num_pairs]]

    

    def raise_uncertainty_on_not_supported(self) -> NoReturn:
        raise ValueError(
            f"""{self.uncertainty_on} not supported.
            `uncertainty_on` should be from `logit`, `probability`, or `label`""",
        )


    def variance_estimate(self, rews1: th.Tensor, rews2: th.Tensor) -> float:
        """Gets the variance estimate from the rewards of a fragment pair.

        Args:
            rews1: rewards obtained by all the ensemble models for the first fragment.
                Shape - (fragment_length, num_ensemble_members)
            rews2: rewards obtained by all the ensemble models for the second fragment.
                Shape - (fragment_length, num_ensemble_members)

        Returns:
            the variance estimate based on the `uncertainty_on` flag.
        """
        if self.uncertainty_on == "logit":
            returns1, returns2 = rews1.sum(0), rews2.sum(0)
            var_estimate = (returns1 - returns2).var().item()
        else:  # uncertainty_on is probability or label
            probs = self.preference_model.probability(rews1, rews2)
            probs_np = probs.cpu().numpy()
            if self.uncertainty_on == "probability":
                var_estimate = probs_np.var()
            elif self.uncertainty_on == "label":  # uncertainty_on is label
                preds = (probs_np > 0.5).astype(np.float32)
                # probability estimate of Bernoulli random variable
                prob_estimate = preds.mean()
                # variance estimate of Bernoulli random variable
                var_estimate = prob_estimate * (1 - prob_estimate)
            else:
                self.raise_uncertainty_on_not_supported()
        return var_estimate


class BasicRewardTrainerVSL(preference_comparisons.BasicRewardTrainer):
    """Train a basic reward model."""
    def __init__(
        self,
        preference_model: preference_comparisons.PreferenceModel,
        loss: preference_comparisons.RewardLoss,
        rng: np.random.Generator,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        epochs: int = 1,
        lr: float = 1e-3,
        custom_logger = None,
        regularizer_factory = None,
    ) -> None:
        
        super().__init__(preference_model, loss, rng, batch_size, minibatch_size, epochs, lr, custom_logger, regularizer_factory)
    
        self.optim = th.optim.AdamW(self._preference_model.parameters(), lr=lr, weight_decay=0.0) #WEIGHT DECAY MUST BE 0 !!!!!!!
        self.regularizer = (
            regularizer_factory(optimizer=self.optim, logger=self.logger)
            if regularizer_factory is not None
            else None
        )

class PreferenceModelTabularVSL(preference_comparisons.PreferenceModel):
    """Class to convert two fragments' rewards into preference probability."""
    """
    Extension of https://imitation.readthedocs.io/en/latest/algorithms/preference_comparisons.html
    """

    def __init__(
        self,
        model: AbstractVSLRewardFunction,
        algorithm: BaseTabularMDPVSLAlgorithm,
        noise_prob: float = 0.0,
        discount_factor: float = 1.0,
        threshold: float = 50,
        alingment=None,
    ) -> None:
        super().__init__(model, noise_prob, discount_factor, threshold)
        self.algorithm = algorithm

        self.state_dim = self.algorithm.env.state_dim
        self.action_dim = self.algorithm.env.action_dim
        self.alignment = alingment
        self.rew_matrix = None

    def forward(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        self.rew_matrix = None
        """Computes the preference probability of the first fragment for all pairs.

        Note: This function passes the gradient through for non-ensemble models.
              For an ensemble model, this function should not be used for loss
              calculation. It can be used in case where passing the gradient is not
              required such as during active selection or inference time.
              Therefore, the EnsembleTrainer passes each member network through this
              function instead of passing the EnsembleNetwork object with the use of
              `ensemble_member_index`.

        Args:
            fragment_pairs: batch of pair of fragments.

        Returns:
            A tuple with the first element as the preference probabilities for the
            first fragment for all fragment pairs given by the network(s).
            If the ground truth rewards are available, it also returns gt preference
            probabilities in the second element of the tuple (else None).
            Reward probability shape - (num_fragment_pairs, ) for non-ensemble reward
            network and (num_fragment_pairs, num_networks) for an ensemble of networks.

        """
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        gt_reward_available = _trajectory_pair_includes_reward(
            fragment_pairs[0])
        if gt_reward_available:
            gt_probs = th.empty(len(fragment_pairs), dtype=th.float32)
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            rews1 = self.rewards(trans1)
            rews2 = self.rewards(trans2)
            probs[i] = self.probability(rews1, rews2)
            if gt_reward_available:
                frag1 = cast(TrajectoryWithRew, frag1)
                frag2 = cast(TrajectoryWithRew, frag2)
                gt_rews_1 = th.from_numpy(frag1.rews)
                gt_rews_2 = th.from_numpy(frag2.rews)
                gt_probs[i] = self.probability(gt_rews_1, gt_rews_2)
            # rews1_real = self.rewards(trans1, real=True, real_alignment=frag1.infos[0]['align_func'])

        predictions = probs.detach() >= 0.5

        ground_truth = gt_probs.detach() >= 0.5

        self.last_accuracy = float(
            (predictions == ground_truth).float().mean().numpy())
        return probs, gt_probs

    def get_last_accuracy(self):
        return self.last_accuracy
    """def eval_on_align_func(
            self,
            align_func: tuple,
            fragment_pairs: Sequence[TrajectoryPair],
    ):
        self.model: AbstractVSLRewardFunction
        with th.no_grad():
            prev_obs = self.observation_matrix.clone()
            prev_model =self.model.copy()
            prev_mode = self.model.mode
            self.observation_matrix = None

            self.model.set_mode(TrainingModes.EVAL)
            self.model.set_alignment_function(align_func)
            
            frag1, frag2 = fragment_pairs[0]
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])

            rews1 = self.rewards(trans1)
            rews2 = self.rewards(trans2)

            frag1 = cast(TrajectoryWithRew, frag1)
            frag2 = cast(TrajectoryWithRew, frag2)
            gt_rews_1 = th.from_numpy(frag1.rews)
            gt_rews_2 = th.from_numpy(frag2.rews)
            
            
            ret = self.forward(fragment_pairs)
            self.model = prev_model
            self.observation_matrix = None
            self.model.set_mode(prev_mode)
            self.model.cur_align_func = None
        return ret"""

    def rewards(self, transitions: Transitions, real=False, real_alignment=None) -> th.Tensor:
        """Computes the reward for all transitions.

        Args:
            transitions: batch of obs-act-obs-done for a fragment of a trajectory.

        Returns:
            The reward given by the network(s) for all the transitions.
            Shape - (num_transitions, ) for Single reward network and
            (num_transitions, num_networks) for ensemble of networks.
        """
        state = types.assert_not_dictobs(transitions.obs)
        action = transitions.acts
        next_state = types.assert_not_dictobs(transitions.next_obs)
        done = transitions.dones
        if self.ensemble_model is not None:
            raise NotImplementedError("Ensemble not implemented")
        else:

            if self.rew_matrix is None:
                self.rew_matrix, _ = self.algorithm.calculate_rewards(self.alignment,
                                                                      grounding=None,
                                                                      obs_mat=self.algorithm.torch_obs_mat,
                                                                      action_mat=self.algorithm.torch_action_mat,
                                                                      obs_action_mat=self.algorithm.torch_obs_action_mat,
                                                                      reward_mode=self.algorithm.training_mode,
                                                                      recover_previous_config_after_calculation=False,
                                                                      use_probabilistic_reward=False, requires_grad=True)
            assert self.rew_matrix.shape == (self.state_dim, self.action_dim)
            if real:
                _, rew_matrix_real = self.algorithm.calculate_rewards(real_alignment,
                                                                      grounding=None,
                                                                      obs_mat=self.algorithm.torch_obs_mat,
                                                                      action_mat=self.algorithm.torch_action_mat,
                                                                      obs_action_mat=self.algorithm.torch_obs_action_mat,
                                                                      reward_mode=TrainingModes.EVAL,
                                                                      recover_previous_config_after_calculation=True,
                                                                      use_probabilistic_reward=False, requires_grad=False)
                rews = rew_matrix_real[state, action]
            else:
                rews = self.rew_matrix[state, action]
            assert len(state) == len(action)
            assert rews.shape == (len(state),)
        return rews
    
    def probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        """Changed from imitation the ability to compare fragments of different lengths
        """
        # check rews has correct shape based on the model
        expected_dims = 2 if self.ensemble_model is not None else 1
        
        assert rews1.ndim == rews2.ndim == expected_dims
        # First, we compute the difference of the returns of
        # the two fragments. We have a special case for a discount
        # factor of 1 to avoid unnecessary computation (especially
        # since this is the default setting).
        if self.discount_factor == 1:
            if len(rews1) == len(rews2):
                returns_diff = (rews2 - rews1).sum(axis=0)
            else:
                returns_diff = rews2.sum(axis=0) - rews1.sum(axis=0)
            
        else:
            device = rews1.device
            assert device == rews2.device
            l1, l2 = len(rews1), len(rews2)
            discounts = self.discount_factor ** th.arange(max(l1,l2), device=device)
            if self.ensemble_model is not None:
                    discounts = discounts.reshape(-1, 1)
                
            if len(rews1) == len(rews2):
                returns_diff = (discounts * (rews2 - rews1)).sum(axis=0)
            else:
                returns_diff = (discounts[0:l2] * rews2).sum(axis=0) - (discounts[0:l1] * rews1).sum(axis=0)
        # Clip to avoid overflows (which in particular may occur
        # in the backwards pass even if they do not in the forward pass).
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        # We take the softmax of the returns. model_probability
        # is the first dimension of that softmax, representing the
        # probability that fragment 1 is preferred.
        model_probability = 1 / (1 + returns_diff.exp())
        probability = self.noise_prob * 0.5 + (1 - self.noise_prob) * model_probability
        if self.ensemble_model is not None:
            assert probability.shape == (self.model.num_members,)
        else:
            assert probability.shape == ()
        
        return probability


class SupportedFragmenters(enum.Enum):
        ACTIVE_FRAGMENTER_LOGIT = 'logit'
        ACTIVE_FRAGMENTER_PROBABILITY = 'probability'
        ACTIVE_FRAGMENTER_LABEL = 'label'
        RANDOM_FRAGMENTER = 'random'
        CONNECTED_FRAGMENTER = 'connected'

class CrossEntropyRewardLossForQualitativePreferences(preference_comparisons.RewardLoss):
    """Compute the cross entropy reward loss."""

    def __init__(self, beta=5) -> None:
        """Create cross entropy reward loss."""
        super().__init__()
        self.beta=beta

    def forward(
        self,
        fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
        preference_model: preference_comparisons.PreferenceModel,
    ) -> preference_comparisons.LossAndMetrics:
        """Computes the loss. Same as Cross Entropy but does not overfit to class certainty."""
        probs, gt_probs = preference_model(fragment_pairs)
        #probs_real, gt_probs_real = preference_model.eval_on_align_func((0.0, 1.0), fragment_pairs)
        
        # TODO(ejnnr): Here and below, > 0.5 is problematic
        #  because getting exactly 0.5 is actually somewhat
        #  common in some environments (as long as sample=False or temperature=0).
        #  In a sense that "only" creates class imbalance
        #  but it's still misleading.
        predictions = probs >= 0.5
        
        preferences_th = th.as_tensor(preferences, dtype=th.float32)
       
        ground_truth = preferences_th >= 0.5

        """if preference_model.model.get_learned_align_function()[1] > 0.96:
            exit(0)"""
        metrics = {}
        metrics["accuracy"] = (predictions == ground_truth).float().mean()

        #misclassified_pairs = predictions != ground_truth
        if gt_probs is not None:
            metrics["gt_reward_loss"] = th.nn.functional.binary_cross_entropy(
                gt_probs,
                preferences_th,
            )
        metrics = {key: value.detach().cpu() for key, value in metrics.items()}


        """return preference_comparisons.LossAndMetrics(
            loss=th.nn.functional.binary_cross_entropy(probs[misclassified_pairs], preferences_th[misclassified_pairs]),
            metrics=metrics,
        )"""
        loss_per_case = th.nn.functional.binary_cross_entropy(probs, preferences_th, reduce=False)
        loss_per_case += -th.multiply(probs, self.beta*th.log(probs)) # Subtract entropy
        # This idea is CONFIDENT PENALTY: https://openreview.net/pdf?id=HyhbYrGYe ICLR 2017
        # TODO: Idea for future work:
        # from pairwise comparisons, minimize divergence from a prior that consists on the expected class probability, 
        # being it estimated online.
        # Under the assumption of convergence, we should find a single only possible function that is equivalent to the original
        # preferences (up to some operation, probably multiplication by a constant).
        loss = th.mean(loss_per_case)
        return preference_comparisons.LossAndMetrics(
            loss=loss,
            metrics=metrics,
        )

class TrajectoryDatasetWithID(preference_comparisons.TrajectoryGenerator):
    """A fixed dataset of trajectories."""

    def __init__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        rng: np.random.Generator,
        custom_logger: Optional[preference_comparisons.imit_logger.HierarchicalLogger] = None,
    ):
        """Creates a dataset loaded from `path`.

        Args:
            trajectories: the dataset of rollouts.
            rng: RNG used for shuffling dataset.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger=custom_logger)
        self._trajectories = trajectories
        for idx, t in enumerate(self._trajectories):
            for s in len(t):
                t.infos[s]['_TID'] = idx
        self.rng = rng
    
    
class PreferenceBasedTabularMDPVSL(BaseTabularMDPVSLAlgorithm):
    def train_vsl_probabilistic(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, n_reward_reps_if_probabilistic_reward, reward_nets_per_target_align_func, target_align_func):
        raise NotImplementedError(
            "Probabilistic reward is not clear in this context...")

    def __init__(self, env,
                 reward_net,
                 vgl_optimizer_cls=th.optim.Adam,
                 vsi_optimizer_cls=th.optim.Adam,
                 vgl_optimizer_kwargs=None,
                 vsi_optimizer_kwargs=None,
                 discount=1,
                 log_interval=100,
                 vgl_expert_policy=None,
                 vsi_expert_policy=None,
                 target_align_func_sampler=...,
                 vsi_target_align_funcs=...,
                 vgl_target_align_funcs=...,
                 training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                 independent_vgl = False,
                 rng=np.random.default_rng(0),
                 discount_factor_preferences=None,
                 use_quantified_preference=False,
                 vgl_reference_policy='random',
                 vsi_reference_policy='random',
                 stochastic_sampling_in_reference_policy=True,
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
        super().__init__(env=env, reward_net=reward_net, vgl_optimizer_cls=vgl_optimizer_cls, policy_approximator=policy_approximator, approximator_kwargs=approximator_kwargs, vsi_optimizer_cls=vsi_optimizer_cls,
                         vgl_optimizer_kwargs=vgl_optimizer_kwargs, vsi_optimizer_kwargs=vsi_optimizer_kwargs, discount=discount,
                         log_interval=log_interval, vgl_expert_policy=vgl_expert_policy, vsi_expert_policy=vsi_expert_policy,
                         target_align_func_sampler=target_align_func_sampler, vsi_target_align_funcs=vsi_target_align_funcs,
                         vgl_target_align_funcs=vgl_target_align_funcs, training_mode=training_mode, custom_logger=custom_logger, learn_stochastic_policy=learn_stochastic_policy,stochastic_expert=expert_is_stochastic)

        self.rng = rng
        self.independent_vgl = independent_vgl
        if discount_factor_preferences is None:
            self.discount_factor_preferences = discount
        else:
            self.discount_factor_preferences = discount_factor_preferences

        self.sample = not use_quantified_preference
        self.vgl_reference_policy = self.vgl_expert_policy if vgl_reference_policy is None else vgl_reference_policy
        self.vsi_reference_policy = self.vsi_expert_policy if vsi_reference_policy is None else vsi_reference_policy

        if self.vgl_reference_policy == 'random':
            self.vgl_reference_policy = VAlignedDictSpaceActionPolicy(env=self.env, policy_per_va_dict={pr: np.ones(
                (self.env.state_dim, self.env.action_dim))/self.env.action_dim for pr in self.vgl_target_align_funcs}, expose_state=True)
        if self.vsi_reference_policy == 'random':
            self.vsi_reference_policy = VAlignedDictSpaceActionPolicy(env=self.env, policy_per_va_dict={pr: np.ones(
                (self.env.state_dim, self.env.action_dim))/self.env.action_dim for pr in self.vsi_target_align_funcs}, expose_state=True)

        self.stochastic_sampling_in_reference_policy = stochastic_sampling_in_reference_policy

        self.vgl_reference_trajs_with_rew_per_profile = None
        self.vsi_reference_trajs_with_rew_per_profile = None

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

    def train_vgl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, epoch_partition = 0.1):
        # return super().train_vgl(max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_funcs)
        reward_net_per_target = dict()
        reference_trajs_per_profile = self.extract_trajectories(
                n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, self.resample_trajectories_if_not_already_sampled,
                random_trajs_proportion=self.random_trajs_proportion)
            
        
        """n_steps_per_change_of_target = int(self.interactive_imitation_iterations*epoch_partition)
        for rep in range(n_steps_per_change_of_target):
            
            if rep > 0:
                self.vgl_reference_policy = self.calculate_learned_policies(self.vgl_target_align_funcs)
            
            
            for vi, target_align_func in enumerate(self.vgl_target_align_funcs):
                
                self._train_global(max_iter, target_align_func,
                                reference_trajs_per_profile, partition=ceil(1/epoch_partition), 
                                starting_t=int(n_steps_per_change_of_target*(rep*2+vi)))

                reward_net_per_target[target_align_func] = self.current_net.copy()
            """
        for vi, target_align_func in enumerate(reversed(self.vgl_target_align_funcs)):
            self._train_global(max_iter, target_align_func,
                                reference_trajs_per_profile,)
            
        for target_align_func in self.vgl_target_align_funcs:
            reward_net_per_target[target_align_func] = self.current_net.copy()
            #print(self.current_net.get_learned_grounding())
        return reward_net_per_target
        
                
    def extract_trajectories(self, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, resample_trajs=False, random_trajs_proportion=0.0):
        seed = int(self.rng.random()*1000)


        if self.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            if self.vgl_reference_trajs_with_rew_per_profile is None or resample_trajs:
                self.vgl_reference_trajs_with_rew_per_profile = {
                    pr: self.vgl_reference_policy.obtain_trajectories(n_seeds=n_seeds_for_sampled_trajectories, seed=seed,
                                                                      stochastic=self.stochastic_sampling_in_reference_policy,
                                                                      repeat_per_seed=n_sampled_trajs_per_seed, align_funcs_in_policy=[
                                                                          pr,],
                                                                      exploration=random_trajs_proportion,
                                                                      t_max=self.env.horizon,
                                                                      with_reward=True,
                                                                      alignments_in_env=[pr,])
                    for pr in self.vgl_target_align_funcs}
            reference_trajs_per_profile = self.vgl_reference_trajs_with_rew_per_profile
        else:
            if self.vsi_reference_trajs_with_rew_per_profile is None or resample_trajs:
                self.vsi_reference_trajs_with_rew_per_profile = {
                    pr: self.vsi_reference_policy.obtain_trajectories(n_seeds=n_seeds_for_sampled_trajectories, seed=seed,
                                                                      stochastic=self.stochastic_sampling_in_reference_policy,
                                                                      repeat_per_seed=n_sampled_trajs_per_seed, align_funcs_in_policy=[
                                                                          pr,],
                                                                      exploration=random_trajs_proportion,
                                                                      t_max=self.env.horizon,
                                                                      with_reward=True,
                                                                      alignments_in_env=[pr,])
                    for pr in self.vsi_target_align_funcs}
            reference_trajs_per_profile = self.vsi_reference_trajs_with_rew_per_profile
        return reference_trajs_per_profile

    def _train_global(self, max_iter, target_align_func, reference_trajs_per_profile, partition=1, starting_t=0):
        self.current_target = target_align_func
        if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            alignment = None
        else:
            alignment = self.current_target
        #idxs = list(range(len(reference_trajs_per_profile[target_align_func])))
        #chosen_trajs = np.random.choice(idxs, size=len(reference_trajs_per_profile[target_align_func])//partition)
        
        traj_dataset = preference_comparisons.TrajectoryDataset(
            reference_trajs_per_profile[target_align_func],
            rng=self.rng
        )
        self.preference_model = PreferenceModelTabularVSL(
            self.current_net, algorithm=self, noise_prob=0, alingment=alignment, discount_factor=self.discount_factor_preferences)

        
        if self.active_fragmenter_on != SupportedFragmenters.RANDOM_FRAGMENTER:
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

            )

        reward_trainer = BasicRewardTrainerVSL(
            preference_model=self.preference_model,
            loss=self.loss_class(**self.loss_kwargs),
            rng=self.rng,
            **self.reward_trainer_kwargs
        )

        self.pref_comparisons = preference_comparisons.PreferenceComparisons(
            trajectory_generator=traj_dataset,
            reward_model=self.current_net,
            num_iterations=self.interactive_imitation_iterations//partition,
            fragmenter=self.fragmenter,
            preference_gatherer=self.gatherer,
            reward_trainer=reward_trainer,
            fragment_length=self.fragment_length,
            transition_oversampling=self.transition_oversampling,
            initial_comparison_frac=self.initial_comparison_frac,
            allow_variable_horizon=True,
            initial_epoch_multiplier=self.initial_epoch_multiplier,
            query_schedule=self.query_schedule,
        )

        metric = self.pref_comparisons.train(
            max_iter//partition, total_comparisons=self.total_comparisons//partition, callback=lambda t: self.train_callback(t+starting_t))
        self.last_accuracies_per_align_func[self.current_target].append(
            metric['reward_accuracy'])

    def train_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func):
        reference_trajs_per_profile = self.extract_trajectories(
            n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, 
            self.resample_trajectories_if_not_already_sampled,
            random_trajs_proportion=self.random_trajs_proportion)

        self._train_global(max_iter, target_align_func,
                           reference_trajs_per_profile)

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
    