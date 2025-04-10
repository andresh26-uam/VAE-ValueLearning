from collections import defaultdict
from copy import deepcopy
import enum
import itertools
from typing import Dict, Iterable, List, Sequence, Union
import imitation
import imitation.algorithms
import imitation.algorithms.base
from imitation.data.types import TrajectoryPair
import imitation.regularization
import imitation.regularization.regularizers

from imitation.util import util


from src.algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.data import TrajectoryWithValueSystemRewsPair
from src.reward_nets.vsl_reward_functions import AbstractVSLRewardFunction, TrainingModes
import torch as th

from typing import (
    Optional,
    Sequence,
    Tuple,
)

from imitation.util import logger as imit_logger

import numpy as np
from imitation.data import rollout, types
from imitation.data.types import (
    TrajectoryPair,
    Transitions,
)
from imitation.algorithms import preference_comparisons


class PrefLossClasses(enum.Enum):
    CROSS_ENTROPY_CLUSTER = 'cross_entropy_cluster'
    SOBA = 'soba'




def discordance(probs: th.Tensor = None, gt_probs: th.Tensor = None, indifference_tolerance=0.0, apply_indifference_in_gt=False, reduce='mean'):
    return 1.0 - calculate_accuracies(probs_vs=probs, gt_probs_vs=gt_probs, indifference_tolerance=indifference_tolerance, apply_indifference_in_gt=apply_indifference_in_gt, reduce=reduce)[0]
def calculate_accuracies(probs_vs: th.Tensor = None, probs_gr: th.Tensor = None, gt_probs_vs: th.Tensor = None, gt_probs_gr: th.Tensor = None, indifference_tolerance=0.0, apply_indifference_in_gt=False, reduce='mean'):
    accuracy_vs = None
    accuracy_gr = None
    misclassified_vs = None
    misclassified_gr = None
    if probs_vs is not None:
        assert gt_probs_vs is not None
        if isinstance(probs_vs, th.Tensor):
            assert isinstance(gt_probs_vs , th.Tensor)
            detached_probs_vs = probs_vs.detach()
        else:
            assert isinstance(gt_probs_vs , np.ndarray)
            detached_probs_vs = probs_vs

        vs_predictions_positive = detached_probs_vs > 0.5
        vs_predictions_negative = detached_probs_vs < 0.5
        vs_predictions_indifferent = abs(
            detached_probs_vs - 0.5) < indifference_tolerance

        if isinstance(gt_probs_gr, th.Tensor):
            gt_detached_probs_vs = gt_probs_vs.detach()
        else:
            gt_detached_probs_vs = gt_probs_vs

        
        gt_predictions_positive = gt_detached_probs_vs > 0.5
        gt_predictions_negative = gt_detached_probs_vs < 0.5
        gt_predictions_indifferent = gt_detached_probs_vs == 0.5 if not apply_indifference_in_gt else abs(
            gt_detached_probs_vs - 0.5) <= indifference_tolerance
        
        
        misclassified_positive = (
            gt_predictions_positive != vs_predictions_positive)
        misclassified_negative = (
            gt_predictions_negative != vs_predictions_negative)
        misclassified_indifferent = (
            gt_predictions_indifferent != vs_predictions_indifferent)
        
        # Combine all misclassified examples
        misclassified_vs = misclassified_positive | misclassified_negative | misclassified_indifferent
        if reduce == 'mean':
            accuracy_vs = (1.0 - sum(misclassified_vs)/len(probs_vs)) 
        else:
            accuracy_vs = (1.0 - misclassified_vs)

    if probs_gr is not None:
        assert gt_probs_gr is not None
        accuracy_gr = []
        misclassified_gr = []
        for j in range(probs_gr.shape[-1]):
            if isinstance(probs_gr, th.Tensor):
                detached_probs_vgrj = probs_gr[:, j].detach()
            else:
                detached_probs_vgrj = probs_gr[:, j]

            vgrj_predictions_positive = detached_probs_vgrj > 0.5
            vgrj_predictions_negative = detached_probs_vgrj < 0.5
            vgrj_predictions_indifferent = abs(
                detached_probs_vgrj - 0.5) < indifference_tolerance
            if isinstance(gt_probs_gr, th.Tensor):
                gt_detached_probs_vgrj = gt_probs_gr[:, j].detach()
            else:
                gt_detached_probs_vgrj = gt_probs_gr[:, j]
            gt_predictions_positive = gt_detached_probs_vgrj > 0.5
            gt_predictions_negative = gt_detached_probs_vgrj < 0.5
            gt_predictions_indifferent = gt_detached_probs_vgrj == 0.5 if not apply_indifference_in_gt else abs(
            gt_detached_probs_vgrj - 0.5) <= indifference_tolerance

            misclassified_positive = (
                gt_predictions_positive != vgrj_predictions_positive)
            misclassified_negative = (
                gt_predictions_negative != vgrj_predictions_negative)
            misclassified_indifferent = (
                gt_predictions_indifferent != vgrj_predictions_indifferent)

            missclassified_vgrj = misclassified_positive | misclassified_negative | misclassified_indifferent


            if reduce == 'mean':
                acc_gr_vi = (1.0 - sum(missclassified_vgrj)/len(detached_probs_vgrj))
            else:
                acc_gr_vi = (1.0 - missclassified_vgrj)
                
            accuracy_gr.append(acc_gr_vi)
            misclassified_gr.append(missclassified_vgrj)

    return accuracy_vs, accuracy_gr, misclassified_vs, misclassified_gr

def total_variation_distance(preferences1, preferences2, p='inf'):
    return th.norm(preferences1-preferences2, p=p)
    return th.mean(th.abs(preferences1 - preferences2))#/len(preferences1)


def likelihood_x_is_target(pred_probs, target_probs, mode='numpy', slope=1, adaptive_slope=True, qualitative_mode=False, indifference_tolerance=0.0):
    # SEE GEOGEBRA: https://www.geogebra.org/calculator/vdn3mj4k
    # to visualize the output probabilities used for the likelihood estimation that pred_probs would correspond to the modelled target_probs.
    # In general, you can use the slope to scale the probability differences, so the likelihood do not tend to 0 under bigger datasets.
    
    assert mode == 'numpy' or mode == 'th'
    minfun = th.min if mode == 'th' else np.minimum
    absfun = th.abs if mode == 'th' else np.abs
    productfun = th.prod if mode == 'th' else np.prod

    if adaptive_slope:
            # Here you can interpret the slope parameter as 1 - the minimum possible probability value that would be given to any input, i.e.
            # if slope = a, the minimum possible probability value would be 1 - a, for instance, if slope = 0.3, the minimum possible probability value would be 0.7
            # If slope is bigger than 1, the minimum possible probability value would be 0, in a segment of the input space bigger as the slope increases.
            probs = 1 - slope * \
                minfun(1/target_probs, 1/(1-target_probs)) * \
                absfun(target_probs - pred_probs)
    else:
            # Here the slope is the slope of the linear , and it will always give bigger or equal probabilities than the adaptive slope. It is then, more lax but "unfair" method
            probs = 1 - slope*absfun(target_probs - pred_probs)
    if qualitative_mode:
        probs [(probs > 0.5) & (target_probs > 0.5)] = 1.0

        probs [(probs < 0.5) & (target_probs < 0.5)] = 1.0
        
        probs[(abs(probs - indifference_tolerance) <= indifference_tolerance) & (abs(target_probs - 0.5) <= indifference_tolerance)] = 1.0
        
    return productfun(probs)


class PreferenceModelClusteredVSL(preference_comparisons.PreferenceModel):
    """Class to convert two fragments' rewards into preference probability."""
    """
    Extension of https://imitation.readthedocs.io/en/latest/algorithms/preference_comparisons.html
    """

    def _slice_all_transitions_into_pairs(v, K):
        """This function returns len(v)/K slices of v of length K,the even slices first, the odd slices second, 

        Args:
            v (np.ndarray): Vector
            K (int): length desired of each slice

        Returns:
            Tuple[np.ndarray, np.ndarray]: Odd slices, even slices.
        """
        # Reshape the vector into N slices of length K
        reshaped_v = v.reshape((-1, K))
        # Calculate the slices we need: indexes K+1 to 2K+1, etc.
        odd_slices = reshaped_v[1::2]
        even_slices = reshaped_v[0::2]
        return even_slices, odd_slices

    def __init__(
        self,
        model: AbstractVSLRewardFunction,
        algorithm: BaseVSLAlgorithm,
        noise_prob: float = 0.0,
        discount_factor: float = 1.0,
        threshold: float = 50,
    ) -> None:
        super().__init__(model, noise_prob, discount_factor, threshold)
        self.algorithm = algorithm

        self.state_dim = self.algorithm.env.state_dim
        self.action_dim = self.algorithm.env.action_dim

        self.dummy_models = []

    def diversity_grounding(self, 
        fragments: Sequence[TrajectoryWithValueSystemRewsPair],
        grounding1: th.nn.Module, grounding2: th.nn.Module,
        value_idx,
        indifference_tolerance=0.0,
        difference_function = total_variation_distance
        ) -> Tuple[th.Tensor, th.Tensor]:

        
        fragments_dict = {'g1': fragments, 'g2': fragments}
        #print(fragments['none'][0])
        if len(self.dummy_models) < 2:
            self.dummy_models.append(self.model.copy())
            self.dummy_models.append(self.model.copy())

        cm_with_different_grounding1= self.dummy_models[0]
        cm_with_different_grounding1.set_network_for_value(value_idx, grounding1)
        cm_with_different_grounding2 = self.dummy_models[1]
        cm_with_different_grounding2.set_network_for_value(value_idx, grounding2)
        
        probs, _ = self.forward(fragment_pairs_per_agent_id=fragments_dict, fragment_pairs_idxs_per_agent_id=None,
                                add_probs_per_agent=False,
                                custom_model_per_agent_id={'g1': cm_with_different_grounding1, 'g2': cm_with_different_grounding2},
                                only_for_alignment_function=self.algorithm.env.basic_profiles[value_idx])
        n = len(fragments)
        probs_with_cluster1, probs_with_cluster2 = probs[0:n], probs[n:2*n]
        
        disc = discordance(probs=probs_with_cluster1, gt_probs= probs_with_cluster2, apply_indifference_in_gt=False, indifference_tolerance=indifference_tolerance)
        
        return difference_function(probs_with_cluster1, probs_with_cluster2), disc

    def diversity_vs(self, 
        fragments: Dict[str, Sequence[TrajectoryWithValueSystemRewsPair]],
        reward_net_per_aid: Dict[str, AbstractVSLRewardFunction],
        vs1: th.nn.Module, vs2: th.nn.Module,
        indifference_tolerance=0.0,
        difference_function = total_variation_distance
        ) -> Tuple[th.Tensor, th.Tensor]:

        # TODO: Implement the conciseness score here. 
        previous_vs = {}
        reward_net_per_aid_1 = dict.fromkeys(reward_net_per_aid.keys(), None)
        reward_net_per_aid_2 = dict.fromkeys(reward_net_per_aid.keys(), None)

        al1 = None
        al2 = None
        for aid in fragments.keys():
            
            rew = reward_net_per_aid[aid]
            rew1 = rew.copy()
            rew2 = rew.copy()
            rew1.set_trained_alignment_function_network(vs1)
            rew2.set_trained_alignment_function_network(vs2)
            al1 = rew1.get_learned_align_function()
            al2 = rew2.get_learned_align_function()
            reward_net_per_aid_1[aid] = rew1
            reward_net_per_aid_2[aid] = rew2


        probs_with_cluster1, _ = self.forward(fragment_pairs_per_agent_id=fragments, fragment_pairs_idxs_per_agent_id=None,
                                add_probs_per_agent=False,
                                custom_model_per_agent_id=reward_net_per_aid_1,
                                only_for_alignment_function=al1)
        probs_with_cluster2, _ = self.forward(fragment_pairs_per_agent_id=fragments, fragment_pairs_idxs_per_agent_id=None,
                                add_probs_per_agent=False,
                                custom_model_per_agent_id=reward_net_per_aid_2,
                                only_for_alignment_function=al2)
        
        disc = discordance(probs=probs_with_cluster1, gt_probs= probs_with_cluster2, apply_indifference_in_gt=False, indifference_tolerance=indifference_tolerance)
        #disc = sum(missed_pairs)/len(probs_with_cluster1)
        return difference_function(probs_with_cluster1, probs_with_cluster2), disc


    def forward(
        self,
        fragment_pairs_per_agent_id: Dict[str, Sequence[TrajectoryWithValueSystemRewsPair]],
        fragment_pairs_idxs_per_agent_id: Dict[str, np.ndarray],
        custom_model_per_agent_id: Union[AbstractVSLRewardFunction,
                                         Dict[str, AbstractVSLRewardFunction]] = None,
        on_specific_agent_ids: Iterable = None,
        only_for_alignment_function=None,
        only_grounding=False,

        add_probs_per_agent=False
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]:
        """Computes the preference probability of the first fragment for all pairs.

        """

        dtype = self.algorithm.reward_net.dtype

        if on_specific_agent_ids is not None:
            fragment_pairs_per_agent_id = {aid: rid for aid, rid in fragment_pairs_per_agent_id.items() if aid in on_specific_agent_ids}
            
        if custom_model_per_agent_id is not None:
            prev_model = self.model

            if isinstance(custom_model_per_agent_id, dict):
                models_per_agent: Dict = custom_model_per_agent_id
            else:
                models_per_agent = {
                    aid: custom_model_per_agent_id for aid in fragment_pairs_per_agent_id.keys()}

        total_number_of_fragment_pairs = sum(
            [len(fpair) for fpair in fragment_pairs_per_agent_id.values()])

        probs_vs = None
        probs_gr = None
        if not only_grounding:
            probs_vs = th.zeros(total_number_of_fragment_pairs, dtype=dtype)
        if only_for_alignment_function is None:
            probs_gr = th.zeros(
                (total_number_of_fragment_pairs, self.algorithm.env.n_values), dtype=dtype)

        probs_vs_per_aid = dict()
        probs_gr_per_aid = dict()

        counter_idx = 0

        #  TODO... Fragments per agent id... But remain in the order said by fragment_pairs
        
        for aid, fragment_pairs_aid in fragment_pairs_per_agent_id.items():
            model = models_per_agent[aid]
            n_fragments = len(fragment_pairs_aid)

            probs_vs_per_aid[aid] = th.empty(n_fragments, dtype=dtype)
            if only_for_alignment_function is None:
                probs_gr_per_aid[aid] = th.zeros(
                    (len(fragment_pairs_aid), fragment_pairs_aid[0][0].value_rews.shape[0]), dtype=dtype)

            all_transitions_aid = rollout.flatten_trajectories(
                [frag for fragment in fragment_pairs_aid for frag in fragment])

            all_rews_vs_aid, all_rews_gr_aid = self.rewards(
                all_transitions_aid, only_with_alignment=False, alignment=only_for_alignment_function, custom_model=model, only_grounding=only_grounding)

            idx = 0

            if self.algorithm.allow_variable_horizon:
                for iad, (f1, f2) in enumerate(fragment_pairs_aid):

                    rews1_vsaid, rews1_graid = all_rews_vs_aid[idx:idx+len(
                        f1)], all_rews_gr_aid[:, idx:idx+len(f1)]
                    assert np.allclose(
                        f1.acts, all_transitions_aid[idx:idx+len(f1)].acts)

                    idx += len(f1)
                    rews2_vsaid, rews2_graid = all_rews_vs_aid[idx:idx+len(
                        f2)], all_rews_gr_aid[:, idx:idx+len(f2)]
                    assert np.allclose(
                        f2.acts, all_transitions_aid[idx:idx+len(f2)].acts)

                    idx += len(f2)
                    if not only_grounding:
                        probs_vs_per_aid[aid][iad] = self.probability(
                            rews1_vsaid, rews2_vsaid)
                    if only_for_alignment_function is None:
                        for j in range(rews1_graid.shape[0]):
                            probs_gr_per_aid[aid][iad, j] = self.probability(
                                rews1_graid[j], rews2_graid[j])
            else:
                fragment_size = len(fragment_pairs_aid[0][0])
                if not only_grounding:
                    rews1_vsaid_all, rews2_vsaid_all = PreferenceModelClusteredVSL._slice_all_transitions_into_pairs(
                        all_rews_vs_aid, fragment_size)
                    probs_vs_per_aid[aid] = self.probability(
                        rews1_vsaid_all, rews2_vsaid_all)
                if only_for_alignment_function is None:
                    for j in range(fragment_pairs_aid[0][0].value_rews.shape[0]):
                        rews1_graid_all_j, rews2_graid_all_j = PreferenceModelClusteredVSL._slice_all_transitions_into_pairs(
                            all_rews_gr_aid[j], fragment_size)
                        probs_gr_per_aid[aid][:, j] = self.probability(
                            rews1_graid_all_j, rews2_graid_all_j)

            if fragment_pairs_idxs_per_agent_id is not None:
                fragment_idxs = fragment_pairs_idxs_per_agent_id[aid]
            else:
                # just put them in order of appearance of each agent. 
                fragment_idxs = np.array(list(range(n_fragments))) + counter_idx
            if not only_grounding:
                probs_vs[fragment_idxs] = probs_vs_per_aid[aid]
            if only_for_alignment_function is None:
                probs_gr[fragment_idxs] = probs_gr_per_aid[aid]
            counter_idx += n_fragments

        if custom_model_per_agent_id is not None:
            self.model = prev_model

        if add_probs_per_agent:
            return probs_vs, probs_gr, probs_vs_per_aid, probs_gr_per_aid
        else:
            return probs_vs, probs_gr,

    def rewards(self, transitions: Transitions, only_with_alignment=False, only_grounding=False, real=False, alignment=None, grounding=None, custom_model=None) -> th.Tensor:
        """Computes the reward for all transitions.

        Args:
            transitions: batch of obs-act-obs-done for a fragment of a trajectory.

        Returns:
            The reward given by the network(s) for all the transitions.
            Shape - (num_transitions, ) for Single reward network and
            (num_transitions, num_networks) for ensemble of networks.
        """
        if custom_model is not None:
            prev_model = self.model
            self.model: AbstractVSLRewardFunction = custom_model

        state = None
        action = None
        next_state = None
        state = deepcopy(util.safe_to_tensor(types.assert_not_dictobs(
                transitions.obs), device=self.model.device, dtype=self.model.dtype))
        action = deepcopy(util.safe_to_tensor(
            transitions.acts, device=self.model.device, dtype=self.model.dtype))
        if self.model.use_next_state:
            next_state = deepcopy(util.safe_to_tensor(types.assert_not_dictobs(
            transitions.next_obs), device=self.model.device, dtype=self.model.dtype))
        
        info = transitions.infos[0] 


        grouped_transitions = {}
        if 'context' in info.keys():
            # group by context...
            # Initialize a dictionary to store grouped transitions
            # Iterate through the transitions and group them by context
            grouped_transitions = defaultdict(list)
            for iinf, info_ in enumerate(transitions.infos):
                grouped_transitions[info_['context']].append(iinf)
        
        else:
            grouped_transitions['no-context1'] = list(range(len(transitions)))

        
        rews = th.zeros((len(transitions.obs), ), device=self.model.device, dtype=self.model.dtype)
        rews_gr = th.zeros((self.algorithm.env.n_values, len(transitions.obs)), device=self.model.device, dtype=self.model.dtype)

        if self.model.mode == TrainingModes.VALUE_GROUNDING_LEARNING and alignment is None:
                # This is a dummy alignment function when we only want the grounding preferences. We set it as tuple to be sure no chages are made here.
                alignment = tuple(self.model.get_learned_align_function())

        #indexes_so_far = []
        for context, indexes in grouped_transitions.items():
            info = transitions.infos[indexes[0]]

            """for i in indexes:
                assert i not in indexes_so_far
            indexes_so_far.extend(indexes)"""

            #self.algorithm.env.contextualize(context)
            #th.testing.assert_close(rews[indexes], th.zeros((len(deepcopy(states_i)),), device=self.model.device, dtype=self.model.dtype))
            #th.testing.assert_close(rews_gr[:,indexes], th.zeros((self.algorithm.env.n_values, len(deepcopy(states_i))), device=self.model.device, dtype=self.model.dtype))

            # done = transitions.dones
            states_i = state[indexes]
            action_i = action[indexes]
            next_states_i = next_state[indexes] if next_state is not None else None

            if only_with_alignment:
                rews[indexes], np_rewards = self.algorithm.calculate_rewards(alignment if self.model.mode == TrainingModes.VALUE_GROUNDING_LEARNING else None,
                                                                            custom_model=self.model,
                                                                            grounding=None,
                                                                            obs_mat=states_i,
                                                                            action_mat=action_i,
                                                                            next_state_obs_mat=next_states_i,
                                                                            obs_action_mat=None,  # TODO?
                                                                            reward_mode=self.algorithm.training_mode,
                                                                            recover_previous_config_after_calculation=False,
                                                                            use_probabilistic_reward=False, requires_grad=True, forward_groundings=False,
                                                                            info=info)
                

                if real:
                    _, rews[indexes] = self.algorithm.calculate_rewards(alignment,
                                                                        custom_model=self.model,
                                                                        grounding=grounding,
                                                                        obs_mat=states_i,
                                                                        action_mat=action_i,
                                                                        next_state_obs_mat=next_states_i,
                                                                        obs_action_mat=None,
                                                                        reward_mode=TrainingModes.EVAL,
                                                                        recover_previous_config_after_calculation=True,
                                                                        use_probabilistic_reward=False, requires_grad=False, info=info)
                    
                
                    

                
            else:
                rews[indexes], np, rews_gr[:, indexes], np2 = self.algorithm.calculate_rewards(alignment if self.model.mode == TrainingModes.VALUE_GROUNDING_LEARNING else None,
                                                                            custom_model=self.model,
                                                                            grounding=None if self.algorithm.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING else grounding,
                                                                            obs_mat=states_i,
                                                                            action_mat=action_i,
                                                                            next_state_obs_mat=next_states_i,
                                                                            obs_action_mat=None,  # TODO?
                                                                            reward_mode=self.algorithm.training_mode,
                                                                            recover_previous_config_after_calculation=False,
                                                                            use_probabilistic_reward=False, requires_grad=True, info=info, forward_groundings=True)

                # th.testing.assert_close(rews_gr, th_rewards_gr)
                #print("NC??", self.algorithm.env.context, info['context'])
                

        
        if custom_model is not None:
            self.model = prev_model
            
        if only_with_alignment:
            return rews, None
    
        assert len(state) == len(action)
        if not only_grounding:
            assert rews.shape == (len(state),)
        assert rews_gr.shape == (
            self.algorithm.env.n_values, len(state))
        
        
        return rews, rews_gr

    def probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        """Changed from imitation the ability to compare fragments of different lengths
        """
        # First, we compute the difference of the returns of
        # the two fragments. We have a special case for a discount
        # factor of 1 to avoid unnecessary computation (especially
        # since this is the default setting). The else part is for calculating probability of batches of pairs of trajectories.
        if len(rews1.shape) <= 1:
            if self.discount_factor == 1:
                if len(rews1) == len(rews2):
                    returns_diff = (rews2 - rews1).sum(axis=0)
                else:
                    returns_diff = rews2.sum(axis=0) - rews1.sum(axis=0)
            else:
                device = rews1.device
                assert device == rews2.device
                l1, l2 = len(rews1), len(rews2)
                discounts = self.discount_factor ** th.arange(
                    max(l1, l2), device=device)
                if self.ensemble_model is not None:
                    discounts = discounts.reshape(-1, 1)
                if len(rews1) == len(rews2):
                    returns_diff = (discounts * (rews2 - rews1)).sum(axis=0)
                else:
                    returns_diff = (
                        discounts[0:l2] * rews2).sum(axis=0) - (discounts[0:l1] * rews1).sum(axis=0)
        else:
            # Batched calculation. Need to have same length trajectories
            assert rews1.shape == rews2.shape
            if self.discount_factor < 1.0:
                device = rews1.device
                assert device == rews2.device

                if not hasattr(self, 'cached_discounts') or len(self.cached_discounts) < rews1.shape[1]:
                    self.cached_discounts = self.discount_factor ** np.arange(
                        rews1.shape[1])

                returns_diff = (self.cached_discounts *
                                (rews2 - rews1)).sum(axis=1)
            else:
                returns_diff = (rews2 - rews1).sum(axis=1)

            assert returns_diff.shape == (rews1.shape[0],)

        # Clip to avoid overflows (which in particular may occur
        # in the backwards pass even if they do not in the forward pass).
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        # We take the softmax of the returns. model_probability
        # is the first dimension of that softmax, representing the
        # probability that fragment 1 is preferred.
        probability = 1.0 / (1.0 + returns_diff.exp())
        if self.noise_prob > 0:
            probability = self.noise_prob * 0.5 + \
                (1 - self.noise_prob) * probability

        return probability


class CrossEntropyRewardLossCluster(preference_comparisons.RewardLoss):
    """Compute the cross entropy reward loss."""
    def gradients(self, scalar_loss: th.Tensor, renormalization: float) -> None:
        scalar_loss *= renormalization
        return scalar_loss.backward()
    def set_parameters(self, *params) -> None:
        pass

    def __init__(self, model_indifference_tolerance, apply_on_misclassified_pairs_only=False, confident_penalty=0.0, cluster_similarity_penalty=0.01) -> None:
        """Create cross entropy reward loss."""
        super().__init__()
        # This is the tolerance in the probability model when in the ground truth two trajectories are deemed equivalent (i.e. if two trajectories are equivalent, the ground truth target is 0.5. The model should output something in between (0.5 - indifference, 0.5 + indifference) to consider it has done a correct preference prediction.)
        self.model_indifference_tolerance = model_indifference_tolerance
        self.apply_on_misclassified_only = apply_on_misclassified_pairs_only
        self.confident_penalty = confident_penalty
        self.cluster_similarity_penalty = cluster_similarity_penalty


    def loss_func(self, probs, target_probs, misclassified_pairs=None):
        if misclassified_pairs is not None and self.apply_on_misclassified_only:
            probs_l = probs[misclassified_pairs == True]
            target_probs_l = target_probs[misclassified_pairs == True]
        else:
            probs_l = probs
            target_probs_l = target_probs
        
        if self.confident_penalty > 0.0:
            return th.mean(th.nn.functional.binary_cross_entropy(probs_l, target_probs_l, reduce=False) - self.confident_penalty*th.multiply(probs_l, th.log(probs_l)))
        else:
            return th.nn.functional.binary_cross_entropy(probs_l, target_probs_l, reduce='mean')
        

    def forward(
        self,
        preferences: np.ndarray,
        preferences_with_grounding: np.ndarray,
        preference_model: PreferenceModelClusteredVSL,
        reward_model_per_agent_id: Dict[str, AbstractVSLRewardFunction],
        # agent_ids: Dict[str, np.ndarray],
        fragment_idxs_per_aid: Dict[str, List[int]],
        fragment_pairs_per_agent_id: Dict[str,
                                          Sequence[TrajectoryPair]] = None,
        preferences_per_agent_id: np.ndarray = None,
        preferences_per_agent_id_with_grounding: np.ndarray = None,

        value_system_network_per_cluster: List =None,
        grounding_per_value_per_cluster: List =None

    ) -> preference_comparisons.LossAndMetrics:
        """Computes the loss. Same as Cross Entropy but does not overfit to class certainty, i.e. does not consider examples that are already correct."""
        # start_time = time.time()

        probs_vs, probs_gr, probs_vs_per_agent, probs_gr_per_agent = preference_model.forward(fragment_pairs_per_agent_id=fragment_pairs_per_agent_id, custom_model_per_agent_id=reward_model_per_agent_id,
                                                                                              fragment_pairs_idxs_per_agent_id=fragment_idxs_per_aid, only_for_alignment_function=None, add_probs_per_agent=True)

        """This tests forward works correctly...
        probs_vs_n, _, gt_probs_vs_n, gt_probs_gr_n  = preference_model.forward(fragment_pairs_per_agent_id={'normal_0': fragment_pairs_per_agent_id['normal_0']}, custom_model_per_agent_id={'normal_0': reward_model_per_agent_id['normal_0']}, 
                                                                                 fragment_pairs_idxs_per_agent_id = {'normal_0': list(range(len(fragment_pairs_per_agent_id['normal_0'])))},
                                                                                 gt_probs_vs=preferences[fragment_idxs_per_aid['normal_0']], 
                                                                                 gt_probs_gr=preferences_with_grounding[fragment_idxs_per_aid['normal_0'], :], only_for_alignment_function=(0.0,1.0))
        
        th.testing.assert_close(probs_gr[:,1][fragment_idxs_per_aid['normal_0']], probs_vs_n)
        th.testing.assert_close(gt_probs_gr[:,1][fragment_idxs_per_aid['normal_0']], preferences_with_grounding[fragment_idxs_per_aid['normal_0'], 1])

        th.testing.assert_close(gt_probs_gr[:,1][fragment_idxs_per_aid['normal_0']], gt_probs_gr_n[:,1])"""
        # end_time = time.time()
        # print(f"Execution time for preference_model.forward: {end_time - start_time} seconds")
        # start_time = time.time()
        accuracy_vs, accuracy_gr, missclassified_vs, missclassified_gr = calculate_accuracies(
            probs_vs, probs_gr, preferences, preferences_with_grounding, indifference_tolerance=self.model_indifference_tolerance)

        accuracy_vs_per_agent, accuracy_gr_per_agent, missclassified_vs_per_agent, missclassified_gr_per_agent = {}, {}, {}, {}
        for aid in preferences_per_agent_id.keys():
            accuracy_vs_per_agent[aid], accuracy_gr_per_agent[aid], missclassified_vs_per_agent[aid], missclassified_gr_per_agent[aid] = calculate_accuracies(
                probs_vs_per_agent[aid], probs_gr_per_agent[aid], preferences_per_agent_id[aid], preferences_per_agent_id_with_grounding[aid], indifference_tolerance=self.model_indifference_tolerance)
            accuracy_gr_per_agent[aid] = np.array(accuracy_gr_per_agent[aid])
            accuracy_vs_per_agent[aid] = float(accuracy_vs_per_agent[aid])

        self.last_accuracy_gr = np.array(accuracy_gr)

        metrics = {}
        metrics["global_accvs"] = accuracy_vs
        metrics["global_accgr"] = np.array(accuracy_gr)

        metrics["accvs"] = accuracy_vs_per_agent
        metrics["accgr"] = accuracy_gr_per_agent
        metrics['loss_per_vi'] = {}
        # misclassified_pairs = predictions != ground_truth

        metrics = {key: value.detach().cpu() if isinstance(
            value, th.Tensor) else value for key, value in metrics.items()}

        # LOSS VALUE SYSTEM.
        example_model = reward_model_per_agent_id[list(
            reward_model_per_agent_id.keys())[0]]
        if reward_model_per_agent_id[list(reward_model_per_agent_id.keys())[0]].mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            loss_vs = th.tensor(
                0.0, device=example_model.device, dtype=example_model.dtype)
        else:
            value_systems_in_each_cluster = th.stack([clust.get_alignment_layer()[0][0] for clust in value_system_network_per_cluster])
            
            # TODO. only take into account the clusters that are used in the current assignment.
            avg_vs = th.mean(value_systems_in_each_cluster, axis=0)
            
            vs_penalty = th.norm(value_systems_in_each_cluster - avg_vs )
            """print(vs_penalty, )
            vs_penalty.backward()
            print("GRAD", value_system_network_per_cluster[0].weight.grad)
            print("GRAD", value_system_network_per_cluster[1].weight.grad)
            print("GRAD", value_system_network_per_cluster[2].weight.grad)
            print("GRAD", value_system_network_per_cluster[3].weight.grad)
            exit(0)"""
            loss_vs = self.loss_func(probs_vs, preferences, missclassified_vs) - self.cluster_similarity_penalty*vs_penalty
            metrics['loss_vs'] = loss_vs
        # LOSS GROUNDING.
        # start_time = time.time()

        loss_gr = th.tensor(0.0, device=example_model.device,
                            dtype=example_model.dtype)
        for vi in range(probs_gr.shape[1]):
            # TODO: This is the modified loss...
            """
            # This idea is CONFIDENT PENALTY: https://openreview.net/pdf?id=HyhbYrGYe ICLR 2017
            # TODO: Idea for future work:
            # from pairwise comparisons, minimize divergence from a prior that consists on the expected class probability, 
            # being it estimated online.
            # Under the assumption of convergence, we should find a single only possible function that is equivalent to the original
            # preferences (up to some operation, probably multiplication by a constant).
            
            """
            # loss_gr += th.mean(th.nn.functional.binary_cross_entropy(prgvi, preferences_with_grounding[:, vi], reduce=False) -th.multiply(prgvi, self.beta*th.log(prgvi)))
            
            nl = self.loss_func(probs_gr[:, vi], preferences_with_grounding[:, vi], misclassified_pairs=missclassified_gr[vi])
            metrics['loss_per_vi'][vi] = nl
            loss_gr += nl
            assert not loss_gr.isnan()
        metrics['loss_gr'] = loss_gr
        # end_time = time.time()
        # print(f"Execution time for loss grounding: {end_time - start_time} seconds")
        return preference_comparisons.LossAndMetrics(
            loss=(loss_vs, loss_gr),
            metrics=metrics,
        )


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
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        regularizer_factory: Optional[imitation.regularization.regularizers.RegularizerFactory] = None,
        optim_cls=th.optim.AdamW,
        optim_kwargs=dict(lr=1e-3, weight_decay=0.0)
    ) -> None:
        lr = optim_kwargs['lr']
        
        super().__init__(preference_model, loss, rng, batch_size,
                         minibatch_size, epochs, lr, custom_logger, regularizer_factory)
        # WEIGHT DECAY MUST BE 0, otherwise not affected networks may decay without reason!!!!!!!
        optim_kwargs['weight_decay'] = 0.0
        self.optim_kwargs = optim_kwargs
        self.optim_cls = optim_cls

        # Default. This should be overriden.
        if self.optim_cls != SobaOptimizer:
            self.reset_optimizer_with_params(
                parameters=self._preference_model.parameters())
        else:
            self.reset_optimizer_with_params(
                parameters={'params_gr': list(self._preference_model.parameters()), 'params_vs': list(self._preference_model.parameters())})
            

        self.regularizer = (
            regularizer_factory(optimizer=self.optim, logger=self.logger)
            if regularizer_factory is not None
            else None
        )

    def reset_optimizer_with_params(self, parameters):
        # WEIGHT DECAY MUST BE 0, otherwise not affected networks may decay without reason!!!!!!!
        assert self.optim_kwargs['weight_decay'] == 0.0
        if isinstance(parameters, dict):
            self.optim = self.optim_cls(params_gr=parameters['params_gr'], params_vs=parameters['params_vs'], **self.optim_kwargs)
        else:
            self.optim = self.optim_cls(parameters, **self.optim_kwargs)


def nth_derivative(f, wrt, n):
    all_grads = []
    for i in range(n):

        grads = th.autograd.grad(f, wrt, create_graph=i <n-1, retain_graph=True)
        all_grads.append(grads)
        f = th.sum(th.cat(tuple(g.flatten() for g in grads)))

    return all_grads
def derivative21(f, wrt1, wrt2, dw2 = None, retain_last = True):
    if dw2 is None:
        grads = th.autograd.grad(f, wrt2, create_graph=True, retain_graph=True, allow_unused=True, materialize_grads=True)
        if grads is None:
            return None, None
    else:
        grads = dw2
    f = th.sum(th.cat(tuple(g.flatten() for g in grads)))
    grads2 = th.autograd.grad(f, wrt1, retain_graph=True, create_graph=True, allow_unused=True, materialize_grads=True)
    
    return grads, grads2
def gradients_soba(wx, wy, vt, goal1_x, goal2_xy):
    
    grad_G_wrt_wx, grad_G_wrt_wx2 = nth_derivative(goal1_x, wx, 2)
    grad_G_wrt_wy, grad_G_wrt_wywx  = derivative21(goal1_x, wx, wy, retain_last = False)
    
    grad_F_wrt_wx = th.autograd.grad(goal2_xy, wx, retain_graph=True, create_graph=True,allow_unused=True, materialize_grads=True)
    grad_F_wrt_wy = th.autograd.grad(goal2_xy, wy, retain_graph=False, create_graph=True, allow_unused=True, materialize_grads=True)

    
    if grad_G_wrt_wywx is None:
        Dlambda = grad_F_wrt_wy
    else:
        Dlambda = []
        cum_sum = sum(th.sum(gryx.mul(vt_i.T)) for gryx, vt_i in zip(grad_G_wrt_wywx, vt))
        for p in grad_F_wrt_wy:
            Dlambda.append(cum_sum + p) 

    Dv = []
    cum_sum2 = sum(th.sum(grx2.matmul(vt_i.T)) for grx2, vt_i in zip(grad_G_wrt_wx2, vt))
    for p in grad_F_wrt_wx:
        Dv.append(cum_sum2 + p) 

    Dtheta = grad_G_wrt_wx
    
    return Dtheta,Dv, Dlambda


class SobaOptimizer(th.optim.Optimizer):
    def __init__(self, params_gr, params_vs, lr=0.001, lr_grounding=None, lr_value_system=None, use_lr_decay=True, **optimizer_kwargs):

        lr_grounding = lr if lr_grounding is None else lr_grounding
        lr_value_system = lr if lr_value_system is None else lr_value_system

        defaults = dict(lr_grounding=lr_grounding, lr_value_system=lr_value_system, lr_vt=lr_value_system)
        self.use_lr_decay = use_lr_decay
        self.optimizer_kwargs = optimizer_kwargs

        self.optimx = th.optim.SGD(params_gr, lr=lr_grounding, **self.optimizer_kwargs)
        self.optimy = th.optim.SGD(params_vs, lr=lr_value_system, **self.optimizer_kwargs)

        self.vt = [th.zeros_like(p) for p in params_gr]
        self.wx = params_gr
        self.wy = params_vs

        super(SobaOptimizer, self).__init__([*params_gr, *params_vs], defaults)

        self.lr_value_system = lr_value_system

        self.time = 0
        self.set_state({'time': self.time, 'vt': self.vt})

    def zero_grad(self, set_to_none = True):

        self.optimx.zero_grad(set_to_none)
        self.optimy.zero_grad(set_to_none)
        self.optimv.zero_grad(set_to_none)

    def get_state(self):
        return {'time': self.time, 'vt': self.vt}
    
    def set_state(self, state):
        if state is not None:
            self.time = state['time']
            self.vt = state['vt']
            self.optimv = th.optim.SGD(state['vt'], lr=self.lr_value_system)

            if self.use_lr_decay:
                self.optimx_scheduler = th.optim.lr_scheduler.LambdaLR(lr_lambda=lambda epoch: 1/np.sqrt(epoch+self.time+1), optimizer=self.optimx)
                self.optimy_scheduler = th.optim.lr_scheduler.LambdaLR(lr_lambda=lambda epoch: 1/np.sqrt(epoch+self.time+1), optimizer=self.optimy)
                self.optimv_scheduler = th.optim.lr_scheduler.LambdaLR(lr_lambda=lambda epoch: 1/np.sqrt(epoch+self.time+1), optimizer=self.optimv)

    def step(self, closure=None):
        self.optimx.step()
        self.optimy.step()
        
        self.optimv.step()

        if self.use_lr_decay:
            self.optimx_scheduler.step()
            self.optimy_scheduler.step()
            self.optimv_scheduler.step() #TODO: This is an issue. After loading a checkpoint, the scheduler is not updated...
        return None


class SobaLoss(CrossEntropyRewardLossCluster):

    def __init__(self, model_indifference_tolerance, apply_on_misclassified_pairs_only=False, confident_penalty=0, cluster_similarity_penalty=0.00, max_grad_norm_gr=1000.0, max_grad_norm_vs=1000.0):
        super().__init__(model_indifference_tolerance, apply_on_misclassified_pairs_only, confident_penalty, cluster_similarity_penalty)
        
        self.max_grad_norm_gr = max_grad_norm_gr
        self.max_grad_norm_vs = max_grad_norm_vs

    def set_parameters(self, wx,wy,vt):
        self.wx = wx
        self.wy = wy
        self.vt = vt

    
    """def forward(self, gr_loss, vs_loss):
        self.gr_loss = gr_loss
        self.vs_loss = vs_loss
        return gr_loss + vs_loss"""
    def parameters(self, recurse = True):
        return self.wx + self.wy + self.vt
    
    def forward(self, preferences, preferences_with_grounding, preference_model, reward_model_per_agent_id, fragment_idxs_per_aid, fragment_pairs_per_agent_id = None, preferences_per_agent_id = None, preferences_per_agent_id_with_grounding = None, value_system_network_per_cluster = None, grounding_per_value_per_cluster = None):
        lossMetrics = super().forward(preferences, preferences_with_grounding, preference_model, reward_model_per_agent_id, fragment_idxs_per_aid, fragment_pairs_per_agent_id, preferences_per_agent_id, preferences_per_agent_id_with_grounding, value_system_network_per_cluster, grounding_per_value_per_cluster)
        self.gr_loss = lossMetrics.loss[1]
        self.vs_loss = lossMetrics.loss[0]
        return lossMetrics

    def gradients(self, scalar_loss: th.Tensor, renormalization: float) -> None:
        Dtheta, Dv, Dlambda = gradients_soba(
            self.wx, self.wy, self.vt, self.gr_loss*renormalization, self.vs_loss*renormalization
        )
        print(len(Dtheta), len(self.wx))
        assert len(Dtheta) == len(self.wx)
        assert len(Dlambda) == len(self.wy)
        assert len(self.vt) == len(Dv)
        for p,pref in zip(self.wx, Dtheta):
            p.grad = pref
        
        for p,pref in zip(self.wy, Dlambda):
            p.grad = pref
        for p,pref in zip(self.vt, Dv):
            p.grad = pref
        
        th.nn.utils.clip_grad_norm_(self.wx, self.max_grad_norm_gr)
        th.nn.utils.clip_grad_norm_(self.wy, self.max_grad_norm_vs)
        th.nn.utils.clip_grad_norm_(self.vt, self.max_grad_norm_gr)