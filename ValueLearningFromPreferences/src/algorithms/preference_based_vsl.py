from copy import deepcopy
import itertools
import os
import dill
import random
import re
import time
from colorama import Style
import imitation
from imitation.data.types import TrajectoryPair, TrajectoryWithRewPair
from imitation.util import logger as imit_logger, util
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from torch.utils import data as data_th
import tqdm
from src.algorithms.clustering_utils import ASSIGNMENT_CHECKPOINTS, ClusterAssignment, ClusterAssignmentMemory, check_assignment_consistency, check_grounding_value_system_networks_consistency_with_optim, check_optimizer_consistency
from src.algorithms.base_vsl_algorithm import BaseVSLAlgorithm
import torch as th


from src.algorithms.preference_based_vsl_lib import BasicRewardTrainerVSL, CrossEntropyRewardLossCluster, PrefLossClasses, PreferenceModelClusteredVSL, SobaLoss, SobaOptimizer, calculate_accuracies, likelihood_x_is_target
from src.algorithms.utils import PolicyApproximators, assign_colors, convert_nested_list_to_tuple, mce_partition_fh
from src.data import VSLPreferenceDataset
from src.policies.vsl_policies import VAlignedDictSpaceActionPolicy, ValueSystemLearningPolicy
from src.reward_nets.vsl_reward_functions import AbstractVSLRewardFunction, ConvexAlignmentLayer, TrainingModes

from imitation.algorithms import preference_comparisons




class ClusteringRewardTrainerVSL(BasicRewardTrainerVSL):
    
    def __init__(self, preference_model: PreferenceModelClusteredVSL, loss: CrossEntropyRewardLossCluster, rng: np.random.RandomState, batch_size=32,
                 minibatch_size: int = None, epochs: int = 1, refining_steps_after_cluster_assignment: int = 1, initial_refining_steps: int = 50,
                 initial_exploration_rate: float = 0.5, qualitative_cluster_assignment: bool = True, custom_logger: imit_logger.HierarchicalLogger = None, 
                 regularizer_factory: Optional[imitation.regularization.regularizers.RegularizerFactory] = None,
                 optim_cls: th.optim.Optimizer = th.optim.AdamW,
                 inner_k_fold_validation_divisions_per_epoch = None,
                 optim_kwargs=dict(lr=1e-3, weight_decay=0.0)):
        """Trains a reward model for a society of agents (clusterizing groundings of values into certain number of clusters)

        Args:
            preference_model (PreferenceModelClusteredVSL): Preference model that estimates the degree of preference of one trajectory over other, based on the different reward models that
            loss (CrossEntropyRewardLossCluster): Loss calculator
            rng (np.random.RandomState): random state generator
            batch_size (int, optional): Batch size. Defaults to 32.
            minibatch_size (int, optional): Minibatch size, optional. Defaults to None.
            epochs (int, optional): Number of epochs to train the model in each iteration, each including: batch selection, cluster assignment, and model refining. Defaults to 1.
            refining_steps_after_cluster_assignment (int, optional): Number of gradient descent updates in the model refining phase after a cluster assignment. Defaults to 1.
            initial_refining_steps (int, optional): In the first iteration, the number of refining steps to take. Defaults to 50.
            initial_exploration_rate (float, optional): Initial exploration rate for the cluster assignment phase. Defaults to 0.5.
            qualitative_cluster_assignment (bool, optional): If True, uses the classification accuracy (number of correctly identified preferences) for cluster assignment, else, a likelihood-based score, which is computationally expensive but useful if the preferences are quantiative and not only qualitative. Defaults to True.
            custom_logger (imit_logger.HierarchicalLogger, optional): Logger from the imitation package. Defaults to None.
            regularizer_factory (Optional[imitation.regularization.regularizers.RegularizerFactory], optional): Regularizer factory for the loss function. TODO: Not used Defaults to None.
            optim_cls (th.optim.Optimizer, optional): Optimizer class. Defaults to th.optim.AdamW.
            optim_kwargs (dict, optional): Keyword arguments for the optimizer. Defaults to dict(lr = 1e-3, weight_decay = 0.0).
        """
        
        self.n_values = preference_model.algorithm.env.n_values
        self.basic_profiles = preference_model.algorithm.env.basic_profiles
        self.train_mode = preference_model.algorithm.training_mode
        self.refining_steps_after_cluster_assignment = refining_steps_after_cluster_assignment
        self.initial_refining_steps = initial_refining_steps
        self.initial_exploration_rate = initial_exploration_rate
        self.qualitative_cluster_assignment = qualitative_cluster_assignment


        self.use_full_batch = batch_size == "full"
        # if batchsize = '5-fold' or '3-fold' or '224-fold', put the numbers in self.k_cross_validation
        if inner_k_fold_validation_divisions_per_epoch == 1:
            self.k_mini_folds = None # no validation set, it will be the full dataset instead.
        else:
            self.k_mini_folds = inner_k_fold_validation_divisions_per_epoch
            
        if self.use_full_batch:
            batch_size = 32  # dummy initialization

        super().__init__(preference_model=preference_model, loss=loss, rng=rng, batch_size=batch_size, minibatch_size=minibatch_size,
                         epochs=epochs, custom_logger=custom_logger, regularizer_factory=regularizer_factory, optim_cls=optim_cls, optim_kwargs=optim_kwargs)
        self._preference_model: PreferenceModelClusteredVSL
        self.cluster_colors = None
        self.cluster_colors_vs = None

    def reset_training(self):
        #self.assignment_scores_gr = [{} for vi in range(self.n_values)]
        self.min_inter_dist_gr_so_far = [float('inf') for vi in range(self.n_values)]
        #self.assignment_scores_vs = {}
        self.min_inter_dist_vs_so_far = float('inf')

    
    def _make_data_loader(self, dataset: data_th.Dataset) -> data_th.DataLoader:
        """Make a dataloader."""
        return data_th.DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            collate_fn=self._preference_collate_fn,
        )

    def _preference_collate_fn(
            self,
        batch: Sequence[Tuple[TrajectoryWithRewPair, float, float]],
    ) -> Tuple[Sequence[TrajectoryWithRewPair], np.ndarray]:
        fragment_pairs, preferences, preferences_per_grounding, agent_ids = zip(
            *batch)
        dtype = self._preference_model.model.dtype
        device = self._preference_model.model.device
        return np.asarray(fragment_pairs), util.safe_to_tensor(np.array(preferences), device=device, dtype=dtype), util.safe_to_tensor(np.array(preferences_per_grounding), device=device, dtype=dtype), np.asarray(agent_ids)

    def train(self, dataset: VSLPreferenceDataset, reward_model_per_agent_id: Mapping[str, AbstractVSLRewardFunction], grounding_per_value_per_cluster: List[List[th.nn.Module]], value_system_per_cluster: List[Any] = [], running_assignment_vs=None, running_assignment_gr=None, epoch_multiplier: float = 1.0, global_iteration=None, global_iterations=None, starting_assignment: ClusterAssignment = 'start_random', assignment_ranking: ClusterAssignmentMemory=None) -> None:
        

        if self.use_full_batch:
            self.batch_size = len(dataset)
            self.minibatch_size = self.batch_size
        if self.k_mini_folds is not None:
            cross_splits = dataset.k_fold_split(self.k_mini_folds)
        else:
            cross_splits =[(dataset, None)]

        if self.cluster_colors is None:
            self.cluster_colors = assign_colors(
                max(len(grounding_per_value_per_cluster[vi]) for vi in range(self.n_values)))
        if self.cluster_colors_vs is None:
            self.cluster_colors_vs = assign_colors(len(value_system_per_cluster))

        epochs = round(self.epochs * epoch_multiplier)

        assert epochs > 0, "Must train for at least one epoch."

    
        if global_iteration == 0:

            self.reset_training()
        
            
        
        general_batch_size = self.batch_size # overriden in each val-train-split
        check_optimizer_consistency(reward_model_per_agent_id, self.optim)
        check_grounding_value_system_networks_consistency_with_optim(grounding_per_value_per_cluster, value_system_per_cluster, self.optim)
                        
        with self.logger.accumulate_means("reward"):
            
            for ie, epoch_num in enumerate(tqdm.tqdm(range(epochs), desc="Training reward model")):
                
                for cvi, (mini_train_dataset, mini_val_dataset) in enumerate(cross_splits):
                    # we train a single model with K splits train_set, val_set
                    # in the outer loop a normal K fold cross validation is done.
                    # Then, the test set evaluation.
                    self.batch_size = min(len(mini_train_dataset), general_batch_size)
                    
                    dataloader = self._make_data_loader(mini_train_dataset)
                    if mini_val_dataset is not None:
                        self.batch_size = len(mini_val_dataset)
                        val_data_loader = self._make_data_loader(mini_val_dataset)
                    else:
                        self.batch_size = len(mini_train_dataset)
                        #assert len(mini_train_dataset) == general_batch_size
                        val_data_loader = self._make_data_loader(mini_train_dataset)

                    self.batch_size = general_batch_size # recover original batch size that is used for training 
                    
                    with self.logger.add_key_prefix(f"ep-{epoch_num}-cv-{cvi}"):
                        train_loss = 0.0
                        accumulated_size = 0
                        self.optim.zero_grad()

                        st_full_batch = time.time()
                        for fragment_pairs, preferences, preferences_per_grounding, agent_ids in dataloader: 
                            

                            print(f"EPOCH: {epoch_num} NOW TRAINING WITH", len(
                                fragment_pairs), "over", len(dataset))
                            # EXPECTATION MAXIMIZTION: REF: https://proceedings.mlr.press/v235/chakraborty24b.html

                            # E step: Cluster assignment.
                            # For each value (indexed by 'vi'), there can be (at most) self.cluster_sizes[vi] different "clusters". Each of those "clusters" is a different grounding function that aims to ground the meaning of each value.
                            # Each agent is assigned to self.n_values clusters. These clusters (grounding functions) should, after trainig, jointly, estimate what the agent believes to be the grounding of the n_values.
                            st = time.time()
                            exploration = self.initial_exploration_rate * (1 - global_iteration / global_iterations)
                            
                            print("Exploration rate: ", exploration, "epoch", epoch_num, "global_iteration", global_iteration)
                            #print(starting_assignment)
                            
                            check_optimizer_consistency(reward_model_per_agent_id, self.optim)
                            check_grounding_value_system_networks_consistency_with_optim(grounding_per_value_per_cluster, value_system_per_cluster, self.optim)
                            check_assignment_consistency(grounding_per_value_per_cluster, value_system_per_cluster, assignment_aid_to_gr_cluster=running_assignment_gr, assignment_aid_to_vs_cluster=running_assignment_vs,reward_models_per_aid= reward_model_per_agent_id)
                            
                            with th.no_grad():
                                new_assignment, fragment_pairs_per_aid, preferences_per_aid, preference_grounding_per_aid, fragment_idxs_per_aid = self.cluster_assignment(
                                        fragment_pairs, preferences, preferences_per_grounding,
                                        reward_model_per_agent_id, grounding_per_value_per_cluster,
                                        agent_ids,
                                        #iteration=global_iteration*epochs+epoch_num, max_iterations=global_iterations*epochs,
                                    running_assignment_vs = running_assignment_vs, running_assignment_gr = running_assignment_gr,
                                    value_system_network_per_cluster=value_system_per_cluster, use_assignment=starting_assignment if epoch_num == 0 else None)
                                    
                                    
                            end = time.time()
                            print("CLUSTER ASSIGNMENT EX TIME: ", end - st)
                            # Ensure optimizer consistency
                            check_assignment_consistency(grounding_per_value_per_cluster, value_system_per_cluster, assignment_aid_to_gr_cluster=running_assignment_gr, assignment_aid_to_vs_cluster=running_assignment_vs,reward_models_per_aid= reward_model_per_agent_id)
                            check_grounding_value_system_networks_consistency_with_optim(grounding_per_value_per_cluster, value_system_per_cluster, self.optim)
                            check_optimizer_consistency(reward_model_per_agent_id, self.optim)
                            # M step: Cluster refinement
                            st = time.time()
                            if __debug__:
                                prev_rews = {aid: rc.copy() for aid, rc in reward_model_per_agent_id.items()}
                                
                            #self.reset_optimizer_with_params({p for clust in grounding_per_value_per_cluster for c in clust for p in c.parameters() }.union({p for c in value_system_per_cluster for p in c.parameters()}))#.union({p for rc in reward_model_per_agent_id.values() for p in rc.parameters()}))
                            #self.optim.zero_grad()
                            
                            DIDASTEP = False
                            
                            for r in range(self.refining_steps_after_cluster_assignment if epoch_num > 0 and starting_assignment!='start_random' else self.initial_refining_steps):
                                with self.logger.add_key_prefix("train"):

                                    
                                    loss, metrics = self._training_inner_loop(
                                       
                                        preferences=preferences,
                                        preferences_with_grounding=preferences_per_grounding,
                                        reward_model_per_agent_id=reward_model_per_agent_id,
                                        grounding_per_value_per_cluster=grounding_per_value_per_cluster,
                                        fragment_idxs_per_aid=fragment_idxs_per_aid,
                                        fragment_pairs_per_aid=fragment_pairs_per_aid,
                                        preferences_per_aid=preferences_per_aid,
                                        preference_grounding_per_aid=preference_grounding_per_aid,
                                        agent_to_gr_cluster_assignments=new_assignment.agent_to_gr_cluster_assignments,
                                        agent_to_vs_cluster_assignments=new_assignment.agent_to_vs_cluster_assignments, 
                                        value_system_network_per_cluster=value_system_per_cluster
                                    )

                                    # Renormalise the loss to be averaged over
                                    # the whole batch size 
                                
                                renormalization = len(fragment_pairs) / self.batch_size

                                train_loss += loss.item()*renormalization

                                self.loss.gradients(scalar_loss = loss, renormalization = renormalization)
                                
                                accumulated_size += len(fragment_pairs)
                                if accumulated_size >= self.batch_size:
                                    # self.optim.step()
                                    if __debug__:
                                        state_dict_old = {aid: deepcopy(rc.state_dict()) for aid, rc in reward_model_per_agent_id.items()}

                                    self.optim.step()
                                    if __debug__:
                                        
                                    
                                        state_dict_new = {aid: rc.state_dict() for aid, rc in reward_model_per_agent_id.items()}
                                        

                                        counter_fails = 0
                                        
                                            
                                        print("Gradient Information:")
                                        params_rew = {p for rew in reward_model_per_agent_id.values() for p in rew.parameters()}
                                        params_vs = {p for c in value_system_per_cluster for p in c.parameters()}
                                        params_gr = {p for clust in grounding_per_value_per_cluster for c in clust for p in c.parameters()}
                                        for param in params_rew:
                                            assert param in {p for g in self.optim.param_groups for p in g['params']}, "Reward model parameter not in optimizer"
                                        one_at_least_requires_grad = False
                                        grad_norm = 0.0
                                        for param in self.optim.param_groups[0]['params']:
                                            if param.requires_grad:
                                                #print(f"Parameter: {param}, Gradient: {param.grad}")
                                                if param.grad is None:
                                                    assert param not in params_rew, "Reward model parameter in optimizer"
                                                else:
                                                    one_at_least_requires_grad = True
                                                    assert param in params_rew or param in params_gr or params_vs, "Optimizer parameter not in reward model"
                                                    grad_norm = max(th.norm(param.grad), grad_norm)
                                        assert one_at_least_requires_grad, "No parameters in optimizer have gradients"
                                        print(grad_norm)
                                        if grad_norm >= 1e-7:
                                            for aid in reward_model_per_agent_id.keys():
                                            
                                                try:
                                                    th.testing.assert_close(reward_model_per_agent_id[aid].state_dict(), prev_rews[aid].state_dict()), "State dicts of rc and raid do not match"
                                                except AssertionError:
                                                    continue
                                                counter_fails+=1
                                            if counter_fails == len(reward_model_per_agent_id.keys()):
                                                raise AssertionError("No changes in reward model???", starting_assignment, new_assignment, new_assignment_copy, "no changes in rw...")

                                            try:
                                                
                                                th.testing.assert_close(state_dict_new, state_dict_old)
                                                raise ValueError("State dicts of reward models have not changed")
                                        
                                            except AssertionError:
                                                pass
                                    self.optim.zero_grad()
                                    accumulated_size = 0
                                    
                            end = time.time()

                            print("BATCH REFINEMENT EX TIME: ", end - st)

                            
                        if accumulated_size != 0:
                            self.optim.step()  # if there remains an incomplete batch
                            self.optim.zero_grad()
                        check_assignment_consistency(grounding_per_value_per_cluster, value_system_per_cluster, assignment_aid_to_gr_cluster=running_assignment_gr, assignment_aid_to_vs_cluster=running_assignment_vs,reward_models_per_aid= reward_model_per_agent_id)
                        check_grounding_value_system_networks_consistency_with_optim(grounding_per_value_per_cluster, value_system_per_cluster, self.optim)
                        check_optimizer_consistency(reward_model_per_agent_id, self.optim)

                        end_full_batch = time.time()
                        print("Epoch Training time: ", end_full_batch - st_full_batch)
                        # Clustering evaluation. Registering the best configurations.
                        st = time.time()
                        #aggs_permutation = np.random.permutation(list(reward_model_per_agent_id.keys()))                            
                        with th.no_grad():
                            val_data_loader : data_th.DataLoader
                            data = [data for data in val_data_loader]
                            val_fragment_pairs, val_preferences, val_preferences_per_grounding, val_agent_ids = map(
                                lambda x: np.concatenate(x, axis=0) if len(data) > 1 else x[0],
                                zip(*data)
                            )
                            
                            new_assignment_copy, val_fragment_pairs_per_aid, val_preferences_per_aid, val_preference_grounding_per_aid, val_fragment_idxs_per_aid = self.cluster_assignment(
                                fragment_pairs=val_fragment_pairs, preferences=val_preferences, preferences_with_grounding=val_preferences_per_grounding,
                                reward_model_per_agent_id=reward_model_per_agent_id, grounding_per_value_per_cluster=grounding_per_value_per_cluster,
                                running_assignment_gr=running_assignment_gr, running_assignment_vs=running_assignment_vs,
                                agent_ids=val_agent_ids,
                                value_system_network_per_cluster=value_system_per_cluster, use_assignment=new_assignment.copy())
                                
                      
                            
                            probs_vs, probs_gr, probs_vs_per_agent, probs_gr_per_agent = self._preference_model.forward(fragment_pairs_per_agent_id=val_fragment_pairs_per_aid, custom_model_per_agent_id=reward_model_per_agent_id,
                                                                                              fragment_pairs_idxs_per_agent_id=val_fragment_idxs_per_aid, only_for_alignment_function=None, add_probs_per_agent=True)

                            
                            vs_intra_dist, vs_inter_dist = self.vs_discordances(reward_model_per_agent_id=reward_model_per_agent_id,
                                                                                vs_cluster_to_agents_assignment=new_assignment_copy.assignment_vs, 
                                                                                value_system_per_cluster=value_system_per_cluster,
                                                                                fragment_pairs_per_aid=val_fragment_pairs_per_aid,
                                                                                probs_per_aid=probs_vs_per_agent, preferences_per_aid=val_preferences_per_aid)
                            
                            # Grounding distances
                            gr_intra_dist, gr_inter_dist = self.gr_discordances(grounding_per_value_per_cluster=grounding_per_value_per_cluster, 
                                                                                gr_cluster_to_agents_assignment=new_assignment_copy.assignment_gr, 
                                                                                fragment_pairs_per_aid= val_fragment_pairs_per_aid,
                                                                                preferences_grounding_per_aid= val_preference_grounding_per_aid, 
                                                                                probs_gr_per_agent=probs_gr_per_agent)
                            
                            
                            new_assignment_copy.inter_discordances_gr = gr_inter_dist
                            new_assignment_copy.intra_discordances_gr = gr_intra_dist
                            
                            
                            new_assignment_copy.inter_discordances_vs = vs_inter_dist
                            new_assignment_copy.intra_discordances_vs = vs_intra_dist
                            if hasattr(self.optim, 'get_state'):
                                new_assignment_copy.optimizer_state = deepcopy(self.optim.get_state())
                            
                        
                        position, old_assignment = assignment_ranking.insert_assignment(new_assignment_copy.copy())

                        # Ensure optimizer consistency
                        check_assignment_consistency(grounding_per_value_per_cluster, value_system_per_cluster, assignment_aid_to_gr_cluster=running_assignment_gr, assignment_aid_to_vs_cluster=running_assignment_vs,reward_models_per_aid= reward_model_per_agent_id)
                        check_grounding_value_system_networks_consistency_with_optim(grounding_per_value_per_cluster, value_system_per_cluster, self.optim)
                        check_optimizer_consistency(reward_model_per_agent_id, self.optim)
                        starting_assignment = None # to not use it again
                        
                        if train_loss > 0.0 and DIDASTEP and grad_norm >= 1e-7 and __debug__:
                            counter_fails = 0
                            for aid in reward_model_per_agent_id.keys():
                                
                                try:
                                    th.testing.assert_close(reward_model_per_agent_id[aid].state_dict(), prev_rews[aid].state_dict()), "State dicts of rc and raid do not match"
                                except AssertionError:
                                    counter_fails = 0
                                    continue
                                counter_fails+=1
                            if counter_fails == len(reward_model_per_agent_id.keys()):
                                raise AssertionError("No changes in reward model???", starting_assignment, new_assignment, new_assignment_copy, "no changes in rw...")

                            assert all(
                                np.allclose(
                                    new_assignment_copy.reward_model_per_agent_id[aid].get_learned_align_function(),
                                    reward_model_per_agent_id[aid].get_learned_align_function()
                                )
                                for aid in reward_model_per_agent_id.keys()
                            ), "Not all keys satisfy the second condition."
                        if position == 0:
                            
                            print("NEW BEST", "old_scores: ", old_assignment.vs_score, old_assignment.gr_score, "new_scores: ", new_assignment_copy.vs_score, new_assignment_copy.gr_score)
                            print(assignment_ranking)
                            time.sleep(5)
                        elif old_assignment is not None:
                            print("Inserted before: ", old_assignment.vs_score, old_assignment.gr_score,"new_scores: ", new_assignment_copy.vs_score, new_assignment_copy.gr_score)
                            print(assignment_ranking)
                            #time.sleep(2)
                        else:
                            print("Not inserted.")
                        
                        end = time.time()
                        print(f"Epoch Validation time: {end-st}")
                        
                        

        # after training all the epochs,
        # record also the final value in a separate key for easy access.
        keys = list(self.logger.name_to_value.keys())
        outer_prefix = self.logger.get_accumulate_prefixes()
        for key in keys:
            # existing prefix + accum_means ctx
            base_path = f"{outer_prefix}reward/"
            # mean for last epoch
            epoch_path = f"mean/{base_path}ep-{epoch_num}-cv-{len(cross_splits)-1}/"
            final_path = f"{base_path}final/"  # path to record last epoch
            pattern = rf"{epoch_path}(.+)"
            if regex_match := re.match(pattern, key):
                (key_name,) = regex_match.groups()
                val = self.logger.name_to_value[key]
                new_key = f"{final_path}{key_name}"
                self.logger.record(new_key, val)
        

        return assignment_ranking

   
    def gr_discordances(self, grounding_per_value_per_cluster, gr_cluster_to_agents_assignment, fragment_pairs_per_aid, preferences_grounding_per_aid, probs_gr_per_agent):
        gr_inter_cluster_distances = [[] for _ in range(self.n_values)]
        agents_assigned_to_each_cluster_pair = [{(c1,c2): (gr_cluster_to_agents_assignment[vi][c1], gr_cluster_to_agents_assignment[vi][c2]) for c1,c2 in itertools.combinations(range(len(gr_cluster_to_agents_assignment[vi])), r=2) } for vi in range(self.n_values)]
        
        for vi in range(self.n_values):
            if agents_assigned_to_each_cluster_pair[vi] == {}:
                agents_assigned_to_each_cluster_pair[vi] = {(0,0): (list(fragment_pairs_per_aid.keys()), [])}

        gr_intra_cluster_distances = [[] for _ in range(self.n_values)]
        fragments_on_each_cluster = {vi : {} for vi in range(self.n_values)}
        preferences_on_each_cluster = {vi : {} for vi in range(self.n_values)}

        gt_is_numpy =  isinstance(preferences_grounding_per_aid[list(preferences_grounding_per_aid.keys())[0]][0], np.ndarray)
        for vi in range(self.n_values):
            for (c1,c2), (assigned_c1, assigned_c2) in agents_assigned_to_each_cluster_pair[vi].items():
                
                for (ci, assigned_ci) in [(c1, assigned_c1), (c2, assigned_c2)]:
                    if ci not in fragments_on_each_cluster[vi].keys():
                        fragments_on_each_cluster[vi][ci] = np.array([pair for aid, fragaids in fragment_pairs_per_aid.items() if (aid in assigned_ci) for pair in fragaids]) 
                        preferences_on_each_cluster[vi][ci] = np.array([pr for aid, praids in preferences_grounding_per_aid.items() if (aid in assigned_ci) for pr in praids[:,vi]]) 
                        # Intra cluster distances
                        for aid in assigned_ci:
                            representativity_error_on_vi = 1.0-calculate_accuracies(probs_vs=probs_gr_per_agent[aid][:,vi].detach().numpy()  if gt_is_numpy else probs_gr_per_agent[aid][:,vi],
                            gt_probs_vs=preferences_grounding_per_aid[aid][:,vi], apply_indifference_in_gt=False, indifference_tolerance=self.loss.model_indifference_tolerance)[0]
                            gr_intra_cluster_distances[vi].append(representativity_error_on_vi)
                            
                            assert len(gr_intra_cluster_distances[vi]) <= len(preferences_grounding_per_aid.keys())
                if len(assigned_c1) > 0 and len(assigned_c2) > 0:
                    fragments_on_v1c1c2 = [*fragments_on_each_cluster[vi][c1] , *fragments_on_each_cluster[vi][c2]]
                    q_div, acc_div = self._preference_model.diversity_grounding(fragments=fragments_on_v1c1c2,
                                                                        grounding1=grounding_per_value_per_cluster[vi][c1],
                                                                        grounding2=grounding_per_value_per_cluster[vi][c2],
                                                                        value_idx=vi,indifference_tolerance=self.loss.model_indifference_tolerance,
                                                                        difference_function = self.loss.loss_func
                                                                        )
                                            
                    gr_inter_cluster_distances[vi].append(acc_div if self.qualitative_cluster_assignment else q_div ) 
                    print(f"VALUE {vi}, Clusters {c1}, {c2}. Inter cluster distances: Quantitative {q_div}, Qualitative {acc_div}" )
                else:
                    if len(assigned_c1) == 0:
                        print(f"VALUE {vi}, Cluster {c1} is of length 0, skipping" )
                    else:
                        print(f"VALUE {vi}, Cluster {c2} is of length 0, skipping" )
                
            if len(gr_inter_cluster_distances[vi]) > 0:
                self.min_inter_dist_gr_so_far[vi] = min(np.min(gr_inter_cluster_distances[vi]), self.min_inter_dist_gr_so_far[vi])
            else:
                gr_inter_cluster_distances[vi] = [float('inf')]
               
        return gr_intra_cluster_distances, gr_inter_cluster_distances
    def vs_discordances(self, reward_model_per_agent_id, value_system_per_cluster, vs_cluster_to_agents_assignment, fragment_pairs_per_aid, preferences_per_aid, probs_per_aid):
        
        fragments_on_each_cluster_vs  = {(c1,c2): (vs_cluster_to_agents_assignment[c1], vs_cluster_to_agents_assignment[c2]) for c1,c2 in itertools.combinations(range(len(vs_cluster_to_agents_assignment)), r=2) }
        vs_inter_cluster_distances = []
        fragments_on_each_cluster_per_aid = {}
        preferences_on_each_cluster_per_aid = {}


        # Intra cluster distances:
        vs_intra_cluster_distances = []

        gt_is_numpy = isinstance(preferences_per_aid[list(preferences_per_aid.keys())[0]], np.ndarray)
        for (c1,c2), (assigned_c1, assigned_c2) in fragments_on_each_cluster_vs.items():
            for (ci, assigned_ci) in [(c1, assigned_c1), (c2, assigned_c2)]:
                #print("CI", ci, fragments_on_each_cluster_per_aid.keys(), assigned_ci, vs_intra_cluster_distances_per_value_per_cluster.keys())
                if ci not in fragments_on_each_cluster_per_aid.keys():
                    fragments_on_each_cluster_per_aid[ci] = {aid: fragment_pairs_per_aid[aid] for aid in assigned_ci }
                    preferences_on_each_cluster_per_aid[ci] = {aid: preferences_per_aid[aid] for aid in assigned_ci }

                    for aid in assigned_ci:
                        representativity_error = 1.0-calculate_accuracies(probs_vs=probs_per_aid[aid].detach().numpy() if gt_is_numpy else probs_per_aid[aid],
                        gt_probs_vs=preferences_per_aid[aid], apply_indifference_in_gt=False, indifference_tolerance=self.loss.model_indifference_tolerance)[0]
                        vs_intra_cluster_distances.append(representativity_error)
                    #assert len(fragments_on_each_cluster_per_aid[ci]) == len(preferences_on_each_cluster_per_aid[ci])
                    #assert len(fragments_on_each_cluster_per_aid[ci]) == len(assigned_ci)
                    #assert all(np.allclose(preferences[fragment_idxs_per_aid[aid]], preferences_per_aid[aid]) for aid in assigned_ci)
                    assert all(np.allclose(reward_model_per_agent_id[aid].get_learned_align_function(), value_system_per_cluster[ci].get_alignment_layer()[0])  for aid in assigned_ci)
                
                    
            if len(assigned_c1) > 0 and len(assigned_c2) > 0:
                fragments_on_each_cluster_c1c2 = fragments_on_each_cluster_per_aid[c1] | fragments_on_each_cluster_per_aid[c2]
                
                q_div_vs, ql_div_vs = self._preference_model.diversity_vs(fragments=fragments_on_each_cluster_c1c2,
                                                                    reward_net_per_aid=reward_model_per_agent_id,
                                                                    vs1=value_system_per_cluster[c1],
                                                                    vs2=value_system_per_cluster[c2],indifference_tolerance=self.loss.model_indifference_tolerance,
                                                                    difference_function = self.loss.loss_func)
                                        
                vs_inter_cluster_distances.append(ql_div_vs if self.qualitative_cluster_assignment else q_div_vs) # TODO.
                print(f"VS Clusters {c1} {c2}. Inter cluster distances: Quantitative {q_div_vs}, Qualitative {ql_div_vs}" )
            else:
                if len(assigned_c1) == 0:
                    print(f"VS Cluster {c1} is of length 0, skipping" )
                if len(assigned_c2) == 0:
                    print(f"VS Cluster {c2} is of length 0, skipping" )
                    
        if len(vs_inter_cluster_distances) == 0:
            # This means K = 1
            vs_inter_cluster_distances = [0.0]  
        assert len(vs_inter_cluster_distances) <= len(probs_per_aid.keys())
        if len(vs_intra_cluster_distances) == 0:
            raise ValueError("Cluster discordances are not well calculated")
        
        return vs_intra_cluster_distances, vs_inter_cluster_distances

    

    def _training_inner_loop(
        self,
        # fragment_pairs: Sequence[TrajectoryPair],
        preferences: np.ndarray,
        preferences_with_grounding: np.ndarray,
        reward_model_per_agent_id: Mapping[str, AbstractVSLRewardFunction],
        grounding_per_value_per_cluster: List[List[th.nn.Module]],
        # agent_ids: np.ndarray[int],
        # iteration: int,
        # max_iterations: int,

        fragment_pairs_per_aid: Dict[str, Sequence[TrajectoryPair]],
        preferences_per_aid: np.ndarray,
        preference_grounding_per_aid: np.ndarray,
        fragment_idxs_per_aid: Dict[str, List[int]],
        agent_to_gr_cluster_assignments: List[Dict[Any, int]],
        agent_to_vs_cluster_assignments: Optional[Dict[Any, int]] = {},
        value_system_network_per_cluster: Optional[List[th.nn.Module]] = [],
        # trajpair_indexes_per_value_per_cluster: np.ndarray
    ) -> th.Tensor:

        # This is the M step. of the EM-like algorithm
        output = self.loss.forward(
            preferences=preferences,
            preferences_with_grounding=preferences_with_grounding,
            preference_model=self._preference_model,
            fragment_pairs_per_agent_id=fragment_pairs_per_aid,
            preferences_per_agent_id_with_grounding=preference_grounding_per_aid,
            preferences_per_agent_id=preferences_per_aid,
            reward_model_per_agent_id=reward_model_per_agent_id,
            fragment_idxs_per_aid=fragment_idxs_per_aid,
            
            value_system_network_per_cluster=value_system_network_per_cluster,
            grounding_per_value_per_cluster=grounding_per_value_per_cluster) # These two provide penalties

        loss_vs, loss_gr = output.loss
        self.logger.record("loss_gr", loss_gr.item())

        self.logger.record("loss_vs", loss_vs.item())

        self.last_metrics = output.metrics

        if self.train_mode == TrainingModes.VALUE_GROUNDING_LEARNING:
            loss = loss_gr

        elif self.train_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
            loss = loss_vs
        else:   
            #raise NotImplementedError("Simultaneous not tested yet")
            loss =  loss_vs  + loss_gr # TODO simultaneous
            # TODO implement bi level optimization
            

        for name, value in output.metrics.items():
            if isinstance(value, dict):
                for v_key, v in value.items():

                    self.logger.record(name + '-' + str(v_key), v)
            else:
                self.logger.record(name, value)

        if __debug__:
            networks_to_train = []
            for aid in reward_model_per_agent_id.keys():
                for vi in range(self.n_values):
                    cs = agent_to_gr_cluster_assignments[aid][vi]
                    networks_to_train.append(
                        grounding_per_value_per_cluster[vi][cs])
                if aid in agent_to_vs_cluster_assignments.keys():
                    cvs_ = agent_to_vs_cluster_assignments[aid]
                    networks_to_train.append(value_system_network_per_cluster[cvs_])
            optim_params = []

            network_params = {
                param for network in networks_to_train for param in network.parameters()}

            # Get the parameters from the optimizer
            optim_params = {
                param for group in self.optim.param_groups for param in group['params']}

            mismatched_params = []
            for i, param in enumerate(network_params):
                if param not in optim_params:
                    mismatched_params.append((i, param))
                    print("MISMATCHED", i, param)

            if not mismatched_params:
                pass
            else:
                print("Mismatch found in the following parameters:")
                for index, param in mismatched_params:
                    print(
                        f"Network parameter at index {index} is not in optimizer parameters.")
                raise RuntimeError(
                    "The optimizer does not optimize the expected model parameters.")

        """TEST the parameter updates work here!"""
        return loss, output.metrics

    def cluster_assignment(self, fragment_pairs, preferences, preferences_with_grounding, 
                           reward_model_per_agent_id: Dict[Tuple, AbstractVSLRewardFunction], 
                           grounding_per_value_per_cluster, agent_ids,
                           running_assignment_vs: Dict, running_assignment_gr: Dict,
                           value_system_network_per_cluster=[], use_assignment: ClusterAssignment=None) -> Tuple[ClusterAssignment, Dict, Dict, Dict, Dict]:
        
        
        using_assignment = isinstance(use_assignment, ClusterAssignment) and use_assignment.agent_to_gr_cluster_assignments != []
        if using_assignment:
            self.update_training_networks_from_assignment(reward_model_per_agent_id, grounding_per_value_per_cluster, value_system_network_per_cluster, use_assignment, prev_agent_to_gr=running_assignment_gr, prev_agent_to_vs=running_assignment_vs)

        else:
            agent_to_gr_cluster_assignment = {}
            agent_to_vs_cluster_assignments = {}

            gr_cluster_to_agents_assignment = [
                [list() for _ in grounding_per_value_per_cluster[vi]] for vi in range(self.n_values)]
            vs_cluster_to_agents_assignment = [list() for _ in value_system_network_per_cluster]
            
        """examples_indexes_per_value_per_cluster = [
            [list() for _ in grounding_per_value_per_cluster[vi]] for vi in range(self.n_values)]"""
        
        
        fragment_pairs_per_aid = {}
        preferences_per_aid = {}
        preference_grounding_per_aid = {}
        fragment_idxs_per_aid = {}


        
            #random_assignment = random.random() <= epsilon

        with th.no_grad():
            for aid, rew_aid in reward_model_per_agent_id.items():
                if not using_assignment:
                    agent_to_gr_cluster_assignment[aid] = dict()
                    agent_to_vs_cluster_assignments[aid] = dict()
                    aid_likelihood_per_value_per_cluster = [
                        [1.0]*len(grounding_per_value_per_cluster[vi]) for vi in range(self.n_values)]
                    aid_likelihood_per_vs_cluster = [1.0]*len(value_system_network_per_cluster)

                """gr_intra_cluster_distances = [
                    [0.0]*len(grounding_per_value_per_cluster[vi]) for vi in range(self.n_values)]"""
                
                """vs_intra_cluster_distances = [0.0]*len(value_system_network_per_cluster)"""

                agent_fragments_idxs = list(np.where(agent_ids == aid)[0])

                agent_fragments = fragment_pairs[agent_fragments_idxs]
                agent_preferences = preferences[agent_fragments_idxs]
                agent_preferences_with_grounding = preferences_with_grounding[
                    agent_fragments_idxs, :]

                fragment_pairs_per_aid[aid] = agent_fragments
                preferences_per_aid[aid] = agent_preferences
                preference_grounding_per_aid[aid] = agent_preferences_with_grounding
                fragment_idxs_per_aid[aid] = agent_fragments_idxs

                if not using_assignment:
                    agent_fragments_idxs_temp_dict = {aid: list(range(len(agent_fragments)))}
                    fragments_aid_temp_dict = {aid: agent_fragments}
                
                for vi, vi_profile in enumerate(self.basic_profiles):
                    # For each agent
                    if not using_assignment:
                        for icluster, network_vi_c in enumerate(grounding_per_value_per_cluster[vi]):

                            if len(grounding_per_value_per_cluster[vi]) > 1:
                                rew_aid.set_network_for_value(
                                    value_id=vi, network=network_vi_c)
                                rew_aid.set_alignment_function(vi_profile)
                                # TODO: probably not needed.
                                rew_aid.set_mode(self.train_mode)
                            probs_vs, _ = self._preference_model.forward(fragments_aid_temp_dict, custom_model_per_agent_id=rew_aid, fragment_pairs_idxs_per_agent_id=agent_fragments_idxs_temp_dict, only_for_alignment_function=vi_profile)
                            gt_probs = agent_preferences_with_grounding[:, vi]
                            # adaptive slope is not necessary if only want the maximum likelihood. In other contexts, though, it could be useful to better discern the probability differences.
                            aid_likelihood_per_value_per_cluster[vi][icluster] = likelihood_x_is_target(probs_vs.detach(), gt_probs.detach(), mode='th', slope=0.3, adaptive_slope=False, qualitative_mode=self.qualitative_cluster_assignment, indifference_tolerance=self.loss.model_indifference_tolerance)
                            #gr_intra_cluster_distances[vi][icluster] = self.loss.loss_func(probs_vs, gt_probs)
                           
                    # MAX ASsIgnMent
                    # assigned_cluster = int(np.argmax(aid_likelihood_per_value_per_cluster[vi]))
                    # PROBABILISTIC ASSIGNMENT # TOO stochastic, but seems to kind of converge to something... Still dependent on initial conditions...
                    # assigned_cluster = int(np.random.choice(a=len(aid_likelihood_per_value_per_cluster[vi]), p=aid_likelihood_per_value_per_cluster[vi]/np.sum(aid_likelihood_per_value_per_cluster[vi])))
                    # TODO: Try epsilon greedy. Should be good enough ?
                        if len(grounding_per_value_per_cluster[vi]) > 1:
                            assigned_cluster = np.argmax(
                                aid_likelihood_per_value_per_cluster[vi])
                                
                            rew_aid.set_network_for_value(
                                value_id=vi, network=grounding_per_value_per_cluster[vi][assigned_cluster])
                        else:
                            assigned_cluster = 0
                    
                        agent_to_gr_cluster_assignment[aid][vi] = assigned_cluster
                        running_assignment_gr[aid][vi] = assigned_cluster
                        gr_cluster_to_agents_assignment[vi][assigned_cluster].append(aid)
                        """examples_indexes_per_value_per_cluster[vi][assigned_cluster].extend(
                        agent_fragments_idxs)"""

                        print(
                            f"ESTIMATED CPrefs FOR {aid} and Value {vi_profile}", end='(')
                        
                        for i, value in enumerate(aid_likelihood_per_value_per_cluster[vi]):
                            color = self.cluster_colors[i]
                            print(f"{color}{value}{Style.RESET_ALL}", end=', ')
                        print(
                            f"), {self.cluster_colors[assigned_cluster]}Cluster {assigned_cluster}: {aid_likelihood_per_value_per_cluster[vi][assigned_cluster]}, {Style.RESET_ALL}")
                        print(Style.RESET_ALL, end='')
                                                 
                if not using_assignment:
                    
                    for ivscluster, network_vs_c in enumerate(value_system_network_per_cluster):

                            rew_aid.set_trained_alignment_function_network(network_vs_c)
                            rew_aid.set_mode(self.train_mode)
                            probs_vs, _ = self._preference_model.forward(fragments_aid_temp_dict, custom_model_per_agent_id=rew_aid, fragment_pairs_idxs_per_agent_id=agent_fragments_idxs_temp_dict, only_for_alignment_function=rew_aid.get_learned_align_function())
                            gt_probs = agent_preferences
                            # adaptive slope is not necessary if only want the maximum likelihood. In other contexts, though, it could be useful to better discern the probability differences.
                            aid_likelihood_per_vs_cluster[ivscluster] = likelihood_x_is_target(probs_vs.detach(), gt_probs.detach(), mode='th', slope=0.3, adaptive_slope=False, qualitative_mode=self.qualitative_cluster_assignment, indifference_tolerance=self.loss.model_indifference_tolerance)
                            
                    if len(value_system_network_per_cluster) > 1:
                        assigned_cluster_vs = np.argmax(
                                aid_likelihood_per_vs_cluster)
                    else:
                        assigned_cluster_vs = 0
                    rew_aid.set_trained_alignment_function_network(value_system_network_per_cluster[assigned_cluster_vs])
                    reward_model_per_agent_id[aid] = rew_aid
                    agent_to_vs_cluster_assignments[aid] = assigned_cluster_vs
                    running_assignment_vs[aid] = assigned_cluster_vs                    
                    vs_cluster_to_agents_assignment[assigned_cluster_vs].append(aid)
                    
                    for i, value in enumerate(aid_likelihood_per_vs_cluster):
                        color = self.cluster_colors_vs[i]
                        print(f"{color}{value}{Style.RESET_ALL}", end=', ')
                    print(f"), {self.cluster_colors_vs[assigned_cluster_vs]} {rew_aid.get_learned_align_function()} - VS Cluster {assigned_cluster_vs}: {aid_likelihood_per_vs_cluster[assigned_cluster_vs]}, {Style.RESET_ALL}")
                check_assignment_consistency(grounding_per_value_per_cluster,value_system_network_per_cluster,running_assignment_gr, running_assignment_vs, reward_model_per_agent_id
                                                 )
        if not using_assignment:
            gr_cluster_to_agents_assignment = convert_nested_list_to_tuple(gr_cluster_to_agents_assignment)
            vs_cluster_to_agents_assignment = convert_nested_list_to_tuple(vs_cluster_to_agents_assignment)
            
            assert agent_to_gr_cluster_assignment == running_assignment_gr
            assert agent_to_vs_cluster_assignments == running_assignment_vs
            
            assignment = ClusterAssignment(reward_model_per_agent_id=reward_model_per_agent_id,
                                    grounding_per_value_per_cluster=grounding_per_value_per_cluster,
                                    value_system_per_cluster=value_system_network_per_cluster,
                                    agent_to_gr_cluster_assignments=agent_to_gr_cluster_assignment,
                                    agent_to_vs_cluster_assignments=agent_to_vs_cluster_assignments,
                                    #TODO; aggregation_on_gr_scores=self.aggregation_on_gr_scores,
                                    assignment_gr=gr_cluster_to_agents_assignment,
                                    assignment_vs=vs_cluster_to_agents_assignment,
                                    )
            
        else:
            
            assignment = use_assignment.copy()
            for aid in reward_model_per_agent_id.keys():
                running_assignment_gr[aid] = assignment.agent_to_gr_cluster_assignments[aid]
                running_assignment_vs[aid] = assignment.agent_to_vs_cluster_assignments[aid]
                assert aid in use_assignment.assignment_vs[assignment.agent_to_vs_cluster_assignments[aid]] 

            assignment.reward_model_per_agent_id = reward_model_per_agent_id
            assignment.value_system_per_cluster = value_system_network_per_cluster
            assignment.grounding_per_value_per_cluster = grounding_per_value_per_cluster
            assignment.agent_to_gr_cluster_assignments = running_assignment_gr
            assignment.agent_to_vs_cluster_assignments = running_assignment_vs


                
                
            check_assignment_consistency(grounding_per_value_per_cluster=assignment.grounding_per_value_per_cluster,
                                          assignment_aid_to_gr_cluster=assignment.agent_to_gr_cluster_assignments,
                                          assignment_aid_to_vs_cluster=assignment.agent_to_vs_cluster_assignments,
                                          reward_models_per_aid=assignment.reward_model_per_agent_id,
                                          value_system_network_per_cluster=assignment.value_system_per_cluster)
            check_assignment_consistency(grounding_per_value_per_cluster,value_system_network_per_cluster,running_assignment_gr, running_assignment_vs, reward_model_per_agent_id
                                                 )
            check_assignment_consistency(assignment.grounding_per_value_per_cluster,value_system_network_per_cluster,running_assignment_gr
                                         , running_assignment_vs, assignment.reward_model_per_agent_id
                                                 )
                                                 

        return assignment, fragment_pairs_per_aid, preferences_per_aid, preference_grounding_per_aid, fragment_idxs_per_aid
    def update_training_networks_from_assignment(self, reward_model_per_agent_id, grounding_per_value_per_cluster, value_system_per_cluster, reference_assignment: ClusterAssignment, prev_agent_to_vs=None, prev_agent_to_gr=None):
        
        check_assignment_consistency(grounding_per_value_per_cluster=reference_assignment.grounding_per_value_per_cluster,value_system_network_per_cluster=reference_assignment.value_system_per_cluster,
                                     assignment_aid_to_gr_cluster=reference_assignment.agent_to_gr_cluster_assignments, assignment_aid_to_vs_cluster=reference_assignment.agent_to_vs_cluster_assignments, reward_models_per_aid=reference_assignment.reward_model_per_agent_id)
        
        check_assignment_consistency(grounding_per_value_per_cluster=grounding_per_value_per_cluster,value_system_network_per_cluster=value_system_per_cluster,
                                     assignment_aid_to_gr_cluster=prev_agent_to_gr, assignment_aid_to_vs_cluster=prev_agent_to_vs, reward_models_per_aid=reward_model_per_agent_id)
        with th.no_grad():
            
                
            for aid, model in reward_model_per_agent_id.items():
                reference_model = reference_assignment.reward_model_per_agent_id[aid]
                if __debug__:
                    th.testing.assert_close(reference_model.get_trained_alignment_function_network().state_dict(), reference_assignment.value_system_per_cluster[reference_assignment.agent_to_vs_cluster_assignments[aid]].state_dict())
               
                model: AbstractVSLRewardFunction
                for vi in range(self.n_values):
                    #th.testing.assert_close(model.get_network_for_value(vi).state_dict(), grounding_per_value_per_cluster[vi][prev_agent_to_gr[aid][vi]].state_dict())
                    
                    grounding_per_value_per_cluster[vi][reference_assignment.agent_to_gr_cluster_assignments[aid][vi]].load_state_dict(deepcopy(reference_model.get_network_for_value(vi).state_dict()))
                    #model.get_network_for_value(vi).load_state_dict(reference_model.get_network_for_value(vi).state_dict())
                    model.set_network_for_value(vi, grounding_per_value_per_cluster[vi][reference_assignment.agent_to_gr_cluster_assignments[aid][vi]])
                    if __debug__:
                        th.testing.assert_close(model.get_network_for_value(vi).state_dict(), reference_model.get_network_for_value(vi).state_dict())
                        th.testing.assert_close(model.get_network_for_value(vi).state_dict(), grounding_per_value_per_cluster[vi][prev_agent_to_gr[aid][vi]].state_dict())
                    prev_agent_to_gr[aid][vi] = deepcopy(reference_assignment.agent_to_gr_cluster_assignments[aid][vi])
                if __debug__:
                    th.testing.assert_close(model.get_trained_alignment_function_network().state_dict(), value_system_per_cluster[prev_agent_to_vs[aid]].state_dict())
                value_system_per_cluster[reference_assignment.agent_to_vs_cluster_assignments[aid]].load_state_dict(deepcopy(reference_model.get_trained_alignment_function_network().state_dict()))
                model.set_trained_alignment_function_network(value_system_per_cluster[reference_assignment.agent_to_vs_cluster_assignments[aid]])
                prev_agent_to_vs[aid] = deepcopy(reference_assignment.agent_to_vs_cluster_assignments[aid])
                if __debug__:
                    th.testing.assert_close(reference_model.get_trained_alignment_function_network().state_dict(), value_system_per_cluster[reference_assignment.agent_to_vs_cluster_assignments[aid]].state_dict())
                    
                    th.testing.assert_close(model.get_trained_alignment_function_network().state_dict(), value_system_per_cluster[reference_assignment.agent_to_vs_cluster_assignments[aid]].state_dict())
                    
                    th.testing.assert_close(model.get_trained_alignment_function_network().state_dict(), reference_model.get_trained_alignment_function_network().state_dict())
                    
                    
                    th.testing.assert_close(model.get_trained_alignment_function_network().state_dict(), reference_model.get_trained_alignment_function_network().state_dict())
                    np.testing.assert_almost_equal(model.get_learned_align_function(), reference_model.get_learned_align_function())
            if __debug__:
                for aid, model in reward_model_per_agent_id.items():
                    reference_model = reference_assignment.reward_model_per_agent_id[aid]
                    vsNetwork: ConvexAlignmentLayer = value_system_per_cluster[reference_assignment.agent_to_vs_cluster_assignments[aid]]
                    th.testing.assert_close(reference_model.get_trained_alignment_function_network().state_dict(), vsNetwork.state_dict())

                    np.testing.assert_allclose(model.get_learned_align_function() , reference_assignment.reward_model_per_agent_id[aid].get_learned_align_function())
                    
                    th.testing.assert_close(model.get_trained_alignment_function_network().state_dict(), vsNetwork.state_dict())
            
        
        if isinstance(self.loss, SobaLoss):
            self.optim.set_state(reference_assignment.optimizer_state)
            # TODO: esto de arriba
            self.loss.set_parameters(wx= self.optim.wx, wy=self.optim.wy, vt=self.optim.vt)
        if __debug__:
            updated_params = {
                param for group in self.optim.param_groups for param in group['params']
            }
            changed_params = {
                param for aid, model in reward_model_per_agent_id.items() for param in model.parameters()
            }
            for vi, cluster in enumerate(grounding_per_value_per_cluster):
                for network in cluster:
                    changed_params.update(network.parameters())
            for network in value_system_per_cluster:
                changed_params.update(network.parameters())

            assert updated_params == changed_params, "Optimizer parameters do not match the changed networks."
            check_assignment_consistency(grounding_per_value_per_cluster=grounding_per_value_per_cluster,value_system_network_per_cluster=value_system_per_cluster,
                                        assignment_aid_to_gr_cluster=prev_agent_to_gr, assignment_aid_to_vs_cluster=prev_agent_to_vs, reward_models_per_aid=reward_model_per_agent_id)
            # Ensure optimizer consistency
            check_grounding_value_system_networks_consistency_with_optim(grounding_per_value_per_cluster, value_system_per_cluster, self.optim)
            check_optimizer_consistency(reward_model_per_agent_id, self.optim)



class PreferenceComparisonVSL(preference_comparisons.PreferenceComparisons):
    def __init__(self, dataset: VSLPreferenceDataset, reward_model: AbstractVSLRewardFunction, num_iterations: int, reward_trainer: ClusteringRewardTrainerVSL, rng,
                 custom_logger=None, allow_variable_horizon=False, query_schedule="constant"):
        self.complete_dataset = dataset
        super().__init__(preference_comparisons.TrajectoryDataset(dataset, rng, custom_logger),
                         reward_model, num_iterations, fragmenter=None, preference_gatherer=None, reward_trainer=reward_trainer,
                         comparison_queue_size=1,
                         fragment_length=1, transition_oversampling=1, initial_comparison_frac=0.0, initial_epoch_multiplier=1,
                         custom_logger=custom_logger, allow_variable_horizon=allow_variable_horizon, rng=rng, query_schedule=query_schedule)
        assert isinstance(self.reward_trainer, ClusteringRewardTrainerVSL)

    
    def train(
        self,
        num_iterations: int,
        reward_model_per_agent_id: Mapping[str, AbstractVSLRewardFunction],
        grounding_per_value_per_cluster: List[List[th.nn.Module]],
        value_system_per_cluster: List[Any],
        original_agent_to_gr_cluster_assignments: Dict, 
        original_agent_to_vs_cluster_assignments: Dict,
        comparisons_per_agent_per_step: int = None,
        callback: Optional[Callable[[int], None]] = None,
        historical_assignments_save_folder = 'training_assignments',
        try_without_replacement=True,
        max_assignment_memory = 10
    ) -> ClusterAssignmentMemory:
        """Train the reward model and the policy if applicable.

        Args:
            total_timesteps: number of environment interaction steps
            total_comparisons: number of preferences to gather in total
            callback: callback functions called at the end of each iteration

        Returns:
            Historical of the scores of the best assignments at every iteration (saved to a folder)
            The assignment memory resulting from the training is also returned.
        """
        
        all_ids = set(self.complete_dataset.agent_data.keys())
        min_number_of_trajectories = float('inf')
        for aid in all_ids:
            min_number_of_trajectories = min(
                len(self.complete_dataset.data_per_agent[aid]), min_number_of_trajectories)
        print("Min number of comparisons perceived by any agent: ",
              min_number_of_trajectories)
        if comparisons_per_agent_per_step is None:
            comparisons_per_agent_per_step = min_number_of_trajectories

        # Compute the number of comparisons to request at each iteration in advance.
        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized_probs = vec_schedule(
            np.linspace(0, 1, self.num_iterations))
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        shares = util.oric(
            probs * comparisons_per_agent_per_step * num_iterations)
        schedule = shares.tolist()

        timesteps_per_iteration, extra_timesteps = divmod(
            num_iterations,
            self.num_iterations,
        )
        reward_loss_vs = None
        reward_accuracy_vs = None
        reward_loss_groundings = None
        reward_accuracy_groundings = None

        best_assignment = None

        best_assignments_list = ClusterAssignmentMemory(max_size=max_assignment_memory)

        self.training_networks = {'rw': reward_model_per_agent_id, "gr": grounding_per_value_per_cluster, "vc": value_system_per_cluster, 'version': 0}

        exploration = 1.0 # initially, random assignment.
        for global_iter, num_pairs in enumerate(schedule):
            ##########################
            # Gather new preferences #
            ##########################

            self.logger.log(
                f"Collecting {num_pairs} pairs of fragments per agent",
            )
            # trajectories = self.trajectory_generator.sample(num_pairs)

            # This assumes there are no fragments missing initial timesteps
            # (but allows for fragments missing terminal timesteps).
            # horizons = (len(traj) for traj in trajectories if traj.terminal)
            
            dataset_batch = VSLPreferenceDataset(
                n_values=self.complete_dataset.n_values, single_agent=False)
            for aid, adata in self.complete_dataset.data_per_agent.items():
                selection = np.random.choice(len(
                    adata), size=num_pairs, replace=True if try_without_replacement and len(adata) < num_pairs else False)
                agent_dataset_batch = adata[selection]
                dataset_batch.push(fragments=agent_dataset_batch[0], preferences=agent_dataset_batch[1], preferences_with_grounding=agent_dataset_batch[2], agent_ids=[
                                   aid]*len(selection), agent_data={aid: self.complete_dataset.agent_data[aid]})

            self.logger.log("Creating fragment pairs")
            # fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)
            with self.logger.accumulate_means("preferences"):
                self.logger.log("Gathering preferences")

                # preferences = self.preference_gatherer(fragment_pairs=dataset_batch[aid], traj_ids=None, value_or_value_system='vs')
            """TODO: online adding trajectories from learning policies...
            self.dataset.push(fragments, preferences)
            self.logger.log(f"Dataset now contains {len(self.dataset)} comparisons")"""

            ##########################
            # Train the reward model #
            ##########################
            
            self.reward_trainer: ClusteringRewardTrainerVSL
            
            if random.random() < exploration:
                starting_assignment = 'start_random'
                if global_iter <= int(num_iterations*0.3):
                    reference_assignment = self.generate_random_assignment(reward_model_per_agent_id, grounding_per_value_per_cluster, value_system_per_cluster, original_agent_to_gr_cluster_assignments, original_agent_to_vs_cluster_assignments)
                else:
                    reference_assignment = self.generate_mutated_assignment(reward_model_per_agent_id, grounding_per_value_per_cluster, value_system_per_cluster, original_agent_to_gr_cluster_assignments, original_agent_to_vs_cluster_assignments)
                self.reward_trainer.update_training_networks_from_assignment(reward_model_per_agent_id, grounding_per_value_per_cluster, value_system_per_cluster, reference_assignment, prev_agent_to_gr=original_agent_to_gr_cluster_assignments, prev_agent_to_vs=original_agent_to_vs_cluster_assignments)
            else:
                starting_assignment = best_assignments_list.get_random_weighted_assignment()
                check_assignment_consistency(grounding_per_value_per_cluster,value_system_per_cluster,
                                             assignment_aid_to_gr_cluster=original_agent_to_gr_cluster_assignments,
                                             assignment_aid_to_vs_cluster=original_agent_to_vs_cluster_assignments, 
                                             reward_models_per_aid=reward_model_per_agent_id)
                check_assignment_consistency(starting_assignment.grounding_per_value_per_cluster,starting_assignment.value_system_per_cluster,
                                             starting_assignment.agent_to_gr_cluster_assignments,
                                             assignment_aid_to_vs_cluster=starting_assignment.agent_to_vs_cluster_assignments, 
                                             reward_models_per_aid=starting_assignment.reward_model_per_agent_id)

                self.reward_trainer.update_training_networks_from_assignment(reward_model_per_agent_id, grounding_per_value_per_cluster, value_system_per_cluster, starting_assignment, prev_agent_to_gr=original_agent_to_gr_cluster_assignments, prev_agent_to_vs=original_agent_to_vs_cluster_assignments)


                # Assert that the optimizer parameters are still those of the changed networks
            exploration = self.reward_trainer.initial_exploration_rate*(1-1/num_iterations)
            print("EXPLORATION RATE", exploration)
            
            best_assignments_list = self.reward_trainer.train(
                dataset_batch, reward_model_per_agent_id=reward_model_per_agent_id, grounding_per_value_per_cluster=grounding_per_value_per_cluster, 
                value_system_per_cluster=value_system_per_cluster, 
                running_assignment_vs=original_agent_to_vs_cluster_assignments, running_assignment_gr=original_agent_to_gr_cluster_assignments,
                epoch_multiplier=1, 
                global_iteration=global_iter, global_iterations=len(schedule),
                starting_assignment = starting_assignment,
                assignment_ranking=best_assignments_list)
            
            print("BEST ASSIGNMENTS so far:", best_assignments_list)
            
            best_assignment: ClusterAssignment = best_assignments_list.get_best_assignment()
            # Save the best assignment of each iteration
            best_assignment.save(historical_assignments_save_folder, f"best_assignment_iter_{global_iter}.pkl")

            
            # TODO: save assign
            # TODO: Overhaul debugging metrics...
            self.logger.name_to_value['cluster_score'] = best_assignment.vs_score, best_assignment.gr_score
            base_key = self.logger.get_accumulate_prefixes() + "reward/final/train"
            assert f"{base_key}/loss_vs" in self.logger.name_to_value
            assert f"{base_key}/global_accvs" in self.logger.name_to_value
            assert f"{base_key}/loss_gr" in self.logger.name_to_value
            assert f"{base_key}/global_accgr" in self.logger.name_to_value
            """
            reward_loss_vs = self.logger.name_to_value[f"{base_key}/loss_vs"]
            reward_accuracy_vs = self.logger.name_to_value[f"{base_key}/global_accvs"]

            reward_loss_groundings = self.logger.name_to_value[f"{base_key}/loss_gr"]
            reward_accuracy_groundings = self.logger.name_to_value[f"{base_key}/global_accgr"]"""
            ###################
            # Train the agent #
            ###################
            num_steps = timesteps_per_iteration
            # if the number of timesteps per iterations doesn't exactly divide
            # the desired total number of timesteps, we train the agent a bit longer
            # at the end of training (where the reward model is presumably best)
            if global_iter == self.num_iterations - 1:
                num_steps += extra_timesteps
            with self.logger.accumulate_means("agent"):
                self.logger.log(f"Training agent for {num_steps} timesteps")
                self.trajectory_generator.train(steps=num_steps)

            self.logger.dump(self._iteration)

            ########################
            # Additional Callbacks #
            ########################
            if callback:
                callback(self._iteration)
            self._iteration += 1

        self.reward_trainer.update_training_networks_from_assignment(reward_model_per_agent_id, grounding_per_value_per_cluster, value_system_per_cluster, prev_agent_to_gr=original_agent_to_gr_cluster_assignments, prev_agent_to_vs=original_agent_to_vs_cluster_assignments, reference_assignment=best_assignment)
            
        return best_assignments_list

    def generate_mutated_assignment(self, reward_model_per_agent_id, grounding_per_value_per_cluster, value_system_per_cluster, assignment_gr, assignment_vs):
        example_model = list(reward_model_per_agent_id.values())[0]
        with th.no_grad():
            grounding_per_value_per_cluster_c = [[None for c in range(len(cs_per_vi))] for cs_per_vi in grounding_per_value_per_cluster]
            value_system_per_cluster_c = None
            if value_system_per_cluster is not None:
                    value_system_per_cluster_c = [None for _ in range(len(value_system_per_cluster))]
            random_reward_model: Dict[str,AbstractVSLRewardFunction] = {}

            agent_to_gr_cluster_assignments = {}
            agent_to_vs_cluster_assignments = {}

            assignment_gr_new = [[list()  for _ in range(len(grounding_per_value_per_cluster_c[vi_]))] for vi_ in range(len(grounding_per_value_per_cluster_c))]
            assignment_vs_new = [list() for _ in range(len(value_system_per_cluster_c))]
            
            grounding_per_value_per_cluster_c = deepcopy(grounding_per_value_per_cluster)
            value_system_per_cluster_c = deepcopy(value_system_per_cluster)
            
            for aid, cluster_vs_aid in assignment_vs.items():
                agent_to_gr_cluster_assignments[aid] = {}
                grclusterby_value_aid = assignment_gr[aid]

                rc: AbstractVSLRewardFunction = reward_model_per_agent_id[aid].copy()
                rc.set_mode(example_model.mode)
                rc.reset_learned_grounding_function()

                assignment_vs_new[cluster_vs_aid].append(aid)
                agent_to_vs_cluster_assignments[aid] = cluster_vs_aid
                for vi, cluster_vi_aid in grclusterby_value_aid.items():
                    assignment_gr_new[vi][cluster_vi_aid].append(aid)
                    agent_to_gr_cluster_assignments[aid][vi] = cluster_vi_aid
                    if len(assignment_gr_new[vi][cluster_vi_aid]) == 1:
                        # TODO: mutate cluster corresponding...
                        
                        for param in grounding_per_value_per_cluster_c[vi][cluster_vi_aid].parameters():
                            if param.requires_grad:
                                mask = th.rand_like(param) < 0.05  # percentage of parameters changed.
                                unifr= th.empty_like(param).uniform_(0, 1.0)
                                param.mul_(th.where(mask, unifr, th.ones_like(param)))
                        # Assert that the original is different from the updated network
                        #grounding_per_value_per_cluster_c[vi][cluster_vi_aid] = rc.get_network_for_value(vi)
                                    
                    rc.set_network_for_value(vi, grounding_per_value_per_cluster_c[vi][cluster_vi_aid])

                if value_system_per_cluster_c is not None and len(assignment_vs_new[cluster_vs_aid]) == 1:
                    rc.reset_learned_alignment_function()

                    value_system_per_cluster_c[cluster_vs_aid] = rc.get_trained_alignment_function_network()
                rc.set_trained_alignment_function_network(value_system_per_cluster_c[cluster_vs_aid])
                random_reward_model[aid]  =rc
                
                    
            
                    
            assert set(random_reward_model.keys()) == set(reward_model_per_agent_id.keys())

            # Check that the assignment is consistent 
            check_assignment_consistency(grounding_per_value_per_cluster,value_system_per_cluster,
                                                assignment_aid_to_gr_cluster=agent_to_gr_cluster_assignments,
                                                assignment_aid_to_vs_cluster=agent_to_vs_cluster_assignments, reward_models_per_aid=reward_model_per_agent_id)
            check_assignment_consistency(grounding_per_value_per_cluster_c,value_system_per_cluster_c,
                                                assignment_aid_to_gr_cluster=agent_to_gr_cluster_assignments,
                                                assignment_aid_to_vs_cluster=agent_to_vs_cluster_assignments, reward_models_per_aid=random_reward_model)
            reference_assignment = ClusterAssignment(reward_model_per_agent_id=random_reward_model, grounding_per_value_per_cluster=grounding_per_value_per_cluster_c,value_system_per_cluster=value_system_per_cluster_c,
                                                            agent_to_gr_cluster_assignments=agent_to_gr_cluster_assignments, agent_to_vs_cluster_assignments=agent_to_vs_cluster_assignments, 
                                                            assignment_gr=assignment_gr_new, assignment_vs=assignment_vs_new)
                                                    
        return reference_assignment
    def generate_random_assignment(self, reward_model_per_agent_id, grounding_per_value_per_cluster, value_system_per_cluster, assignment_gr, assignment_vs):
        example_model = list(reward_model_per_agent_id.values())[0]

        grounding_per_value_per_cluster_c = [[None for c in range(len(cs_per_vi))] for cs_per_vi in grounding_per_value_per_cluster]
        for vi, cs_per_vi in enumerate(grounding_per_value_per_cluster):
            for cs in range(len(cs_per_vi)):
                rc: AbstractVSLRewardFunction = example_model.copy()
                rc.set_mode(example_model.mode)
                        # This creates a new random grounding function.
                rc.reset_learned_grounding_function()
                grounding_per_value_per_cluster_c[vi][cs] = rc.get_network_for_value(vi)
        value_system_per_cluster_c = None
        if value_system_per_cluster is not None:
            value_system_per_cluster_c = []
            for cs in range(len(value_system_per_cluster)):
                rc: AbstractVSLRewardFunction =  example_model.copy()
                        # This creates a new random alignment function.
                rc.set_mode(example_model.mode)
                rc.reset_learned_alignment_function()
                value_system_per_cluster_c.append(
                            rc.get_trained_alignment_function_network())


        random_reward_model: Dict[str,AbstractVSLRewardFunction] = {}
                
        agent_to_gr_cluster_assignments = {}
        agent_to_vs_cluster_assignments = {}
        assignment_gr = [[list()  for _ in range(len(grounding_per_value_per_cluster_c[vi]))] for vi_ in range(len(grounding_per_value_per_cluster_c))]
        assignment_vs = [list() for _ in range(len(value_system_per_cluster_c))]
        for aid in self.complete_dataset.agent_data.keys():
            agent_to_gr_cluster_assignments[aid] = {}
            random_reward_model[aid] = reward_model_per_agent_id[aid].copy()
            random_reward_model[aid].set_mode(example_model.mode)
            random_reward_model[aid].reset_learned_alignment_function()
            random_reward_model[aid].reset_learned_grounding_function()
            for vi in range(example_model.values_net.num_outputs):
                cs = np.random.choice(len(grounding_per_value_per_cluster_c[vi]))
                assignment_gr[vi][cs].append(aid)
                agent_to_gr_cluster_assignments[aid][vi] = cs
                random_reward_model[aid].set_network_for_value(
                            value_id=vi, network=grounding_per_value_per_cluster_c[vi][cs])
                        
            if value_system_per_cluster_c is not None:
                cs = np.random.choice(len(value_system_per_cluster_c))
                assignment_vs[cs].append(aid)
                agent_to_vs_cluster_assignments[aid] = cs
                random_reward_model[aid].set_trained_alignment_function_network(
                            value_system_per_cluster_c[cs])
                
        assert set(random_reward_model.keys()) == set(reward_model_per_agent_id.keys())
                
        check_assignment_consistency(grounding_per_value_per_cluster_c,value_system_per_cluster_c,
                                             assignment_aid_to_gr_cluster=agent_to_gr_cluster_assignments,
                                             assignment_aid_to_vs_cluster=agent_to_vs_cluster_assignments, reward_models_per_aid=random_reward_model)
        reference_assignment = ClusterAssignment(reward_model_per_agent_id=random_reward_model, grounding_per_value_per_cluster=grounding_per_value_per_cluster_c,value_system_per_cluster=value_system_per_cluster_c,
                                                        agent_to_gr_cluster_assignments=agent_to_gr_cluster_assignments, agent_to_vs_cluster_assignments=agent_to_vs_cluster_assignments, 
                                                        assignment_gr=assignment_gr, assignment_vs=assignment_vs)
                                                
        return reference_assignment

def load_historic_assignments(experiment_name):
    save_folder = os.path.join(ASSIGNMENT_CHECKPOINTS,experiment_name)
    if not os.path.exists(save_folder):
        raise FileNotFoundError(f"Historic assignments directory not found: {save_folder}")
    
    historic_assignments = []

    file_list = os.listdir(save_folder)
    env_state = None
    if 'env_state.pkl' in file_list:

        with open(os.path.join(save_folder, 'env_state.pkl'), 'rb') as f:
            env_state = dill.load(f)
            print("Loaded environment state:", env_state)
        file_list.remove('env_state.pkl')

    for file_name in sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0])):
        file_path = os.path.join(save_folder, file_name)
        if os.path.isfile(file_path) and file_name.startswith('best_assignment_iter_') and file_name.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                historic_assignments.append(dill.load(f))
    if env_state is not None:
        for a in historic_assignments:
            for aid, r in a.reward_model_per_agent_id.items():
                r.set_env(env_state)
                # TODO : check this works! GENERALIZE!
    print(f"Historic assignments loaded from {save_folder}")
    return historic_assignments

class PreferenceBasedClusteringTabularMDPVSL(BaseVSLAlgorithm):
    
    def __init__(self, env,
                 reward_net,
                 optimizer_cls=th.optim.Adam,
                 optimizer_kwargs=None,
                 discount=1,
                 log_interval=100,
                 dataset: VSLPreferenceDataset = None,
                 training_mode=TrainingModes.VALUE_GROUNDING_LEARNING,
                 rng=np.random.default_rng(0),
                 discount_factor_preferences=None,
                 use_quantified_preference=False,
                 query_schedule="hyperbolic",
                 learn_stochastic_policy=True,
                 expert_is_stochastic=True,
                 cluster_sizes=None,
                 vs_cluster_sizes=None,
                 vgl_target_align_funcs=[],
                 approximator_kwargs={},
                 policy_approximator=PolicyApproximators.MCE_ORIGINAL,
                 # 0 for deterministic preference sampling, 1 for totally random according to softmax probabilities
                 preference_sampling_temperature=0,
                 assume_variable_horizon=False,
                 reward_trainer_kwargs={
                     'epochs': 5, 'lr': 0.05, 'regularizer_factory': None, 'batch_size': 32, 'minibatch_size': None, },
                 loss_class=preference_comparisons.CrossEntropyRewardLoss, loss_kwargs={},

                 *, custom_logger=None):

        self.cluster_sizes = cluster_sizes
        
        self.vs_L_clusters = vs_cluster_sizes if vs_cluster_sizes is not None else []
        self.allow_variable_horizon = not assume_variable_horizon

        vsi_target_align_funcs_per_agent = set()
        vgl_target_align_funcs_per_agent = set()
        for aid, adata in dataset.agent_data.items():
            vsi_target_align_funcs_per_agent.add(
                tuple([aid, tuple(adata['value_system'])]))
            for profile in vgl_target_align_funcs:
                vgl_target_align_funcs_per_agent.add(
                    tuple([aid, tuple(profile)]))
        vgl_target_align_funcs_per_agent = list(
            vgl_target_align_funcs_per_agent)
        vsi_target_align_funcs_per_agent = list(
            vsi_target_align_funcs_per_agent)
        super().__init__(env=env, reward_net=reward_net, vgl_optimizer_cls=optimizer_cls,
                         vsi_optimizer_cls=optimizer_cls,
                         vgl_optimizer_kwargs=optimizer_kwargs, vsi_optimizer_kwargs=optimizer_kwargs, discount=discount,
                         log_interval=log_interval, vgl_expert_policy=None, vsi_expert_policy=None, vsi_target_align_funcs=vsi_target_align_funcs_per_agent,
                         vgl_target_align_funcs=vgl_target_align_funcs_per_agent, training_mode=training_mode, custom_logger=custom_logger, learn_stochastic_policy=learn_stochastic_policy, stochastic_expert=expert_is_stochastic)
        self.policy_approximator = policy_approximator
        self.approximator_kwargs = approximator_kwargs
        self.rng = rng
        if discount_factor_preferences is None:
            self.discount_factor_preferences = discount
        else:
            self.discount_factor_preferences = discount_factor_preferences

        self.sample = not use_quantified_preference

        self.learned_policy_per_va = VAlignedDictSpaceActionPolicy(env=self.env, policy_per_va_dict={pr: np.ones(
            (self.env.state_dim, self.env.action_dim))/self.env.action_dim for pr in self.vgl_target_align_funcs}, expose_state=True)

        self.dataset = dataset

        self.reward_trainer_kwargs = reward_trainer_kwargs
        self.temperature = preference_sampling_temperature
        self.query_schedule = query_schedule

        self.loss_class = loss_class
        self.loss_kwargs = loss_kwargs

        if self.loss_class == PrefLossClasses.CROSS_ENTROPY_CLUSTER.value:
            self.loss_class = CrossEntropyRewardLossCluster
        elif self.loss_class == PrefLossClasses.SOBA.value:
            # TODO: modified versions here...
            self.loss_class = SobaLoss
        else:
            raise ValueError(
                "Unsupported for clustering VSL or unrecognized loss_class: ", self.loss_class)

        """self.active_fragmenter_on = active_fragmenter_on
        for k in SupportedFragmenters:
            if active_fragmenter_on == k or active_fragmenter_on == k.value:
                self.active_fragmenter_on = k
        assert self.active_fragmenter_on in [f for f in SupportedFragmenters]"""

    @property
    def logger(self):
        return self.pref_comparisons.logger
    

    def train(self, max_iter=5000, mode=TrainingModes.VALUE_GROUNDING_LEARNING, assumed_grounding=None,
              trajectory_batch_size=100, use_probabilistic_reward=False, n_reward_reps_if_probabilistic_reward=10,
              resample_trajectories_if_not_already_sampled=True,max_assignment_memory=10,
              initial_epoch_multiplier=10, experiment_name='test', **kwargs):

        self.resample_trajectories_if_not_already_sampled = resample_trajectories_if_not_already_sampled
        self.fragment_length = self.env.horizon
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.max_assignment_memory = max_assignment_memory
        self.historical_assignments_save_folder  = os.path.join(ASSIGNMENT_CHECKPOINTS,experiment_name)

        """self.gatherer = ClusteredGatherer(preferece_vs=self.dataset.preferences, preference_per_value=self.dataset.preferences_with_grounding, rng=self.rng,
                                                                 discount_factor=self.discount_factor_preferences,
                                                                 sample=self.sample, temperature=self.temperature)"""

        """if self.active_fragmenter_on == SupportedFragmenters.RANDOM_FRAGMENTER:
            self.fragmenter = RandomFragmenterVariableHorizon(
                warning_threshold=1,
                rng=self.rng
            )
            pass # It will be initialized in _train_global."""
        self.last_vs_accuracies_per_agent = {aid: []
                                             for aid in self.dataset.agent_data.keys()}
        self.last_gr_accuracies_per_agent = {aid: []
                                             for aid in self.dataset.agent_data.keys()}

        self.last_accuracies_vs_global = []
        self.last_accuracies_gr_global = []
        # {al: [] for al in (self.vsi_target_align_funcs if mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION else self.vgl_target_align_funcs if mode == TrainingModes.VALUE_GROUNDING_LEARNING else self.all_targets)}

        ret = super().train(max_iter=max_iter,
                             mode=mode, assumed_grounding=assumed_grounding,
                             n_seeds_for_sampled_trajectories=trajectory_batch_size,
                             n_sampled_trajs_per_seed=1,
                             use_probabilistic_reward=use_probabilistic_reward,
                             n_reward_reps_if_probabilistic_reward=n_reward_reps_if_probabilistic_reward,
                             **kwargs)
        historic_assignments = load_historic_assignments(experiment_name)
        return *ret, historic_assignments

    def train_callback(self, t):
        for aid in self.dataset.agent_data.keys():
            self.last_vs_accuracies_per_agent[aid].append(
                self.pref_comparisons.reward_trainer.last_metrics['accvs'][aid])
            self.last_gr_accuracies_per_agent[aid].append(
                self.pref_comparisons.reward_trainer.last_metrics['accgr'][aid])
        self.last_accuracies_vs_global.append(
            self.pref_comparisons.reward_trainer.last_metrics['global_accvs'])
        self.last_accuracies_gr_global.append(
            self.pref_comparisons.reward_trainer.last_metrics['global_accgr'])

        if t % self.log_interval == 0:
            self.logger.record("iteration", t)
            for aid, adata in self.dataset.agent_data.items():

                if self.training_mode == TrainingModes.VALUE_SYSTEM_IDENTIFICATION:
                    self.logger.record(
                        f"Agent {aid} VS target", adata['value_system'])
                    self.logger.record(f"Agent {aid} VS learned: ", tuple(
                        [float("{0:.3f}".format(v)) for v in self.training_reward_nets_per_agent[aid].get_learned_align_function()]))
                else:
                    self.logger.record(
                        f"Agent {aid} grounding", adata['grounding'])
                    self.logger.record(f"Agent {aid} learned: ",
                                       self.training_reward_nets_per_agent[aid].get_learned_grounding())

    def train_simultaneous_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed):
        # TODO: K FOLD CROSS VALIDATION?

        self.training_mode = TrainingModes.SIMULTANEOUS
        assert self.reward_net.mode == TrainingModes.SIMULTANEOUS
        assert len(self.cluster_sizes) == self.current_net.values_net.num_outputs
        grounding_per_value_per_cluster = [[]]*len(self.cluster_sizes)
        for vi, cs_per_vi in enumerate(self.cluster_sizes):
            grounding_per_value_per_cluster[vi] = []
            for cs in range(cs_per_vi):
                rc: AbstractVSLRewardFunction = self.current_net.copy()
                rc.set_mode(self.training_mode)
                # This creates a new random grounding function.
                rc.reset_learned_grounding_function()
                grounding_per_value_per_cluster[vi].append(
                    rc.get_learned_grounding().networks[vi])
        value_system_network_per_cluster = None
        
        if self.vs_L_clusters is not None:
            value_system_network_per_cluster = []
            for cs in range(self.vs_L_clusters):
                rc: AbstractVSLRewardFunction = self.current_net.copy()
                # This creates a new random alignment function.
                rc.set_mode(self.training_mode)
                rc.reset_learned_alignment_function()
                value_system_network_per_cluster.append(
                    rc.get_trained_alignment_function_network())


        self.training_reward_nets_per_agent = {}
        
        for aid in self.dataset.agent_data.keys():
            self.training_reward_nets_per_agent[aid] = self.current_net.copy()
            r: AbstractVSLRewardFunction = self.training_reward_nets_per_agent[aid]
            r.set_mode(self.training_mode)
            r.reset_learned_alignment_function()
            r.reset_learned_grounding_function()

        # Initial Random Assignment of agents to clusters.
        networks_to_train = []
        networks_gr = []
        networks_vs = []
        assignment_aid_to_gr_cluster = {}
        assignment_aid_to_vs_cluster = {}

        for aid in self.dataset.agent_data.keys():
            assignment_aid_to_gr_cluster[aid] = {}
            for vi in range(self.current_net.values_net.num_outputs):
                cs = np.random.choice(self.cluster_sizes[vi])
                assignment_aid_to_gr_cluster[aid][vi] = cs
                self.training_reward_nets_per_agent[aid].set_network_for_value(
                    value_id=vi, network=grounding_per_value_per_cluster[vi][cs])
                if self.training_mode in [TrainingModes.VALUE_GROUNDING_LEARNING, TrainingModes.SIMULTANEOUS]:
                    for cs_ in range(self.cluster_sizes[vi]):
                        networks_to_train.append(
                            grounding_per_value_per_cluster[vi][cs_])
                        networks_gr.append(
                            grounding_per_value_per_cluster[vi][cs_])
            if value_system_network_per_cluster is not None:
                cs = np.random.choice(self.vs_L_clusters)
                assignment_aid_to_vs_cluster[aid] = cs
                self.training_reward_nets_per_agent[aid].set_trained_alignment_function_network(
                    value_system_network_per_cluster[cs])
                if self.training_mode in [TrainingModes.VALUE_SYSTEM_IDENTIFICATION, TrainingModes.SIMULTANEOUS]:
                    for cs_ in range(self.vs_L_clusters):
                        networks_to_train.append(
                            value_system_network_per_cluster[cs_])
                        networks_vs.append(
                            value_system_network_per_cluster[cs_])
        all_params = [
            param for network in networks_to_train for param in network.parameters()]
        params_gr = [
            param for network in networks_gr for param in network.parameters()]
        params_vs = [
            param for network in networks_vs for param in network.parameters() if network is not None]
        
        optim_cls=self.vgl_optimizer_cls if self.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING else self.vsi_optimizer_cls
        optim_kwargs = self.vgl_optimizer_kwargs if self.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING else self.vsi_optimizer_kwargs

        # TODO: Divide parameters into VGL and VSL.

        reward_models_per_aid = self.training_reward_nets_per_agent
        check_assignment_consistency(grounding_per_value_per_cluster, value_system_network_per_cluster, assignment_aid_to_gr_cluster, assignment_aid_to_vs_cluster, reward_models_per_aid)

        best_assignment = self._train_global(max_iter, parameters={'params_gr': params_gr, 'params_vs': params_vs} if optim_cls==SobaOptimizer else all_params,
                           assignment_aid_to_vs_cluster=assignment_aid_to_vs_cluster,
                           assignment_aid_to_gr_cluster=assignment_aid_to_gr_cluster,
                           grounding_per_value_per_cluster=grounding_per_value_per_cluster, 
                           value_system_per_cluster=value_system_network_per_cluster,
                           reward_nets_per_agent=self.training_reward_nets_per_agent,
                           historical_assignments_save_folder=self.historical_assignments_save_folder,
                           optim_cls=optim_cls, optim_kwargs=optim_kwargs,max_assignment_memory=self.max_assignment_memory)
        assert self.training_reward_nets_per_agent.keys() == self.training_reward_nets_per_agent.keys()
        for aid in self.training_reward_nets_per_agent.keys():
            th.testing.assert_close(self.training_reward_nets_per_agent[aid].state_dict() , best_assignment.reward_model_per_agent_id[aid].state_dict())
        self.best_assignment = best_assignment

        return self.training_reward_nets_per_agent 

    def train_vgl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed):
        self.training_mode = TrainingModes.VALUE_GROUNDING_LEARNING
        return self.train_simultaneous_vsl(max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed)

    def _train_global(self, max_iter, optim_cls, optim_kwargs, parameters: Dict[str, List[th.nn.Parameter]],
                      assignment_aid_to_vs_cluster,
                      assignment_aid_to_gr_cluster,
                      reward_nets_per_agent, grounding_per_value_per_cluster, value_system_per_cluster=[], starting_t=0, max_assignment_memory=10,historical_assignments_save_folder='train_assignments'):
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
        
        
        self.preference_model = PreferenceModelClusteredVSL(
            model=self.current_net,
            algorithm=self,
            noise_prob=0,
            discount_factor=self.discount_factor_preferences,
            threshold=50,
        )
        reward_trainer = ClusteringRewardTrainerVSL(
            preference_model=self.preference_model,
            loss=self.loss_class(**self.loss_kwargs),
            rng=self.rng,
            optim_cls=optim_cls,
            optim_kwargs=optim_kwargs,
            **self.reward_trainer_kwargs
        )

        self.pref_comparisons = PreferenceComparisonVSL(
            dataset=self.dataset,
            reward_model=self.current_net,
            num_iterations=max_iter,
            reward_trainer=reward_trainer,
            allow_variable_horizon=self.allow_variable_horizon,
            query_schedule=self.query_schedule,
            rng=self.rng,
            custom_logger=None
        )
        self.pref_comparisons.reward_trainer.reset_optimizer_with_params(
            parameters=parameters)
        
        check_assignment_consistency(grounding_per_value_per_cluster, value_system_per_cluster, assignment_aid_to_gr_cluster, assignment_aid_to_vs_cluster, reward_nets_per_agent)


        assignment_memory = self.pref_comparisons.train(num_iterations=max_iter, reward_model_per_agent_id=reward_nets_per_agent,
                                              grounding_per_value_per_cluster=grounding_per_value_per_cluster, 
                                              value_system_per_cluster=value_system_per_cluster,
                                              original_agent_to_gr_cluster_assignments=assignment_aid_to_gr_cluster,
                                              original_agent_to_vs_cluster_assignments=assignment_aid_to_vs_cluster,
                                              try_without_replacement=True,
                                              historical_assignments_save_folder=historical_assignments_save_folder,
                                                callback=lambda t: self.train_callback(t+starting_t),max_assignment_memory=max_assignment_memory)
        return assignment_memory.get_best_assignment()

    def train_vsl(self, max_iter, n_seeds_for_sampled_trajectories, n_sampled_trajs_per_seed, target_align_func):
        raise NotImplementedError("train vsl need args in train global...")
        self._train_global(max_iter, target_align_func,
                           batch_size=n_seeds_for_sampled_trajectories*n_sampled_trajs_per_seed,)

        return self.current_net.get_learned_align_function()

    def calculate_learned_policies(self, target_align_funcs) -> ValueSystemLearningPolicy:
        self.learned_policy_per_va = VAlignedDictSpaceActionPolicy(
            policy_per_va_dict={}, env=self.env, state_encoder=None, expose_state=True)
        rewards = self.state_action_callable_reward_from_reward_net_per_target_align_func(
            targets=target_align_funcs)

        for target_align_func in target_align_funcs:
            """aid, target_profile = target_align_func
            learned_al_function = target_profile if self.training_mode == TrainingModes.VALUE_GROUNDING_LEARNING else self.target_agent_and_align_func_to_learned_ones[
                aid]
            reward_net: AbstractVSLRewardFunction = self.training_reward_nets_per_agent[aid]
            reward_net.set_alignment_function(learned_al_function)

            rewards = np.zeros_like(self.env.reward_matrix)
            next_states = th.tensor(self._resample_next_states(), dtype=th.int64)
            for a in range(self.env.action_dim):

                _, rewards[:, a] = self.calculate_rewards(align_func=learned_al_function, obs_mat=th.arange(self.env.state_dim), action_mat=th.tensor([a]*self.env.state_dim), next_state_obs_mat=next_states[:, a], 
                                                       reward_mode=self.training_mode, recover_previous_config_after_calculation=True, requires_grad=False, custom_model=reward_net, forward_groundings=False,)"""

            _, _, pi = mce_partition_fh(self.env, reward=rewards(target_align_func)(),
                                        discount=self.discount, deterministic=not self.learn_stochastic_policy)

            # self.learned_policy_per_va.set_policy_for_va(target_align_func, pi)

            self.learned_policy_per_va.set_policy_for_va(
                self.target_agent_and_align_func_to_learned_ones[target_align_func], pi)

        return self.learned_policy_per_va

    def get_metrics(self):
        metrics = super().get_metrics()
        metrics.update({'accuracy_vs_aid': self.last_vs_accuracies_per_agent, 'accuracy_gr_aid': self.last_gr_accuracies_per_agent,
                       'global_accvs': self.last_accuracies_vs_global, 'global_accgr': self.last_accuracies_gr_global, 'assignment': self.best_assignment})
        return metrics
    