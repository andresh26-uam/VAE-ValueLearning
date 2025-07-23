from copy import deepcopy
from functools import cmp_to_key
import math
import os
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Self, Set, Tuple

import dill
import numpy as np
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.morl_custom_reward import MOCustomRewardVector
from src.algorithms.clustering_utils_simple import ClusterAssignment, ClusterAssignmentMemory, check_grounding_value_system_networks_consistency_with_optim, generate_random_assignment
from src.algorithms.preference_based_vsl_lib import BaseVSLClusterRewardLoss, VSLCustomLoss, VSLOptimizer, likelihood_x_is_target
from src.dataset_processing.data import VSLPreferenceDataset
from src.reward_nets.vsl_reward_functions import AbstractVSLRewardFunction, ConvexAlignmentLayer, VectorModule, LinearAlignmentLayer, TrainingModes
import torch as th


def probability_BT(x: th.Tensor, y: th.Tensor) -> th.Tensor:
    return 1.0 / (1.0 + th.exp(x - y) + 1e-8)

class PVSL(object):
    def __init__(self, Lmax, moagent_class, moagent_kwargs,  grounding_network: VectorModule, alignment_layer_class=ConvexAlignmentLayer, 
                  loss_class: BaseVSLClusterRewardLoss=VSLCustomLoss, loss_kwargs: Dict[str, Any] = {}, optim_class=VSLOptimizer, optim_kwargs: Dict[str, Any] = {}, 
                  discount_factor_preferences=1.0, qualitative_cluster_assignment=True, online_policy_update=True, lexicographic_vs_first=False, debug_mode=True, **kwargs):
        
        self.Lmax = Lmax
        self.moagent_class = moagent_class
        self.moagent_kwargs = moagent_kwargs
        self.kwargs = kwargs
        self.discount_factor_preferences = discount_factor_preferences
        self.alignment_layer_class = alignment_layer_class
        self.grounding_network = grounding_network
        self.value_system_per_cluster = [alignment_layer_class() for _ in range(Lmax)]

        self.loss: VSLCustomLoss = loss_class(**loss_kwargs)
        self.optim: VSLOptimizer = optim_class( **optim_kwargs)

        self.debug_mode = debug_mode
        self.online_policy_update = online_policy_update
        self.qualitative_cluster_assignment = qualitative_cluster_assignment
        self.lexicographic_vs_first = lexicographic_vs_first
        
    def update_training_networks_from_assignment(self, value_system_per_cluster: List[LinearAlignmentLayer], grounding: VectorModule, reference_assignment: ClusterAssignment):
        
        with th.no_grad():
            for c in range(len(value_system_per_cluster)):
                value_system_per_cluster[c].load_state_dict(deepcopy(reference_assignment.weights_per_cluster[c].state_dict()))
            grounding.load_state_dict(deepcopy(reference_assignment.grounding.state_dict()))
            
           
        
        if isinstance(self.loss, VSLCustomLoss):
            assert isinstance(self.optim, VSLOptimizer)
            self.optim.set_state(reference_assignment.optimizer_state)
            
            if self.debug_mode:
                should_be_wx = {param for network in [grounding.get_network_for_value(i) for i in range(self.n_values)] for param in network.parameters()}
                should_be_wy = {param for al in value_system_per_cluster for param in al.parameters()}
                
                assert should_be_wx.issubset(self.optim.params_gr), "Mismatch in wx parameters"
                assert should_be_wy.issubset(self.optim.params_vs), "Mismatch in wy parameters"
                assert self.optim.params_gr == {param for network in [grounding.get_network_for_value(i) for i in range(self.n_values)] for param in network.parameters()}, "Mismatch in wx parameters"
                assert self.optim.params_vs == {param for network in value_system_per_cluster for param in network.parameters()}, "Mismatch in wy parameters"
            
            self.loss.set_parameters(params_gr=self.optim.params_gr, params_vs=self.optim.params_vs, optim_state=self.optim.get_state())
        if self.debug_mode:
            updated_params = {
                param for group in self.optim.param_groups for param in group['params']
            }
            changed_params = {
                param for model in value_system_per_cluster for param in model.parameters()
            }
            for netwo in enumerate(grounding.networks):
                changed_params.update(netwo.parameters())
            

            assert updated_params == changed_params, "Optimizer parameters do not match the changed networks."
            # Ensure optimizer consistency
            check_grounding_value_system_networks_consistency_with_optim(grounding, value_system_per_cluster, self.optim)

    

    def train(self, experiment_name: str, dataset: VSLPreferenceDataset, train_env, max_iter, epochs_per_iteration, initial_reward_iterations, reward_iterations, eval_env=None, max_assignment_memory=10, resume_from=0, policy_train_kwargs: Dict = {}, **kwargs):
        self.train_env = train_env
        self.eval_env = eval_env if eval_env is not None else train_env
        self.policy_train_kwargs = policy_train_kwargs
        

        policy_iterations = self.policy_train_kwargs['total_timesteps'] // max_iter if self.online_policy_update else self.policy_train_kwargs['total_timesteps']
        self.policy_train_kwargs['total_timesteps'] = policy_iterations

        self.policy_per_cluster: MOAgent | MOPolicy | MOCustomRewardVector = self.initialize_policy(train_env)

        best_assignments_list = ClusterAssignmentMemory(
            max_size=max_assignment_memory,n_values=dataset.n_values
        )
        self.historic = dict()
        for global_iter in range(resume_from, max_iter):
            if global_iter == 0:
                best_assignments_list.initializing = True
                while len(best_assignments_list) < best_assignments_list.max_size:
                    self.current_assignment = generate_random_assignment(dataset, l_max=self.Lmax, alignment_layer_class=self.alignment_layer_class, ref_grounding=self.grounding_network)
                    self.update_training_networks_from_assignment(self.value_system_per_cluster, self.grounding_network, self.current_assignment)
                    best_assignments_list.insert_assignment(self.current_assignment.copy())
            elif resume_from == global_iter:
                best_assignments_list, historic, self.policy_per_cluster = PVSL.load_state(ename=experiment_name)
                self.current_assignment = historic[global_iter]
                self.update_training_networks_from_assignment(self.value_system_per_cluster, self.grounding_network, self.current_assignment)
            
            # Training logic for the algorithm
            new_assignment = self.selection(best_assignments_list)
            self.update_training_networks_from_assignment(self.value_system_per_cluster, self.grounding_network, new_assignment)
            for epoch in range(epochs_per_iteration):
                self.cluster_assignment(dataset) # E-step
                self.train_reward_models(dataset, iterations=reward_iterations if global_iter>0 else initial_reward_iterations) # M-step
                if self.online_policy_update:
                    self.update_policy(dataset, self.value_system_per_cluster, iterations=policy_iterations) # Policy update
            self.historic[global_iter] = best_assignments_list.get_best_assignment(lexicographic_vs_first=False)
                
        if self.online_policy_update is False:
            self.update_policy(dataset, self.value_system_per_cluster, iterations=policy_iterations)
        PVSL.save_state(experiment_name, best_assignments_list, historic)
        return best_assignments_list, historic, self.policy_per_cluster

    def save_state(ename, best_assignments_list: ClusterAssignmentMemory, historic, policy_per_cluster):
        folder = os.path.join("train_results", ename)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "best_assignments_list.pkl"), "wb") as file:
            dill.dump(best_assignments_list, file)
        # Save the historic assignments in different files
        for global_iter, assignment in historic.items():
            with open(os.path.join(folder, 'historic', f"assignment_{global_iter}.pkl"), "wb") as file:
                dill.dump(assignment, file)
        # Save policy
        policy_per_cluster.save(save_dir=folder, filename="policy", save_replay_buffer=True)

    def load_state(ename, policy_class: MOPolicy):
        folder = os.path.join("train_results", ename)
        with open(os.path.join(folder, "best_assignments_list.pkl"), "rb") as file:
            best_assignments_list = dill.load(file)
        historic = {}
        for filename in os.listdir(os.path.join(folder, 'historic')):
            if filename.endswith('.pkl'):
                global_iter = int(filename.split('_')[1].split('.')[0])
                with open(os.path.join(folder, 'historic', filename), "rb") as file:
                    historic[global_iter] = dill.load(file)
        policy = policy_class.load(path=os.path.join(folder, "policy"), load_replay_buffer=True)
        
        return best_assignments_list, historic, policy
        
    def initialize_policy(self, env):
        """
        Initialize a policy for a given cluster.
        This method should be implemented to return a policy agent for the specified cluster.
        """
        return self.moagent_class(env=env, **self.moagent_kwargs)
    
    def selection(self, best_assignments_list: ClusterAssignmentMemory) -> ClusterAssignment:
        if random.random() > self.exploration_rate:
            # Randomly select an assignment
            return best_assignments_list.get_random_weighted_assignment(consider_only_unexplored=True, lexicographic_vs_first=self.lexicographic_vs_first)
        else:
            # TODO Select two assignments, crossover them, then mutate
            raise NotImplementedError("Selection method not implemented yet.")
        
    def cluster_assignment(self, dataset: VSLPreferenceDataset):
        grounding1 = self.grounding_network.forward(dataset.fragments1) 
        grounding2 = self.grounding_network.forward(dataset.fragments2)

        agent_to_vs_cluster_assignments = dict()
        assignment_vs = [[] for _ in range(self.Lmax)]
        with th.no_grad():
            for aid in dataset.agent_data.keys():
                fidxs = dataset.fidxs_per_agent[aid]
                agent_to_vs_cluster_assignments[aid] = dict()
                agent_preferences = dataset.preferences[fidxs]
                aid_likelihood_per_vs_cluster = [1.0]*self.Lmax
                for c in range(self.Lmax):
                    vs = self.value_system_per_cluster[c]
                    probs = probability_BT(vs.forward(grounding1[fidxs]), vs.forward(grounding2[fidxs]))
                    gt_probs = agent_preferences
                    aid_likelihood_per_vs_cluster[c] = likelihood_x_is_target(probs.detach(), gt_probs.detach(), mode='th', slope=0.3, adaptive_slope=False, qualitative_mode=self.qualitative_cluster_assignment, indifference_tolerance=self.loss.model_indifference_tolerance)
                best_cluster = np.argmax(aid_likelihood_per_vs_cluster)
                agent_to_vs_cluster_assignments[aid] = self.value_system_per_cluster[best_cluster]
                assignment_vs[best_cluster].append(aid)

        self.current_assignment.agent_to_vs_cluster_assignments = agent_to_vs_cluster_assignments
        self.current_assignment.assignment_vs = assignment_vs
        self.current_assignment.grounding = self.grounding_network
        self.current_assignment.weights_per_cluster = self.value_system_per_cluster
        return self.current_assignment
    
    def train_reward_models(self, dataset: VSLPreferenceDataset, iterations=10, batch_size_per_agent=None):
        
        for i in range(iterations):

            self.optim.zero_grad()

            if batch_size_per_agent is not None:
                dataset_b= dataset.select_batch(batch_size_per_agent)
            else:
                dataset_b= dataset.select_batch(len(dataset))
            
            fragment_pairs, preferences, preferences_with_grounding, agent_ids = dataset_b

            output = self.loss.forward( # TODO this will fail.
                preferences=preferences,
                preferences_with_grounding=preferences_with_grounding,
                
                fragment_pairs_per_agent_id={a: f for a,f in zip(agent_ids, fragment_pairs)},
                grounding=self.grounding_network,
                value_system_per_cluster=self.value_system_per_cluster,
                fragment_idxs_per_aid=dataset.fidxs_per_agent,
                agent_to_vs_cluster_assignments=self.current_assignment.agent_to_vs_cluster_assignments) # These two provide penalties

            loss_vs, loss_gr, loss_gr_per_vi = output.loss

            self.last_metrics = output.metrics

            loss =  loss_vs  + loss_gr # + loss_policy??? TODO.

            self.loss.gradients(scalar_loss = loss, renormalization = batch_size_per_agent/math.ceil(len(dataset)//dataset.n_agents))
            
            self.optim.step()
            if self.debug_mode:
                print(f"Loss: {loss_vs.item():.4f} (VS), {loss_gr.item():.4f} (GR)")
        
        if isinstance(self.loss, VSLCustomLoss):
            self.current_assignment.optimizer_state = self.optim.get_state()
        
        if self.debug_mode:
            print("Reward models trained successfully.")
        

    def update_policy(self, dataset: VSLPreferenceDataset, iterations=10):
        """
        Update the policies for each cluster based on the current value systems.
        This method should be implemented to return updated policies for each cluster.
        """
        if self.debug_mode:
            print("Updating policies for each cluster...")
        
        for c in range(self.Lmax):
            agent: MOAgent | MOPolicy | MOCustomRewardVector = self.policy_per_cluster

            agent.env = self.env 
            agent.train(**self.policy_train_kwargs)
            
        
        if self.debug_mode:
            print("Policies updated successfully.")
        
        return self.policy_per_cluster