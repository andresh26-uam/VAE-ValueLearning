

from copy import deepcopy
import random
from typing import Any, List, Mapping, Tuple

from src.reward_nets.vsl_reward_functions import AbstractVSLRewardFunction

import numpy as np
import torch as th


def check_grounding_value_system_networks_consistency_with_optim(grounding_per_value_per_cluster, value_system_per_cluster, optimizer):
        """Checks if the optimizer parameters match the networks' parameters."""
        optimizer_params = {param for group in optimizer.param_groups for param in group['params']}
        network_params = {param for cluster in grounding_per_value_per_cluster for network in cluster for param in network.parameters()}
        network_params.update({param for network in value_system_per_cluster for param in network.parameters()})
        assert optimizer_params == network_params, "Optimizer parameters do not match the networks' parameters."


def check_optimizer_consistency(reward_model_per_agent_id, optimizer):
    """Checks if the optimizer parameters match the reward model parameters."""
    optimizer_params = {param for group in optimizer.param_groups for param in group['params']}
    model_params = {param for model in reward_model_per_agent_id.values() for param in model.parameters()}
    assert optimizer_params.issuperset(model_params)
    if optimizer_params != model_params:
        missing_in_optimizer = model_params - optimizer_params
        extra_in_optimizer = optimizer_params - model_params
        error_message = (
            "Optimizer parameters do not match the reward model parameters.\n"
            f"Missing in optimizer: {missing_in_optimizer}\n"
            f"Extra in optimizer: {extra_in_optimizer}"
        )
        if len(missing_in_optimizer) > 0:
            raise AssertionError(error_message)

def check_assignment_consistency(grounding_per_value_per_cluster, value_system_network_per_cluster, assignment_aid_to_gr_cluster, assignment_aid_to_vs_cluster, reward_models_per_aid):
        

        
        for aid, model in reward_models_per_aid.items():
                
                model: AbstractVSLRewardFunction
                vsNetwork: ConvexAlignmentLayer = value_system_network_per_cluster[assignment_aid_to_vs_cluster[aid]]
                th.testing.assert_close(model.get_trained_alignment_function_network().state_dict(), vsNetwork.state_dict())

                np.testing.assert_allclose(model.get_learned_align_function(), vsNetwork.get_alignment_layer()[0].detach()[0])

                assignment_per_value = assignment_aid_to_gr_cluster[aid]

                model_params = {param for param in model.parameters()}
                gNetworksParams = set()
                for vi in range(len(model.get_learned_align_function())):
                    grNetwork: ConvexAlignmentLayer = grounding_per_value_per_cluster[vi][assignment_per_value[vi]]
                    th.testing.assert_close(model.get_network_for_value(vi).state_dict(), grNetwork.state_dict()) # TODO: New class of base clustering vsl algorithm? or gather per grounding and then per agent?
                    network_params = {param for param in grNetwork.parameters()}
                    gNetworksParams.update(network_params)
                all_should_be_params = gNetworksParams.union({p for p in vsNetwork.parameters()})    
                
                if all_should_be_params != model_params:
                    missing_in_optimizer = model_params - all_should_be_params
                    extra_in_optimizer = all_should_be_params - model_params
                    error_message = (
                        "Reward model parameters do not match the ones in the networks.\n"
                        f"Missing in reward model: {missing_in_optimizer}\n"
                        f"Extra in networks: {extra_in_optimizer}"
                    )
                    raise AssertionError(error_message)
                
        model_params = {param for model in reward_models_per_aid.values() for param in model.parameters()}
        
        network_params = {param for cluster in grounding_per_value_per_cluster for network in cluster for param in network.parameters()}
        network_params.update({param for network in value_system_network_per_cluster for param in network.parameters()})
        assert model_params.issubset(network_params)

        #assert network_params.issubset(model_params)
        #assert model_params == network_params, "reward model per aid has different parameters than the networks in the grounding and value system networks."


class ClusterAssignment():

    def _combined_cluster_score(inter_cluster_distances, intra_cluster_distances, n_actual_clusters):
        if n_actual_clusters <= 1:
            return ClusterAssignment._representativity(intra_cluster_distances)
        return ClusterAssignment._conciseness(inter_cluster_distances, n_actual_clusters) / ClusterAssignment._representativity(intra_cluster_distances)

    def _conciseness(inter_cluster_distances, n_actual_clusters):
        if n_actual_clusters <= 1:
            return 1.0
        distances_non_zero = [d for d in inter_cluster_distances if d > 0]
        if len(distances_non_zero) > 0:
            conciseness = min(distances_non_zero)
        else:
            conciseness = 0.0 # ?????
        return conciseness

    def _representativity(intra_cluster_distances):
        return 1.0 - np.mean(np.asarray(intra_cluster_distances))  # TODO. Representativity is the average of the negated intra cluster distances, but these are distances from each agent to its cluster, change that at vs_score().

    def __init__(self, reward_model_per_agent_id: Mapping[str, AbstractVSLRewardFunction] = {},
                 grounding_per_value_per_cluster: List[List[th.nn.Module]] = [],
                 value_system_per_cluster: List[Any] = [],
                 intra_discordances_vs=None,
                 inter_discordances_vs=None,
                 intra_discordances_gr=None,
                 inter_discordances_gr=None,
                 assignment_gr: List[List[str]] = [], assignment_vs: List[str] = [],
                 agent_to_gr_cluster_assignments: Mapping[str, List] = {},
                 agent_to_vs_cluster_assignments: Mapping[str, int] = {},
                 aggregation_on_gr_scores=lambda list_scores: np.mean(list_scores)):
        self.grounding_per_value_per_cluster = grounding_per_value_per_cluster
        self.value_system_per_cluster = value_system_per_cluster

        self.intra_discordances_vs = intra_discordances_vs
        self.inter_discordances_vs = inter_discordances_vs

        self.agent_to_gr_cluster_assignments = agent_to_gr_cluster_assignments
        self.agent_to_vs_cluster_assignments = agent_to_vs_cluster_assignments

        self.intra_discordances_gr = intra_discordances_gr
        self.inter_discordances_gr = inter_discordances_gr

        self.reward_model_per_agent_id = reward_model_per_agent_id
        self.assignment_gr = assignment_gr
        self.assignment_vs = assignment_vs
        self.aggregation_on_gr_scores = aggregation_on_gr_scores

    def copy(self):

        new_models = {}
        new_groundigs_per_value_per_cluster = deepcopy(self.grounding_per_value_per_cluster)
        new_value_system_per_cluster = deepcopy(self.value_system_per_cluster)

        for aid, raid in self.reward_model_per_agent_id.items():
            rc: AbstractVSLRewardFunction = raid.copy()
            th.testing.assert_close(rc.state_dict(), raid.state_dict()), "State dicts of rc and raid do not match"
            rc.set_mode(raid.mode)
            for vi in range(self.n_values):
                cluster_of_aid = self.agent_to_gr_cluster_assignments[aid][vi]
                assert aid in self.assignment_gr[vi][cluster_of_aid]
                rc.set_network_for_value(vi, new_groundigs_per_value_per_cluster[vi][cluster_of_aid])
                #new_groundigs_per_value_per_cluster[vi][cluster_of_aid] = rc.get_network_for_value(vi)

            cluster_of_aid_vs = self.agent_to_vs_cluster_assignments[aid]
            assert aid in self.assignment_vs[cluster_of_aid_vs]
            rc.set_trained_alignment_function_network(new_value_system_per_cluster[cluster_of_aid_vs])
            #new_value_system_per_cluster[cluster_of_aid_vs] = rc.get_trained_alignment_function_network()
            new_models[aid] = rc

        check_assignment_consistency(grounding_per_value_per_cluster=self.grounding_per_value_per_cluster,
                                     value_system_network_per_cluster=self.value_system_per_cluster,
                                     assignment_aid_to_gr_cluster=self.agent_to_gr_cluster_assignments,
                                     assignment_aid_to_vs_cluster=self.agent_to_vs_cluster_assignments,
                                     reward_models_per_aid=self.reward_model_per_agent_id)

        check_assignment_consistency(grounding_per_value_per_cluster=new_groundigs_per_value_per_cluster,
                                     value_system_network_per_cluster=new_value_system_per_cluster,
                                     assignment_aid_to_gr_cluster=self.agent_to_gr_cluster_assignments,
                                     assignment_aid_to_vs_cluster=self.agent_to_vs_cluster_assignments,
                                     reward_models_per_aid=new_models)

        clust = ClusterAssignment(reward_model_per_agent_id=new_models,
                                  grounding_per_value_per_cluster=new_groundigs_per_value_per_cluster,
                                  value_system_per_cluster=new_value_system_per_cluster,
                          intra_discordances_vs=deepcopy(self.intra_discordances_vs),
                          inter_discordances_vs=deepcopy(self.inter_discordances_vs),
                          intra_discordances_gr=deepcopy(self.intra_discordances_gr),
                          inter_discordances_gr=deepcopy(self.inter_discordances_gr),
                          assignment_gr=deepcopy(self.assignment_gr),
                          assignment_vs=deepcopy(self.assignment_vs),
                          agent_to_gr_cluster_assignments=deepcopy(self.agent_to_gr_cluster_assignments),
                          agent_to_vs_cluster_assignments=deepcopy(self.agent_to_vs_cluster_assignments),
                          aggregation_on_gr_scores=self.aggregation_on_gr_scores)

        check_assignment_consistency(grounding_per_value_per_cluster=clust.grounding_per_value_per_cluster,
                                     value_system_network_per_cluster=clust.value_system_per_cluster,
                                     assignment_aid_to_gr_cluster=clust.agent_to_gr_cluster_assignments,
                                     assignment_aid_to_vs_cluster=clust.agent_to_vs_cluster_assignments,
                                     reward_models_per_aid=clust.reward_model_per_agent_id)
        return clust

    @property
    def L(self):
        return sum(1 for c in self.assignment_vs if len(c) > 0) # TODO: take into account inter cluster distances?, if they are 0 they are actually the same cluster...


    @property
    def K(self):
        return [sum(1 for c in self.assignment_gr[value_i] if len(c) > 0)for value_i in range(self.n_values)] # TODO: take into account inter cluster distances?, if they are 0 they are actually the same cluster...
    @property
    def n_values(self):
        return len(self.assignment_gr)
    @property
    def vs_score(self):
        if self.inter_discordances_vs == float('inf') or self.L == 1:
            return self.representativity_vs()
        else:
            return self.combined_cluster_score_vs()
    @property
    def gr_score(self):
        # TODO: for now, it is the average (or other aggregation) on the intra scores of the value-based clusters (accuracies)
        return self.combined_cluster_score_gr_aggr()


    def representativities_gr(self):

        return [ClusterAssignment._representativity(np.array(self.intra_discordances_gr[i])) for i in range(self.n_values)]
    def representativity_gr_aggr(self):
        return self.aggregation_on_gr_scores(self.representativities_gr())

    def representativity_vs(self):
        return ClusterAssignment._representativity(self.intra_discordances_vs)

    def concisenesses_gr(self):
        return [ClusterAssignment._conciseness(np.array(self.inter_discordances_gr[i]), self.K[i]) for i in range(self.n_values)]
    def conciseness_gr_aggr(self):
        return self.aggregation_on_gr_scores(self.concisenesses_gr())
    def conciseness_vs(self):
        return ClusterAssignment._conciseness(self.inter_discordances_vs, self.L)

    def combined_cluster_score_gr(self):
        return [ClusterAssignment._combined_cluster_score(self.inter_discordances_gr[i], self.intra_discordances_gr[i], self.K[i]) for i in range(self.n_values)]

    def combined_cluster_score_vs(self):
        return ClusterAssignment._combined_cluster_score(self.inter_discordances_vs, self.intra_discordances_vs, self.L)

    def combined_cluster_score_gr_aggr(self):
        return self.combined_cluster_score_gr() # TODO FUTURE WORK aggregation of combined scores is this, or dividing the aggregation?


    def __str__(self):
        result = "Cluster Assignment:\n"
        result += "Grounding Clusters:\n"
        for vi, clusters in enumerate(self.assignment_gr):
            result += f"Value {vi}:\n"
            if self.K[vi] == 1:
                result += f"  Single GR Cluster: {[cix for cix in range(len(clusters)) if len(clusters[cix]) >0][0] } \n"
            else:
                for cluster_idx, agents in enumerate(clusters):
                    if len(agents) > 0:
                        result += f"  Cluster {cluster_idx}: {agents}\n"
        result += "\nValue System Clusters:\n"
        for cluster_idx, agents in enumerate(self.assignment_vs):
            if self.L == 1:
                result += f"  Single VS Cluster: {cluster_idx}\n"
            else:
                if len(agents) > 0:
                    result += f"  Cluster {cluster_idx} {self.reward_model_per_agent_id[agents[0]].get_learned_align_function()}: {agents}\n"
        result += "\nScores:\n"
        try:
            result += f"Representativities (Grounding): {self.representativities_gr()}\n"
            result += f"Concisenesses (Grounding): {self.concisenesses_gr()}\n"
            result += f"Combined Scores (Grounding): {self.combined_cluster_score_gr()}\n"
            result += f"Representativity (Value System): {self.representativity_vs()}\n"
            result += f"Conciseness (Value System): {self.conciseness_vs()}\n"
            result += f"Combined Score (Value System): {self.combined_cluster_score_vs()}\n"
        except TypeError:
            result += f"Not available\n"
        return result

    def __repr__(self):
        return self.__str__()


class ClusterAssignmentMemory():

    def __str__(self):
        result = "Cluster Assignment Memory:\n"
        for i, assignment in enumerate(self.memory):
            result += f"Assignment {i}:"
            result += f" {assignment.combined_cluster_score_vs(), assignment.combined_cluster_score_gr_aggr()}"
            result += "\n"
        return result

    def __len__(self):
        return len(self.memory)
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory: List[ClusterAssignment] = []
    def compare_assignments(x: ClusterAssignment, y: ClusterAssignment) -> float:

        # first on different grounding scores... then on value system scores.
        assert x.n_values == y.n_values
        assert x.n_values > 0

        difs = []
        has1 = False
        hasmorethan1 = False
        xK = x.K
        yK = y.K

        xrepr_per_value = x.representativities_gr()
        yrepr_per_value = y.representativities_gr()

        x_combined_per_value, y_combined_per_value = x.combined_cluster_score_gr(), y.combined_cluster_score_gr()

        for i in range(x.n_values):
            if xK[i] == 1 and yK[i] == 1:
                dif_gr_i = xrepr_per_value[i] - yrepr_per_value[i]
                assert x_combined_per_value[i] == xrepr_per_value[i] # this should be the same if the functions are correct because conciseness is 1 when K[i] = 1
                has1 = True
            else:
                hasmorethan1 = True
                dif_gr_i = x_combined_per_value[i] - y_combined_per_value[i]
            difs.append(dif_gr_i)
            #TODO: how to aggregate if there are cases where K(i) == 1 and K(j) > 1...
            assert not (has1 and hasmorethan1) # we need to come up with something here. For ECAI we have 1 grounding always, so no problem yet
        gr_score_dif = x.aggregation_on_gr_scores(difs) # TODO... maybe aggregation on scores should be modelled outside these two?
        if gr_score_dif != 0: # TODO relax this comparison? Lexicograhic is very strict... But this is how it is modelled.
            return gr_score_dif
        else:
            if x.L == 1 or y.L == 1:
                return x.representativity_vs() - y.representativity_vs()
            else:
                return x.combined_cluster_score_vs() - y.combined_cluster_score_vs()

    def insert_assignment(self, assignment) -> Tuple[int, ClusterAssignment]:
        index = 0
        while index < len(self.memory) and ClusterAssignmentMemory.compare_assignments(self.memory[index], assignment) > 0:
            index += 1


        if index == len(self.memory):
            self.memory.append(assignment)
            old = self.memory[index-1]
        else:
            self.memory.insert(index, assignment)
            old = self.memory[index]
        if len(self.memory) > self.max_size:
            self.memory.pop()
        return index, old
    def get_random_weighted_assignment(self)-> ClusterAssignment:
        if len(self.memory) == 0:
            return None
        weights = [i + 1 for i in range(len(self.memory))] # TODO: do something w.r.t. scores? But again K = 1...
        return random.choices(self.memory, weights=weights, k=1)[0]
    def get_best_assignment(self) -> ClusterAssignment:
        if len(self.memory) == 0:
            return None
        return self.memory[0]