

from copy import deepcopy
import itertools
import os
import random
import sys
from typing import Any, List, Mapping, Self, Tuple

from colorama import Fore, init
import dill
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

from src.reward_nets.vsl_reward_functions import AbstractVSLRewardFunction, ConvexAlignmentLayer, LinearAlignmentLayer

import numpy as np
import torch as th

from utils import CHECKPOINTS


from scipy.spatial.distance import euclidean

ASSIGNMENT_CHECKPOINTS = os.path.join(CHECKPOINTS, "historic_assignments/")

def assign_colors_matplotlib(num_coordinates):
    colors = plt.cm.tab10.colors  # Use the 'tab10' colormap from matplotlib
    assigned_colors = [colors[i % len(colors)] for i in range(num_coordinates)]
    return assigned_colors
def assign_colors(num_coordinates):
    init()
    colors = [Fore.RED, Fore.GREEN, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.YELLOW, Fore.WHITE, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTWHITE_EX]
    assigned_colors = [colors[i % len(colors)] for i in range(num_coordinates)]
    return assigned_colors
def check_grounding_value_system_networks_consistency_with_optim(grounding_per_value_per_cluster, value_system_per_cluster, optimizer):
    if __debug__:
        """Checks if the optimizer parameters match the networks' parameters."""
        optimizer_params = {param for group in optimizer.param_groups for param in group['params']}
        network_params = {param for cluster in grounding_per_value_per_cluster for network in cluster for param in network.parameters()}
        network_params.update({param for network in value_system_per_cluster for param in network.parameters()})
        assert optimizer_params == network_params, "Optimizer parameters do not match the networks' parameters."


def check_optimizer_consistency(reward_model_per_agent_id, optimizer):
    if __debug__:
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
        
    if __debug__:
        
        for aid, model in reward_models_per_aid.items():
                
                model: AbstractVSLRewardFunction
                vsNetwork: LinearAlignmentLayer = value_system_network_per_cluster[assignment_aid_to_vs_cluster[aid]]
                th.testing.assert_close(model.get_trained_alignment_function_network().state_dict(), vsNetwork.state_dict())

                np.testing.assert_allclose(model.get_learned_align_function(), vsNetwork.get_alignment_layer()[0].detach()[0])

                assignment_per_value = assignment_aid_to_gr_cluster[aid]

                model_params = {param for param in model.parameters()}
                gNetworksParams = set()
                for vi in range(len(model.get_learned_align_function())):
                    grNetwork: LinearAlignmentLayer = grounding_per_value_per_cluster[vi][assignment_per_value[vi]]
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

def extract_cluster_coordinates(inter_cluster_dists, used_clusters):

    # Step 1: Extract all unique node IDs
    nodes = sorted(set(i for pair in inter_cluster_dists for i in pair))
    index_map = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    D = np.zeros((n, n))
    for (i, j), dij in inter_cluster_dists.items():
        idx_i, idx_j = index_map[i], index_map[j]
        D[idx_i][idx_j] = dij
        D[idx_j][idx_i] = dij  # Make it symmetric

    if n > 0:
        # MDS to compute positions most likely to be sort of the same separations...
        embedding = MDS(n_components=2, max_iter=10000,dissimilarity='precomputed', random_state=42, eps=1e-12)
        coords = embedding.fit_transform(D)

        print("Computed distances between embedded points:\n")
        calculated_distances = dict()
        for (i, j), target_dist in inter_cluster_dists.items():
            idx_i, idx_j = index_map[i], index_map[j]
            point_i, point_j = coords[idx_i], coords[idx_j]
            dist = euclidean(point_i, point_j)
            print(f"Nodes ({i}, {j}): Target = {target_dist:.4f}, Actual = {dist:.4f}")
            calculated_distances[(i,j)] = dist
    else:
        # Case 1 cluster
        nodes = used_clusters
        coords = [[0,0]]
        calculated_distances = []
    return nodes,coords, calculated_distances

class ClusterAssignment():
    def __init__(self, reward_model_per_agent_id: Mapping[str, AbstractVSLRewardFunction] = {},
                 grounding_per_value_per_cluster: List[List[th.nn.Module]] = [],
                 value_system_per_cluster: List[Any] = [],
                 intra_discordances_vs=None,
                 inter_discordances_vs=None,
                 intra_discordances_gr=None,
                 inter_discordances_gr=None,
                 intra_discordances_gr_per_agent = None,
                    intra_discordances_vs_per_agent = None,
                    inter_discordances_gr_per_cluster_pair = None,
                    inter_discordances_vs_per_cluster_pair = None,
                 assignment_gr: List[List[str]] = [], assignment_vs: List[str] = [],
                 agent_to_gr_cluster_assignments: Mapping[str, List] = {},
                 agent_to_vs_cluster_assignments: Mapping[str, int] = {},
                 aggregation_on_gr_scores=None):
        self.grounding_per_value_per_cluster = grounding_per_value_per_cluster
        self.value_system_per_cluster = value_system_per_cluster

        self.intra_discordances_vs = intra_discordances_vs
        self.inter_discordances_vs = inter_discordances_vs

        self.agent_to_gr_cluster_assignments = agent_to_gr_cluster_assignments
        self.agent_to_vs_cluster_assignments = agent_to_vs_cluster_assignments

        self.intra_discordances_gr = intra_discordances_gr
        self.inter_discordances_gr = inter_discordances_gr

        self.intra_discordances_gr_per_agent = intra_discordances_gr_per_agent
        self.intra_discordances_vs_per_agent = intra_discordances_vs_per_agent

        self.inter_discordances_gr_per_cluster_pair = inter_discordances_gr_per_cluster_pair
        self.inter_discordances_vs_per_cluster_pair = inter_discordances_vs_per_cluster_pair
        self.reward_model_per_agent_id = reward_model_per_agent_id
        self.assignment_gr = assignment_gr
        self.assignment_vs = assignment_vs

        self.optimizer_state = None # This is useful when saving and loading cluster assignments.
        if aggregation_on_gr_scores is None:
            
            aggregation_on_gr_scores = ClusterAssignment._default_aggr_on_gr_scores
        self.aggregation_on_gr_scores = aggregation_on_gr_scores


    def get_remove_env(self):
        example_model = self.reward_model_per_agent_id[list(self.reward_model_per_agent_id.keys())[0]]
        env_state = example_model.remove_env()

        for aid, rewid in self.reward_model_per_agent_id.items():
                rewid.remove_env() # TODO might be needed to keep copies of the env?

        return env_state
    
    def set_env(self, env):
        for aid, rewid in self.reward_model_per_agent_id.items():
            rewid.set_env(env) # TODO above
            
        
    def save(self, path: str, file_name: str = "cluster_assignment.pkl"):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, file_name)

        env_state = self.get_remove_env()
        
        if env_state is not None:
            env_path  = os.path.join(path, "env_state.pkl")
            with open(env_path, 'wb') as fe:
                dill.dump(env_state, fe)

            
            with open(save_path, 'wb') as f:
                dill.dump(self, f)

            self.set_env(env_state)
        else:
            with open(save_path, 'wb') as f:
                dill.dump(self, f)

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
    
    
    def plot_vs_assignments(self, save_path=None):
        """
        Plots the agents-to-value-system (VS) assignments in 2D space.
        Each cluster is represented as a point, and agents are plotted around the cluster center
        based on their intra-cluster distances. Clusters are separated by their inter-cluster distances.

        Args:
            save_path (str, optional): Path to save the plot. If None, the plot is shown interactively.
        """
        if self.inter_discordances_vs is None or self.intra_discordances_vs is None:
            raise ValueError("Inter-cluster and intra-cluster distances must be defined to plot VS assignments.")

        # Normalize inter-cluster distances for visualization
        
        # Create a 2D space for clusters
        # Define a function to calculate the total error in distances
        cluster_idx_to_label, cluster_positions, calculated_distances = extract_cluster_coordinates(self.inter_discordances_vs_per_cluster_pair, [cid for (cid,_) in self.active_vs_clusters()])
        # Plot clusters and agents
        
        cluster_colors_vs = assign_colors_matplotlib(self.L)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        max_intra_dist = max(max(self.intra_discordances_vs ), 1.0)
        for idx,(x, y)  in enumerate(cluster_positions):
            cluster_idx = cluster_idx_to_label[idx]
            # Plot cluster center
            ax.scatter(x, y, color=cluster_colors_vs[idx], label=f"Cluster {cluster_idx}", s=100, zorder=3, marker='x')

            # Plot agents around the cluster center
            agents = self.assignment_vs[cluster_idx]
            intra_distances = self.intra_discordances_vs_per_agent
            if len (calculated_distances) > 0:
                min_inter_dist = min(d for (i,j), d in calculated_distances.items() if i == cluster_idx or j == cluster_idx)
            else:
                min_inter_dist = 1.0
            # Plot a circumference around the cluster center
            radius =  min_inter_dist / 2.0
            circle = plt.Circle((x, y), radius, color=cluster_colors_vs[idx], fill=False, linestyle='--', alpha=0.5)
            ax.add_artist(circle)

            for agent_idx, agent in enumerate(agents):
                # Place agents around the cluster center based on intra-cluster distances
                agent_angle = 2 * np.pi * agent_idx / len(agents) 
                
                agent_x = x + ((intra_distances[agent]/max_intra_dist)*min_inter_dist/2 )* np.cos(agent_angle)
                agent_y = y + ((intra_distances[agent]/max_intra_dist)*min_inter_dist/2) * np.sin(agent_angle)
                ax.scatter(agent_x, agent_y, color=cluster_colors_vs[idx], s=50, zorder=2)
                #ax.text(agent_x, agent_y, agent, fontsize=8, ha="center", va="center", zorder=4)

        # Add labels and legend
        ax.set_title("Agents-to-VS Assignments")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_aspect('equal', adjustable='datalim')  
        ax.set_xlim(min(-0.5, ax.get_xlim()[0]), max(0.5, ax.get_xlim()[1]))
        ax.set_ylim(min(-0.5, ax.get_ylim()[0]), max(0.5, ax.get_ylim()[1]))  
        ax.legend()
        ax.grid(True)

        # Save or show the plot
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()
        
    def _default_aggr_on_gr_scores(x):
                return np.mean(x, axis=0)
    def copy(self):

        new_models = {}
        new_groundigs_per_value_per_cluster = deepcopy(self.grounding_per_value_per_cluster)
        new_value_system_per_cluster = deepcopy(self.value_system_per_cluster)

        for aid, raid in self.reward_model_per_agent_id.items():
            rc: AbstractVSLRewardFunction = raid.copy()
            #th.testing.assert_close(rc.state_dict(), raid.state_dict()), "State dicts of rc and raid do not match"
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

                          intra_discordances_vs_per_agent=deepcopy(self.intra_discordances_vs_per_agent),
                          inter_discordances_vs_per_cluster_pair=deepcopy(self.inter_discordances_vs_per_cluster_pair),
                          intra_discordances_gr_per_agent=deepcopy(self.intra_discordances_gr_per_agent),
                          inter_discordances_gr_per_cluster_pair=deepcopy(self.inter_discordances_gr_per_cluster_pair),
                          assignment_gr=deepcopy(self.assignment_gr),
                          assignment_vs=deepcopy(self.assignment_vs),
                          agent_to_gr_cluster_assignments=deepcopy(self.agent_to_gr_cluster_assignments),
                          agent_to_vs_cluster_assignments=deepcopy(self.agent_to_vs_cluster_assignments),
                          aggregation_on_gr_scores=self.aggregation_on_gr_scores)
        clust.optimizer_state = deepcopy(self.optimizer_state)
        check_assignment_consistency(grounding_per_value_per_cluster=clust.grounding_per_value_per_cluster,
                                     value_system_network_per_cluster=clust.value_system_per_cluster,
                                     assignment_aid_to_gr_cluster=clust.agent_to_gr_cluster_assignments,
                                     assignment_aid_to_vs_cluster=clust.agent_to_vs_cluster_assignments,
                                     reward_models_per_aid=clust.reward_model_per_agent_id)
        
        
        return clust

    @property
    def L(self):
        return sum(1 for c in self.assignment_vs if len(c) > 0) # TODO: take into account inter cluster distances?, if they are 0 they are actually the same cluster...

    def active_vs_clusters(self):
        return [(i, len(self.assignment_vs[i])) for i in range(len(self.assignment_vs)) if len(self.assignment_vs[i]) > 0]
    
    def active_gr_clusters(self):
        return [[(i, len(self.assignment_gr[value_i][i])) for i in range(len(self.assignment_gr[value_i])) if len(self.assignment_gr[value_i][i]) > 0] for value_i in range(self.n_values)]
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
    
    def is_equivalent_assignment(self, other: Self):
        if self.L != other.L:
            return False
        if self.K != other.K:
            return False
        return self.agent_distribution_gr() == other.agent_distribution_gr() and self.agent_distribution_vs() == other.agent_distribution_vs()

    def agent_distribution_gr(self):

        dist = [set(tuple(cluster) for cluster in self.assignment_gr[vi]) for vi in range(len(self.assignment_gr))]
        return dist
    def agent_distribution_vs(self):
        dist = set([tuple(cluster) for cluster in self.assignment_vs])
        return dist

class ClusterAssignmentMemory():

    # TODO: Cambiar a lista de soluciones no dominadas (que sea más grande que antes...)
    # TODO: Escoger y ordenar por mejor ratio, que sea mejorable (ver abajo) teniendo en cuenta que
        # TODO: La conciseness de los de 1 es la mejor conciseness hasta ahora (DE TODOS LOS QUE SE HAN INTENTADO INSERTAR > 1).
        # TODO: si no hay un referente, se escoje representativity, no queda otra. 
        # Asi ya son directamente comparables sin la movida del if, que no es un orden.
    # TODO: Cuando coges uno para mejorar:
        # TODO: si mejora a sí mismo, se borra el anterior.
        # TODO: si no mejora tras un ciclo de epochs, se marca como no mejorable.
        # TODO: Si todos son no mejorables, se escoge con mutaciones.
        # Hasta end de iteraciones.
    # Cuando insertas, borrar a todos los que la solución domine. Si hay dos assignment iguales no dominados, no pasa nada.
    # Si domina a algunas pero no a todas...?
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory: List[ClusterAssignment] = []
        self.common_env = None
    def __str__(self):
        result = "Cluster Assignment Memory:\n"
        for i, assignment in enumerate(self.memory):
            result += f"Assignment {i}:"
            result += f" VS: {assignment.combined_cluster_score_vs()}|{assignment.representativity_vs()}, GR: {assignment.combined_cluster_score_gr_aggr()}, K: {assignment.K}, L: {assignment.L} \n"
            result += f" GR Clusters: {assignment.active_gr_clusters()} VS Clusters: {assignment.active_vs_clusters()}\n"
            result += "\n"
        return result

    
    def __len__(self):
        return len(self.memory)
    
    def compare_assignments(x: ClusterAssignment, y: ClusterAssignment, lexicographic_vs_first=False, priotitize_representativity=False) -> float:

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
        #pareto
        vs_score_dif = 0
        repr_dif = x.representativity_vs() - y.representativity_vs()
        vs_score_dif = x.combined_cluster_score_vs() - y.combined_cluster_score_vs()
        vs_score_dif = repr_dif if x.L == 1 or y.L == 1 else vs_score_dif

        if lexicographic_vs_first:
            if priotitize_representativity:
                if repr_dif != 0.0:
                    return repr_dif
            
            if vs_score_dif != 0.0: # TODO relax this comparison? Lexicograhic is very strict... But this is how it is modelled.
                    return vs_score_dif
            else:
                    
                    return gr_score_dif
        else:
            if gr_score_dif != 0.0: # TODO relax this comparison? Lexicograhic is very strict... But this is how it is modelled.
                    return gr_score_dif
            else:
                    if priotitize_representativity:
                        if repr_dif != 0.0:
                            return repr_dif
                    return vs_score_dif
        """elif lexicographic_or_pareto == 'pareto':
            if (gr_score_dif < 0 and vs_score_dif <= 0) or (gr_score_dif <= 0 and vs_score_dif < 0):
                return -1
            elif (gr_score_dif > 0 and vs_score_dif >= 0) or (gr_score_dif >= 0 and vs_score_dif > 0):
                return 1
            else:
                return 0"""
    def insert_assignment(self, assignment: ClusterAssignment) -> Tuple[int, ClusterAssignment]:
        index = 0
        dont_insert = False

        if __debug__:
            for a,b in itertools.combinations(range(len(self.memory)), 2):
                assert not self.memory[a].is_equivalent_assignment(self.memory[b]), f"Assignments {a} and {b} are not equivalent. {a} vs {b}"
        
        dont_insert = False
        while index < len(self.memory):
            cmp = ClusterAssignmentMemory.compare_assignments(self.memory[index], assignment) 
            eq = self.memory[index].is_equivalent_assignment(assignment)
            if cmp <= 0:
                if eq: # If is a single cluster, you need to go down until the representativity is low enough.
                    dont_insert = True # The assignment is equivalent to the one in the memory, but it is better.
                break
            else:
                if eq:
                    return len(self.memory), None # There is already an equivalent assignment in the memory that is better.
                index += 1
        
        old = None
        
        if dont_insert:
            old = self.memory[index]

            assert self.memory[index].is_equivalent_assignment(assignment)
            self.memory[index] = assignment
        
        elif index < self.max_size:
            if index == len(self.memory):
                self.memory.append(assignment)
                old = self.memory[index-1]
            else:
                old = self.memory[index]
                self.memory.insert(index, assignment)
        else:
            pass # Here it means the assignment is worse than the worst one in the memory, so we do not insert it.
        remaining = index + 1      
        while remaining < len(self.memory):
            #cmp = ClusterAssignmentMemory.compare_assignments(self.memory[remaining], assignment) 
            
            if self.memory[remaining].is_equivalent_assignment(assignment):
                self.memory.pop(remaining)
            else:
                remaining+=1
                
        if len(self.memory) > self.max_size:
            self.memory.pop()
        
    
        if assignment.L == 1:
            from functools import cmp_to_key
            self.memory = sorted(self.memory, key=cmp_to_key(lambda x, y: ClusterAssignmentMemory.compare_assignments(x,y,priotitize_representativity=True)), reverse=True)
        
        
        return index, old
    def get_random_weighted_assignment(self)-> ClusterAssignment:
        if len(self.memory) == 0:
            return None
        weights = [i + 1 for i in range(len(self.memory))] # TODO: do something w.r.t. scores? But again K = 1...
        assignment =  random.choices(self.memory, weights=weights, k=1)[0]
        return self.assignment_with_env(assignment)

    def get_best_assignment(self) -> ClusterAssignment:
        if len(self.memory) == 0:
            return None
        assignment = self.memory[0]
        return self.assignment_with_env(assignment)
    
    def assignment_with_env(self, assignment: ClusterAssignment) -> ClusterAssignment:
        if self.common_env is not None:
            assignment.set_env(self.common_env)
        
        return assignment