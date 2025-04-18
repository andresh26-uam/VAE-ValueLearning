

from copy import deepcopy
from functools import cmp_to_key
import itertools
import os
import random
import sys
from typing import Any, List, Mapping, Self, Set, Tuple

from colorama import Fore, init
import dill
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.axes
from ordered_set import OrderedSet
from sklearn.manifold import MDS

from src.reward_nets.vsl_reward_functions import AbstractVSLRewardFunction, ConvexAlignmentLayer, LinearAlignmentLayer

import numpy as np
import torch as th

from defines import CHECKPOINTS


from scipy.spatial.distance import euclidean

ASSIGNMENT_CHECKPOINTS = os.path.join(CHECKPOINTS, "historic_assignments/")

def assign_colors_matplotlib(num_coordinates,color_map=plt.cm.tab10.colors):
    colors =  color_map # Use the 'tab10' colormap from matplotlib
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
                gNetworksParams = OrderedSet()
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
                
        model_params = OrderedSet(param for model in reward_models_per_aid.values() for param in model.parameters())
        
        network_params = OrderedSet(param for cluster in grounding_per_value_per_cluster for network in cluster for param in network.parameters())
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

    if n > 2:
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
    elif n > 1:
        # Case 2 clusters
        nodes = used_clusters
        coords = np.array([[0, -list(inter_cluster_dists.values())[0]/2.0], [0, list(inter_cluster_dists.values())[0]/2.0]])
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import permutations
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

        self.explored = False

        self.optimizer_state = None # This is useful when saving and loading cluster assignments.
        if aggregation_on_gr_scores is None:
            
            aggregation_on_gr_scores = ClusterAssignment._default_aggr_on_gr_scores
        self.aggregation_on_gr_scores = aggregation_on_gr_scores

    @property
    def n_agents(self):
        return len(self.reward_model_per_agent_id)
    def get_value_system(self, cluster_idx):
        vs =tuple(self.value_system_per_cluster[cluster_idx].get_alignment_layer()[0][0].detach().numpy().tolist())
        if len(self.assignment_vs[cluster_idx]) > 0:
            if self.reward_model_per_agent_id[self.assignment_vs[cluster_idx][0]].get_learned_align_function() != vs:
                raise ValueError(f"Value system {vs} does not match the learned alignment function of the agents in cluster {cluster_idx}.")
        return vs
    
    def average_value_system(self):
        average_vs = np.array([0.0]*self.n_values)
        for cluster_idx in range(len(self.assignment_vs)):
            if len(self.assignment_vs[cluster_idx]) > 0:
                vs = np.array(list(self.get_value_system(cluster_idx)))*len(self.assignment_vs[cluster_idx])
                average_vs += vs
        average_vs /= self.n_agents
        return average_vs
                
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

    def _combined_cluster_score(inter_cluster_distances, intra_cluster_distances, n_actual_clusters, conciseness_if_1_cluster=None):
        if n_actual_clusters <= 1:
            if (conciseness_if_1_cluster is None) or (conciseness_if_1_cluster == float('-inf')):
                return ClusterAssignment._representativity(intra_cluster_distances)
            else:
                return conciseness_if_1_cluster/ ClusterAssignment._representativity(intra_cluster_distances)
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
    
    
    def plot_vs_assignments(self, save_path="demo.pdf", show = False,subfig_multiplier=5.0, values_color_map=plt.cm.tab10.colors, 
                            values_names=None, 
                                                   values_short_names=None, fontsize=12):
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
        
        fig, ax = plt.subplots(figsize=(12, 12))
        max_intra_dist = max(max(self.intra_discordances_vs), 1.0)
        sum_radius = 0
        max_radius = 0

        print(cluster_idx_to_label,cluster_positions)

        for idx, (x, y) in enumerate(cluster_positions):
            cluster_idx = cluster_idx_to_label[idx]
            if len(calculated_distances) > 0:
                min_inter_dist = min(d for (i, j), d in calculated_distances.items() if i == cluster_idx or j == cluster_idx)
            else:
                min_inter_dist = 1.0
            # Plot a circumference around the cluster center
            radius = min_inter_dist / 2.0
            max_radius = max(radius, max_radius)

        for idx, (x, y) in enumerate(cluster_positions):
            cluster_idx = cluster_idx_to_label[idx]
            # Plot cluster center
            ax.scatter(x, y, color=cluster_colors_vs[idx], label=f"Cluster {cluster_idx}", s=100, zorder=3, marker='x')

            # Plot agents around the cluster center
            agents = self.assignment_vs[cluster_idx]
            intra_distances = self.intra_discordances_vs_per_agent
            if len(calculated_distances) > 0:
                min_inter_dist = min(d for (i, j), d in calculated_distances.items() if i == cluster_idx or j == cluster_idx)
            else:
                min_inter_dist = 1.0

            # Plot a circumference around the cluster center
            radius = min_inter_dist / 2.0
            circle = plt.Circle((x, y), radius, color=cluster_colors_vs[idx], fill=False, linestyle='--', alpha=0.5)
            ax.add_artist(circle)
            sum_radius = radius + sum_radius
            
            for agent_idx, agent in enumerate(agents):
                # Place agents around the cluster center based on intra-cluster distances
                agent_angle = 2 * np.pi * agent_idx / len(agents)
                agent_x = x + ((intra_distances[agent] / max_intra_dist) * min_inter_dist / 2) * np.cos(agent_angle)
                agent_y = y + ((intra_distances[agent] / max_intra_dist) * min_inter_dist / 2) * np.sin(agent_angle)
                ax.scatter(agent_x, agent_y, color=cluster_colors_vs[idx], s=50, zorder=2)

            # Plot histogram of intra-cluster distances
            cluster_intra_distances = [intra_distances[agent] for agent in agents]
            

            hist_ax = inset_axes(ax,
                    width=max_radius*2*subfig_multiplier,                     # inch
                    height=max_radius*2*subfig_multiplier,                    # inch
                    bbox_transform=ax.transData, # data coordinates
                    bbox_to_anchor=(x+radius +fontsize/200,y),    # data coordinates
                    loc='center left')
            
            # Add the histogram at the transformed position
            
            # Plot histogram of intra-cluster distances
            hist_ax.hist(cluster_intra_distances, bins=5, color=cluster_colors_vs[idx], alpha=1.0)
            hist_ax.set_xlim(0, 1.0)
            hist_ax.set_ylim(0, len(self.assignment_vs[cluster_idx]))

            hist_ax.tick_params(axis='both', which='major', labelsize=fontsize)
            hist_ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            hist_ax.set_yticks(np.linspace(0, len(self.assignment_vs[cluster_idx]), num=8, endpoint=True, dtype=np.int64))
            hist_ax.set_title(f"{[float('{0:.3f}'.format(t)) for t in self.get_value_system(cluster_idx)]}", fontsize=fontsize)

            # Add a pie chart for value system weights
            pie_ax: matplotlib.axes.Axes = inset_axes(ax,
                                width=max_radius * 2 * subfig_multiplier,  # inch
                                height=max_radius * 2 * subfig_multiplier,  # inch
                                bbox_transform=ax.transData,  # data coordinates
                                bbox_to_anchor=(x - radius - fontsize/200, y),  # data coordinates
                                loc='center right')

            value_system_weights = self.get_value_system(cluster_idx)
            pie_ax.pie(value_system_weights, labels=[f"V{i}" for i in range(len(value_system_weights))] if values_short_names is None else [values_short_names[i] for i in range(len(value_system_weights))],
                       autopct='%f', 
                       startangle=90, colors=assign_colors_matplotlib(self.n_values,color_map=values_color_map), textprops={'fontsize': fontsize})
            pie_ax.set_title("Value System", fontsize=fontsize)  # Add labels and legend
        ax.set_title("Agents-to-VS Assignments")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_aspect('equal', adjustable='datalim')  
        ax.set_xlim(min(-3*max_radius*1.3 - fontsize/200, ax.get_xlim()[0]), max(3*max_radius*1.3 + fontsize/200, ax.get_xlim()[1]))
        ax.set_ylim(min(-3*max_radius*1.0, ax.get_ylim()[0]), max(3*max_radius*1.0, ax.get_ylim()[1]))  
        ax.legend()
        ax.grid(False)

        # Save or show the plot
       
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        if show or save_path is None:
            plt.show()
        plt.close()

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
        clust.explored = self.explored
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

    def combined_cluster_score_gr(self, conciseness_if_K_is_1=None):
        return [ClusterAssignment._combined_cluster_score(self.inter_discordances_gr[i], self.intra_discordances_gr[i], self.K[i], conciseness_if_1_cluster=conciseness_if_K_is_1[i] if conciseness_if_K_is_1 is not None else None) for i in range(self.n_values)]

    def combined_cluster_score_vs(self,conciseness_if_L_is_1=None):
        return ClusterAssignment._combined_cluster_score(self.inter_discordances_vs, self.intra_discordances_vs, self.L, conciseness_if_1_cluster=conciseness_if_L_is_1)

    def combined_cluster_score_gr_aggr(self, conciseness_if_K_is_1=None):
        return self.combined_cluster_score_gr(conciseness_if_K_is_1) # TODO FUTURE WORK aggregation of combined scores is this, or dividing the aggregation?


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
                    result += f"  Cluster {cluster_idx} {self.get_value_system(cluster_idx=cluster_idx)}: {agents}\n"
        result += "\nScores:\n"
        try:
            result += f"Representativities (Grounding): {self.representativities_gr()}\n"
            result += f"Concisenesses (Grounding): {self.concisenesses_gr()}\n"
            result += f"Combined Scores (Grounding): {self.combined_cluster_score_gr_aggr()}\n"
            result += f"Representativity (Value System): {self.representativity_vs()}\n"
            result += f"Conciseness (Value System): {self.conciseness_vs()}\n"
            result += f"Combined Score (Value System): {self.combined_cluster_score_vs()}\n"
        except TypeError:
            result += f"Not available\n"
        return result

    def __repr__(self):
        return self.__str__()
    
    def is_equivalent_assignment(self, other: Self):
        l = self.L 
        l_other = other.L
        k_t = tuple(self.K)
        k_t_other = tuple(other.K)
        if self.n_values != other.n_values:
            return False
        
        if l != l_other:
            return False
        if k_t != k_t_other:
            return False
        
        if l == 1 and l_other == 1 and k_t == tuple([1]*self.n_values) and k_t_other == tuple([1]*self.n_values):
            return True
        
        return self.agent_distribution_gr() == other.agent_distribution_gr() and self.agent_distribution_vs() == other.agent_distribution_vs()

    def cluster_similarity(self, other: Self):
        l = self.L 
        l_other = other.L
        k_t = tuple(self.K)
        k_t_other = tuple(other.K)
        if self.n_values != other.n_values:
            return 0.0
        
        if l != l_other:
            return 0.0
        if k_t != k_t_other:
            return 0.0
        one_grounding = (k_t == tuple([1]*self.n_values) and k_t_other == tuple([1]*self.n_values))
        if l == 1 and l_other == 1 and one_grounding:
            return 1.0
        if self.n_agents != other.n_agents:
            raise ValueError("Number of agents is different between the two cluster assignments. This needs a workaround.")
        
        # self.agent_distribution_vs() is a set of tuples. I want to use the edit distance to compare the two distributions
        total_differences = []

        a1 = self.agent_distribution_vs()
        a2 = other.agent_distribution_vs()
        min_total_edit_distance = float('inf')

        # Generate all possible permutations of clusters in a2
        for perm in permutations(a2):
            total_edit_distance = 0
            for cluster1, cluster2 in zip(a1, perm):
                total_edit_distance += len(set(cluster1).symmetric_difference(set(cluster2)))
            min_total_edit_distance = min(min_total_edit_distance, total_edit_distance)
        assert min_total_edit_distance <= 2*self.n_agents
        total_differences .append(min_total_edit_distance)
        if not one_grounding:
            gr_dists = self.agent_distribution_gr()
            gr_dists_other = other.agent_distribution_gr()
            for a1, a2 in zip(gr_dists, gr_dists_other):
                a1 = self.agent_distribution_vs()
                a2 = other.agent_distribution_vs()
                min_total_edit_distance = float('inf')
                for perm in permutations(a2):
                    total_edit_distance = 0
                    for cluster1, cluster2 in zip(a1, perm):
                        total_edit_distance += len(set(cluster1).symmetric_difference(cluster2))
                min_total_edit_distance = min(min_total_edit_distance, total_edit_distance)
                total_differences .append(min_total_edit_distance)
        diff = np.mean(1.0 - np.array(total_differences)/ (2*self.n_agents) )
        if diff == 1.0:
            assert self.is_equivalent_assignment(other)
        else:
            assert not self.is_equivalent_assignment(other)
        return diff # TODO: separate grounding?
    
    def agent_distribution_gr(self):

        dist = [set(tuple(cluster) for cluster in self.assignment_gr[vi]) for vi in range(len(self.assignment_gr))]
        return dist
    def agent_distribution_vs(self) -> Set[Tuple]:
        dist = set([tuple(cluster) for cluster in self.assignment_vs])
        return dist

    


class ClusterAssignmentMemory():

    def __init__(self, max_size, n_values):
        self.max_size = max_size
        self.memory: List[ClusterAssignment] = []
        self.common_env = None
        self.maximum_conciseness_vs = float('-inf')
        self.maximum_conciseness_gr = [float('-inf') for _ in range(n_values)]

        self.maximum_grounding_coherence = [float('-inf') for _ in range(n_values)]

        self.non_improvable_assignments = []

    def __str__(self):
        result = "Cluster Assignment Memory:\n"
        mgr = self.maximum_conciseness_vs 
        mgr_gr = self.maximum_conciseness_gr

        for i, assignment in enumerate(self.memory):
            result += f"Assignment {i} (Explored: {assignment.explored}):"
            result += f" VS: {assignment.combined_cluster_score_vs(conciseness_if_L_is_1=mgr)}|{assignment.representativity_vs()}, GR: {assignment.combined_cluster_score_gr_aggr(conciseness_if_K_is_1=mgr_gr)}, K: {assignment.K}, L: {assignment.L} \n"
            result += f" GR Clusters: {assignment.active_gr_clusters()} VS Clusters: {assignment.active_vs_clusters()}\n"
            result += "\n"
        return result

    
    def __len__(self):
        return len(self.memory)

    
    
    def compare_assignments(self, x: ClusterAssignment, y: ClusterAssignment, lexicographic_vs_first=False) -> float:

        # first on different grounding scores... then on value system scores.
        assert x.n_values == y.n_values
        assert x.n_values > 0

        mcvs = self.maximum_conciseness_vs
        mcgr = self.maximum_conciseness_gr

        difs = []
        has1 = False
        hasmorethan1 = False
        xK = x.K
        yK = y.K

        x_combined_per_value, y_combined_per_value = x.combined_cluster_score_gr(conciseness_if_K_is_1=mcgr), y.combined_cluster_score_gr(conciseness_if_K_is_1=mcgr)

        for i in range(x.n_values):
            if xK[i] == 1 and yK[i] == 1:
                has1 = True
            else:
                hasmorethan1 = True
            dif_gr_i = x_combined_per_value[i] - y_combined_per_value[i]
            difs.append(dif_gr_i)
            assert not (has1 and hasmorethan1) # we need to come up with something here. For ECAI we have 1 grounding always, so no problem yet
        gr_score_dif = x.aggregation_on_gr_scores(difs) # TODO... maybe aggregation on scores should be modelled outside these two?
        #pareto
        vs_score_dif = x.combined_cluster_score_vs(conciseness_if_L_is_1=mcvs) - y.combined_cluster_score_vs(conciseness_if_L_is_1=mcvs)
        repr_dif = x.representativity_vs() - y.representativity_vs()
        #TODO: TEST PARETO TAKING INTO ACOUNT REPRESENTATIVITY TOO?
        
        pareto_score = 0.0
        lexic_diff = 0.0
        if (gr_score_dif > 0.0 and vs_score_dif >= 0.0 and repr_dif >=0) or (gr_score_dif >= 0.0 and vs_score_dif > 0.0 and repr_dif >=0) or (gr_score_dif >= 0.0 and vs_score_dif >= 0.0 and repr_dif > 0):
                pareto_score = 1.0
        elif (gr_score_dif < 0.0 and vs_score_dif <= 0.0 and repr_dif <=0) or (gr_score_dif <= 0.0 and vs_score_dif < 0.0 and repr_dif <=0) or (gr_score_dif <= 0.0 and vs_score_dif <= 0.0 and repr_dif < 0):
            pareto_score    = -1.0
        else:
            pareto_score = 0.0

        if lexicographic_vs_first:
                
                if abs(vs_score_dif) > 0: 
                        lexic_diff = vs_score_dif
                else:
                    if abs(repr_dif) > 0:
                        lexic_diff = repr_dif
                    else:
                        lexic_diff = gr_score_dif
        else:
            if abs(gr_score_dif) > 0: 
                lexic_diff = gr_score_dif  
            else:
                if abs(repr_dif) > 0:
                        lexic_diff = repr_dif
                else:
                    lexic_diff = vs_score_dif
        return lexic_diff, pareto_score
       
    def insert_assignment(self, assignment: ClusterAssignment) -> Tuple[int, ClusterAssignment]:

        self.update_maximum_conciseness(assignment)
            
        
        
        #lexico_diffs = []
        equivalent_assignments = []
        pareto_diffs = []

        dominated_indices = []
        changes_made = False
        is_dominated = False
        admit_insertion = True

        l_assignment = assignment.L # if it is 1, need to have only one.
        for i in range(len(self.memory)):
            cmp_lexico, cmp_pareto = self.compare_assignments(self.memory[i], assignment,lexicographic_vs_first=False,) 
            eq = self.memory[i].is_equivalent_assignment(assignment)
            equivalent_assignments.append(eq)
            #lexico_diffs.append(cmp_lexico)
            pareto_diffs.append(cmp_pareto)

            if cmp_pareto > 0:
                is_dominated = True

            if (cmp_pareto < 0 and eq) or (cmp_pareto < 0 and self.memory[i].explored):
                dominated_indices.append(i) # Dominated that also equivalent

            if l_assignment == 1 and self.memory[i].L == 1:
                if cmp_lexico > 0 or cmp_pareto > 0:
                    admit_insertion = False
                else:
                    if i not in dominated_indices:
                        dominated_indices.append(i)
        if assignment.L == 1:
            print("L1" ,admit_insertion, dominated_indices, self.maximum_conciseness_vs)            
        # Insert the new one if all the pareto diffs are less than or equal than 0 (pareto dominates someone or is non dominated).
        pareto_diffs = np.array(pareto_diffs)
        if admit_insertion and ((all(pareto_diffs <= 0)) or (not is_dominated) or (equivalent_assignments.count(True) == 0) or all([asa.explored for asa in self.memory])):
            changes_made = True
            self.memory.append(assignment)  
            # Eliminate the assignments in the dominated_indices
            for idx in sorted(dominated_indices, reverse=True):
                    changes_made=True
                    self.memory.pop(idx)  

        if changes_made:
            self.sort_lexicographic(lexicographic_vs_first=True)
            if len(self.memory) > self.max_size:
                self.clean_memory(exhaustive=False)
            
        assignment.explored = not changes_made

        if __debug__:
            for a,b in itertools.combinations(range(len(self.memory)), 2):
                assert not (self.memory[a].is_equivalent_assignment(self.memory[b]) and self.compare_assignments(self.memory[a],self.memory[b])[1] != 0), f"Assignments {a} and {b} are not equivalent. {a} vs {b}"
        
        return

    def update_maximum_conciseness(self, assignment: ClusterAssignment):

        gr_diffs = np.array(self.maximum_grounding_coherence) - np.array(assignment.gr_score)
        better_grounding_precondition = all(gr_diffs <= 0.0)
        if better_grounding_precondition:
            self.maximum_grounding_coherence = assignment.gr_score

        if better_grounding_precondition:
            if assignment.L > 1: 
                new_max_c = max(self.maximum_conciseness_vs, assignment.conciseness_vs()) 
                if new_max_c != self.maximum_conciseness_vs:
                    changes_made = True
                self.maximum_conciseness_vs = new_max_c
            
            for vi in range(len(assignment.K)):
                if assignment.K[vi] > 1:
                    new_max_c = max(assignment.concisenesses_gr()[vi], self.maximum_conciseness_gr[vi])
                    if new_max_c != self.maximum_conciseness_gr[vi]:
                        changes_made = True
                    self.maximum_conciseness_gr[vi] = new_max_c

    def clean_memory(self, exhaustive=True):
        pareto_dominated_counts = [0] * len(self.memory)
        equivalent_assignments_counts = [0] * len(self.memory)
        similarity_index = [0] * len(self.memory)
        # Calculate pareto dominance and equivalence
        
        for i in reversed(list(range(len(self.memory)))):
            eliminated_i = False
            for j in range(len(self.memory)):
                if eliminated_i:
                    continue
                if i != j:
                    _, cmp_pareto = self.compare_assignments(self.memory[j], self.memory[i], lexicographic_vs_first=True)
                    sim = self.memory[i].cluster_similarity(self.memory[j])
                    similarity_index[i] += sim
                    if cmp_pareto > 0:
                        
                        pareto_dominated_counts[i] += 1
                    if sim == 1.0:
                        equivalent_assignments_counts[i] += 1 
                    if (cmp_pareto > 0 and sim==0.0) or (cmp_pareto > 0 and self.memory[i].explored): 
                        # This is always something that needs to be done.
                        # If explored and is dominated is of no interest.
                        # In any case, an equivalent assignment that is dominated is not useful.
                        self.memory.pop(i)
                        pareto_dominated_counts.pop(i)
                        equivalent_assignments_counts.pop(i)
                        similarity_index.pop(i)
                        eliminated_i = True
            # Also try to eliminate any element that is pareto dominated and equivalent to another
            # This is not compulsory, though. 
            if not eliminated_i:
                if exhaustive or len(self.memory) > self.max_size:
                    if pareto_dominated_counts[i] > 0 and equivalent_assignments_counts[i] > 0:
                        self.memory.pop(i)
                        pareto_dominated_counts.pop(i)
                        equivalent_assignments_counts.pop(i)
                        similarity_index.pop(i)
                        if not exhaustive:
                            return
        
        # If still too many examples:
        # Eliminate the one pareto dominated by the most others, or all if exhaustive (only at the end or under all examples explored)
        if len(self.memory) > self.max_size:
            max_dominated_count = max(pareto_dominated_counts)
            while max_dominated_count > 0:
                idx_to_remove = pareto_dominated_counts.index(max_dominated_count)
                self.memory.pop(idx_to_remove)
                pareto_dominated_counts.pop(idx_to_remove)
                equivalent_assignments_counts.pop(idx_to_remove)
                similarity_index.pop(i)
                max_dominated_count = max(pareto_dominated_counts)
                if not exhaustive:
                    break
        # If still too many, remove the first assignment that is the most equivalent to any other
        if len(self.memory) > self.max_size:
            #sorted_indices = [i[0] for i in sorted(enumerate(self.memory), key=lambda x: equivalent_assignments_counts[x[0]], reverse=True)]
            sorted_indices = [i[0] for i in sorted(enumerate(self.memory), key=lambda x: (similarity_index[x[0]], -x[1].combined_cluster_score_vs(conciseness_if_L_is_1=self.maximum_conciseness_vs)), reverse=True)]
            worst = sorted_indices[0]
            self.memory.pop(worst)
        return
        
    def sort_lexicographic(self, lexicographic_vs_first=False):
        self.memory = sorted(self.memory, key=cmp_to_key(lambda x, y: self.compare_assignments(x,y,lexicographic_vs_first=lexicographic_vs_first)[0]), reverse=True)
    
    def get_random_weighted_assignment(self, override_explore=True)-> ClusterAssignment:
        non_explored_assignments = [assignment for assignment in self.memory if not assignment.explored]
        if len(non_explored_assignments) == 0:
            self.clean_memory(exhaustive=True)
            return None
        weights = list(reversed([i+1 for i in range(len(non_explored_assignments))])) # TODO: do something w.r.t. scores? But again K = 1...
        assignment_index =  random.choices(list(range(len(non_explored_assignments))), weights=weights, k=1)[0]
        assignment = non_explored_assignments[assignment_index]
        return self.assignment_with_env(assignment, override_explore)

    def get_best_assignment(self, consider_only_unexplored=False, override_explore_state=True) -> ClusterAssignment:
        if not consider_only_unexplored:
            return self.assignment_with_env(assignment=self.memory[0], override_explore=override_explore_state)
        non_explored_assignments = [assignment for assignment in self.memory if not assignment.explored]
        if len(non_explored_assignments) == 0:
            self.clean_memory(exhaustive=True)
            return None
        
        assignment = non_explored_assignments[0]
        return self.assignment_with_env(assignment, override_explore_state)
    
    def assignment_with_env(self, assignment: ClusterAssignment, override_explore) -> ClusterAssignment:
        if override_explore:
            assignment.explored = True
        if self.common_env is not None:
            assignment.set_env(self.common_env)
        
        return assignment