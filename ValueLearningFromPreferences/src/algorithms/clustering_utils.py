

from itertools import permutations
from copy import deepcopy
from functools import cmp_to_key
import os
import random
from typing import Any, Dict, List, Mapping, Self, Set, Tuple

from colorama import Fore, init
import dill
from matplotlib import pyplot as plt
from ordered_set import OrderedSet
from sklearn.manifold import MDS

from src.reward_nets.vsl_reward_functions import AbstractVSLRewardFunction, LinearAlignmentLayer

import numpy as np
import torch as th

from defines import CHECKPOINTS


from scipy.spatial.distance import euclidean

ASSIGNMENT_CHECKPOINTS = os.path.join(CHECKPOINTS, "historic_assignments/")


def assign_colors_matplotlib(num_coordinates, color_map=plt.cm.tab10.colors):
    colors = color_map  # Use the 'tab10' colormap from matplotlib
    assigned_colors = [colors[i % len(colors)] for i in range(num_coordinates)]
    return assigned_colors


def assign_colors(num_coordinates):
    init()
    colors = [Fore.RED, Fore.GREEN, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.YELLOW, Fore.WHITE, Fore.LIGHTRED_EX,
              Fore.LIGHTGREEN_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTWHITE_EX]
    assigned_colors = [colors[i % len(colors)] for i in range(num_coordinates)]
    return assigned_colors


def check_grounding_value_system_networks_consistency_with_optim(grounding_per_value_per_cluster, value_system_per_cluster, optimizer):
    if __debug__:
        """Checks if the optimizer parameters match the networks' parameters."""
        optimizer_params = {
            param for group in optimizer.param_groups for param in group['params']}
        network_params = {
            param for cluster in grounding_per_value_per_cluster for network in cluster for param in network.parameters()}
        network_params.update(
            {param for network in value_system_per_cluster for param in network.parameters()})
        assert optimizer_params == network_params, "Optimizer parameters do not match the networks' parameters."


def check_optimizer_consistency(reward_model_per_agent_id, optimizer):
    if __debug__:
        """Checks if the optimizer parameters match the reward model parameters."""
        optimizer_params = {
            param for group in optimizer.param_groups for param in group['params']}
        model_params = {param for model in reward_model_per_agent_id.values()
                        for param in model.parameters()}
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
            vsNetwork: LinearAlignmentLayer = value_system_network_per_cluster[
                assignment_aid_to_vs_cluster[aid]]
            th.testing.assert_close(model.get_trained_alignment_function_network(
            ).state_dict(), vsNetwork.state_dict())

            np.testing.assert_allclose(model.get_learned_align_function(
            ), vsNetwork.get_alignment_layer()[0].detach()[0])

            assignment_per_value = assignment_aid_to_gr_cluster[aid]

            model_params = {param for param in model.parameters()}
            gNetworksParams = OrderedSet()
            for vi in range(len(model.get_learned_align_function())):
                grNetwork: LinearAlignmentLayer = grounding_per_value_per_cluster[
                    vi][assignment_per_value[vi]]
                # TODO: New class of base clustering vsl algorithm? or gather per grounding and then per agent?
                th.testing.assert_close(model.get_network_for_value(
                    vi).state_dict(), grNetwork.state_dict())
                network_params = {param for param in grNetwork.parameters()}
                gNetworksParams.update(network_params)
            all_should_be_params = gNetworksParams.union(
                {p for p in vsNetwork.parameters()})

            if all_should_be_params != model_params:
                missing_in_optimizer = model_params - all_should_be_params
                extra_in_optimizer = all_should_be_params - model_params
                error_message = (
                    "Reward model parameters do not match the ones in the networks.\n"
                    f"Missing in reward model: {missing_in_optimizer}\n"
                    f"Extra in networks: {extra_in_optimizer}"
                )
                raise AssertionError(error_message)

        model_params = OrderedSet(
            param for model in reward_models_per_aid.values() for param in model.parameters())

        network_params = OrderedSet(
            param for cluster in grounding_per_value_per_cluster for network in cluster for param in network.parameters())
        network_params.update(
            {param for network in value_system_network_per_cluster for param in network.parameters()})
        assert model_params.issubset(network_params)

        # assert network_params.issubset(model_params)
        # assert model_params == network_params, "reward model per aid has different parameters than the networks in the grounding and value system networks."


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
        embedding = MDS(n_components=2, max_iter=10000,
                        dissimilarity='precomputed', random_state=42, eps=1e-12)
        coords = embedding.fit_transform(D)

        print("Computed distances between embedded points:\n")
        calculated_distances = dict()
        for (i, j), target_dist in inter_cluster_dists.items():
            idx_i, idx_j = index_map[i], index_map[j]
            point_i, point_j = coords[idx_i], coords[idx_j]
            dist = euclidean(point_i, point_j)
            print(
                f"Nodes ({i}, {j}): Target = {target_dist:.4f}, Actual = {dist:.4f}")
            calculated_distances[(i, j)] = dist
    elif n > 1:
        # Case 2 clusters
        nodes = used_clusters
        coords = np.array([[0, -list(inter_cluster_dists.values())[0]/2.0],
                          [0, list(inter_cluster_dists.values())[0]/2.0]])
        calculated_distances = dict()
        for (i, j), target_dist in inter_cluster_dists.items():
            idx_i, idx_j = index_map[i], index_map[j]
            point_i, point_j = coords[idx_i], coords[idx_j]
            dist = euclidean(point_i, point_j)
            print(
                f"Nodes ({i}, {j}): Target = {target_dist:.4f}, Actual = {dist:.4f}")
            calculated_distances[(i, j)] = dist
    else:
        # Case 1 cluster
        nodes = used_clusters
        coords = [[0, 0]]
        calculated_distances = []
    return nodes, coords, calculated_distances


class ClusterAssignment():
    def __init__(self, reward_model_per_agent_id: Mapping[str, AbstractVSLRewardFunction] = {},
                 grounding_per_value_per_cluster: List[List[th.nn.Module]] = [
    ],
            value_system_per_cluster: List[Any] = [],
            intra_discordances_vs=None,
            inter_discordances_vs=None,
            intra_discordances_gr=None,
            inter_discordances_gr=None,
            intra_discordances_gr_per_agent=None,
            intra_discordances_vs_per_agent=None,
            inter_discordances_gr_per_cluster_pair=None,
            inter_discordances_vs_per_cluster_pair=None,
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

        # This is useful when saving and loading cluster assignments.
        self.optimizer_state = None
        if aggregation_on_gr_scores is None:

            aggregation_on_gr_scores = ClusterAssignment._default_aggr_on_gr_scores
        self.aggregation_on_gr_scores = aggregation_on_gr_scores
        self.n_training_steps = 0

    @property
    def n_agents(self):
        return len(self.reward_model_per_agent_id)

    def get_value_system(self, cluster_idx):
        vs = tuple(self.value_system_per_cluster[cluster_idx].get_alignment_layer()[
                   0][0].detach().numpy().tolist())
        if len(self.assignment_vs[cluster_idx]) > 0:
            if self.reward_model_per_agent_id[self.assignment_vs[cluster_idx][0]].get_learned_align_function() != vs:
                raise ValueError(
                    f"Value system {vs} does not match the learned alignment function of the agents in cluster {cluster_idx}.")
        return vs

    def average_value_system(self):
        average_vs = np.array([0.0]*self.n_values)
        for cluster_idx in range(len(self.assignment_vs)):
            if len(self.assignment_vs[cluster_idx]) > 0:
                vs = np.array(list(self.get_value_system(cluster_idx))
                              )*len(self.assignment_vs[cluster_idx])
                average_vs += vs
        average_vs /= self.n_agents
        return average_vs

    def get_remove_env(self):
        example_model = self.reward_model_per_agent_id[list(
            self.reward_model_per_agent_id.keys())[0]]
        env_state = example_model.remove_env()

        for aid, rewid in self.reward_model_per_agent_id.items():
            rewid.remove_env()  # TODO might be needed to keep copies of the env?

        return env_state

    def set_env(self, env):
        for aid, rewid in self.reward_model_per_agent_id.items():
            rewid.set_env(env)  # TODO above

    def save(self, path: str, file_name: str = "cluster_assignment.pkl"):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, file_name)

        env_state = self.get_remove_env()
        env_path = os.path.join(path, "env_state.pkl")

        if env_state is not None and not os.path.exists(env_path):

            with open(env_path, 'wb') as fe:
                dill.dump(env_state, fe)

            with open(save_path, 'wb') as f:
                dill.dump(self, f)

            self.set_env(env_state)
        else:
            with open(save_path, 'wb') as f:
                dill.dump(self, f)

    def _combined_cluster_score(inter_cluster_distances, intra_cluster_distances_per_agent, n_actual_clusters, cluster_to_agents, conciseness_if_1_cluster=None, normalized=False, aggr_repr=np.max):
        n_actual_clusters = sum(
            [1 for c in range(len(cluster_to_agents)) if len(cluster_to_agents[c]) > 0], 0)
        represent = ClusterAssignment._representativity(
            intra_cluster_distances_per_agent, cluster_to_agents, aggr=aggr_repr)

        if n_actual_clusters <= 1:
            if (conciseness_if_1_cluster is None) or (conciseness_if_1_cluster == float('-inf')):
                return represent
            else:
                conc = (conciseness_if_1_cluster)
        else:
            conc = ClusterAssignment._conciseness(
                inter_cluster_distances, n_actual_clusters)

        if normalized:
            val = ((conc + 1.0) / (1.0 - represent + 1.0) - 0.5)/1.5
            assert val <= 1.0 and val >= 0.0, f"val {val} is negative, conciseness: {conc}, Representativity {represent}"
        else:
            val = conc/(1.0-represent+1e-8)
        return val

    def _intra_cluster_discordances(intra_cluster_distances_per_agent, cluster_to_agents):

        intra_cluster_discordances = [
            0.0 for _ in range(len(cluster_to_agents))]
        for c, agents in enumerate(cluster_to_agents):
            if len(agents) == 0:
                continue
            intra_cluster_distances_c = sum(
                intra_cluster_distances_per_agent[agent] for agent in agents)
            intra_cluster_distances_c /= len(agents)
            intra_cluster_discordances[int(c)] = intra_cluster_distances_c

        return intra_cluster_discordances

    def _conciseness(inter_cluster_distances, n_actual_clusters):
        if n_actual_clusters <= 1:
            return 1.0
        if len(inter_cluster_distances) > 0:
            conciseness = min(inter_cluster_distances)
        else:
            conciseness = 0.0  # ?????
        return conciseness

    def _representativity_cluster(intra_cluster_distances):
        return np.mean(1.0 - np.asarray(intra_cluster_distances))

    def _representativity(intra_cluster_distances_per_agent: Dict[str, float], cluster_to_agents: List[List[str]], aggr=np.mean):
        disc_total = 0.0
        for aid, disc in intra_cluster_distances_per_agent.items():
            disc_total += disc
        disc_total /= len(intra_cluster_distances_per_agent)
        return 1 - disc_total

    def plot_vs_assignments(self, save_path="demo.pdf", pie_and_hist_path="pie_and_histograms.pdf", show=False, subfig_multiplier=5.0, values_color_map=plt.cm.tab10.colors,
                            values_names=None, values_short_names=None, fontsize=12):

        if self.inter_discordances_vs is None or self.intra_discordances_vs is None:
            raise ValueError(
                "Inter-cluster and intra-cluster distances must be defined to plot VS assignments.")

        # Extract cluster coordinates
        cluster_idx_to_label, cluster_positions, calculated_distances = extract_cluster_coordinates(
            self.inter_discordances_vs_per_cluster_pair, [
                cid for (cid, _) in self.active_vs_clusters()]
        )

        cluster_colors_vs = assign_colors_matplotlib(self.L)

        # Create the figure for the cluster plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        max_intra_dist = max(max(self.intra_discordances_vs), 1.0)
        max_radius = 0

        for idx, (x, y) in enumerate(cluster_positions):
            cluster_idx = cluster_idx_to_label[idx]
            if len(calculated_distances) > 0:
                min_inter_dist = min(d for (i, j), d in calculated_distances.items(
                ) if i == cluster_idx or j == cluster_idx)
            else:
                min_inter_dist = 1.0
            radius = min_inter_dist / 2.0
            max_radius = max(radius, max_radius)

        for idx, (x, y) in enumerate(cluster_positions):
            cluster_idx = cluster_idx_to_label[idx]
            ax.scatter(
                x, y, color=cluster_colors_vs[idx], label=f"Cluster {cluster_idx}", s=100, zorder=3, marker='x')

            agents = self.assignment_vs[cluster_idx]
            intra_distances = self.intra_discordances_vs_per_agent
            if len(calculated_distances) > 0:
                min_inter_dist = min(d for (i, j), d in calculated_distances.items(
                ) if i == cluster_idx or j == cluster_idx)
            else:
                min_inter_dist = 1.0

            radius = min_inter_dist / 2.0
            circle = plt.Circle(
                (x, y), radius, color=cluster_colors_vs[idx], fill=False, linestyle='--', alpha=0.5)
            ax.add_artist(circle)

            for agent_idx, agent in enumerate(agents):
                agent_angle = 2 * np.pi * agent_idx / len(agents)
                agent_x = x + \
                    ((intra_distances[agent] / max_intra_dist)
                     * min_inter_dist / 2) * np.cos(agent_angle)
                agent_y = y + \
                    ((intra_distances[agent] / max_intra_dist)
                     * min_inter_dist / 2) * np.sin(agent_angle)
                ax.scatter(agent_x, agent_y,
                           color=cluster_colors_vs[idx], s=50, zorder=2)

        ax.set_title("Agents-to-VS Assignments")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlim(min(-3 * max_radius * 1.3 - fontsize / 200, ax.get_xlim()[0]),
                    max(3 * max_radius * 1.3 + fontsize / 200, ax.get_xlim()[1]))
        ax.set_ylim(min(-3 * max_radius * 1.0, ax.get_ylim()[0]),
                    max(3 * max_radius * 1.0, ax.get_ylim()[1]))
        ax.legend()
        ax.grid(False)

        # Save or show the cluster plot
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        if show or save_path is None:
            plt.show()
        plt.close()

        # Create the figure for combined pie charts and histograms
        fig_combined = plt.figure(figsize=(12, 12))
        n_clusters = len(cluster_idx_to_label)
        for idx, cluster_idx in enumerate(cluster_idx_to_label):
            # Pie chart
            pie_ax = fig_combined.add_subplot(
                2, n_clusters, idx + 1, aspect='equal')
            value_system_weights = self.get_value_system(cluster_idx)
            pie_ax.pie(value_system_weights,
                       labels=[f"V{i}" for i in range(len(value_system_weights))] if values_short_names is None else [
                           values_short_names[i] for i in range(len(value_system_weights))],
                       autopct='%f',
                       startangle=90, colors=assign_colors_matplotlib(self.n_values, color_map=values_color_map),
                       textprops={'fontsize': fontsize})
            pie_ax.set_title(
                f"Cluster {cluster_idx} Value System", fontsize=fontsize)

            # Histogram
            hist_ax = fig_combined.add_subplot(
                2, n_clusters, n_clusters + idx + 1)
            agents = self.assignment_vs[cluster_idx]
            cluster_representativity = [
                1.0 - self.intra_discordances_vs_per_agent[agent] for agent in agents]
            hist_ax.hist(cluster_representativity, bins=5,
                         color=cluster_colors_vs[idx], alpha=1.0)
            hist_ax.set_xlim(0, 1.0)
            hist_ax.set_ylim(0, len(agents))
            hist_ax.tick_params(axis='both', which='major', labelsize=fontsize)
            hist_ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            hist_ax.set_yticks(np.linspace(
                0, len(agents), num=8, endpoint=True, dtype=np.int64))
            hist_ax.set_title(
                f"Cluster {cluster_idx} Representativity", fontsize=fontsize)

        # Ensure the pie chart and histogram have the same width
        pie_ax.set_box_aspect(1)
        # Save or show the combined pie charts and histograms
        hist_ax.set_box_aspect(1)
        if pie_and_hist_path is not None:
            os.makedirs(os.path.dirname(pie_and_hist_path), exist_ok=True)
            plt.savefig(pie_and_hist_path, bbox_inches="tight")
        if show or pie_and_hist_path is None:
            plt.show()
        plt.close()

    def _default_aggr_on_gr_scores(x):
        return np.mean(x, axis=0)

    def copy(self):

        new_models = {}
        new_groundigs_per_value_per_cluster = deepcopy(
            self.grounding_per_value_per_cluster)
        new_value_system_per_cluster = deepcopy(self.value_system_per_cluster)

        for aid, raid in self.reward_model_per_agent_id.items():
            rc: AbstractVSLRewardFunction = raid.copy()
            # th.testing.assert_close(rc.state_dict(), raid.state_dict()), "State dicts of rc and raid do not match"
            rc.set_mode(raid.mode)
            for vi in range(self.n_values):
                cluster_of_aid = self.agent_to_gr_cluster_assignments[aid][vi]
                assert aid in self.assignment_gr[vi][cluster_of_aid]
                rc.set_network_for_value(
                    vi, new_groundigs_per_value_per_cluster[vi][cluster_of_aid])
                # new_groundigs_per_value_per_cluster[vi][cluster_of_aid] = rc.get_network_for_value(vi)

            cluster_of_aid_vs = self.agent_to_vs_cluster_assignments[aid]
            assert aid in self.assignment_vs[cluster_of_aid_vs]
            rc.set_trained_alignment_function_network(
                new_value_system_per_cluster[cluster_of_aid_vs])
            # new_value_system_per_cluster[cluster_of_aid_vs] = rc.get_trained_alignment_function_network()
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
                                  intra_discordances_vs=deepcopy(
                                      self.intra_discordances_vs),
                                  inter_discordances_vs=deepcopy(
                                      self.inter_discordances_vs),
                                  intra_discordances_gr=deepcopy(
                                      self.intra_discordances_gr),
                                  inter_discordances_gr=deepcopy(
                                      self.inter_discordances_gr),

                                  intra_discordances_vs_per_agent=deepcopy(
                                      self.intra_discordances_vs_per_agent),
                                  inter_discordances_vs_per_cluster_pair=deepcopy(
                                      self.inter_discordances_vs_per_cluster_pair),
                                  intra_discordances_gr_per_agent=deepcopy(
                                      self.intra_discordances_gr_per_agent),
                                  inter_discordances_gr_per_cluster_pair=deepcopy(
                                      self.inter_discordances_gr_per_cluster_pair),
                                  assignment_gr=deepcopy(self.assignment_gr),
                                  assignment_vs=deepcopy(self.assignment_vs),
                                  agent_to_gr_cluster_assignments=deepcopy(
                                      self.agent_to_gr_cluster_assignments),
                                  agent_to_vs_cluster_assignments=deepcopy(
                                      self.agent_to_vs_cluster_assignments),
                                  aggregation_on_gr_scores=self.aggregation_on_gr_scores)
        clust.explored = bool(self.explored)
        clust.optimizer_state = deepcopy(self.optimizer_state)
        if hasattr(self, "n_training_steps"):
            clust.n_training_steps = int(self.n_training_steps)
        else:
            clust.n_training_steps = 0
        check_assignment_consistency(grounding_per_value_per_cluster=clust.grounding_per_value_per_cluster,
                                     value_system_network_per_cluster=clust.value_system_per_cluster,
                                     assignment_aid_to_gr_cluster=clust.agent_to_gr_cluster_assignments,
                                     assignment_aid_to_vs_cluster=clust.agent_to_vs_cluster_assignments,
                                     reward_models_per_aid=clust.reward_model_per_agent_id)

        return clust

    @property
    def L(self):
        # TODO: take into account inter cluster distances?, if they are 0 they are actually the same cluster...
        return sum(1 for c in self.assignment_vs if len(c) > 0)

    def active_vs_clusters(self):
        return [(i, len(self.assignment_vs[i])) for i in range(len(self.assignment_vs)) if len(self.assignment_vs[i]) > 0]

    def active_gr_clusters(self):
        return [[(i, len(self.assignment_gr[value_i][i])) for i in range(len(self.assignment_gr[value_i])) if len(self.assignment_gr[value_i][i]) > 0] for value_i in range(self.n_values)]

    @property
    def K(self):
        # TODO: take into account inter cluster distances?, if they are 0 they are actually the same cluster...
        return [sum(1 for c in self.assignment_gr[value_i] if len(c) > 0)for value_i in range(self.n_values)]

    @property
    def n_values(self):
        return len(self.assignment_gr)

    @property
    def vs_score(self):
        if self.inter_discordances_vs == float('inf') or self.L == 1:
            return self.representativity_vs(aggr=np.mean)
        else:
            return self.combined_cluster_score_vs()

    @property
    def gr_score(self):
        # TODO: for now, it is the average (or other aggregation) on the intra scores of the value-based clusters (accuracies)
        return self.combined_cluster_score_gr_aggr()

    def representativities_gr(self):

        return [ClusterAssignment._representativity(self.intra_discordances_gr_per_agent[i], self.assignment_gr[i]) for i in range(self.n_values)]

    def representativity_gr_aggr(self):
        return self.aggregation_on_gr_scores(self.representativities_gr())

    def representativity_vs(self, aggr=np.mean):
        return ClusterAssignment._representativity(self.intra_discordances_vs_per_agent, self.assignment_vs, aggr=aggr)

    def concisenesses_gr(self):
        return [ClusterAssignment._conciseness(np.array(self.inter_discordances_gr[i]), self.K[i]) for i in range(self.n_values)]

    def conciseness_gr_aggr(self):
        return self.aggregation_on_gr_scores(self.concisenesses_gr())

    def conciseness_vs(self):
        return ClusterAssignment._conciseness(self.inter_discordances_vs, self.L)

    def combined_cluster_score_gr(self, conciseness_if_K_is_1=None, normalized=False,  aggr_repr=np.mean):
        return [ClusterAssignment._combined_cluster_score(self.inter_discordances_gr[i], self.intra_discordances_gr_per_agent[i], self.K[i], self.assignment_gr[i], normalized=normalized, conciseness_if_1_cluster=conciseness_if_K_is_1[i] if conciseness_if_K_is_1 is not None else None, aggr_repr=aggr_repr) for i in range(self.n_values)]

    def combined_cluster_score_vs(self, conciseness_if_L_is_1=None, normalized=False,  aggr_repr=np.min):
        return ClusterAssignment._combined_cluster_score(self.inter_discordances_vs, self.intra_discordances_vs_per_agent, self.L, self.assignment_vs, normalized=normalized, conciseness_if_1_cluster=conciseness_if_L_is_1, aggr_repr=aggr_repr)

    def combined_cluster_score_gr_aggr(self, conciseness_if_K_is_1=None, normalized=False):
        # TODO FUTURE WORK aggregation of combined scores is this, or dividing the aggregation?
        return self.combined_cluster_score_gr(conciseness_if_K_is_1, normalized=normalized)

    def __str__(self):
        result = "Cluster Assignment:\n"
        result += "Grounding Clusters:\n"
        for vi, clusters in enumerate(self.assignment_gr):
            result += f"Value {vi}:\n"
            if self.K[vi] == 1:
                result += f"  Single GR Cluster: {[cix for cix in range(len(clusters)) if len(clusters[cix]) > 0][0]} \n"
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
            result += f"Representativity (Value System) MIN: {self.representativity_vs(aggr=np.min)}\n"
            result += f"Representativity (Value System) AVG: {self.representativity_vs(aggr=np.mean)}\n"
            result += f"Representativity (Value System) GLOBAL: {self.representativity_vs(aggr='weighted')}\n"
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
        one_grounding = (k_t == tuple(
            [1]*self.n_values) and k_t_other == tuple([1]*self.n_values))

        if l == 1 and l_other == 1 and one_grounding:
            return 1.0
        if self.n_agents != other.n_agents:
            raise ValueError(
                "Number of agents is different between the two cluster assignments. This needs a workaround.")

        # self.agent_distribution_vs() is a set of tuples. I want to use the edit distance to compare the two distributions
        total_differences = []

        a1 = self.agent_distribution_vs()
        a2 = other.agent_distribution_vs()
        min_total_edit_distance = float('inf')

        # Generate all possible permutations of clusters in a2
        a1_sorted = sorted(a1, key=len)
        a2_sorted = sorted(a2, key=len)
        min_total_edit_distance = sum(len(set(cluster1).symmetric_difference(
            set(cluster2))) for cluster1, cluster2 in zip(a1_sorted, a2_sorted))
        assert min_total_edit_distance <= 2*self.n_agents
        total_differences .append(min_total_edit_distance)
        if not one_grounding:
            gr_dists = self.agent_distribution_gr()
            gr_dists_other = other.agent_distribution_gr()
            for a1, a2 in zip(gr_dists, gr_dists_other):
                a1 = self.agent_distribution_vs()
                a2 = other.agent_distribution_vs()
                raise ValueError("Do the trick from the value systems")
                min_total_edit_distance = float('inf')
                for perm in permutations(a2):
                    total_edit_distance = 0
                    for cluster1, cluster2 in zip(a1, perm):
                        total_edit_distance += len(
                            set(cluster1).symmetric_difference(cluster2))
                min_total_edit_distance = min(
                    min_total_edit_distance, total_edit_distance)
                total_differences .append(min_total_edit_distance)
        diff = np.mean(1.0 - np.array(total_differences) / (2*self.n_agents))
        if diff == 1.0:
            assert self.is_equivalent_assignment(other)
        else:
            assert not self.is_equivalent_assignment(other)
        return diff  # TODO: separate grounding?

    def agent_distribution_gr(self):

        dist = [set(tuple(cluster) for cluster in self.assignment_gr[vi])
                for vi in range(len(self.assignment_gr))]
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

        self.maximum_grounding_coherence = [
            float('-inf') for _ in range(n_values)]

        self.last_selected_assignment = None
        self.initializing = True

    def __str__(self):

        self.sort_lexicographic(lexicographic_vs_first=True)
        result = "Cluster Assignment Memory:\n"
        mgr = self.maximum_conciseness_vs
        mgr_gr = self.maximum_conciseness_gr

        for i, assignment in enumerate(self.memory):
            result += f"Assignment {i} (Explored: {assignment.explored}, {assignment.n_training_steps if hasattr(assignment, 'n_training_steps') else 'unk'}):"
            result += f" VS: DI={assignment.combined_cluster_score_vs(conciseness_if_L_is_1=mgr, aggr_repr=np.min):1.4f}|RP={assignment.representativity_vs(aggr=np.min):.4f},RPav={assignment.representativity_vs(aggr=np.mean):.4f}|CN={assignment.conciseness_vs() if assignment.L > 1 else mgr:.4f}, GR: {[f"{float(g):.3f}" for g in assignment.combined_cluster_score_gr_aggr(
                conciseness_if_K_is_1=mgr_gr)]}, K: {assignment.K}, L: {assignment.L} \n"
            result += f" GR Clusters: {assignment.active_gr_clusters()}, VS Clusters: {assignment.active_vs_clusters()}\n"
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

        x_combined_per_value, y_combined_per_value = x.combined_cluster_score_gr(
            conciseness_if_K_is_1=mcgr), y.combined_cluster_score_gr(conciseness_if_K_is_1=mcgr)

        for i in range(x.n_values):
            if xK[i] == 1 and yK[i] == 1:
                has1 = True
            else:
                hasmorethan1 = True
            dif_gr_i = x_combined_per_value[i] - y_combined_per_value[i]
            difs.append(dif_gr_i)
            # we need to come up with something here. For ECAI we have 1 grounding always, so no problem yet
            assert not (has1 and hasmorethan1)
        # TODO... maybe aggregation on scores should be modelled outside these two?
        gr_score_dif = x.aggregation_on_gr_scores(difs)
        # pareto
        vs_score_dif = x.combined_cluster_score_vs(
            conciseness_if_L_is_1=mcvs) - y.combined_cluster_score_vs(conciseness_if_L_is_1=mcvs)
        conc_proxy = (self.maximum_conciseness_vs if self.maximum_conciseness_vs != float(
            '-inf') else 0.0)
        conc_dif = (x.conciseness_vs() if x.L > 1 else conc_proxy) - \
            (y.conciseness_vs() if y.L > 1 else conc_proxy)
        repr_dif = x.representativity_vs(
            aggr=np.min) - y.representativity_vs(np.min)
        l_diff = x.L - y.L
        # TODO: TEST PARETO TAKING INTO ACOUNT REPRESENTATIVITY TOO?

        pareto_score = 0.0
        lexic_diff = 0.0
        if (l_diff <= 0 and gr_score_dif > 0.0 and conc_dif >= 0.0 and repr_dif >= 0) or (l_diff <= 0 and gr_score_dif >= 0.0 and conc_dif > 0.0 and repr_dif >= 0) or (l_diff <= 0 and gr_score_dif >= 0.0 and conc_dif >= 0.0 and repr_dif > 0) or (l_diff < 0 and gr_score_dif >= 0.0 and conc_dif >= 0.0 and repr_dif >= 0):
            pareto_score = 1.0
        elif (l_diff >= 0 and gr_score_dif < 0.0 and conc_dif <= 0.0 and repr_dif <= 0) or (l_diff >= 0 and gr_score_dif <= 0.0 and conc_dif < 0.0 and repr_dif <= 0) or (l_diff >= 0 and gr_score_dif <= 0.0 and conc_dif <= 0.0 and repr_dif < 0) or (l_diff > 0 and gr_score_dif <= 0.0 and conc_dif <= 0.0 and repr_dif <= 0):
            pareto_score = -1.0
        else:
            pareto_score = 0.0

        if lexicographic_vs_first:

            if abs(vs_score_dif) > 0.00:
                lexic_diff = vs_score_dif
            else:
                lexic_diff = gr_score_dif
        else:
            if abs(gr_score_dif) > 0.00:
                lexic_diff = gr_score_dif
            else:
                lexic_diff = vs_score_dif
        return lexic_diff, pareto_score

    def insert_assignment(self, assignment: ClusterAssignment, sim_threshold=0.95) -> Tuple[int, ClusterAssignment]:

        self.update_maximum_conciseness(assignment)

        if all([asa.explored for asa in self.memory]) and len(self.memory) >= self.max_size:
            # self.clean_memory(exhaustive=True)
            for i in range(len(self.memory)):
                self.memory[i].explored = False

        # if it is 1, need to have only one.
        l_assignment_1 = assignment.L == 1
        k_assignment_all_1 = all(np.asarray(assignment.K) == 1)
        override_and_insert = False

        if l_assignment_1 and k_assignment_all_1:
            l1_exists = False
            for i in range(len(self.memory)):
                if self.memory[i].L == 1 and all(np.asarray(self.memory[i].K) == 1):
                    cmp_lexico, cmp_pareto = self.compare_assignments(
                        self.memory[i], assignment, lexicographic_vs_first=False,)
                    l1_exists = True
                    if self.memory[i].n_training_steps <= assignment.n_training_steps and (cmp_pareto <= 0 or cmp_lexico < 0):
                        self.memory[i] = assignment
                        self.memory[i].explored = False

            if not l1_exists:
                self.memory.append(assignment)
        elif self.initializing:
            self.memory.append(assignment)
            self.last_selected_assignment = None
            if len(self.memory) == self.max_size:
                self.initializing = False

            return
        else:  # general case.
            last_index = self.last_selected_assignment
            override_and_insert = False

            if last_index is None:
                override_and_insert = False
            # and #not any([self.memory[i].L == self.memory[last_index].L  for i in range(len(self.memory)) if i != last_index]):
            elif assignment.L != self.memory[last_index].L:
                override_and_insert = False
            else:
                # try to override the last one.
                cmp_lexico, cmp_pareto = self.compare_assignments(
                    self.memory[last_index], assignment, lexicographic_vs_first=False,)
                # self.memory[last_index].is_equivalent_assignment

                if (cmp_pareto < 0):
                    override_and_insert = True

            if override_and_insert:
                # del self.memory[last_index]
                self.memory[last_index] = assignment
                self.memory[last_index].explored = False
            else:
                # if last_index is not None: self.memory[last_index].explored = True
                self.memory.append(assignment)
                self.memory[-1].explored = False

        if len(self.memory) > self.max_size:
            self.clean_memory(exhaustive=False,
                              append_made=not override_and_insert)
            print("Memory cleaned")
        self.last_selected_assignment = None

        return

    def update_maximum_conciseness(self, assignment: ClusterAssignment):

        gr_diffs = np.array(self.maximum_grounding_coherence) - \
            np.array(assignment.gr_score)
        better_grounding_precondition = all(gr_diffs <= 0.0)
        if better_grounding_precondition:
            self.maximum_grounding_coherence = assignment.gr_score
        changes_made = False
        if better_grounding_precondition:
            if assignment.L > 1:
                new_max_c = max(self.maximum_conciseness_vs,
                                assignment.conciseness_vs())
                if new_max_c != self.maximum_conciseness_vs:
                    changes_made = True
                self.maximum_conciseness_vs = new_max_c

            for vi in range(len(assignment.K)):
                if assignment.K[vi] > 1:
                    new_max_c = max(assignment.concisenesses_gr()[
                                    vi], self.maximum_conciseness_gr[vi])
                    if new_max_c != self.maximum_conciseness_gr[vi]:
                        changes_made = True
                    self.maximum_conciseness_gr[vi] = new_max_c

        return changes_made

    def clean_memory(self, exhaustive=False, sim_threshold=0.95, append_made=False):
        if len(self.memory) < 3:
            return
        pareto_dominated_counts = [0] * len(self.memory)
        equivalent_assignments_counts = [0] * len(self.memory)
        similarity_index = [0] * len(self.memory)
        explored_and_pareto_dominated = [False] * len(self.memory)
        explored = [False] * len(self.memory)
        longevity = [0] * len(self.memory)
        grounding_score = [0] * len(self.memory)
        vs_score = [0] * len(self.memory)
        # Calculate pareto dominance and equivalence
        for i in reversed(list(range(len(self.memory)))):
            if self.memory[i].L == 1 and all(np.asarray(self.memory[i].K) == 1):
                pareto_dominated_counts[i] = 0
                equivalent_assignments_counts[i] = 0
                similarity_index[i] = 0
                continue
            if self.memory[i].explored:
                explored_and_pareto_dominated[i] = True
                explored[i] = True
            longevity[i] = self.memory[i].n_training_steps
            grounding_score[i] = np.mean(self.memory[i].combined_cluster_score_gr_aggr(
                conciseness_if_K_is_1=self.maximum_conciseness_gr))
            vs_score[i] = self.memory[i].combined_cluster_score_vs(
                conciseness_if_L_is_1=self.maximum_conciseness_vs)
            for j in range(len(self.memory)):
                if i != j:
                    _, cmp_pareto = self.compare_assignments(
                        self.memory[j], self.memory[i], lexicographic_vs_first=True)
                    sim = self.memory[i].cluster_similarity(self.memory[j])

                    similarity_index[i] += sim
                    if cmp_pareto > 0:
                        pareto_dominated_counts[i] += 1
                    else:
                        explored_and_pareto_dominated[i] = False
                    
                    if sim >= sim_threshold:
                        equivalent_assignments_counts[i] += 1

        if append_made:
            last_assignment_and_not_pareto = [1 if self.last_selected_assignment is not None and self.last_selected_assignment == i and self.compare_assignments(
                self.memory[-1], self.memory[self.last_selected_assignment])[1] >= 0 else 0 for i in range(len(self.memory))]
        else:
            last_assignment_and_not_pareto = [0] * len(self.memory)
        if (len(self.memory) > self.max_size) or (exhaustive and (max(pareto_dominated_counts) > 0 or sum(equivalent_assignments_counts) > 0)):
            # This is the elimination protocol.
            sorted_indices = sorted(list(range(len(self.memory))), key=lambda x: (
                equivalent_assignments_counts[x],
                similarity_index[x],
                # This tries to remove the last selected assignment if it is pareto equivalent or dominated by the last inserted.
                last_assignment_and_not_pareto[x],

                pareto_dominated_counts[x],

                # -longevity[x],
                -grounding_score[x],
                -vs_score[x]
            ), reverse=True)

            eiminated_index_in_sorted_indices = 0
            # Special cases guards
            best_sorted_indices_by_grounding_then_vs = [i[0] for i in sorted(enumerate(self.memory), key=lambda x: (
                np.mean(x[1].combined_cluster_score_gr_aggr(
                    conciseness_if_K_is_1=self.maximum_conciseness_gr)),
                x[1].combined_cluster_score_vs(
                    conciseness_if_L_is_1=self.maximum_conciseness_vs),
            ), reverse=True)]
            best_sorted_indices_by_vs_then_grounding = [i[0] for i in sorted(enumerate(self.memory), key=lambda x: (

                x[1].combined_cluster_score_vs(
                    conciseness_if_L_is_1=self.maximum_conciseness_vs),
                np.mean(x[1].combined_cluster_score_gr_aggr(
                    conciseness_if_K_is_1=self.maximum_conciseness_gr)),
            ), reverse=True)]

            print("Best sorted indices", best_sorted_indices_by_grounding_then_vs,
                  best_sorted_indices_by_vs_then_grounding)

            eiminated_index_in_sorted_indices = 0
            while eiminated_index_in_sorted_indices < len(self.memory):
                eliminated_index = sorted_indices[eiminated_index_in_sorted_indices]
                c1 = self.memory[eliminated_index].L == 1 and all(
                    np.asarray(self.memory[eliminated_index].K) == 1)
                c2 = best_sorted_indices_by_grounding_then_vs[0] == eliminated_index
                c3 = best_sorted_indices_by_vs_then_grounding[0] == eliminated_index

                if c1 or c2 or c3:
                    eiminated_index_in_sorted_indices += 1
                else:
                    break

            eliminated_index = sorted_indices[eiminated_index_in_sorted_indices % len(
                sorted_indices)]
            self.memory.pop(eliminated_index)
            
            if self.last_selected_assignment is not None:
                if self.last_selected_assignment == eliminated_index:
                    self.last_selected_assignment = None
                elif self.last_selected_assignment > eliminated_index:
                    self.last_selected_assignment -= 1
            if exhaustive:
                self.last_selected_assignment = None
                self.clean_memory(exhaustive=exhaustive,
                                  sim_threshold=sim_threshold, append_made=False)
        return

    def sort_lexicographic(self, lexicographic_vs_first=False):
        sorted_memory_with_indices = sorted(enumerate(self.memory), key=lambda x: cmp_to_key(
            lambda a, b: self.compare_assignments(a, b, lexicographic_vs_first=lexicographic_vs_first)[0])(x[1]), reverse=True)
        self.memory = [item[1] for item in sorted_memory_with_indices]
        new_indices = [item[0] for item in sorted_memory_with_indices]
        if self.last_selected_assignment is not None:
            self.last_selected_assignment = new_indices[self.last_selected_assignment]
        return new_indices

    def get_random_weighted_assignment(self, consider_only_unexplored=False, lexicographic_vs_first=True) -> ClusterAssignment:
        self.sort_lexicographic(lexicographic_vs_first=lexicographic_vs_first)
        if consider_only_unexplored:
            indices_selectable = [i for i, assignment in enumerate(
                self.memory) if not assignment.explored]
        else:
            indices_selectable = [i for i in range(len(self.memory))]

        if len(indices_selectable) == 0:
            self.clean_memory(exhaustive=False)
            for i, assignment in enumerate(self.memory):
                assignment.explored = False
            return self.memory[0]
        n = len(indices_selectable)
        weights = [2*(n-i)/(n*(n+1))
                   for i in range(n)]  # Linear rank selection Goldberg
        assignment_index = random.choices(
            indices_selectable, weights=weights, k=1)[0]
        assignment = self.memory[assignment_index]
        self.last_selected_assignment = assignment_index
        return self.assignment_with_env(assignment)

    def get_best_assignment(self, consider_only_unexplored=False, lexicographic_vs_first=True) -> ClusterAssignment:
        self.sort_lexicographic(lexicographic_vs_first=lexicographic_vs_first)
        indices_non_explored = [i for i, assignment in enumerate(
            self.memory) if not assignment.explored]
        if not consider_only_unexplored:
            return self.assignment_with_env(assignment=self.memory[0])
        if len(indices_non_explored) == 0:
            self.clean_memory(exhaustive=False)
            for i, assignment in enumerate(self.memory):
                assignment.explored = False
            return self.memory[0]

        assignment = self.memory[indices_non_explored[0]]
        self.last_selected_assignment = indices_non_explored[0]
        return self.assignment_with_env(assignment)

    def assignment_with_env(self, assignment: ClusterAssignment) -> ClusterAssignment:
        if self.common_env is not None:
            # assignment.get_remove_env()
            assignment.set_env(self.common_env)

        return assignment
