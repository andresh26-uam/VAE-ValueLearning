import csv
from functools import partial
import pprint
import matplotlib.pyplot as plt
import itertools
from matplotlib import cm, pyplot as plt
import numpy as np
import torch

from src.algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.algorithms.utils import  mce_partition_fh, mce_occupancy_measures
from src.policies.vsl_policies import VAlignedDictSpaceActionPolicy
import os


def get_color_gradient(c1, c2, mix):
    """
    Given two hex colors, returns a color gradient corresponding to a given [0,1] value
    """
    c1_rgb = np.array(c1)
    c2_rgb = np.array(c2)
    mix = torch.softmax(torch.tensor(np.array(mix)),dim=0).detach().numpy()
    return (mix[0]*c1_rgb + ((1-mix[0])*c2_rgb))


def get_linear_combination_of_colors(keys, color_from_keys, mix_weights):

    return np.average(np.array([color_from_keys[key] for key in keys]), 
                      weights=torch.softmax(torch.tensor(np.array(mix_weights)),dim=0).detach().numpy(), axis=0)


def pad(array, length):
    new_arr = np.zeros((length,))
    new_arr[0:len(array)] = np.asarray(array)
    if len(new_arr) > len(array):
        new_arr[len(array):] = array[-1]
    return new_arr

"""
def plot_learning_curves(algo: BaseVSLAlgorithm, historic_metric, name_metric='Linf', name_method='test_lc', align_func_colors=lambda al: 'black', ylim=(0.0,1.1), show=False, usecmap='viridis'):
    plt.figure(figsize=(6, 6))
    plt.title(
        f"Learning curve for {name_metric}\nover {len(historic_metric)} repetitions.")
    plt.xlabel("Training Iteration")
    plt.ylabel(f"{name_metric}")
    n_possible_al = len(historic_metric[0].keys())
    viridis = cm.get_cmap(usecmap, n_possible_al)  # Get 'viridis' colormap with number of AL strategies
    

    for idx, al in enumerate(historic_metric[0].keys()):
        if al not in historic_metric[0].keys() or len(historic_metric[0][al]) == 0 :
            continue
        if usecmap is None or (np.sum(al) == 1.0 and 1.0 in al):
            color = align_func_colors(al)
        else:
            color = viridis(idx / (n_possible_al - 1))

        max_length = np.max([len(historic_metric[rep][al])
                            for rep in range(len(historic_metric))])

        vals = np.asarray([pad(historic_metric[rep][al], max_length)
                          for rep in range(len(historic_metric))])
        avg_vals = np.mean(vals, axis=0)
        std_vals = np.std(vals, axis=0)

        #color = align_func_colors(al)
        plt.plot(avg_vals,
                 color=color,
                 label=f'{tuple([float("{0:.3f}".format(t)) for t in al])} Last: {float(avg_vals[-1]):0.2f}'
                 )
        # plt.plot(avg_grad_norms,color=align_func_colors.get(tuple(al), 'black'), label=f'Grad Norm: {float(avg_grad_norms[-1]):0.2f}'

        plt.fill_between([i for i in range(len(avg_vals))], avg_vals-std_vals,
                         avg_vals+std_vals, edgecolor=color, alpha=0.3, facecolor=color)
        # plt.fill_between([i for i in range(len(avg_grad_norms))], avg_grad_norms-std_grad_norms, avg_grad_norms+std_grad_norms,edgecolor='black',alpha=0.1, facecolor='black')
        
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.grid()
    plt.savefig(f'plots/Learning_curves_{name_method}.pdf')
    if show:
        plt.show()
    plt.close()"""


def plot_learned_to_expert_policies(expert_policy, vsl_algo: BaseVSLAlgorithm, vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='mce_vsl_test', show=False, learnt_policy=None, targets=None):
    targets = targets if targets is not None else (vsl_algo.vsi_target_align_funcs if vsi_or_vgl in ['vsi','sim'] else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs))
    learnt_policy = learnt_policy if learnt_policy is not None else vsl_algo.learned_policy_per_va

    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle(f'Predicted Policy Matrix ({vsi_or_vgl} - {namefig})')
    subfigs[1].suptitle(f'Real Reward Matrix ({vsi_or_vgl} - {namefig})')

    axesUp = subfigs[0].subplots(nrows=1, ncols=len(targets), sharey=True)
    axesDown = subfigs[1].subplots(nrows=1, ncols=len(targets), sharey=True)

    for i, al in enumerate(targets):
        # Plot the first matrix
        all_learned_al = al if vsi_or_vgl == 'vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
            target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]
        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]

        if isinstance(learnt_policy, list):
            if isinstance(target_align_funcs_to_learned_align_funcs, list):
                pol_per_round = [learnt_policy[j].policy_per_va(
                    all_learned_al[j]) for j in range(len(learnt_policy))]
            else:
                pol_per_round = [learnt_policy[j].policy_per_va(
                    all_learned_al) for j in range(len(learnt_policy))]
            lpol = np.mean(pol_per_round, axis=0)
            std_lpol = np.mean(np.std(pol_per_round, axis=0))
            # We plot the average policy and the average learned alignment function which may not correspond to each other directly.

        else:
            learned_al = al if vsi_or_vgl == 'vgl' else target_align_funcs_to_learned_align_funcs[
                al]
            
            lpol = learnt_policy.policy_per_va(learned_al if isinstance(learned_al[0],str) else learned_al)
        if len(lpol.shape) == 3:
            lpol = lpol[0, :, :]

        im1 = axesUp[i].imshow(lpol, cmap='viridis', vmin=0, vmax=1,
                               interpolation='nearest', aspect=lpol.shape[1]/lpol.shape[0])
        learned_al = learned_al[1] if isinstance(learned_al[0],str) else learned_al

        axesUp[i].set_title(
            f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel('Action')
        axesUp[i].set_ylabel(
            f'State\nSTD: {float("{0:.4f}".format(std_lpol)) if isinstance(learnt_policy, list) else 0.0}')

        
        # Plot the second matrix
        pol = expert_policy.policy_per_va(al)
        if len(pol.shape) == 3:
            pol = pol[0, :, :]

        im2 = axesDown[i].imshow(pol, cmap='viridis', interpolation='nearest',
                                 vmin=0, vmax=1, aspect=pol.shape[1]/pol.shape[0])

        axesDown[i].set_title(f'{al}')
        axesDown[i].set_xlabel('Action')
        axesDown[i].set_ylabel('State')

    subfigs[0].colorbar(im1, ax=axesUp, orientation='vertical',
                        label='State-Action Prob.')
    subfigs[1].colorbar(
        im2, ax=axesDown, orientation='vertical', label='State-Action Prob.')
    # Adjust layout to prevent overlap
    # fig.tight_layout()
    dirr = os.path.join('test_results', namefig)
    os.makedirs(dirr, exist_ok=True)
    fig.savefig('test_results/' + namefig + '_policy_dif.pdf')
    # Show the plot
    if show:
        fig.show()
        plt.show()
    plt.close()


def remove_outliers(data1, data2, threshold=1.1):
    """Remove outliers using the IQR method, applied to both datasets together."""
    # Compute IQR for both datasets
    q1_data1, q3_data1 = np.percentile(data1, [25, 75])
    iqr_data1 = q3_data1 - q1_data1
    lower_bound_data1 = q1_data1 - threshold * iqr_data1
    upper_bound_data1 = q3_data1 + threshold * iqr_data1

    q1_data2, q3_data2 = np.percentile(data2, [25, 75])
    iqr_data2 = q3_data2 - q1_data2
    lower_bound_data2 = q1_data2 - threshold * iqr_data2
    upper_bound_data2 = q3_data2 + threshold * iqr_data2

    # Create masks for both datasets
    mask_data1 = (data1 >= lower_bound_data1) & (data1 <= upper_bound_data1)
    mask_data2 = (data2 >= lower_bound_data2) & (data2 <= upper_bound_data2)

    # Combine the masks: only keep points that are not outliers in both datasets
    combined_mask = mask_data1 & mask_data2
    return combined_mask


def filter_values(data1, data2, min_value=-10):
    """Filter out values smaller than min_value in both datasets."""
    mask_data1 = data1 >= min_value
    mask_data2 = data2 >= min_value
    # Combine the masks: only keep points where both data1 and data2 are >= min_value
    combined_mask = mask_data1 & mask_data2
    return combined_mask


def plot_learned_and_expert_reward_pairs(vsl_algo: BaseVSLAlgorithm, learned_rewards_per_al_func, vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='reward_pairs', show=False, targets=None, trajs_sampled=None):
    # Select target alignment functions based on vsi_or_vgl
    targets = targets if targets is not None else (vsl_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs))

    num_targets = len(targets)

    # Adjust subplot grid based on the number of targets
    if num_targets > 3:
        cols = (num_targets + 1) // 2  # Calculate the number of rows needed
        rows = 2  # We want 2 columns
    else:
        rows = 1  # If there are 3 or fewer targets, use a single row
        cols = num_targets
        
    # Create figure and subplots with dynamic layout
    fig, axes = plt.subplots(rows, cols, figsize=(
        16, 8), constrained_layout=True)
    fig.suptitle(
        f'Reward Coincidence Plots ({vsi_or_vgl} - {namefig})', fontsize=16)
    fig.supylabel("Learned Rewards")
    fig.supxlabel("Ground Truth Rewards")

    # Flatten axes if needed (for consistency when accessing them)
    axes = axes.flatten() if num_targets > 1 else [axes]

    for i, al in enumerate(targets):
        ax = axes[i]  # Get the current subplot axis

        # Get learned rewards
        all_learned_al = al if vsi_or_vgl == 'vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
            target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]

        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]
        
        # TODO: put it callable learner reward?
        if hasattr(vsl_algo.policy, 'state_dim'):
            if isinstance(learned_rewards_per_al_func, list):
                
                
                lr_per_round = [learned_rewards_per_al_func[j](
                al)() for j in range(len(learned_rewards_per_al_func))]
                learned_reward_al = np.mean(lr_per_round, axis=0)
                
                std_reward_al = np.mean(np.std(lr_per_round, axis=0))
            else:
                learned_reward_al = learned_rewards_per_al_func(al)()
                #print("SH", np.asarray(learned_reward_al).shape)
                #exit(0)

            # Get expert rewards
            expert_reward_al = vsl_algo.env.reward_matrix_per_align_func(al)
            

            # Check shape consistency
            assert learned_reward_al.shape == expert_reward_al.shape, "Learned and expert rewards must have the same shape"

            # if sampling trajs, plot the aggregated rewards.
            if trajs_sampled is not None:
                traj_expert_rewards_al =  np.array([np.sum(expert_reward_al[trajs_sampled[j].obs[:-1], trajs_sampled[j].acts]) for j in range(len(trajs_sampled))])
                traj_learned_rewards_al = np.array([np.sum(learned_reward_al[trajs_sampled[j].obs[:-1], trajs_sampled[j].acts]) for j in range(len(trajs_sampled))])
            print(traj_learned_rewards_al[0:5])
            exit(0)
        else:
            # If the environment does not have state_dim, we assume it is a single value
            
            if isinstance(learned_rewards_per_al_func, list):
                lr_per_round = [learned_rewards_per_al_func[j](
                    al) for j in range(len(learned_rewards_per_al_func))]
                
            else:
                lr_per_round  = [learned_rewards_per_al_func(al), ]
            traj_expert_rewards_al = [0.0 for _ in range(len(lr_per_round))]
            traj_learned_rewards_al = [0.0 for _ in range(len(lr_per_round))]
            expert_reward_al = partial(vsl_algo.env.get_reward_per_align_func, align_func=al)
                
            
            for round_ in range(len(lr_per_round)):
                learned_reward_al = lr_per_round[round_]
                traj_expert_rewards_al[round_] = np.array(
                    [np.sum(
                        [float(expert_reward_al(state=s, action=a, next_state=ns, done=None)) for s,a,ns in zip(
                            trajs_sampled[j].obs[:-1], trajs_sampled[j].acts, trajs_sampled[j].obs[1:])])
                              for j in range(len(trajs_sampled))])
                traj_learned_rewards_al[round_] = np.array(
                    [np.sum(
                        [float(learned_reward_al(state=s, action=a, next_state=ns, done=None)) for s,a,ns in zip(
                            trajs_sampled[j].obs[:-1], trajs_sampled[j].acts, trajs_sampled[j].obs[1:])])
                              for j in range(len(trajs_sampled))])
            
            traj_learned_rewards_al = np.mean(traj_learned_rewards_al, axis=0)
            traj_expert_rewards_al = np.mean(traj_expert_rewards_al, axis=0)
            

        # Flatten rewards for plotting as pairs
        learned_rewards_flat = traj_learned_rewards_al.flatten()
        expert_rewards_flat = traj_expert_rewards_al.flatten()

        # Remove outliers and filter values smaller than a threshold
        """combined_mask = remove_outliers(
            expert_rewards_flat, learned_rewards_flat)
        learned_rewards_flat = learned_rewards_flat[combined_mask]
        expert_rewards_flat = expert_rewards_flat[combined_mask]"""

        combined_mask = filter_values(
            expert_rewards_flat, learned_rewards_flat, min_value=-1000)
        learned_rewards_flat = learned_rewards_flat[combined_mask]
        expert_rewards_flat = expert_rewards_flat[combined_mask]

        # Scatter plot for reward pairs
        ax.scatter(expert_rewards_flat, learned_rewards_flat,
                   color='blue', alpha=0.5, label='Reward Pairs')

        # Plot x = y line
        min_val = min(min(expert_rewards_flat), min(learned_rewards_flat))
        max_val = max(max(expert_rewards_flat), max(learned_rewards_flat))
        ax.plot([min_val, max_val], [min_val, max_val],
                color='red', linestyle='--', alpha=0.7, label='x=y')

        # Set labels and title for each subplot
        al = al[1] if isinstance(al[0],str) else al 
        learned_al = learned_al[1] if isinstance(learned_al[0],str) else learned_al
        if vsi_or_vgl != 'vgl':
            ax.set_title(
                f'Original: {tuple([float("{0:.3f}".format(v)) for v in al])}\nLearned: {tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        else:
            ax.set_title(
                f'VS aggregation: {tuple([float("{0:.3f}".format(v)) for v in al])}')
        
        ax.legend()

    # Remove unused axes if the number of targets is odd
    if num_targets % 2 != 0 and num_targets > 3:
        fig.delaxes(axes[-1])

    # Save the figure
    dirr = os.path.join('test_results', namefig)
    os.makedirs(dirr, exist_ok=True)
    fig.savefig(os.path.join(dirr, 'reward_dif_correlation.pdf'))

    # Show the plot if requested
    if show:
        fig.show()
        plt.show()
    plt.close()


def plot_learned_and_expert_rewards(vsl_algo, learned_rewards_per_al_func, cmap='viridis',  vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='mce_vsl_test', show=False, targets=None):
    targets = targets if targets is not None else (vsl_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs))

    # fig, axes = plt.subplots(2, len(targets), figsize=(16, 8))
    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle(f'Predicted Reward Matrix ({vsi_or_vgl} - {namefig})')
    subfigs[1].suptitle(f'Real Reward Matrix ({vsi_or_vgl} - {namefig})')

    axesUp = subfigs[0].subplots(nrows=1, ncols=len(targets), sharey=True)
    axesDown = subfigs[1].subplots(nrows=1, ncols=len(targets), sharey=True)
    for i, al in enumerate(targets):
        # Plot the learned matrix
        all_learned_al = al if vsi_or_vgl == 'vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
            target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]
        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]

        if isinstance(learned_rewards_per_al_func, list):
            lr_per_round = [learned_rewards_per_al_func[j](
                al)() for j in range(len(learned_rewards_per_al_func))]
            learned_reward_al = np.mean(lr_per_round, axis=0)
            std_reward_al = np.mean(np.std(lr_per_round, axis=0))
        else:
            learned_reward_al = learned_rewards_per_al_func(al)()
        im1 = axesUp[i].imshow(learned_reward_al, cmap=cmap, interpolation='nearest',
                               aspect=learned_reward_al.shape[1]/learned_reward_al.shape[0])
        learned_al = learned_al[1] if isinstance(learned_al[0],str) else learned_al

        axesUp[i].set_title(
            f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel('Action')
        axesUp[i].set_ylabel(
            f'State\nSTD: {float("{0:.4f}".format(std_reward_al)) if isinstance(learned_rewards_per_al_func, list) else 0.0}')

        # Plot the expert matrix
        im2 = axesDown[i].imshow(vsl_algo.env.reward_matrix_per_align_func(
            al), cmap=cmap, interpolation='nearest', aspect=learned_reward_al.shape[1]/learned_reward_al.shape[0])
        axesDown[i].set_title(f'{al}')
        axesDown[i].set_xlabel('Action')
        axesDown[i].set_ylabel('State')

    subfigs[0].colorbar(im1, ax=axesUp, orientation='vertical', label='Reward')
    subfigs[1].colorbar(
        im2, ax=axesDown, orientation='vertical', label='Reward')
    # subfigs[0].tight_layout(pad=3.0)
    # subfigs[1].tight_layout(pad=3.0)
    # Adjust layout to prevent overlap
    dirr = os.path.join('test_results', namefig)
    os.makedirs(dirr, exist_ok=True)
    fig.savefig(os.path.join(dirr, 'reward_dif.pdf'))
    # Show the plot
    if show:
        fig.show()
        plt.show()
    plt.close()

def plot_learned_and_expert_occupancy_measures(vsl_algo: BaseVSLAlgorithm, 
        learned_rewards_per_al_func,  vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, 
        namefig='mce_vsl_test', show=False, targets=None, assumed_expert_pi: VAlignedDictSpaceActionPolicy = None):
    targets = targets if targets is not None else (vsl_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs))

    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle(
        f'Predicted Occupancy Matrix ({vsi_or_vgl} - {namefig})')
    subfigs[1].suptitle(f'Real Occupancy Matrix ({vsi_or_vgl} - {namefig})')

    axesUp = subfigs[0].subplots(nrows=1, ncols=len(targets), sharey=True)
    axesDown = subfigs[1].subplots(nrows=1, ncols=len(targets), sharey=True)

    assert isinstance(vsl_algo.learned_policy_per_va, VAlignedDictSpaceActionPolicy)
    for i, al in enumerate(targets):

        # Plot the first matrix
        all_learned_al = al if vsi_or_vgl == 'vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
            target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]
        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]

        use_action_visitations = vsl_algo.reward_net.use_action or vsl_algo.reward_net.use_next_state

        if isinstance(learned_rewards_per_al_func, list):
            occupancies = []
            for j in range(len(learned_rewards_per_al_func)):
                occupancies.append(mce_occupancy_measures(
                    env=vsl_algo.env,
                    reward=learned_rewards_per_al_func[j](al)(),
                    pi = vsl_algo.learned_policy_per_va.policy_per_va(target_align_funcs_to_learned_align_funcs[al]),
                    discount=vsl_algo.discount,
                    deterministic=not vsl_algo.learn_stochastic_policy,
                    initial_state_distribution=vsl_algo.env.initial_state_dist,
                    use_action_visitations=use_action_visitations)[1])
            learned_oc = np.mean(occupancies, axis=0)
            std_oc = np.mean(np.std(occupancies, axis=0))

        else:

            std_oc = 0.0
            learned_oc = mce_occupancy_measures(env=vsl_algo.env,
                                                reward=learned_rewards_per_al_func(al)(),
                                                discount=vsl_algo.discount,
                                                pi=vsl_algo.learned_policy_per_va.policy_per_va(target_align_funcs_to_learned_align_funcs[al]),
                                                deterministic=not vsl_algo.learn_stochastic_policy,
                                                initial_state_distribution=vsl_algo.env.initial_state_dist,
                                                use_action_visitations=use_action_visitations)[1]
        ocs = np.transpose(learned_oc)

        

        eocs = np.transpose(mce_occupancy_measures(env=vsl_algo.env,
                                                   reward=vsl_algo.env.reward_matrix_per_align_func(al),
                                                   discount=vsl_algo.discount,
                                                   pi=assumed_expert_pi.policy_per_va(al),
                                                   deterministic=not vsl_algo.learn_stochastic_policy,
                                                   initial_state_distribution=vsl_algo.env.initial_state_dist,
                                                   use_action_visitations=use_action_visitations)[1])
        # eocs2 = np.transpose(vsl_algo.mce_occupancy_measures(pi=expert_policy.policy_per_va(al), deterministic=not vsl_algo.expert_is_stochastic,  use_action_visitations=use_action_visitations)[1])
        # eocs3 = np.transpose(vsl_algo.mce_occupancy_measures(pi=assumed_expert_pi, deterministic=not vsl_algo.expert_is_stochastic,  use_action_visitations=use_action_visitations)[1])

        # assert np.allclose(eocs3, eocs2)
        # assert np.allclose(eocs , eocs2)

        if not use_action_visitations:
            ocs = ocs[:, None]
            eocs = eocs[:, None]

        im1 = axesUp[i].imshow(ocs, cmap='viridis', interpolation='nearest',
                               aspect=vsl_algo.env.state_dim/vsl_algo.env.action_dim)
        learned_al = learned_al[1] if isinstance(learned_al[0],str) else learned_al
        axesUp[i].set_title(
            f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel(
            f'State\nSTD: {float("{0:.4f}".format(std_oc)) if isinstance(learned_rewards_per_al_func, list) else 0.0}')
        if use_action_visitations:
            axesUp[i].set_ylabel('Act')

        # Plot the second matrix
        im2 = axesDown[i].imshow(eocs, cmap='viridis', interpolation='nearest',
                                 aspect=vsl_algo.env.state_dim/vsl_algo.env.action_dim)
        axesDown[i].set_title(f'{al}')
        axesDown[i].set_xlabel('State')
        if use_action_visitations:
            axesDown[i].set_ylabel('Act')

    subfigs[0].colorbar(
        im1, ax=axesUp, orientation='vertical', label='Visitation Freq.')
    subfigs[1].colorbar(
        im2, ax=axesDown, orientation='vertical', label='Visitation Freq.')
    # Adjust layout to prevent overlap
    # fig.tight_layout()
    dirr = os.path.join('test_results', namefig)
    os.makedirs(dirr, exist_ok=True)
    fig.savefig(os.path.join(dirr, 'occupancy_dif.pdf'))
    # Show the plot
    if show:
        fig.show()
        plt.show()
    plt.close()


def compute_stats(data_dict, metric_name='acc'):
        ratios = []

        if isinstance(data_dict[list(data_dict.keys())[0]][metric_name][list(data_dict[list(data_dict.keys())[0]][metric_name].keys())[0]][0], dict):
            epsilons = data_dict[list(data_dict.keys())[0]][metric_name][list(data_dict[list(data_dict.keys())[0]][metric_name].keys())[0]][0].keys()
        
            means_per_al = {eps: {al: {} for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()} for eps in epsilons}
            stds_per_al = {eps: {al: {} for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()} for eps in epsilons}
            labels_per_al = {eps: {al: [] for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()} for eps in epsilons} 
            
            for ratio in sorted(data_dict.keys()):
                ratios.append(ratio)
                for eps in epsilons:
                    for al, values in data_dict[ratio][metric_name].items():
                        n = len(values)
                        values_for_eps = [v[eps] for v in values]
                        
                        means_per_al[eps][al][ratio] = np.mean(values_for_eps)
                        stds_per_al[eps][al][ratio] = np.std(values_for_eps)
                        labels_per_al[eps][al].append(f'{tuple([float("{0:.3f}".format(t)) for t in al])}')  # Convert the tuple key to a string for labeling
            #vector of means, stds and labels and number of repetitions per ratio.
            
            return ratios, means_per_al, stds_per_al, labels_per_al, n
        
        else:
            means_per_al = {al: {} for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()}
            stds_per_al = {al: {} for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()}
            labels_per_al = {al: [] for al in data_dict[list(data_dict.keys())[0]][metric_name].keys()}
            ratios = []
            
            for ratio in sorted(data_dict.keys()):
                ratios.append(ratio)
                for al, values in data_dict[ratio][metric_name].items():
                    n = len(values)
                    means_per_al[al][ratio] = np.mean(values)
                    stds_per_al[al][ratio] = np.std(values)
                    labels_per_al[al].append(f'{tuple([float("{0:.3f}".format(t)) for t in al])}')  # Convert the tuple key to a string for labeling
            #vector of means, stds and labels and number of repetitions per ratio.
            
            return ratios, means_per_al, stds_per_al, labels_per_al, n

def plot_vs_preference_metrics(metrics_per_ratio, namefig='test_metrics', show=False,
                                        align_func_colors=None,
                                        usecmap = 'viridis',
                                        value_expectations_per_ratio=None,
                                        value_expectations_per_ratio_expert=None,
                                        target_align_funcs_to_learned_align_funcs = None,
                                        values_names = None,
                                    ):
    
    # Compute stats for 'f1' and 'ce'
    #ratios, f1_means, f1_stds, f1_labels, n = compute_stats(metrics_per_ratio, 'f1')
    #ratios, jsd_means, jsd_stds, ce_labels, n = compute_stats(metrics_per_ratio, 'jsd')
    ratios, acc_means_per_epsilon, acc_stds_per_epsilon, acc_labels_per_epsilon, n = compute_stats(metrics_per_ratio, 'acc')
    _, n_repescados_means_per_epsilon, n_repescados_stds_per_epsilon, _, n = compute_stats(metrics_per_ratio, 'repescados')
    _, jsd_means_per_epsilon, jsd_stds_per_epsilon, jsd_labels_per_epsilon, n = compute_stats(metrics_per_ratio, 'jsd')
    
    # Plot 'acc'
    namefig_base = namefig
    for eps in acc_means_per_epsilon.keys():
        namefig = namefig_base +f'_EPS{eps}_'
        plt.figure(figsize=(16, 8))
        
        viridis = cm.get_cmap(usecmap, len(acc_means_per_epsilon[eps]))  # Get 'viridis' colormap with number of AL strategies
        target_al_func_ro_mean_align_func = None

        for idx, al in enumerate(acc_means_per_epsilon[eps].keys()):
            if usecmap is None or (np.sum(al) == 1.0 and 1.0 in al):
                color = align_func_colors(al)
            else:
                color = viridis(idx / (len(acc_means_per_epsilon[eps]) - 1))

            if target_align_funcs_to_learned_align_funcs is not None:
                if target_al_func_ro_mean_align_func is None:
                    target_al_func_ro_mean_align_func = {}
                all_learned_al = [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
                    target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]

                if isinstance(target_align_funcs_to_learned_align_funcs, list):
                    learned_al = np.mean(all_learned_al, axis=0)
                    std_learned_al = np.std(all_learned_al, axis=0)
                else:
                    learned_al = all_learned_al
                    std_learned_al = [0 for _ in learned_al]
                
                orig_al = tuple([float("{0:.3f}".format(v)) for v in al])
                std_learned_al = tuple([float("{0:.3f}".format(v)) for v in std_learned_al])
                learned_al = tuple([float("{0:.3f}".format(v)) for v in learned_al])
                label = f'Original: {orig_al}\nLearned: {learned_al}\nSTD: {std_learned_al}'
                target_al_func_ro_mean_align_func[al] = learned_al
            else:
                label = f'Target al: {tuple([float("{0:.3f}".format(v)) for v in al])}'
            assert ratios == sorted(ratios)
            plt.errorbar(ratios, [acc_means_per_epsilon[eps][al][r] for r in ratios], yerr=[acc_stds_per_epsilon[eps][al][r] for r in ratios], label=label
                        , capsize=5, marker='o',color=color,ecolor=color)

        plt.title(f'Avg. preference comparisons accuracy over {n} runs (tol: {eps})')
        #plt.ylabel('F1 score (weighted)')
        plt.ylabel('Accuracy')
        plt.xlabel('Proportion of pairs of trajectories chosen by the expert versus randomly')
        plt.ylim((0.0,1.1))
        
        
        plt.legend(title="VS aggregation function", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        #plt.savefig('results/'+ 'f1_score_' + namefig + f'_{n}_runs.pdf')
        dirr = os.path.join('test_results', namefig)
        os.makedirs(dirr, exist_ok=True)
        plt.savefig(os.path.join(dirr, 'acc_score_' + f'_{n}_runs.pdf'))
        # Show the plot
        if show:
            plt.show()

        plt.figure(figsize=(16, 8))
        

        for idx, al in enumerate(jsd_means_per_epsilon[eps].keys()):

            if usecmap is None or (np.sum(al) == 1.0 and 1.0 in al):
                color = align_func_colors(al)
            else:
                color = viridis(idx / (len(jsd_means_per_epsilon[eps]) - 1))


            if target_align_funcs_to_learned_align_funcs is not None:
                all_learned_al = [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(
                    target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al]

                if isinstance(target_align_funcs_to_learned_align_funcs, list):
                    learned_al = np.mean(all_learned_al, axis=0)
                    std_learned_al = np.std(all_learned_al, axis=0)
                else:
                    learned_al = all_learned_al
                    std_learned_al = [0 for _ in learned_al]

                orig_al = tuple([float("{0:.3f}".format(v)) for v in al])
                std_learned_al = tuple([float("{0:.3f}".format(v)) for v in std_learned_al])
                learned_al = tuple([float("{0:.3f}".format(v)) for v in learned_al])
                label = f'Original: {orig_al}\nLearned: {learned_al}\nSTD: {std_learned_al}'
            else:
                label = f'Target al: {tuple([float("{0:.3f}".format(v)) for v in al])}'
            assert ratios == sorted(ratios)
            plt.errorbar(ratios,[jsd_means_per_epsilon[eps][al][r] for r in ratios], yerr=[jsd_stds_per_epsilon[eps][al][r] for r in ratios], label=label
                        , capsize=5, marker='o',color=color,ecolor=color)

        plt.title(f'Avg. Jensen Shannon div. over {n} runs')
        plt.ylabel('JSD')
        #plt.title(f'Avg. number of reinstated pairs: negative pairs treated as positives due to the tolerance {eps} over {n} runs')
        plt.xlabel('Proportion of pairs of trajectories chosen by the expert versus randomly')
        plt.legend(title="VS aggregation function", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        #plt.savefig('results/' + 'jsd_score_' + namefig + f'_{n}_runs.pdf')
        dirr = os.path.join('test_results', namefig)
        os.makedirs(dirr, exist_ok=True)
        plt.savefig(os.path.join(dirr, 'jsd_score_' + f'_{n}_runs.pdf'))
        # Show the plot
        if show:
            plt.show()
            
        plt.close()

        save_stats_to_csv_and_latex(acc_means_per_epsilon[eps], acc_stds_per_epsilon[eps], jsd_means_per_epsilon[eps], jsd_stds_per_epsilon[eps], n_repescados_means_per_epsilon[eps], n_repescados_stds_per_epsilon[eps],  acc_labels_per_epsilon[eps], namefig, n, value_expectations_per_ratio, value_expectations_per_ratio_expert, values_names, target_align_funcs_to_learned_align_funcs=target_al_func_ro_mean_align_func)


def save_stats_to_csv_and_latex(metric1_means, metric1_stds, metric2_means, metric2_stds, metric3_means, metric3_stds, labels, namefig, n, value_expectations_per_ratio, value_expectations_per_ratio_expert, values_names, target_align_funcs_to_learned_align_funcs=None):
    # File names
    csv_file_1 = f'results/tables/{namefig}_{n}_runs_metrics_table.csv'
    csv_file_2 = f'results/tables/{namefig}_{n}_runs_expected_alignments_table.csv'
    latex_file_1 = f'results/tables/{namefig}_{n}_runs_metrics_table.tex'
    latex_file_2 = f'results/tables/{namefig}_{n}_runs_expected_alignments_table.tex'

    # Process value expectations for the learned and expert policies
    value_expectations_learned = value_expectations_per_ratio
    value_expectations_expert = value_expectations_per_ratio_expert

    # Initialize data rows for each table
    metrics_rows = []
    expected_alignments_rows = []

    for target_vs_function in labels:
        # Gather F1 and JSD stats for metrics table
        m1_at_0 = f'{metric1_means[target_vs_function][0.0]:.3f} ± {metric1_stds[target_vs_function][0.0]:.2f}'
        m1_at_1 = f'{metric1_means[target_vs_function][1.0]:.3f} ± {metric1_stds[target_vs_function][1.0]:.2f}'
        m2_at_0 = f'{metric2_means[target_vs_function][0.0]:.2e} ± {metric2_stds[target_vs_function][0.0]:.2e}'
        m2_at_1 = f'{metric2_means[target_vs_function][1.0]:.2e} ± {metric2_stds[target_vs_function][1.0]:.2e}'

        m3_at_0 = f'{metric3_means[target_vs_function][0.0]:.2e} ± {metric3_stds[target_vs_function][0.0]:.2e}'
        m3_at_1 = f'{metric3_means[target_vs_function][1.0]:.2e} ± {metric3_stds[target_vs_function][1.0]:.2e}'

        # Prepare metrics row with conditional "Learned VS" column
        metrics_row = [str(target_vs_function), m1_at_0, m1_at_1, m2_at_0, m2_at_1,m3_at_0, m3_at_1]
        if target_align_funcs_to_learned_align_funcs:
            learned_vs = target_align_funcs_to_learned_align_funcs[target_vs_function]
            metrics_row.insert(1, learned_vs)  # Insert "Learned VS" as the second column
        metrics_rows.append(metrics_row)

        # Prepare expert and learned policy averages for policy table
        expert_values = []
        learned_values = []
        for alb in values_names.keys():
            #expert_data = [(np.mean(data_rep[alb])-np.mean([d[alb] for d in value_expectations_expert[alb]]))/np.std([d[alb] for d in value_expectations_expert[alb]]) for data_rep in value_expectations_expert[target_vs_function]]
            #learned_data = [(np.mean(data_rep[alb])-np.mean([d[alb] for d in value_expectations_expert[alb]]))/np.std([d[alb] for d in value_expectations_expert[alb]]) for data_rep in value_expectations_learned[target_vs_function]]
            expert_data = [data_rep[alb] for data_rep in value_expectations_expert[target_vs_function]] 
            learned_data = [data_rep[alb] for data_rep in value_expectations_learned[target_vs_function]]
            
            expert_values.append(f'{np.mean(expert_data):.3f} ± {np.std(expert_data):.3f}')
            learned_values.append(f'{np.mean(learned_data):.3f} ± {np.std(learned_data):.3f}')
        
        expected_alignments_row = [str(target_vs_function), *expert_values, *learned_values]
        if target_align_funcs_to_learned_align_funcs:
            expected_alignments_row.insert(1, str(learned_vs))  # Insert "Learned VS" as the second column
        expected_alignments_rows.append(expected_alignments_row)
        

    # Write metrics table to CSV
    with open(csv_file_1, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['VS Function']
        if target_align_funcs_to_learned_align_funcs:
            header.append('Learned VS')
        #header.extend(['F1 (random)', 'F1 (expert)', 'JSD (random)', 'JSD (expert)'])
        header.extend(['Acc (random)', 'Acc (expert)', 'JSD (random)', 'JSD (expert)', '#Reins (random)', '#Reins (expert)'])
        
        writer.writerow(header)
        writer.writerows(metrics_rows)

    # Write policy table to CSV
    with open(csv_file_2, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['VS Function']
        if target_align_funcs_to_learned_align_funcs:
            header.append('Learned VS')
        header.extend([f'Expert Policy Avg. \A_{{{v}}}' for v in values_names.values()])
        header.extend([f'Learned Policy Avg. \A_{{{v}}}' for v in values_names.values()])
        writer.writerow(header)
        writer.writerows(expected_alignments_rows)

    # Write metrics table to LaTeX
    with open(latex_file_1, 'w') as f:
        f.write('\\begin{table}[ht]\n')
        f.write('\\centering\n')
        f.write(f'\\caption{{Metrics Results for {namefig} over {n} runs}}\n')
        col_format = '|l|'
        if target_align_funcs_to_learned_align_funcs:
            col_format += 'l|'
        col_format += 'c|c|c|c|'
        f.write(f'\\begin{{tabular}}{{{col_format}}}\n')
        f.write('\\hline\n')
        header_latex = 'VS Function'
        if target_align_funcs_to_learned_align_funcs:
            header_latex += ' & Learned VS'
        #header_latex += ' & F1 (random) & F1 (expert) & JSD (random) & JSD (expert)'
        header_latex += ' & Acc (random) & Acc (expert) & JSD (random) & JSD (expert) & #Reins (random) & #Reins (expert)'
        f.write(header_latex + ' \\\\\n')
        f.write('\\hline\n')
        for row in metrics_rows:
            f.write(' & '.join([str(v) for v in row]) + ' \\\\\n')
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')

    # Write policy table to LaTeX
    with open(latex_file_2, 'w') as f:
        f.write('\\begin{table}[ht]\n')
        f.write('\\centering\n')
        f.write(f'\\caption{{Policy Averages for {namefig} over {n} runs}}\n')
        col_format = '|l|'
        if target_align_funcs_to_learned_align_funcs:
            col_format += 'l|'
        col_format += 'c|' * (len(values_names) * 2)
        f.write(f'\\begin{{tabular}}{{{col_format}}}\n')
        f.write('\\hline\n')
        header_latex = 'VS Function'
        if target_align_funcs_to_learned_align_funcs:
            header_latex += ' & Learned VS'
        header_latex += ''.join([f' & Expert Policy Avg. $\\A_{{{v}}}$' for v in values_names.values()])
        header_latex += ''.join([f' & Learned Policy Avg. $\\A_{{{v}}}$' for v in values_names.values()])
        f.write(header_latex + ' \\\\\n')
        f.write('\\hline\n')
        for row in expected_alignments_rows:
            f.write(' & '.join([str(v) for v in row]) + ' \\\\\n')
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')

    print(f"Results saved in CSV: {csv_file_1}, {csv_file_2} and LaTeX: {latex_file_1}, {latex_file_2}")