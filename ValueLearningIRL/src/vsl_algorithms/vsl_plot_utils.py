import matplotlib.pyplot as plt
import itertools
from matplotlib import pyplot as plt
import numpy as np

from src.vsl_algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.vsl_algorithms.me_irl_for_vsl import MaxEntropyIRLForVSL, mce_partition_fh, mce_occupancy_measures
from src.vsl_policies import VAlignedDictDiscreteStateActionPolicyTabularMDP


def get_color_gradient(c1, c2, mix):
    """
    Given two hex colors, returns a color gradient corresponding to a given [0,1] value
    """
    c1_rgb = np.array(c1)
    c2_rgb = np.array(c2)

    return ((1-mix)*c1_rgb + (mix*c2_rgb))


def get_linear_combination_of_colors(keys, color_from_keys, mix_weights):

    return np.average(np.array([color_from_keys[key] for key in keys]), weights=np.array(mix_weights), axis=0)


def pad(array, length):
    print(array)
    new_arr = np.zeros((length,))
    new_arr[0:len(array)] = np.asarray(array)
    new_arr[len(array):] = array[-1]
    return new_arr


def plot_learning_curves(algo: BaseVSLAlgorithm, historic_metric, name_metric='Linf', name_method='test_lc', align_func_colors=lambda al: 'black'):
    plt.figure(figsize=(6, 6))
    plt.title(
        f"Learning curve for {name_metric}\nover {len(historic_metric)} repetitions.")
    plt.xlabel("Training Iteration")
    plt.ylabel(f"{name_metric}")

    for al in historic_metric[0].keys():
        if al not in algo.vgl_target_align_funcs:
            continue
        max_length = np.max([len(historic_metric[rep][al])
                            for rep in range(len(historic_metric))])

        vals = np.asarray([pad(historic_metric[rep][al], max_length)
                          for rep in range(len(historic_metric))])
        avg_vals = np.mean(vals, axis=0)
        std_vals = np.std(vals, axis=0)

        color = align_func_colors(al)
        plt.plot(avg_vals,
                 color=color,
                 label=f'{tuple([float("{0:.3f}".format(t)) for t in al])} Last: {float(avg_vals[-1]):0.2f}'
                 )
        # plt.plot(avg_grad_norms,color=align_func_colors.get(tuple(al), 'black'), label=f'Grad Norm: {float(avg_grad_norms[-1]):0.2f}'

        plt.fill_between([i for i in range(len(avg_vals))], avg_vals-std_vals,
                         avg_vals+std_vals, edgecolor=color, alpha=0.3, facecolor=color)
        # plt.fill_between([i for i in range(len(avg_grad_norms))], avg_grad_norms-std_grad_norms, avg_grad_norms+std_grad_norms,edgecolor='black',alpha=0.1, facecolor='black')

    plt.legend()
    plt.grid()
    plt.savefig(f'plots/Learning_curves_{name_method}.pdf')


def plot_learned_to_expert_policies(expert_policy, vsl_algo: MaxEntropyIRLForVSL, vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='mce_vsl_test', show=False, learnt_policy=None):
    targets = vsl_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs)
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

            lpol = learnt_policy.policy_per_va(learned_al)
        if len(lpol.shape) == 3:
            lpol = lpol[0, :, :]

        im1 = axesUp[i].imshow(lpol, cmap='viridis', vmin=0, vmax=1,
                               interpolation='nearest', aspect=lpol.shape[1]/lpol.shape[0])
        axesUp[i].set_title(
            f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel('Action')
        axesUp[i].set_ylabel(
            f'State\nSTD: {float("{0:.4f}".format(std_lpol)) if isinstance(learnt_policy, list) else 0.0}')

        # fig.colorbar(im1, ax=axesUp[i], orientation='vertical', label='Value')

        # Plot the second matrix
        # print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
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
    fig.savefig('results/' + namefig + '_policy_dif.pdf')
    # Show the plot
    if show:
        fig.show()


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


def plot_learned_and_expert_reward_pairs(vsl_algo, learned_rewards_per_al_func, vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='reward_pairs', show=False):
    # Select target alignment functions based on vsi_or_vgl
    targets = vsl_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs)

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

        if isinstance(learned_rewards_per_al_func, list):
            lr_per_round = [learned_rewards_per_al_func[j](
                al)() for j in range(len(learned_rewards_per_al_func))]
            learned_reward_al = np.mean(lr_per_round, axis=0)
            std_reward_al = np.mean(np.std(lr_per_round, axis=0))
        else:
            learned_reward_al = learned_rewards_per_al_func(al)()

        # Get expert rewards
        expert_reward_al = vsl_algo.env.reward_matrix_per_align_func(al)

        # Check shape consistency
        assert learned_reward_al.shape == expert_reward_al.shape, "Learned and expert rewards must have the same shape"

        # Flatten rewards for plotting as pairs
        learned_rewards_flat = learned_reward_al.flatten()
        expert_rewards_flat = expert_reward_al.flatten()

        # Remove outliers and filter values smaller than a threshold
        combined_mask = remove_outliers(
            expert_rewards_flat, learned_rewards_flat)
        learned_rewards_flat = learned_rewards_flat[combined_mask]
        expert_rewards_flat = expert_rewards_flat[combined_mask]

        combined_mask = filter_values(
            expert_rewards_flat, learned_rewards_flat, min_value=-100)
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
        ax.set_title(
            f'Original: {tuple([float("{0:.3f}".format(v)) for v in al])}\nLearned: {tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        ax.legend()

    # Remove unused axes if the number of targets is odd
    if num_targets % 2 != 0 and num_targets > 3:
        fig.delaxes(axes[-1])

    # Save the figure
    fig.savefig('results/' + namefig + '_reward_dif_correlation.pdf')

    # Show the plot if requested
    if show:
        plt.show()


def plot_learned_and_expert_rewards(vsl_algo, learned_rewards_per_al_func, cmap='viridis',  vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='mce_vsl_test', show=False):
    targets = vsl_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs)

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
        axesUp[i].set_title(
            f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel('Action')
        axesUp[i].set_ylabel(
            f'State\nSTD: {float("{0:.4f}".format(std_reward_al)) if isinstance(learned_rewards_per_al_func, list) else 0.0}')

        # Plot the expert matrix
        # print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
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
    fig.savefig('results/' + namefig + '_reward_dif.pdf')
    # Show the plot
    if show:
        fig.show()


def plot_learned_and_expert_occupancy_measures(vsl_algo: MaxEntropyIRLForVSL, expert_policy: VAlignedDictDiscreteStateActionPolicyTabularMDP, learned_rewards_per_al_func, cmap='viridis',  vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='mce_vsl_test', show=False):
    targets = vsl_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else vsl_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(
        vsl_algo.vsi_target_align_funcs, vsl_algo.vgl_target_align_funcs)

    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle(
        f'Predicted Occupancy Matrix ({vsi_or_vgl} - {namefig})')
    subfigs[1].suptitle(f'Real Occupancy Matrix ({vsi_or_vgl} - {namefig})')

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

        use_action_visitations = vsl_algo.reward_net.use_action or vsl_algo.reward_net.use_next_state

        if isinstance(learned_rewards_per_al_func, list):
            occupancies = []
            for j in range(len(learned_rewards_per_al_func)):
                occupancies.append(mce_occupancy_measures(
                    reward_matrix=learned_rewards_per_al_func[j](al)(),
                    discount=vsl_algo.discount,
                    deterministic=not vsl_algo.learn_stochastic_policy,
                    approximator_kwargs=vsl_algo.approximator_kwargs,
                    policy_approximator=vsl_algo.policy_approximator,
                    initial_state_distribution=vsl_algo.initial_state_distribution_test,
                    use_action_visitations=use_action_visitations)[1])
            learned_oc = np.mean(occupancies, axis=0)
            std_oc = np.mean(np.std(occupancies, axis=0))

        else:

            std_oc = 0.0
            learned_oc = mce_occupancy_measures(reward_matrix=learned_rewards_per_al_func(al)(),
                                                discount=vsl_algo.discount,
                                                deterministic=not vsl_algo.learn_stochastic_policy,
                                                approximator_kwargs=vsl_algo.approximator_kwargs,
                                                policy_approximator=vsl_algo.policy_approximator,
                                                initial_state_distribution=vsl_algo.initial_state_distribution_test,
                                                use_action_visitations=use_action_visitations)[1]
        ocs = np.transpose(learned_oc)

        _, _, assumed_expert_pi = mce_partition_fh(vsl_algo.env, discount=vsl_algo.discount,
                                                   reward=vsl_algo.env.reward_matrix_per_align_func(
                                                       al),
                                                   approximator_kwargs={
                                                       'value_iteration_tolerance': 0.00001, 'iterations': 1000},
                                                   policy_approximator=vsl_algo.policy_approximator, deterministic=not vsl_algo.expert_is_stochastic)

        eocs = np.transpose(mce_occupancy_measures(reward_matrix=vsl_algo.env.reward_matrix_per_align_func(al)(),
                                                   discount=vsl_algo.discount,
                                                   deterministic=not vsl_algo.learn_stochastic_policy,
                                                   approximator_kwargs=vsl_algo.approximator_kwargs,
                                                   policy_approximator=vsl_algo.policy_approximator,
                                                   initial_state_distribution=vsl_algo.initial_state_distribution_test,
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
        axesUp[i].set_title(
            f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel(
            f'State\nSTD: {float("{0:.4f}".format(std_oc)) if isinstance(learned_rewards_per_al_func, list) else 0.0}')
        if use_action_visitations:
            axesUp[i].set_ylabel('Act')

        # Plot the second matrix
        # print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
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
    fig.savefig('results/' + namefig + '_occupancy_dif.pdf')
    # Show the plot
    if show:
        fig.show()
