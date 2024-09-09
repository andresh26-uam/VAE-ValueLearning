import itertools
from matplotlib import pyplot as plt
import numpy as np

from src.me_irl_for_vsl import MaxEntropyIRLForVSL

def plot_learned_to_expert_policies_vsi(expert_policy, max_entropy_algo: MaxEntropyIRLForVSL, target_align_funcs_to_learned_align_funcs):
    fig, axes = plt.subplots(2, len(max_entropy_algo.vsi_target_align_funcs), figsize=(16, 8))
    for i, al in enumerate(max_entropy_algo.vsi_target_align_funcs):
        # Plot the first matrix
        lpol = max_entropy_algo.learned_policy_per_va.policy_per_va(target_align_funcs_to_learned_align_funcs[al])
        if len(lpol.shape) == 3:
            lpol = lpol[0,:,:]

        im1 = axes[0, i].imshow(lpol, cmap='viridis', vmin=0, vmax=1, interpolation='none', aspect=lpol.shape[1]/lpol.shape[0])
        axes[0,i].set_title(f'VSI: Predicted Policy Matrix {tuple([float("{0:.3f}".format(v)) for v in target_align_funcs_to_learned_align_funcs[al]])})')
        axes[0,i].set_xlabel('Action')
        axes[0,i].set_ylabel('State')
        fig.colorbar(im1, ax=axes[0,i], orientation='vertical', label='Value')

        # Plot the second matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        pol = expert_policy.policy_per_va(al)
        if len(pol.shape) == 3:
            pol = pol[0,:,:]

        im2 = axes[1,i].imshow(pol, cmap='viridis', interpolation='none', vmin=0, vmax=1, aspect=pol.shape[1]/pol.shape[0])
        axes[1,i].set_title(f'VSI: Real Policy Matrix {al}')
        axes[1,i].set_xlabel('Action')
        axes[1,i].set_ylabel('State')
        fig.colorbar(im2, ax=axes[1,i], orientation='vertical', label='Value')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_learned_to_expert_policies_vgl(expert_policy, max_entropy_algo):
    fig, axes = plt.subplots(len(max_entropy_algo.vgl_target_align_funcs), 2, figsize=(16, 8))
    for i, al in enumerate(max_entropy_algo.vgl_target_align_funcs):
        # Ensure that you do not exceed the number of subplots
        # Plot the first matrix
        lpol = max_entropy_algo.learned_policy_per_va.policy_per_va(al)
        if len(lpol.shape) == 3:
            lpol = lpol[0,:,:]

        im1 = axes[i, 0].imshow(lpol, cmap='viridis', vmin=0, vmax=1, interpolation='none', aspect=lpol.shape[1]/lpol.shape[0])
        axes[i, 0].set_title(f'Predicted Policy Matrix {al}')
        axes[i, 0].set_xlabel('Action')
        axes[i, 0].set_ylabel('State')
        fig.colorbar(im1, ax=axes[i, 0], orientation='vertical', label='Value')

        # Plot the second matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        pol = expert_policy.policy_per_va(al)
        if len(pol.shape) == 3:
            pol = pol[0,:,:]

        im2 = axes[i, 1].imshow(pol, cmap='viridis', interpolation='none', vmin=0, vmax=1, aspect=pol.shape[1]/pol.shape[0])
        axes[i, 1].set_title(f'Real Policy Matrix {al}')
        axes[i, 1].set_xlabel('Action')
        axes[i, 1].set_ylabel('State')
        fig.colorbar(im2, ax=axes[i, 1], orientation='vertical', label='Value')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_learned_and_expert_rewards(env_real, max_entropy_algo, learned_rewards_per_al_func, cmap='viridis',  vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None):
    targets = max_entropy_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else max_entropy_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(max_entropy_algo.vsi_target_align_funcs, max_entropy_algo.vgl_target_align_funcs)

    fig, axes = plt.subplots(2, len(targets), figsize=(16, 8))
    for i, al in enumerate(targets):
        # Plot the learned matrix
        learned_al = al if vsi_or_vgl=='vgl' else target_align_funcs_to_learned_align_funcs[al]
        im1 = axes[0,i].imshow(learned_rewards_per_al_func(al), cmap=cmap, interpolation='none', aspect=learned_rewards_per_al_func(al).shape[1]/learned_rewards_per_al_func(al).shape[0])
        axes[0,i].set_title(f'VSI: Predicted Reward Matrix {tuple([float("{0:.2f}".format(v)) for v in learned_al])}')
        axes[0,i].set_xlabel('Action')
        axes[0,i].set_ylabel('State')
        fig.colorbar(im1, ax=axes[0,i], orientation='vertical', label='Value')

        # Plot the expert matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        im2 = axes[1,i].imshow(env_real.reward_matrix_per_align_func(al), cmap=cmap, interpolation='none', aspect=learned_rewards_per_al_func(al).shape[1]/learned_rewards_per_al_func(al).shape[0])
        axes[1,i].set_title(f'VSI: Real Reward Matrix {al}')
        axes[1,i].set_xlabel('Action')
        axes[1,i].set_ylabel('State')
        fig.colorbar(im2, ax=axes[1,i], orientation='vertical', label='Value')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_learned_and_expert_occupancy_measures(env_real, max_entropy_algo, expert_policy, learned_rewards_per_al_func, cmap='viridis',  vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None):
    targets = max_entropy_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else max_entropy_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(max_entropy_algo.vsi_target_align_funcs, max_entropy_algo.vgl_target_align_funcs)

    fig, axes = plt.subplots(2, len(targets), figsize=(16, 8))
    for i, al in enumerate(targets):

        # Plot the first matrix
        learned_al = al if vsi_or_vgl=='vgl' else target_align_funcs_to_learned_align_funcs[al]
        ocs = np.transpose(max_entropy_algo.mce_occupancy_measures(reward_matrix=learned_rewards_per_al_func(al))[1][:,None])

        eocs = np.transpose(max_entropy_algo.mce_occupancy_measures(reward_matrix=env_real.reward_matrix_per_align_func(al))[1][:,None])

        im1 = axes[0, i].imshow(ocs, cmap='viridis', interpolation='none',aspect=env_real.state_dim/env_real.action_dim)
        axes[0, i].set_title(f'Learned Occupancies {tuple([float("{0:.3f}".format(v)) for v in learned_al])}')
        axes[0, i].set_xlabel('State')
        fig.colorbar(im1, ax=axes[0, i], orientation='horizontal', label='Value', )

        # Plot the second matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        im2 = axes[1, i].imshow(eocs, cmap='viridis', interpolation='none', aspect=env_real.state_dim/env_real.action_dim)
        axes[1, i].set_title(f'Real Occupancies {al}')
        axes[1, i].set_xlabel('State')
        fig.colorbar(im2, ax=axes[1, i], orientation='horizontal', label='Value')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()