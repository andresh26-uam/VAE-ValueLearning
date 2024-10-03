import itertools
from typing import Iterable
from matplotlib import pyplot as plt
import numpy as np

from src.envs.tabularVAenv import TabularVAPOMDP
from src.me_irl_for_vsl import MaxEntropyIRLForVSL, mce_partition_fh
from src.vsl_policies import VAlignedDictDiscreteStateActionPolicyTabularMDP


def plot_learned_to_expert_policies(expert_policy, max_entropy_algo: MaxEntropyIRLForVSL, vsi_or_vgl = 'vsi', target_align_funcs_to_learned_align_funcs=None, namefig='mce_vsl_test', show=False, learnt_policy=None):
    targets = max_entropy_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else max_entropy_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(max_entropy_algo.vsi_target_align_funcs, max_entropy_algo.vgl_target_align_funcs)
    learnt_policy = learnt_policy if learnt_policy is not None else max_entropy_algo.learned_policy_per_va
    
    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle(f'Predicted Policy Matrix ({vsi_or_vgl} - {namefig})')
    subfigs[1].suptitle(f'Real Reward Matrix ({vsi_or_vgl} - {namefig})')

    axesUp = subfigs[0].subplots(nrows=1, ncols=len(targets), sharey=True)
    axesDown = subfigs[1].subplots(nrows=1, ncols=len(targets), sharey=True)
    
    for i, al in enumerate(targets):
        # Plot the first matrix
        all_learned_al = al if vsi_or_vgl=='vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al] 
        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]

        if isinstance(learnt_policy, list):
            if isinstance(target_align_funcs_to_learned_align_funcs, list):
                pol_per_round = [learnt_policy[j].policy_per_va(all_learned_al[j]) for j in range(len(learnt_policy))]
            else:
                pol_per_round = [learnt_policy[j].policy_per_va(all_learned_al) for j in range(len(learnt_policy))]
            lpol = np.mean(pol_per_round, axis=0)
            std_lpol = np.mean(np.std(pol_per_round, axis=0))
            # We plot the average policy and the average learned alignment function which may not correspond to each other directly.

        else:
            learned_al = al if vsi_or_vgl=='vgl' else target_align_funcs_to_learned_align_funcs[al]
            
            lpol = learnt_policy.policy_per_va(learned_al)
        if len(lpol.shape) == 3:
            lpol = lpol[0,:,:]

        im1 = axesUp[i].imshow(lpol, cmap='viridis', vmin=0, vmax=1, interpolation='nearest', aspect=lpol.shape[1]/lpol.shape[0])
        axesUp[i].set_title(f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel('Action')
        axesUp[i].set_ylabel(f'State\nSTD: {float("{0:.4f}".format(std_lpol)) if isinstance(learnt_policy, list) else 0.0}')

        #fig.colorbar(im1, ax=axesUp[i], orientation='vertical', label='Value')

        # Plot the second matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        pol = expert_policy.policy_per_va(al)
        if len(pol.shape) == 3:
            pol = pol[0,:,:]

        im2 = axesDown[i].imshow(pol, cmap='viridis', interpolation='nearest', vmin=0, vmax=1, aspect=pol.shape[1]/pol.shape[0])
        
        axesDown[i].set_title(f'{al}')
        axesDown[i].set_xlabel('Action')
        axesDown[i].set_ylabel('State')
        
    subfigs[0].colorbar(im1, ax=axesUp, orientation='vertical', label='State-Action Prob.')
    subfigs[1].colorbar(im2, ax=axesDown, orientation='vertical', label='State-Action Prob.')
    # Adjust layout to prevent overlap
    #fig.tight_layout()
    fig.savefig('results/' + namefig + '_policy_dif.pdf')
    # Show the plot
    if show:
        fig.show()


def plot_learned_and_expert_rewards(env_real, max_entropy_algo, learned_rewards_per_al_func, cmap='viridis',  vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='mce_vsl_test',show=False):
    targets = max_entropy_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else max_entropy_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(max_entropy_algo.vsi_target_align_funcs, max_entropy_algo.vgl_target_align_funcs)

    #fig, axes = plt.subplots(2, len(targets), figsize=(16, 8))
    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle(f'Predicted Reward Matrix ({vsi_or_vgl} - {namefig})')
    subfigs[1].suptitle(f'Real Reward Matrix ({vsi_or_vgl} - {namefig})')

    axesUp = subfigs[0].subplots(nrows=1, ncols=len(targets), sharey=True)
    axesDown = subfigs[1].subplots(nrows=1, ncols=len(targets), sharey=True)
    for i, al in enumerate(targets):
        # Plot the learned matrix
        all_learned_al = al if vsi_or_vgl=='vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al] 
        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]

        if isinstance(learned_rewards_per_al_func, list):
            lr_per_round = [learned_rewards_per_al_func[j](al) for j in range(len(learned_rewards_per_al_func))]
            learned_reward_al = np.mean(lr_per_round, axis=0)
            std_reward_al = np.mean(np.std(lr_per_round, axis=0))
        else:
            learned_reward_al = learned_rewards_per_al_func(al)
        im1 = axesUp[i].imshow(learned_reward_al, cmap=cmap, interpolation='nearest', aspect=learned_reward_al.shape[1]/learned_reward_al.shape[0])
        axesUp[i].set_title(f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel('Action')
        axesUp[i].set_ylabel(f'State\nSTD: {float("{0:.4f}".format(std_reward_al)) if isinstance(learned_rewards_per_al_func, list) else 0.0}')
        

        # Plot the expert matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        im2 = axesDown[i].imshow(env_real.reward_matrix_per_align_func(al), cmap=cmap, interpolation='nearest', aspect=learned_reward_al.shape[1]/learned_reward_al.shape[0])
        axesDown[i].set_title(f'{al}')
        axesDown[i].set_xlabel('Action')
        axesDown[i].set_ylabel('State')
        
    subfigs[0].colorbar(im1, ax=axesUp, orientation='vertical', label='Reward')
    subfigs[1].colorbar(im2, ax=axesDown, orientation='vertical', label='Reward')
    #subfigs[0].tight_layout(pad=3.0)
    #subfigs[1].tight_layout(pad=3.0)
    # Adjust layout to prevent overlap
    fig.savefig('results/' + namefig + '_reward_dif.pdf')
    # Show the plot
    if show:
        fig.show()


def plot_learned_and_expert_occupancy_measures(env_real: TabularVAPOMDP, max_entropy_algo: MaxEntropyIRLForVSL, expert_policy: VAlignedDictDiscreteStateActionPolicyTabularMDP, learned_rewards_per_al_func, cmap='viridis',  vsi_or_vgl='vsi', target_align_funcs_to_learned_align_funcs=None, namefig='mce_vsl_test', show=False):
    targets = max_entropy_algo.vsi_target_align_funcs if vsi_or_vgl == 'vsi' else max_entropy_algo.vgl_target_align_funcs if vsi_or_vgl == 'vgl' else itertools.chain(max_entropy_algo.vsi_target_align_funcs, max_entropy_algo.vgl_target_align_funcs)

    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle(f'Predicted Occupancy Matrix ({vsi_or_vgl} - {namefig})')
    subfigs[1].suptitle(f'Real Occupancy Matrix ({vsi_or_vgl} - {namefig})')

    axesUp = subfigs[0].subplots(nrows=1, ncols=len(targets), sharey=True)
    axesDown = subfigs[1].subplots(nrows=1, ncols=len(targets), sharey=True)

    for i, al in enumerate(targets):

        # Plot the first matrix
        all_learned_al = al if vsi_or_vgl=='vgl' else [ta_to_la[al] for ta_to_la in target_align_funcs_to_learned_align_funcs] if isinstance(target_align_funcs_to_learned_align_funcs, list) else target_align_funcs_to_learned_align_funcs[al] 
        if isinstance(target_align_funcs_to_learned_align_funcs, list):
            learned_al = np.mean(all_learned_al, axis=0)
            std_learned_al = np.std(all_learned_al, axis=0)
        else:
            learned_al = all_learned_al
            std_learned_al = [0 for _ in learned_al]

        use_action_visitations=max_entropy_algo.reward_net.use_action or max_entropy_algo.reward_net.use_next_state
        
        if isinstance(learned_rewards_per_al_func, list):
            occupancies = [] 
            for j in range(len(learned_rewards_per_al_func)):
                occupancies.append(max_entropy_algo.mce_occupancy_measures(reward_matrix=learned_rewards_per_al_func[j](al), deterministic=not max_entropy_algo.learn_stochastic_policy, use_action_visitations=use_action_visitations)[1])
            learned_oc = np.mean(occupancies, axis=0)
            std_oc = np.mean(np.std(occupancies, axis=0))
            
        else:
            
            std_oc = 0.0
            learned_oc = max_entropy_algo.mce_occupancy_measures(reward_matrix=learned_rewards_per_al_func(al), deterministic=not max_entropy_algo.learn_stochastic_policy, use_action_visitations=use_action_visitations)[1]
        ocs = np.transpose(learned_oc)
        
        _,_, assumed_expert_pi = mce_partition_fh(env_real, discount=max_entropy_algo.discount,
                                             reward=env_real.reward_matrix_per_align_func(al),
                                             approximator_kwargs={'value_iteration_tolerance': 0.00001, 'iterations': 1000},
                                             policy_approximator=max_entropy_algo.policy_approximator,deterministic= not max_entropy_algo.expert_is_stochastic )
        
        eocs = np.transpose(max_entropy_algo.mce_occupancy_measures(reward_matrix=env_real.reward_matrix_per_align_func(al), deterministic=not max_entropy_algo.expert_is_stochastic,  use_action_visitations=use_action_visitations)[1])
        eocs2 = np.transpose(max_entropy_algo.mce_occupancy_measures(pi=expert_policy.policy_per_va(al), deterministic=not max_entropy_algo.expert_is_stochastic,  use_action_visitations=use_action_visitations)[1])
        eocs3 = np.transpose(max_entropy_algo.mce_occupancy_measures(pi=assumed_expert_pi, deterministic=not max_entropy_algo.expert_is_stochastic,  use_action_visitations=use_action_visitations)[1])
        
        assert np.allclose(eocs3, eocs2)
        assert np.allclose(eocs , eocs2) 

        if not use_action_visitations:
            ocs = ocs[:, None]
            eocs = eocs[:, None]

        im1 = axesUp[i].imshow(ocs, cmap='viridis', interpolation='nearest',aspect=env_real.state_dim/env_real.action_dim)
        axesUp[i].set_title(f'{tuple([float("{0:.3f}".format(v)) for v in learned_al])}\nSTD: {tuple([float("{0:.3f}".format(v)) for v in std_learned_al])}')
        axesUp[i].set_xlabel(f'State\nSTD: {float("{0:.4f}".format(std_oc)) if isinstance(learned_rewards_per_al_func, list) else 0.0}')
        if use_action_visitations:
            axesUp[i].set_ylabel('Act')
            
        # Plot the second matrix
        #print(env.real_env.calculate_rewards(env.real_env.translate(152),0,env.real_env.translate(152)))
        im2 = axesDown[i].imshow(eocs2, cmap='viridis', interpolation='nearest', aspect=env_real.state_dim/env_real.action_dim)
        axesDown[i].set_title(f'{al}')
        axesDown[i].set_xlabel('State')
        if use_action_visitations:
            axesDown[i].set_ylabel('Act')
    
    subfigs[0].colorbar(im1, ax=axesUp, orientation='vertical', label='Visitation Freq.')
    subfigs[1].colorbar(im2, ax=axesDown, orientation='vertical', label='Visitation Freq.')
    # Adjust layout to prevent overlap
    #fig.tight_layout()
    fig.savefig('results/' + namefig + '_occupancy_dif.pdf')
    # Show the plot
    if show:
        fig.show()

