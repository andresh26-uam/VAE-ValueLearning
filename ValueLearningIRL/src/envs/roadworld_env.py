from typing_extensions import override
from src.envs.tabularVAenv import TabularVAMDP
from roadworld_env_use_case.network_env import RoadWorldGymPOMDP
import numpy as np


from roadworld_env_use_case.values_and_costs import BASIC_PROFILES


class FixedDestRoadWorldGymPOMDP(TabularVAMDP):

    def _get_reward_matrix_for_profile(self, profile: tuple):

        if profile in self.reward_matrix_dict.keys() and self.reward_matrix_dict[profile].shape == (self.real_environ.state_dim, self.real_environ.action_dim):
            return self.reward_matrix_dict[profile]
        elif profile in BASIC_PROFILES and profile not in self.reward_matrix_dict.keys():

            reward_matrix = np.zeros(
                (self.real_environ.state_dim, self.real_environ.action_dim), dtype=np.float32)
            for s in range(self.real_environ.state_dim):
                for a in range(self.real_environ.action_dim):
                    reward_matrix[s, a] = self.real_environ.get_reward(
                        s, a, self.real_environ.cur_des, profile=profile)

            self.reward_matrix_dict[profile] = reward_matrix
            return self.reward_matrix_dict[profile]
        else:
            return np.sum([profile[i]*self._get_reward_matrix_for_profile(bp) for i, bp in enumerate(BASIC_PROFILES)], axis=0)

        

    def __init__(self, env: RoadWorldGymPOMDP, with_destination=None, done_when_horizon_is_met=False, trunc_when_horizon_is_met=True, **kwargs):
        self.reward_matrix_dict = dict()
        env.cur_des = with_destination
        reward_matrix_per_va = self._get_reward_matrix_for_profile
        self.real_environ = env
        self._invalid_states = [s for s in range(
            env.n_states) if s not in env.valid_edges]
        
        

        super().__init__(env.transition_matrix,
                         env.observation_matrix,
                         reward_matrix_per_va,
                         reward_matrix_per_va(env.last_profile),
                         env.initial_state_dist, env.horizon, done_when_horizon_is_met, trunc_when_horizon_is_met, **kwargs)
        self.cur_align_func = env.last_profile
        self._goal_states = [env.cur_des,]
        
    def step(self, action):
        s, r, d, t, i = super().step(action)
        d = self.state in self.goal_states
        return s, r, d, t, i
    def valid_actions(self, state, align_func=None):
        return self.real_environ.get_action_list(state)
    
    def get_state_actions_with_known_reward(self, align_func):
        return self.real_environ.state_actions_with_known_reward

    @property
    def goal_states(self):
        return self._goal_states

    @property
    def invalid_states(self):
        return self._invalid_states

    @override
    def render(self, caminos_by_value={'sus': [], 'sec': [], 'eff': []}, file='dummy.png', show=True, show_edge_weights=False, custom_weights: dict = None, custom_weights_dest: int = None):
        self.real_environ.render(caminos_by_value=caminos_by_value, file=file, show=show, show_edge_weights=show_edge_weights,
                                 custom_weights=custom_weights, custom_weights_dest=custom_weights_dest)
