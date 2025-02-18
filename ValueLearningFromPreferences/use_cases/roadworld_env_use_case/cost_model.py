
from roadworld_env_use_case.network_env import RoadWorldGym
from roadworld_env_use_case.values_and_costs import BASIC_PROFILES, VALUE_COSTS_PRE_COMPUTED_FEATURES
from src.vsl_reward_functions import LinearVSLRewardFunction, TrainingModes, squeeze_r
import torch as th
import numpy as np

def cost_model_from_reward_net(reward_net: LinearVSLRewardFunction, env: RoadWorldGym, precalculated_rewards_per_pure_pr=None):

    def apply(profile, normalization):
        if np.allclose(list(reward_net.get_learned_profile()), list(profile)):
            def call(state_des):
                data = th.tensor(
                    [[VALUE_COSTS_PRE_COMPUTED_FEATURES[v](*env.get_separate_features(
                        state_des[0], state_des[1], normalization)) for v in VALUE_COSTS_PRE_COMPUTED_FEATURES.keys()],],
                    dtype=reward_net.dtype)

                return -squeeze_r(reward_net.forward(None, None, data, th.tensor([state_des[0] == state_des[1], ])))
            return call
        else:
            if precalculated_rewards_per_pure_pr is not None:

                def call(state_des):
                    if profile in BASIC_PROFILES:
                        return -precalculated_rewards_per_pure_pr[profile][state_des[0], state_des[1]]

                    final_cost = 0.0
                    for i, pr in enumerate(BASIC_PROFILES):
                        final_cost += profile[i] * \
                            precalculated_rewards_per_pure_pr[pr][state_des[0],
                                                                  state_des[1]]
                    return -final_cost
                return call
            else:
                def call(state_des):
                    prev_mode = reward_net.mode
                    reward_net.set_mode(TrainingModes.VALUE_GROUNDING_LEARNING)
                    reward_net.set_alignment_function(profile)
                    data = th.tensor(
                        [[VALUE_COSTS_PRE_COMPUTED_FEATURES[v](*env.get_separate_features(
                            state_des[0], state_des[1], normalization)) for v in VALUE_COSTS_PRE_COMPUTED_FEATURES.keys()],],
                        dtype=reward_net.dtype)
                    result = squeeze_r(reward_net.forward(
                        None, None, data, th.tensor([state_des[0] == state_des[1], ])))
                    reward_net.set_mode(prev_mode)
                    return -result
                return call

    return apply