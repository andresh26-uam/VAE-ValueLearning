
from firefighters_use_case.env import HighRiseFireEnv


def protocol_alignment_prediction(policy, discount_factor, max_steps, max_iterations, avg=False):

    Alignment_Value_1 = 0
    Alignment_Value_2 = 0

    if avg:
        discount_factor = 1.0

    for _ in range(max_iterations):
        _, returns_1, returns_2 = example_execution(policy, None, max_steps=max_steps, discount_factor=discount_factor, verbose=False, store_all_together=True, avg=avg)

        Alignment_Value_1 += returns_1
        Alignment_Value_2 += returns_2

    return Alignment_Value_1 / max_iterations, Alignment_Value_2 / max_iterations


def example_execution(policy, q,max_steps=100, discount_factor=1.0, avg=False, show_q=False,verbose=True, store_all_together=False):
    """

    Simulation of the environment without learning. The L Agent moves according to the parameter policy provided.

    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """
    env = HighRiseFireEnv()
    state = env.reset()
    done = False

    state_list = list()
    action_list = list()
    prof_list = list()
    prox_list = list()
    transitions_list = list()

    R_0 = 0
    R_1 = 0

    steps = 0

    if avg:
        discount_factor = 1.0

    current_discount_factor = 1.0

    while not done and steps < max_steps:

        steps += 1

        scalar_state = env.encrypt(state)
        state_list.append(scalar_state)
        transitions_list.append(scalar_state)
        if show_q:
            print(state, q[scalar_state])
        action = policy[scalar_state]

        action_list.append(action)
        transitions_list.append(action)

        next_state, reward, done, info = env.step(action)

        R_0 += reward[0]*current_discount_factor
        R_1 += reward[1]*current_discount_factor

        current_discount_factor *= discount_factor

        prof_list.append(reward[0])
        prox_list.append(reward[1])

        transitions_list.append(reward[0])
        transitions_list.append(reward[1])

        if verbose:
            state_part = ""
            for i in range(len(state)):
                state_part += env.states[i] + ": " + str(state[i]) + ". "

            action_part = "|| Action: " + env.actions[action]
            rewards_part = "|| Profesionalism algn.: " + str(reward[0]) + ". Proximity algn.: " + str(reward[1])
            print(state_part + action_part + rewards_part)

        state = next_state

    scalar_state = env.encrypt(state)

    state_list.append(scalar_state)
    transitions_list.append(scalar_state)

    if avg:
        R_0 /= steps
        R_1 /= steps

    if verbose:
        state_part = ""
        for i in range(len(state)):
            state_part += env.states[i] + ": " + str(state[i]) + ". "
        print(state_part)
    print("Final returns obtained : ", 0.3*R_0, R_1)

    if store_all_together:
        return transitions_list, R_0, R_1
    else:
        return state_list, action_list, prox_list, prof_list


if __name__ == "__main__":

    import numpy as np

    policy = np.load("pareto_policy.npy")

    print("-------------------")
    print("We Proceed to show the learnt policy.")
    example_execution(policy, None, discount_factor=0.7, max_steps=20, verbose=True)
    print("-------------------")