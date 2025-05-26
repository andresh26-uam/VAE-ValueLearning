import numpy as np
import convexhull
from Environment import Environment


def Q_function_calculator(env, state, V, discount_factor):
    """

    Calculates the (partial convex hull)-value of applying each action to a given state.
    Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the new convex obtained after checking for each action (this is the operation hull of unions)
    """

    state_translated = env.translate_state(state[0], state[1], state[2])
    hulls = list()

    initial_state = state[0] == 10 and state[1] == 11 and state[2] == 8
    if not initial_state:
        for action in env.all_actions:

            env.hard_reset(state_translated[0], state_translated[1], state_translated[2])
            next_state, rewards, _ = env.step([action])

            V_state = V[next_state[0]][next_state[1]][next_state[2]].copy()

            hull_sa = convexhull.translate_hull(rewards, discount_factor, V_state)

            for point in hull_sa:
                hulls.append(point)

        hulls = np.unique(np.array(hulls), axis=0)

        new_hull = convexhull.get_hull(hulls)

    else:
        hulls = list()
        for action in env.all_actions:
            env.hard_reset(state_translated[0], state_translated[1], state_translated[2])
            next_state, rewards, _ = env.step([action])

            env.hard_reset(state_translated[0], state_translated[1], state_translated[2])
            next_state2, rewards2, _ = env.step([action, 4])


            for i in range(len(rewards)):
                rewards[i]  *= 0.5
                rewards2[i] *= 0.5

            V_state = 0.5*V[next_state[0]][next_state[1]][next_state[2]].copy()
            V_state2 = 0.5*V[next_state2[0]][next_state2[1]][next_state2[2]].copy()

            hull_sa1 = convexhull.translate_hull(rewards, discount_factor, V_state)
            hull_sa2 = convexhull.translate_hull(rewards2, discount_factor, V_state2)

            hull_sa = convexhull.sum_hulls(hull_sa1, hull_sa2)

            for point in hull_sa:
                hulls.append(point)

        hulls = np.unique(np.array(hulls), axis=0)

        new_hull = convexhull.get_hull(hulls)

    return new_hull


def partial_convex_hull_value_iteration(env, discount_factor=1.0, max_iterations=5):
    """
    Partial Convex Hull Value Iteration algorithm adapted from "Convex Hull Value Iteration" from
    Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Calculates the partial convex hull for each state of the MOMDP

    :param env: the environment encoding the MOMDP
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :return: value function storing the partial convex hull for each state
    """

    n_cells = 12

    V = list()
    for i in range(n_cells):
        V.append(list())
        for j in range(n_cells):
            V[i].append(list())
            for k in range(n_cells):
                V[i][j].append(np.array([]))

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        # Sweep for every state
        for cell_L in env.states_agent_left:
            for cell_R in env.states_agent_right:
                for cell_G in env.states_garbage:
                        if cell_L != cell_R:
                            # calculate the value of each action for the state
                            V[cell_L][cell_R][cell_G] = Q_function_calculator(env, [cell_L, cell_R, cell_G], V, discount_factor)
        print("Iterations: ", iteration, "/", max_iterations)
    return V


if __name__ == "__main__":
    env = Environment()

    v = partial_convex_hull_value_iteration(env, discount_factor=0.7)

    # Returns partial convex hull of initial state: [[4.67, -10], [2.27, 0], [0.59, 2.4]]
    print(v[10][11][8])




