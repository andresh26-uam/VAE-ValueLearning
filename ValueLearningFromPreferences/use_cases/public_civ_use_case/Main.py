from CHVI import partial_convex_hull_value_iteration
import numpy as np


def ethical_embedding_state(hull):
    """
    Ethical embedding operation for a single state. Considers the points in the hull of a given state and returns
    the ethical weight that guarantees optimality for the ethical point of the hull

    :param hull: set of 2-D points, coded as a numpy array
    :return: the etical weight w, a positive real number
    """

    w = 0.0

    if len(hull) < 2:
        return w
    else:
        ethically_sorted_hull = hull[hull[:,1].argsort()]

        best_ethical = ethically_sorted_hull[-1]
        second_best_ethical = ethically_sorted_hull[-2]

        individual_delta = second_best_ethical[0] - best_ethical[0]
        ethical_delta = best_ethical[1] - second_best_ethical[1]

        if ethical_delta != 0:
            w = individual_delta/ethical_delta

        return w


def ethical_embedding(hull, epsilon):
    """
    Repeats the ethical embedding process for each state in order to select the ethical weight that guarantees
    that all optimal policies are ethical.

    :param hull: the convex-hull-value function storing a partial convex hull for each state. The states are adapted
    to the public civility game.
    :param epsilon: the epsilon positive number considered in order to guarantee ethical optimality (it does not matter
    its value as long as it is greater than 0).
    :return: the desired ethical weight
    """
    ""

    w = 0.0

    for cell_L in env.states_agent_left:
        for cell_R in env.states_agent_right:
            for cell_G in env.states_garbage:
                if cell_L != cell_R:
                    w = max(w, ethical_embedding_state(hull[cell_L][cell_R][cell_G]))

    return w + epsilon


def Ethical_Environment_Designer(env, epsilon, discount_factor=1.0, max_iterations=5):
    """
    Calculates the Ethical Environment Designer in order to guarantee ethical
    behaviours in value alignment problems.


    :param env: Environment of the value alignment problem encoded as an MOMDP
    :param epsilon: any positive number greater than 0. It guarantees the success of the algorithm
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :return: the ethical weight that solves the ethical embedding problem
    """

    hull = partial_convex_hull_value_iteration(env, discount_factor, max_iterations)

    print("-----")
    print("Partial Convex hull of the initial state s_0:")
    print(hull[10][11][8])
    print("-----")
    print("Ethical weight for initial state s_0: ", ethical_embedding_state(hull[10][11][8]))
    print("-----")
    ethical_weight = ethical_embedding(hull, epsilon)

    return ethical_weight


if __name__ == "__main__":

    from Environment import Environment
    env = Environment()
    epsilon = 0.1
    discount_factor = 0.7
    max_iterations = 5

    w_E = Ethical_Environment_Designer(env, epsilon, discount_factor, max_iterations)

    print("Ethical weight if every state can be initial: ", w_E)