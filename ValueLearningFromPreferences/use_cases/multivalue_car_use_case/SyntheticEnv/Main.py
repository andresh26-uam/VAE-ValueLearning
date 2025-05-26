from CHVI import partial_convex_hull_value_iteration
from Lex import lex_max
from WeightsFinder import minimal_weight_computation, minimal_weight_computation_all_states
import numpy as np

def ethical_embedding_state(hull_state, l_ordering, individual_weight=0, epsilon=0.0):
    """
    Ethical embedding operation for a single state. Considers the points in the hull of a given state and returns
    the ethical weight that guarantees optimality for the ethical point of the hull

    :param hull: set of 2-D points, coded as a numpy array
    :return: the etical weight w, a positive real number
    """

    best_ethical_index = lex_max(hull_state, l_ordering, iWantIndex=True)

    #print("Ethical policy : ", hull_state[best_ethical_index])

    w = minimal_weight_computation(hull_state, best_ethical_index, individual_weight, epsilon)

    return w


def ethical_embedding(hull, l_ordering, initial_states, individual_weight=0, epsilon=0.0):
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
    best_ethical_indexes = list()

    for state in initial_states:
        hull_state = hull[state]
        best_ethical_index = lex_max(hull_state, l_ordering, iWantIndex=True)
        # print("Ethical policy : ", hull_state[best_ethical_index])
        best_ethical_indexes.append(best_ethical_index)

    #print(best_ethical_indexes)

    w = minimal_weight_computation_all_states(hull, best_ethical_indexes, initial_states, individual_weight, epsilon)

    return w


def Ethical_Environment_Designer(env, l_ordering, epsilon, discount_factor=1.0, max_iterations=5):
    """
    Calculates the Ethical Environment Designer in order to guarantee ethical
    behaviours in value alignment problems.


    :param env: Environment of the value alignment problem encoded as an MOMDP
    :param epsilon: any positive number greater than 0. It guarantees the success of the algorithm
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :return: the ethical weight that solves the ethical embedding problem
    """

    initial_states = [0, 1]
    try:
        hull = np.load("v_function.npy", allow_pickle=True)
    except:
        hull = partial_convex_hull_value_iteration(env, discount_factor, max_iterations)

    print("-----")
    print("Partial Convex hull of the initial state s_0:")
    hull_s0 = hull[initial_states[0]]
    print(hull_s0)
    print("-----")

    individual_obj = 0

    ethical_weight = ethical_embedding_state(hull_s0, l_ordering, individual_obj, epsilon)
    print("Ethical weight for initial state s_0: ", ethical_weight)
    print("-----")
    ethical_weight = ethical_embedding(hull, l_ordering, initial_states, individual_obj, epsilon)

    return ethical_weight


if __name__ == "__main__":

    from walkroom import WalkRoom
    s, d = 8, 3
    env = WalkRoom(s, d)
    epsilon = 1.0
    lex_ordering = [1, 2, 0]
    discount_factor = 1.0
    max_iterations = 12

    w_E = Ethical_Environment_Designer(env, lex_ordering, epsilon, discount_factor, max_iterations)

    print("Ethical weights found: ", w_E)