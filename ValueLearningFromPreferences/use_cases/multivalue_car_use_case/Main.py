from CHVI import partial_convex_hull_value_iteration
from Lex import lex_max
from WeightsFinder import minimal_weight_computation, minimal_weight_computation_all_states
import numpy as np
import pickle

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
        hull_state = hull[state[0]][state[1]][state[2]]
        best_ethical_index = lex_max(hull_state, l_ordering, iWantIndex=True)
        # print("Ethical policy : ", hull_state[best_ethical_index])
        best_ethical_indexes.append(best_ethical_index)

    #print(best_ethical_indexes)

    w = minimal_weight_computation_all_states(hull, best_ethical_indexes, initial_states, individual_weight, epsilon)

    return w


def Ethical_Environment_Designer(env, l_ordering, epsilon, discount_factor=1.0, max_iterations=15):
    """
    Calculates the Ethical Environment Designer in order to guarantee ethical
    behaviours in value alignment problems.


    :param env: Environment of the value alignment problem encoded as an MOMDP
    :param epsilon: any positive number greater than 0. It guarantees the success of the algorithm
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :return: the ethical weight that solves the ethical embedding problem
    """

    initial_states = [[43, 45, 31]]
    try:
        with open(r"v_function.pickle", "rb") as input_file:
            hull = pickle.load(input_file)
    except:
        hull = partial_convex_hull_value_iteration(env, discount_factor, max_iterations)

    #print("-----")
    #print("Partial Convex hull of the initial state s_0:") # això és un comentari per veure canvis al git
    hull_s0 = hull[initial_states[0][0]][initial_states[0][1]][initial_states[0][2]]
    #print(hull_s0)
    #print("-----")

    individual_obj = env.individual_objective

    ethical_weight = ethical_embedding_state(hull_s0, l_ordering, individual_obj, epsilon)
    #print("Ethical weight for initial state s_0: ", ethical_weight)
    #print("-----")
    ethical_weight = ethical_embedding(hull, l_ordering, initial_states, individual_obj, epsilon)

    return ethical_weight


if __name__ == "__main__":

    from ADS_Environment import Environment
    env = Environment()
    epsilon = 0.1
    lex_ordering = [2, 0, 1] # order the correct values!! [1,2,0]
    # Sembla que: 0: individual, 1: internal, 2: external!!
    discount_factor = 1.0
    max_iterations = 15

    w_E = Ethical_Environment_Designer(env, lex_ordering, epsilon, discount_factor, max_iterations)

    print("Ethical weights found: ", w_E)
    # 0.12: [1.0, 0.08799674999999998, 1e-05]
    # 0.13: [1.0, 0.08699675000000001, 9.999999999999999e-06]
    # 0.15: [1.0, 0.08499675000000001, 9.999999999999999e-06]
    # 0.2: [1.0, 0.07999675000000002, 1.0000000000000003e-05]
    # 0.22: [1.0, 0.07466666666666667, 0.01025641025641028]
    # 0.23: [1.0, 0.07200000000000001, 0.0153846153846154]
    # 0.24: [1.0, 0.06933333333333333, 0.020512820512820516]
    # 0.249: [1.0, 0.06693333333333333, 0.008000000000000007]