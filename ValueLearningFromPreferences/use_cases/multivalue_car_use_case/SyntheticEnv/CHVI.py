import numpy as np
import convexhull
from walkroom import WalkRoom



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

    state_v = env.state_scalar_to_vector(state)
    hulls = list()

    for action in range(env.action_space.n):
        env.reset(state_v)
        next_state, rewards, d, _ = env.step(action)

        next_state_sc = env.state_vector_to_scalar(next_state)

        V_state = V[next_state_sc].copy()

        hull_sa = convexhull.translate_hull(rewards, discount_factor, V_state)


        for point in hull_sa:
            hulls.append(point)

    hulls = np.unique(np.array(hulls), axis=0)

    new_hull = convexhull.get_hull(hulls)

    return new_hull


def partial_convex_hull_value_iteration(env, discount_factor=1.0, max_iterations=12):
    """
    Partial Convex Hull Value Iteration algorithm adapted from "Convex Hull Value Iteration" from
    Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Calculates the partial convex hull for each state of the MOMDP

    :param env: the environment encoding the MOMDP
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :return: value function storing the partial convex hull for each state
    """

    V = [np.array([])]*env.n_states
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        # Sweep for every state
        for scalar_state in range(env.n_states):
            if not env.is_terminal_scalar(scalar_state):
                V[scalar_state] = Q_function_calculator(env, scalar_state, V, discount_factor)

        print("Iterations: ", iteration, "/", max_iterations)

        print("---")

    np.save("v_function.npy", V)
    return V


def learn_and_do():
    size_room, dimensions_room = 7, 4
    env = WalkRoom(size_room, dimensions_room)


    print(env.terminal_list())



    v = partial_convex_hull_value_iteration(env)


    print("ooooooooooooooooooooooooooooo")



if __name__ == "__main__":

    learn = True

    if learn:
        learn_and_do()

    # Returns partial convex hull of initial state
    v_func = np.load("v_function.npy", allow_pickle=True)
    #print(v_func[43][31][31])
    print("--")
    #print(v_func[43][38][38])
    interesting_hull = v_func[0]

    for h in interesting_hull:
        print(np.array(h) - np.array([7, 7, 7, 7]))



