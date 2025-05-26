import numpy as np
from walkroom import WalkRoom


def translate_action(action):
    """
    Specific to the public civility game environmment, translates what each action number means

    :param action: int number identifying the action
    :return: string with the name of the action
    """

    part_1 = ""
    part_2 = ""

    if action < 3:
        part_1 = "MOVE SLOW "
    else:
        part_1 = "MOVE FAST "

    if action % 3 == 0:
        part_2 = "RIGHT"
    elif action % 3 == 1:
        part_2 = "FORWARD"
    else:
        part_2 = "LEFT"

    action_name = part_1 + part_2
    return action_name


def scalarisation_function(values, w):
    """
    Scalarises the value of a state using a linear scalarisation function

    :param values: the different components V_0(s), ..., V_n(s) of the value of the state
    :param w:  the weight vector of the scalarisation function
    :return:  V(s), the scalarised value of the state
    """

    f = 0
    for objective in range(len(values)):
        f += w[objective]*values[objective]

    return f


def scalarised_Qs(Q_state, w):
    """
    Scalarises the value of each Q(s,a) for a given state using a linear scalarisation function

    :param Q_state: the different Q(s,a) for the state s, each with several components
    :param w: the weight vector of the scalarisation function
    :return: the scalarised value of each Q(s,a)
    """

    n_actions = len(Q_state)
    scalarised_Q = np.zeros(n_actions)
    for action in range(n_actions):
        scalarised_Q[action] = scalarisation_function(Q_state[action], w)

    return scalarised_Q



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
    Q_state = np.zeros((env.action_space.n, env.dimensions))

    for action in range(env.action_space.n):
        env.reset(state_v)
        next_state, rewards, d, _ = env.step(action)

        next_state_sc = env.state_vector_to_scalar(next_state)

        V_state = V[next_state_sc].copy()

        for objective in range(len(rewards)):
            Q_state[action, objective] = rewards[objective] + discount_factor * V_state[objective]

    return Q_state


def randomized_argmax(v):
    return np.random.choice(np.where(v == v.max())[0])


def deterministic_optimal_policy_calculator(Q, env, weights):
    """
    Create a deterministic policy using the optimal Q-value function

    :param Q: optimal Q-function that has the optimal Q-value for each state-action pair (s,a)
    :param env: the environment, needed to know the number of states (adapted to the public civility game)
    :param weights: weight vector, to know how to scalarise the Q-values in order to select the optimals
    :return: a policy that for each state returns an optimal action
    """
    n_states = len(Q)
    policy = np.zeros(n_states)
    for state in range(n_states):
        policy[state] = int(np.argmax(scalarised_Qs(Q[state], weights)))

    return policy


def value_iteration(env, weights, theta=1.0, discount_factor=0.7):
    """
    Value Iteration Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 4.4,
    (1998).

    It has been adapted to the particularities of the public civility game, a deterministic envirnoment, and also
     adapted to a MOMDP environment, having a reward function with several components (but assuming the linear scalarisation
    function is known).

    :param env: the environment encoding the (MO)MDP
    :param weights: the weight vector of the known linear scalarisation function
    :param theta: convergence parameter, the smaller it is the more precise the algorithm
    :param discount_factor: discount factor of the (MO)MPD, can be set at discretion
    :return:
    """

    V = np.zeros((env.n_states, env.dimensions))
    Q = np.zeros((env.n_states, env.action_space.n, env.dimensions))

    iteration = 0

    while True:
        delta = 0
        iteration += 1
        # Sweep for every state
        for scalar_state in range(env.n_states):
            if not env.is_terminal_scalar(scalar_state):
                Q[scalar_state] = Q_function_calculator(env, scalar_state, V, discount_factor)

                best_action = np.argmax(scalarised_Qs(Q[scalar_state], weights))
                best_action_value = Q[scalar_state, best_action]
                delta += np.abs(scalarisation_function(best_action_value, weights) - scalarisation_function(V[scalar_state], weights))

                # Update the state value function
                V[scalar_state] = best_action_value

        # Check if we can finish the algorithm

        if delta < theta:
            print('Delta = ' + str(round(delta, 3)) + " < Theta = " + str(theta))
            print("Learning Process finished!")
            break
        else:
            print('Delta = ' + str(round(delta, 3)) + " > Theta = " + str(theta))


    # Output a deterministic optimal policy
    policy = deterministic_optimal_policy_calculator(Q, env, weights)

    return policy, V, Q

def example_execution(env, policy, Q):
    """

    Simulation of the environment without learning. The L Agent moves according to the parameter policy provided.

    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """

    env.reset()
    state = env.pos

    done = False

    print("Start position: " + str(state))

    while not done:

        state_scalar = env.state_vector_to_scalar(state)
        action_chosen = int(policy[state_scalar])

        state, reward, done, _ = env.step(action_chosen)

        print("Agent position: " + str(state) + ". Action: " + str(action_chosen) + ". Received reward: ", reward)


if __name__ == "__main__":

    weights = [1.0, 1.0, 1.0]

    s, d = 8, 3

    env = WalkRoom(s, d)

    policy, v, q = value_iteration(env, weights=weights, discount_factor=1.0)

    print("-------------------")
    print("-------------------")

    print(v[0])
    print("-------------------")
    print("We Proceed to show the learnt policy. Please use the image PCG_positions.png provided to identify the agent and garbage positions:")
    print()

    example_execution(env, policy, q)

    print("-------------------")



