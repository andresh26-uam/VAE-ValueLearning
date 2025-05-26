import numpy as np
from Environment import Environment

RIGHT = 0
UP = 1
LEFT = 2

def translate_action(action):
    """
    Specific to the public civility game environmment, translates what each action number means

    :param action: int number identifying the action
    :return: string with the name of the action
    """

    part_1 = ""
    part_2 = ""

    if action < 3:
        part_1 = "MOVE "
    else:
        part_1 = "PUSH GARBAGE "

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


def scalarised_Qs(env, Q_state, w):
    """
    Scalarises the value of each Q(s,a) for a given state using a linear scalarisation function

    :param Q_state: the different Q(s,a) for the state s, each with several components
    :param w: the weight vector of the scalarisation function
    :return: the scalarised value of each Q(s,a)
    """

    scalarised_Q = np.zeros(len(env.all_actions))
    for action in range(len(Q_state)):
        scalarised_Q[action] = scalarisation_function(Q_state[action], w)

    return scalarised_Q


def Q_function_calculator(env, state, V, discount_factor):
    """

    Calculates the value of applying each action to a given state. Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the value obtained for each action
    """

    Q_state = np.zeros([len(env.all_actions), len(V[0,0,0])])
    for action in env.all_actions:
        state_translated = env.translate_state(state[0], state[1], state[2])
        env.hard_reset(state_translated[0], state_translated[1], state_translated[2])
        next_state, rewards, _ = env.step([action])
        for objective in range(len(rewards)):
            Q_state[action, objective] = rewards[objective] + discount_factor * V[next_state[0], next_state[1], next_state[2], objective]
    return Q_state


def deterministic_optimal_policy_calculator(Q, env, weights):
    """
    Create a deterministic policy using the optimal Q-value function

    :param Q: optimal Q-function that has the optimal Q-value for each state-action pair (s,a)
    :param env: the environment, needed to know the number of states (adapted to the public civility game)
    :param weights: weight vector, to know how to scalarise the Q-values in order to select the optimals
    :return: a policy that for each state returns an optimal action
    """
    #
    policy = np.zeros([12, 12, 12])
    for cell_L in env.states_agent_left:
        for cell_R in env.states_agent_right:
            for cell_G in env.states_garbage:
                if cell_L != cell_R:
                    # One step lookahead to find the best action for this state
                    policy[cell_L, cell_R, cell_G] = np.argmax(scalarised_Qs(env, Q[cell_L, cell_R, cell_G], weights))

    return policy


def choose_action(st, eps, q_table, weights, infoQ):
    """

    :param st: the current state in the environment
    :param eps: the epsilon value
    :param q_table:  q_table or q_function the algorithm is following
    :return:  the most optimal action for the current state or a random action
    """

    eps = max(0.1, eps**infoQ[st[0],st[1],st[2]])
    NB_ACTIONS = 6

    if np.random.random() <= eps:
        return np.random.randint(NB_ACTIONS)
    else:
        maxi = np.max(scalarised_Qs(env, q_table[st[0], st[1], st[2]], weights))

        possible_actions = list()
        for act in range(NB_ACTIONS):
            q_A = scalarisation_function(q_table[st[0],st[1],st[2],act], weights)
            if q_A == maxi:
                possible_actions.append(act)

        return possible_actions[np.random.randint(len(possible_actions))]

def update_q_table(q_table, alpha, gamma, action, state, new_state, reward):

    for objective in range(len(reward)):
        best_action = np.argmax(scalarised_Qs(env, q_table[new_state[0], new_state[1], new_state[2]], weights))

        q_table[state[0], state[1], state[2], action, objective] += alpha * (
            reward[objective] + gamma * q_table[new_state[0], new_state[1], new_state[2], best_action, objective] - q_table[state[0], state[1], state[2], action, objective])

def q_learning(env, weights, alpha=0.8, gamma=0.7):
    """
    Q-Learning Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 6.5,
    (1998).

    It has been adapted to the particularities of the public civility game, a deterministic environment, and also
     adapted to a MOMDP environment, having a reward function with several components (but assuming the linear scalarisation
    function is known).

    :param env: the environment encoding the (MO)MDP
    :param weights: the weight vector of the known linear scalarisation function
    :param alpha: the learning rate of the algorithm, can be set at discretion
    :param gamma: discount factor of the (MO)MPD, can be set at discretion (notice that this will change the Q-values)
    :return: the learnt policy and its associated state-value (V) and state-action-value (Q) functions
    """

    n_objectives = 2
    n_actions = 6
    n_cells = 12
    V = np.zeros([n_cells, n_cells, n_cells, n_objectives])
    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])

    max_episodes = 5000
    max_steps = 20

    epsilon = 0.999
    eps_reduc = epsilon/max_episodes
    infoQ = np.zeros([n_cells, n_cells, n_cells])
    alpha_reduc = 0.5*alpha/max_episodes

    for_graphics = list()

    for episode in range(1, max_episodes + 1):
        done = False

        env.hard_reset()

        state = env.get_state()

        if episode % 100 == 0:
            print("Episode : ", episode)

        step_count = 0

        alpha -= alpha_reduc

        R_big = [0, 0]

        while not done and step_count < max_steps:

            step_count += 1
            actions = list()

            actions.append(choose_action(state, epsilon, Q, weights, infoQ))
            infoQ[state[0],state[1],state[2]] += 1.0

            if state[0] == 10 and state[1] == 11 and state[2] == 8:

                #print(graphiker)
                if np.random.randint(2) < 1:
                    actions.append(1)
                else:
                    if env.is_deterministic:
                        n = 1
                    else:
                        n = 3
                    actions.append(n)

            new_state, reward, dones = env.step(actions)

            R_big[0] += reward[0]*(gamma**(step_count-1))
            R_big[1] += reward[1]*(gamma**(step_count-1))

            update_q_table(Q, alpha, gamma, actions[0], state, new_state, reward)

            state = new_state
            done = dones[0]
        q = Q[10, 11, 8].copy()
        sq = scalarised_Qs(env, q, weights)

        #a = np.argmax(sq)
        #print(q[a])
        #for_graphics.append(q[a])


    # Now that we have Q, it is straightforward to obtain V
    for cell_L in env.states_agent_left:
        for cell_R in env.states_agent_right:
            for cell_G in env.states_garbage:
                if cell_L != cell_R:
                    best_action = np.argmax(scalarised_Qs(env, Q[cell_L, cell_R, cell_G], weights))
                    V[cell_L, cell_R, cell_G] = Q[cell_L, cell_R, cell_G, best_action]


    # Output a deterministic optimal policy
    policy = deterministic_optimal_policy_calculator(Q, env, weights)

    #np_graphics = np.array(for_graphics)
    #np.save('example.npy', np_graphics)

    return policy, V, Q


def example_execution(env, policy):
    """

    Simulation of the environment without learning. The L Agent moves according to the parameter policy provided.

    :param env: the environment encoding the (MO)MDP
    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """
    state = env.get_state()
    returns = list()
    gamma = 0.7
    original_gamma = gamma
    done = False

    ethical_objective_fulfilled = False
    individual_objective_fulfilled = False

    while not done:


        actions = list()
        actions.append(policy[state[0], state[1], state[2]])  # L agent uses the learnt policy
        if state[0] == 10 and state[1] == 11 and state[2] == 8:
            if np.random.randint(2) < 1:
                actions.append(1)
            else:
                if env.is_deterministic:
                    n = 1
                else:
                    n = 3
                actions.append(n)

        action_recommended = translate_action(actions[0])
        print("L Agent position: " + str(state[0]) + ". R Agent position: " + str(state[1]) + ". Garbage position: " + str(
            state[2]) + ". L Action: " + action_recommended)

        if state[2] == 2:
            if not ethical_objective_fulfilled:
                ethical_objective_fulfilled = True
                print("====Ethical objective fulfilled! Garbage in wastebasket====")

        state, rewards, dones = env.step(actions)
        done = dones[0]  # R Agent does not interfere

        if len(returns) == 0:
            returns = rewards
        else:
            for i in range(len(rewards)):
                returns[i] += gamma*rewards[i]

            gamma *= original_gamma

        if done:
            if not individual_objective_fulfilled:
                individual_objective_fulfilled = True

                print("L Agent position: " + str(state[0]) + ". Garbage position: " + str(
                    state[2]) + ".")
                print("====Individual objective fulfilled! Agent in goal position====")

    print("Policy Value: ", returns)


if __name__ == "__main__":

    det = False
    env = Environment(is_deterministic=det)
    w_E = 0.71
    print("-------------------")
    print("L(earning) Agent will learn now using Q-Learning in the Public Civility Game.")
    print("The Ethical Weight of the Scalarisation Function is set to W_E = " + str(w_E) + ", found by our Algorithm.")
    print("-------------------")
    print("Learning Process started. Will finish when Episode = 5000.")
    weights = [1.0, w_E]

    policy, v, q = q_learning(env, weights)

    print("-------------------")
    print("The Learnt Policy has the following Value:")
    policy_value = v[10,11,8]
    print("Individual Value V_0 = " + str(round(policy_value[0],2)))
    print("Ethical Value (V_N + V_E) = " + str(round(policy_value[1],2)))
    if v[10, 11, 8][1] >= 2.4:
        print("Since V_N + V_E = 2.4, the L Agent has learnt the Ethical Policy.")
    print("-------------------")
    if v[10, 11, 8][1] >= -10:
        print("We Proceed to show the learnt policy. Please use the image PCG_positions.png provided to identify the agent and garbage positions:")
        print()
        env = Environment(is_deterministic=det)
        example_execution(env, policy)

        print("-------------------")



