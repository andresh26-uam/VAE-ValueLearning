import numpy as np
from ADS_Environment import Environment
import threading
import time
import matplotlib.pyplot as plt


RIGHT = 0
UP = 1
LEFT = 2


def translate_action(action):
    """
    Specific to the public civility game environment, translates what each action number means

    :param action: int number identifying the action
    :return: string with the name of the action
    """

    part_1 = ""
    part_2 = ""

    if action < 3:
        part_1 = "MOVE "
    else:
        part_1 = "Go Fast "

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
        f += w[objective] * values[objective]

    return f


def scalarised_Qs(env, Q_state, w):
    """
    Scalarises the value of each Q(s,a) for a given state using a linear scalarisation function

    :param Q_state: the different Q(s,a) for the state s, each with several components
    :param w: the weight vector of the scalarisation function
    :return: the scalarised value of each Q(s,a)
    """

    scalarised_Q = np.zeros(env.n_actions)
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

    Q_state = np.zeros([len(env.all_actions), len(V[0, 0, 0])])
    for action in env.all_actions:
        state_translated = env.translate_state(state[0], state[1], state[2])
        env.hard_reset(state_translated[0], state_translated[1], state_translated[2])
        next_state, rewards, _ = env.step([action])
        for objective in range(len(rewards)):
            Q_state[action, objective] = rewards[objective] + discount_factor * V[
                next_state[0], next_state[1], next_state[2], objective]
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
    # print ()

    policy = np.zeros(Q.shape[:-2])
    V = np.zeros(Q.shape[:-2] + (Q.shape[4],)) # this should work in theory, all cells + objectives

    for cell_car in  range(env.map_num_cells): # es podria acotar nombre de cel·les al nombre de cel·les a les que poden accedir??
        for cell_pedestrian_1 in range(env.map_num_cells):
            for cell_pedestrian_2 in range(env.map_num_cells):
                # One step lookahead to find the best action for this state
                best_action = randomized_argmax(scalarised_Qs(env, Q[cell_car, cell_pedestrian_1, cell_pedestrian_2], weights))
                policy[cell_car, cell_pedestrian_1, cell_pedestrian_2] =  best_action
                V[cell_car, cell_pedestrian_1, cell_pedestrian_2] = Q[cell_car, cell_pedestrian_1, cell_pedestrian_2, best_action]

    return policy, V


def choose_action(env, state, eps, q_table, weights):
    """

    :param state: the current state in the environment
    :param eps: the epsilon value
    :param q_table:  q_table or q_function the algorithm is following
    :return:  the most optimal action for the current state or a random action
    """

    NB_ACTIONS = 6

    if np.random.random() <= eps:
        return np.random.randint(NB_ACTIONS)
    else:
        return randomized_argmax(scalarised_Qs(env, q_table[state[0], state[1], state[2]], weights))


def update_q_table(q_table, env, weights, alpha, gamma, action, state, new_state, reward):
    best_action = np.argmax(scalarised_Qs(env, q_table[new_state[0], new_state[1], new_state[2]], weights))

    for objective in range(len(reward)):
        q_table[state[0], state[1], state[2], action, objective] += alpha * (
                reward[objective] + gamma * q_table[new_state[0], new_state[1], new_state[2], best_action, objective] -
                q_table[state[0], state[1], state[2], action, objective])


def q_learning(env, weights, alpha=0.98, gamma=1.0, max_weights=5000, max_episodes = 20000):
    """
    Q-Learning Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 6.5,
    (1998).

    It has been adapted to the particularities of environment, a deterministic environment, and also
     adapted to a MOMDP environment, having a reward function with several components (but assuming the linear scalarisation
    function is known).

    :param env: the environment encoding the (MO)MDP
    :param weights: the weight vector of the known linear scalarisation function
    :param alpha: the learning rate of the algorithm, can be set at discretion
    :param gamma: discount factor of the (MO)MPD, can be set at discretion (notice that this will change the Q-values)
    :param max_episodes: episodes taken into account in each q_learning
    :return: the learnt policy and its associated state-value (V) and state-action-value (Q) functions
    """

    n_objectives = env.n_objectives
    n_actions = env.n_actions
    n_cells = env.map_num_cells
    V = np.zeros([n_cells, n_cells, n_cells, n_objectives])
    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])

    max_steps = 25

    epsilon = 0.99
    #eps_reduc = epsilon / max_episodes  # per trobar les accions (exploration vs exploitation)
    infoQ = np.zeros([n_cells, n_cells, n_cells])
    #alpha_reduc = 0.5 * alpha / max_episodes  # pel pas de cada correcció
    valpha = []
    vepsilon = []
    verror0 = []
    verror1 = []
    verror2 = []

    #current_alpha = 0.5
    current_eps = 0.3
    reward = [0,0,0]

    for episode in range(1, max_episodes + 1):
        done = False

        env.easy_reset()  # comencem de 0

        state = env.get_state()

        if episode % 100 == 0:
            print("Episode : ", episode)
            print(Q[43, 45, 31])
            #print(Q[43, 45, 31])
            #print(infoQ[43, 45, 31])
            #valpha.append(current_alpha)
            #vepsilon.append(current_eps)
            #verror0.append(Q[43,45,31][4][0])
            #verror1.append(Q[43, 45, 31][4][1])
            #verror2.append(Q[43, 45, 31][4][2])


        step_count = 0

        #alpha -= alpha_reduc  # per intentar aproximar-nos el màxim de ràpid i més fiablement a la funció òptima



        while not done and step_count < max_steps:
            step_count += 1
            actions = list()
            infoQ[state[0], state[1], state[2]] += 1.0

            current_eps = max(0.1, epsilon - (0.001 * infoQ[state[0], state[1], state[2]]))

            actions.append(choose_action(env, state, current_eps, Q, weights))

            current_max = 0.1

            if episode > 20000:
                alpha = 0.2
            elif episode > 30000:
                alpha = 0.1
            elif episode > 40000:
                alpha = 0.05

            #current_alpha = max(current_max, alpha - (0.0001 * infoQ[state[0]][state[1]][state[2]]))

            new_state, reward, dones = env.step(actions)  # we take the actions

            # R_big[0] += reward[0]*(gamma**(step_count-1))
            # R_big[1] += reward[1]*(gamma**(step_count-1))

            # we update the table
            update_q_table(Q, env, weights, alpha, gamma, actions[0], state, new_state, reward)

            state = new_state
            done = dones[0]

        q = Q[43, 45, 31].copy()
        sq = scalarised_Qs(env, q, weights)

        # a = np.argmax(sq)
        # print(q[a])
        # for_graphics.append(q[a])

    # Output a deterministic optimal policy
    policy, V = deterministic_optimal_policy_calculator(Q, env, weights)

    # Plot the information gathered:
    '''plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(valpha)), valpha, label="Alpha")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(vepsilon)), vepsilon, label="Espilon")
    plt.legend()

    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(verror0)), verror0, label="0")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(verror1)), verror1, label="1")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(verror2)), verror2, label="2")
    plt.legend()
    plt.show()'''


    # np_graphics = np.array(for_graphics)
    # np.save('example.npy', np_graphics)

    return policy, V, Q


def example_execution(env, policy, render=False, stop=False):
    """

    Simulation of the environment without learning.

    :param env: the environment encoding the (MO)MDP
    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """
    max_timesteps = 200

    n_incidents = 0
    n_peatons_atacs = 0
    for i in range(10):
        timesteps = 0
        env.hard_reset()

        state = env.get_state()

        print("State :", state)
        done = False

        #env.set_stats(i + 1, 99, 99, 99, 99)
        if render:
            if not env.drawing_paused():
                time.sleep(0.5)
                env.update_window()

        while (timesteps < max_timesteps) and (not done):
            timesteps += 1

            actions = list()
            actions.append(policy[state[0], state[1], state[2]])

            if stop:
                actions = [LEFT, RIGHT, RIGHT]

            state, rewards, dones = env.step(actions)
            print("State :", state)

            print(actions, rewards)

            done = dones[0]  # R Agent does not interfere

            if render:
                if not env.drawing_paused():
                    time.sleep(0.5)
                    env.update_window()

class QLearner:
    """
    A Wrapper for the Q-learning method, which uses multithreading
    in order to handle the game rendering.
    """

    def __init__(self, environment, policy, drawing=False):

        threading.Thread(target=example_execution, args=(environment, policy, drawing,)).start()
        if drawing:
            env.render('Evaluating')


if __name__ == "__main__":

    if True:
        env = Environment(obstacles=-1)
        max_weights = 15000
        print("-------------------")
        print("L(earning) Agent will learn now using Q-Learning in the Public Civility Game.")
        # print("The Ethical Weight of the Scalarisation Function is set to W_E = " + str(w_E) + ", found by our Algorithm.")
        print("-------------------")

        # Parameters to change in each run
        alpha = 0.8
        weights = [1.0, 0.0, 0.0]
        lexicographic_order = "000"
        max_episodes = 50000
        save = True

        # Training.
        policy, v, q = q_learning(env, weights, alpha=alpha, max_weights=max_weights, max_episodes=max_episodes)
        print("-------------------")
        print("The Learnt Policy has the following Value for alpha = ", alpha, " is:")

        if save:
            np.save("./Policies/policy_lex" + lexicographic_order + ".npy", policy) #changepo

        print("-------------------")
        print("Finnished!!!")
        policy_value = v[43, 45, 31]
        print(policy_value)

    policy = np.load("policies/policy_012.npy")

    env = Environment(obstacles=2)
    QLearner(env, policy, drawing=True)
