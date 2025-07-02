import numpy as np
from use_cases.multivalue_car_use_case.ADS_Environment import Environment
import gc
from use_cases.multivalue_car_use_case.Lex import lex_max

agent_2_actions = Environment.pedestrian_move_map

def translate_action(action):
    """
    Specific to the public civility game environmment, translates what each action number means

    :param action: int number identifying the action
    :return: string with the name of the action
    """

    part_1 = ""
    part_2 = ""

    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    n_originals = len([RIGHT, UP, LEFT, DOWN])

    if action < n_originals:
        part_1 = "MOVE SLOW "
    else:
        part_1 = "MOVE FAST "

    if action % n_originals == RIGHT:
        part_2 = "RIGHT"
    elif action % n_originals == UP:
        part_2 = "FORWARD"
    elif action % n_originals == LEFT:
        part_2 = "LEFT"
    else:
        part_2 = "DOWN"

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


def Q_function_calculator(env, state, V, discount_factor, model_used=None):
    """

    Calculates the value of applying each action to a given state. Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the value obtained for each action
    """
    Q_state = np.zeros((env.n_actions, env.n_objectives))
    state_translated = env.translate_state(state)
    action2 = agent_2_actions[state_translated[1][0]][state_translated[1][1]]
    action3 = agent_2_actions[state_translated[2][0]][state_translated[2][1]]

    dividing_factor = float(1./(len(action2)*len(action3)))

    for action in range(env.n_actions):
        if action != 6 or state[0] < 2*env.map_width: # can only move DOWN in the first two ROWS
            for act2 in action2:
                for act3 in action3:
                    if model_used is not None:
                        all_things = model_used[state[0], state[1], state[2], action, act2, act3]

                        next_state = [int(all_things[i]) for i in range(3, 6)]
                        rewards = all_things[:3]
                    else:
                        env.easy_reset(state_translated[0], state_translated[1], state_translated[2])
                        next_state, rewards, _ = env.step([action, act2, act3])
                        if len(next_state) > 3:
                            next_state = next_state[2:]

                    for objective in range(len(rewards)):
                        Q_state[action, objective] += dividing_factor*(rewards[objective] + discount_factor * V[next_state[0], next_state[1], next_state[2], objective])

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
    #
    policy = np.zeros([env.map_num_cells, env.map_num_cells, env.map_num_cells])
    for cell_L in env.states_agent_left:
        for cell_R in env.states_agent_right:
            for cell_J in env.states_agent_right:
                #amx = np.argmax(scalarised_Qs(env, Q[cell_L, cell_R, cell_G], weights))
                if True: #cell_L < 2*env.map_width:
                    best_action = randomized_argmax(scalarised_Qs(Q[cell_L, cell_R, cell_J], weights))
                else:
                    best_action = randomized_argmax(scalarised_Qs(Q[cell_L, cell_R, cell_J][:-1], weights))

                policy[cell_L, cell_R, cell_J] = best_action
    return policy


def generate_model(env):

    n_objectives = env.n_objectives
    n_actions = env.n_actions
    n_cells = env.map_num_cells
    model = np.zeros([53, 53, 53, n_actions, n_actions + 2, n_actions + 2, n_objectives + 4])

    print(n_cells, len(env.states_agent_left), len(env.states_agent_right))
    print("Number of states:", len(env.states_agent_left)*len(env.states_agent_right)**2)

    states_agent = env.states_agent_left
    states_pedestrian = env.states_agent_right

    n_states = 0
    # Sweep for every state
    for cell_L in states_agent:
        for cell_R in states_pedestrian:
            for cell_J in states_pedestrian:
                if cell_R >= cell_J:
                    n_states += 1
                    if n_states % 100 == 0:
                        print(n_states)
                    state_translated = env.translate_state([cell_L, cell_R, cell_J])

                    for action in range(env.n_actions):
                        for action2 in [0, 1, 2, 7]:
                            for action3 in [0, 1, 2, 7]:
                                env.easy_reset(*state_translated)

                                next_state, rewards, done = env.step([action, action2, action3])

                                model[cell_L, cell_R, cell_J, action, action2, action3, 0:len(rewards)] = rewards

                                model[cell_L, cell_R, cell_J, action, action2, action3, len(rewards):-1] = next_state

                                model[cell_L, cell_R, cell_J, action, action2, action3, -1] = int(done[0])

    np.save("model.npy", model)
    return model, n_actions, n_cells, n_objectives

def value_iteration(env, weights, lex=None, theta=1.0, discount_factor=0.7, model_used=None):
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

    n_objectives = env.n_objectives
    n_actions = env.n_actions
    n_cells = env.map_num_cells
    V = np.zeros([n_cells, n_cells, n_cells, n_objectives])
    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])

    print(len(env.states_agent_left), len(env.states_agent_right)) # com es que ho fem així? no tenim "3" agents?
    print("Number of states:", len(env.states_agent_left)*len(env.states_agent_right)**2)

    states_agent = env.states_agent_left
    states_pedestrian = env.states_agent_right

    if lex is not None:
        len_l = len(lex)
        weights = [10**(len_l-lex[i]) for i in range(len_l)]

    max_iterations = 100
    for _ in range(max_iterations):
        # Threshold delta
        delta = 0
        max_delta = 0
        #max_delta_s = []
        n_states = 0
        # Sweep for every state
        for cell_L in states_agent:
            for cell_R in states_pedestrian:
                for cell_J in states_pedestrian:
                    if cell_R >= cell_J: # perquè?
                        n_states += 1

                        #if n_states % 1000 == 0:
                        #    print(n_states)

                        # calculate the value of each action for the state
                        Q[cell_L, cell_R, cell_J] = Q_function_calculator(env, [cell_L, cell_R, cell_J], V, discount_factor, model_used)
                        
                        
                        # compute the best action for the state
                        if lex is None:
                            if True: #cell_L < 2 * env.map_width:
                                best_action = np.argmax(scalarised_Qs(Q[cell_L, cell_R, cell_J], weights))
                            else:
                                best_action = np.argmax(scalarised_Qs(Q[cell_L, cell_R, cell_J][:-1], weights))
                        else:
                            if True: #cell_L < 2* env.map_width:
                                best_action = lex_max(Q[cell_L, cell_R, cell_J][:-1], lex, iWantIndex=True)
                            else:
                                best_action = lex_max(Q[cell_L, cell_R, cell_J], lex, iWantIndex=True)
                        best_action_value = scalarisation_function(Q[cell_L, cell_R, cell_J, best_action], weights)
                        # Recalculate delta
                        delta += np.abs(best_action_value - scalarisation_function(V[cell_L, cell_R, cell_J], weights))
                        if delta > max_delta:
                            max_delta = delta
                            #max_delta_s = [cell_L, cell_R, cell_J]
                        # Update the state value function
                        V[cell_L, cell_R, cell_J] = Q[cell_L, cell_R, cell_J, best_action]

        # Check if we can finish the algorithm

        if delta < theta:
            print('Delta = ' + str(round(delta, 3)) + " < Theta = " + str(theta))
            print("Learning Process finished!")
            break
        else:
            print('Delta = ' + str(round(delta, 3)) + " > Theta = " + str(theta))


    # Output a deterministic optimal policy
    env = Environment()

    policy = deterministic_optimal_policy_calculator(Q, env, weights)

    return policy, V, Q

def example_execution(policy, q):
    """

    Simulation of the environment without learning. The L Agent moves according to the parameter policy provided.

    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """
    env = Environment()
    state = env.get_state()

    done = False

    individual_objective_fulfilled = False

    while not done:

        print(state, q[state[0], state[1], state[2]])
        actions = list()
        action_chosen = policy[state[0], state[1], state[2]]
        actions.append(action_chosen)  # L agent uses the learnt policy
        #actions.append(-1)  # R Agent does not interfere


        action_recommended = translate_action(policy[state[0], state[1], state[2]])
        print("L Agent position: " + str(state[0]) + ". Action: " + action_recommended + " " + str(action_chosen))

        state, reward, dones = env.step(actions)
        print("Received reward: ", reward)
        done = dones[0]  # R Agent does not interfere

        if done:
            if not individual_objective_fulfilled:
                individual_objective_fulfilled = True

                print("L Agent position: " + str(state[0]) + ".")
                print("====Individual objective fulfilled! Agent in goal position====")


if __name__ == "__main__":
    env = Environment(obstacles=9)
    w_E = 0.0

    print("TO DO: No hace falta que todos los estados sean diferentes para los dos peatones. Si los cambias de sitio al final es lo mismo.")
    print("Y tambien ayudaria en temas de velocidad tener un model.npy")

    print("-------------------")
    print("L(earning) Agent will learn now using Value Iteration in the Public Civility Game.")
    print("The Ethical Weight of the Scalarisation Function is set to W_E = " + str(w_E) + ", found by our Algorithm.")
    print("-------------------")
    print("Learning Process started. Will finish when Delta < Theta.")
    weights = [10.0, 100.0, 1.0]




    #weights = [1.0, w_E, w_E]

    #generate_model(env)
    #fasf

    policy, v, q = value_iteration(env, weights=weights, discount_factor=1.0, model_used=None)
    np.save("policy_test.npy", policy)
    np.save("v_test.npy", v)
    np.save("q_test.npy", q)

    policy = np.load("policy_test.npy")
    v = np.load("v_test.npy")
    q = np.load("q_test.npy")
    print("-------------------")
    print("-------------------")

    car = 43
    person = 45
    person2 = 31
    print(v[car][person][person2])

    if True:
        print(v[43, 45, 31])
        print(v[44, 38, 24])
        print(v[30, 31, 23])
        print(v[16, 24, 22])
        print(v[18, 21, 17])

        print("-------------------")
        print("We Proceed to show the learnt policy. Please use the image PCG_positions.png provided to identify the agent and garbage positions:")
        print()

        example_execution(policy, q)

        print("-------------------")



