from ADS_Environment import Environment
import csv
from VI import translate_action, scalarisation_function
import numpy as np


SPEED = 0
INTERNAL_SAFETY = 1
EXTERNAL_SAFETY = 2

obstacles_type =  [[],
                  [[2, 5]],
                  [[1, 5]],
                  [[4, 1]],
                  [[2, 2]],
                  [[2, 1]],
                  [[4, 1], [2, 1]],
                  [[4, 1], [2, 2]],
                  [[4, 1], [1, 2]],
                  [[4, 1], [1, 5]],
                  [[4, 1], [2, 5]],
                  [[5, 2], [2, 5]],
                  [[4, 1], [2, 4]],
                  [[2, 2], [2, 5], [4, 1]],
                  [[1, 2], [1, 5], [4, 2]],
                  [[1, 1], [2, 5], [4, 2]],
                  [[1, 1], [1, 5], [4, 2]],
                  [[1, 1], [1, 5], [5, 2]],
                  [[2, 2], [2, 5], [5, 1]],
                  [[1, 2], [1, 5], [5, 2]],
                  [[1, 1], [2, 5], [5, 2]]]

possible_inits = [[[6,1], [4, 3], [6,3]],
                  [[6,1], [4, 3], [5,3]],
                  [[6,2], [4, 3], [6,3]],
                  [[6,2], [4, 3], [5,3]],
                  [[6, 1], [3, 3], [5, 3]],
                  [[6, 2], [3, 3], [6, 3]],
                  [[6, 1], [3, 3], [4, 3]],
                  [[6, 2], [3, 3], [4, 3]],
                  [[6, 1], [4, 3], [4, 3]],
                  [[6, 2], [4, 3], [4, 3]],
                  [[6, 1], [5, 3], [4, 3]],
                  [[6, 2], [5, 3], [4, 3]],
                  ]


def example_execution(lex_order):
    """
    Simulation of the environment without learning. The L Agent moves according to the parameter policy provided.

    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """

    f = open('resultsAlt'+str(lex_order) +'.csv', 'w', newline='')
    writer = csv.writer(f)
    for i in range(len(obstacles_type)):
        if i == 7 or i == 13 or i == 18:
            pass
        else:
            policy = np.load("Policies_Celeste/Unpolicy_" + str(lex_order) + "_v" + str(i) + ".npy")
            for _ in range(10):
                for j in possible_inits:
                    env = Environment(obstacles=i)

                    env.easy_reset(j[0], j[1], j[2])

                    state = env.get_state()

                    done = False

                    individual_objective_fulfilled = False
                    n_timesteps = 0
                    while not done:
                        n_timesteps +=1
                        #print(state, q[state[0], state[1], state[2]])
                        actions = list()
                        action_chosen = policy[state[0], state[1], state[2]]
                        actions.append(action_chosen)  # L agent uses the learnt policy
                        #actions.append(-1)  # R Agent does not interfere


                        action_recommended = translate_action(policy[state[0], state[1], state[2]])
                        if action_recommended == 7:
                            print("found!")
                        #print("L Agent position: " + str(state[0]) + ". Action: " + action_recommended + " " + str(action_chosen))

                        state_v = env.translate_state(state)
                        next_state, rewards, dones = env.step(actions)

                        if state[0] == next_state[0]:
                            action_recommended = "STOP"


                        next_state_v = env.translate_state(next_state)
                        #print("State :", state_v, "Action : ", action_recommended, "Reward : ", rewards, scalarisation_function(rewards, weights),"Next state :", next_state_v, "Goal reached :", dones[0])
                        writer.writerow([state_v, action_recommended, next_state_v, rewards, dones[0], obstacles_type[i]])
                        done = dones[0]  # R Agent does not interfere
                        state = next_state
                        if done:
                            if not individual_objective_fulfilled:
                                individual_objective_fulfilled = True
                        if n_timesteps > 9:
                            print(i, n_timesteps)


    f.close()


if __name__ == "__main__":

    lex_orders = [[SPEED, INTERNAL_SAFETY, EXTERNAL_SAFETY],
                  [SPEED, EXTERNAL_SAFETY, INTERNAL_SAFETY],
                  [INTERNAL_SAFETY, SPEED, EXTERNAL_SAFETY],
                  [INTERNAL_SAFETY, EXTERNAL_SAFETY, SPEED],
                  [EXTERNAL_SAFETY, INTERNAL_SAFETY, SPEED],
                  [EXTERNAL_SAFETY, SPEED, INTERNAL_SAFETY]
                  ]


for lex_order in lex_orders:
    print(lex_order)
    example_execution(lex_order)
