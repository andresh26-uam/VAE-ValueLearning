import firefighters_use_case.pareto_front as pareto_front
from firefighters_use_case.env import HighRiseFireEnv
from firefighters_use_case.scalarisation import *
import pickle



def Q_function_calculator(env, state, V, discount_factor, model_used=None, epsilon=-1.0, steps=-1, pareto=False):
    """

    Calculates the value of applying each action to a given state in terms of PMOVI.
    Adapted to deterministic environemnts

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the next Pareto front obtained after checking for each action (this is the operation hull of unions)
    """

    hulls = list()

    state_translated = env.translate(state)


    for action in range(env.action_space.n):
        hull_sa_all = []

        if model_used is not None:
           pass

        else:
            env.reset(force_new_state=state_translated)
            next_state, rewards, _, _ = env.step(action)

        if steps >= 0:
            for i in range(len(rewards)):
                rewards[i] /= steps
            discount_factor = (steps -1.0)/steps

        encrypted_next_state = env.encrypt(next_state)

        hull_sa = pareto_front.translate_hull(rewards, discount_factor, V[encrypted_next_state])
        hull_sa_all = pareto_front.sum_hulls(hull_sa, hull_sa_all, epsilon=epsilon, pareto=pareto)

        for point in hull_sa_all:
            hulls.append(point)

    hulls = np.unique(np.array(hulls), axis=0)

    new_hull = pareto_front.get_hull(hulls, epsilon=epsilon, pareto=pareto)

    return new_hull

def get_full_q_function(env, v, discount_factor=0.7):
    """

    From a Pareto front of Value functions, it returns the Pareto front of Q-Value functions

    :param env: original gymnasium environment
    :param v: Pareto front of Value functions
    :param discount_factor: discount factor that was used to compute v
    :return: Pareto front of Q-Value functions
    """
    n_states = env.n_states
    n_actions = env.action_space.n

    Q = list()
    for i in range(n_states):
        Q.append(list())
        state_translated = env.translate(i)

        for action in range(n_actions):

            env.reset(force_new_state=state_translated)

            next_state, rewards, _, _ = env.step(action)
            encrypted_next_state = env.encrypt(next_state)

            hull_sa = pareto_front.translate_hull(rewards, discount_factor, v[encrypted_next_state])
            Q[i].append(hull_sa)

    return Q


def get_particular_policy(q_hull, env, discount_factor, reference_point, reference_state):

    """
    Policy-tracking-method from Roijer's and Whiteson's "Multi-Objective Decision-Making"

    Retrieves a policy from its Q-Value from a pareto front of Q-value functions

    :param q_hull: Pareto front of Q-Value functions
    :param env: original gymnasium environment
    :param discount_factor: gamma used to compute q_hull
    :param reference_point: initial state of the env
    :param reference_state: value of the policy that we want to retrieve at the reference point
    :return:
    """
    n_states = env.n_states
    n_actions = env.action_space.n

    policy = np.zeros(n_states)
    chosen_action = -1
    done = False

    while not done:
        policy[reference_state] = -9999
        for action in range(n_actions):

            for point in q_hull[reference_state][action]:
                if np.linalg.norm(np.array(reference_point)-np.array(point)) < 0.001:
                    policy[reference_state] = action
                    chosen_action = action
                    break

        #print("----")
        #print(reference_state, reference_point, env.actions[policy[reference_state]])

        state_translated = env.translate(reference_state)
        env.reset(force_new_state=state_translated)
        state_translated, rewards, done, _ = env.step(chosen_action)

        reference_state = env.encrypt(state_translated)
        reference_point = (np.array(reference_point) - np.array(rewards)) / discount_factor

    return policy

def scalarise_q_function(q_hull, objectives, weights):

    """
    Auxiliary function to compute scalar q-function that maximises a given weight vector

    :param q_hull: pareto front of q_functions for a given state-action pair (i.e., list of vectors)
    :param objectives: integer
    :param weights: weight vector in R^n
    :return:
    """
    scalarised_q = np.zeros((len(q_hull), len(q_hull[0]), objectives))

    for state in range(len(q_hull)):
        for action in range(len(q_hull[0])):
            best_value = randomized_argmax(scalarised_Qs(q_hull[state][action], weights))
            for obj in range(objectives):
                scalarised_q[state, action, obj] = q_hull[state][action][best_value][obj]
    return scalarised_q


def pareto_multi_objective_value_iteration(env, discount_factor=1.0, max_iterations=50, model_used=None, from_scratch=True, pareto=False, normalized=False):
    """
    From Roijers and Whiteson's "Multi-Objective Decision-Making" (2017)


    Calculates the Pareto front for each state of the MOMDP

    :param env: the environment encoding the MOMDP
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :param model_used: containing the transition function of the MOMDP. Should be a .pickle file
    :param from_scratch: if false, it attempts to load a .pickle file containing the Pareto front of V-functions
    :return: value function storing the partial convex hull for each state
    :pareto: if true, computes the pareto front, if false: computes the convex hull (i.e., convex region of PF)
    :normalized: if true, returns the values of the Pareto front in a range between 0 and 1
    """

    n_states = env.n_states


    V = list()
    for i in range(n_states):
        V.append(list())

    if not from_scratch:
        try:
            with open(r"v_function.pickle", "rb") as input_file:
                V = pickle.load(input_file)
        except:
            print("Warning: Pickle file storing values does not exist!")
    iteration = 0
    print("Number of states:", env.n_states)
    origi_eps = -1.0
    eps = 1.0

    theta = 1000

    while iteration < max_iterations and theta > 0.1:
        iteration += 1
        theta = 0

        if origi_eps >= 0.0:
            eps *= origi_eps
        else:
            eps = -1.0
        # Sweep for every state

        steps = -1

        for i in range(n_states):
            if i % 100 == 0:
                print(i)

            if not env.is_done(env.translate(i)):
                oldie = V[i].copy()
                V[i] = Q_function_calculator(env, i, V, discount_factor, model_used, eps, steps, pareto)

                aux_theta = abs(len(V[i])-len(oldie))
                if aux_theta > 0:
                    theta = max(theta, aux_theta)
                else:
                    theta = max(theta, np.sum(np.absolute(np.array(V[i])-np.array(oldie))))

        print("Iterations: ", iteration, "/", max_iterations, ". Theta : ", theta)
        print(V[323])
        print()



    Q = get_full_q_function(env, V, discount_factor)

    if normalized:
        if discount_factor >= 1:
            print("Warning!!! Wrong normalization due to incorrect discount factor!")
        for s in range(len(V)):
            V[s] = (1.0-discount_factor)*np.array(V[s])
            for a in range(len(Q[s])):
                Q[s][a] = (1.0-discount_factor)*np.array(Q[s][a])

    return V, Q



def learn_and_do():
    env = HighRiseFireEnv()
    v, q = pareto_multi_objective_value_iteration(env, discount_factor=0.7, model_used=None, pareto=True, normalized=False)
    with open(r"v_hull.pickle", "wb") as output_file:
        pickle.dump(v, output_file)

    with open(r"q_hull.pickle", "wb") as output_file:
        pickle.dump(q, output_file)


if __name__ == "__main__":

    # Set to false if you already have a pretrained protocol in .pickle format
    learn = True

    if learn:
        learn_and_do()

    # Returns pareto front of initial state in V-table format
    with open(r"v_hull.pickle", "rb") as input_file:
        v_func = pickle.load(input_file)

    # Returns pareto front of initial state in Q-table format
    with open(r"q_hull.pickle", "rb") as input_file:
        q_func = pickle.load(input_file)

    print("--")

    initial_state = 323
    normalisation_factor = 0.3 # because discount factor is 1 - 0.3 = 0.7
    # Shows (normalised) Pareto front of initial state
    print("Pareto front of initial state : ", v_func[initial_state]*normalisation_factor)