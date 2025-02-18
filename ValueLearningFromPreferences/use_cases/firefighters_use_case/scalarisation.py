import numpy as np


def randomized_argmax(v):
    return np.random.choice(np.where(v == v.max())[0])

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


def deterministic_optimal_policy_calculator(Q, weights):
    """
    Create a deterministic policy using the optimal Q-value function

    :param Q: optimal Q-function that has the optimal Q-value for each state-action pair (s,a)
    :param weights: weight vector, to know how to scalarise the Q-values in order to select the optimals
    :return: a policy that for each state returns an optimal action
    """
    #
    policy = np.zeros(len(Q))
    for i in range(len(policy)):
        best_action = randomized_argmax(scalarised_Qs(Q[i], weights))
        policy[i] = best_action
    return policy

def stochastic_optimal_policy_calculator(Q, weights, deterministic=False, tau =1.0):
    """
    Create a possibly stochastic policy using the optimal Q-value function

    :param Q: optimal Q-function that has the optimal Q-value for each state-action pair (s,a)
    :param weights: weight vector, to know how to scalarise the Q-values in order to select the optimals
    :param deterministic: whether to put all probability in the actions with maximum Q-values.
    :param tau: Temperature factor for grading the Q values in the softmax transformation to probabilities when setting the deterministic option to False. Tau close to zero makes the policiy more greedy and viceversa.
    :return: a policy in form of a matrix that for each state-action returns its probability
    """
    #
    policy = np.zeros((len(Q), Q.shape[1]))
    for i in range(policy.shape[0]):

        qs = scalarised_Qs(Q[i], weights).copy()
        
        if deterministic:
            where_max_q = np.where(qs == qs.max())[0]
            probs = np.zeros_like(qs)
            probs[where_max_q] = 1.0/len(where_max_q)
        else: 
            exp_qs_tau = np.exp(qs/tau)
            probs = exp_qs_tau/np.sum(exp_qs_tau)
        if __debug__: np.allclose(np.sum(probs), 1.0)
        policy[i, :] = probs
    return policy
    