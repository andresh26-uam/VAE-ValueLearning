from firefighters_use_case.execution import example_execution
from firefighters_use_case.pmovi import *
import matplotlib
import matplotlib.pyplot as plt

"""

This file contains examples on how to perform each kind of relevant operation

1. Computing the Pareto front

2. Retrieving a Pareto-optimal protocol from the convex region of the Pareto front

3. Retrieving a Pareto-optimal protocol from the non-convex region (or any region) of the Pareto front

4. Evaluating a particular protocol

5. Plotting the whole Pareto front

"""

"""
1. Computing the Pareto front
"""

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
discount_factor = 0.7
normalisation_factor = 1 - discount_factor  # because discount factor is 1 - 0.3 = 0.7
# Shows (normalised) Pareto front of initial state
print("Pareto front of initial state : ", v_func[initial_state] * normalisation_factor)


"""
2. Recovering protocol from convex region
"""

"If we want to recover a specific policy from the convex part of the Pareto front, we do as follows"
w = [0.8, 0.2]
scalarised_q = scalarise_q_function(q_func, objectives=2, weights=w)
pi = deterministic_optimal_policy_calculator(scalarised_q, w)

print("Fist action recommended by the protocol with weights [", str(w[0]), "," + str(w[1]) + "] to each objective: ",
      pi[initial_state])


"""
3. Recovering protocol from non-convex region
"""

"If we want to recover a specific policy from the non-convex part of the Pareto front, we do as follows"

selected_policy_index = 2

recovered_pi = get_particular_policy(q_func, HighRiseFireEnv(), discount_factor=discount_factor,
                                     reference_point=v_func[initial_state][selected_policy_index], reference_state=initial_state)

"""
4. We now observed a full execution of this policy
"""

example_execution(recovered_pi, None, discount_factor=discount_factor, max_steps=20, verbose=True)

"We store it"
np.save("pareto_policy.npy", recovered_pi)



"""
5. Finally, we print the whole Pareto front
"""


x, y = (v_func[initial_state] * normalisation_factor).T
plt.scatter(x, y)
matplotlib.rcParams.update({'font.size': 22})

plt.title('Pareto Front', fontsize=20)
plt.xlabel('Professionalism alignment', fontsize=18)
plt.ylabel('Proximity alignment', fontsize=18)
plt.show()