# Multi-Objective Reinforcement Learning for Designing Ethical Environments

Code for the paper "Multi-Objective Reinforcement Learning for Designing Ethical Environments", presented at the 30th International Joint Conference on Artificial Intelligence (IJCAI-21). It is also the code for its extended journal paper "Instilling moral value alignment by means of multi‑objective 
reinforcement learning" published online in 2022 in the Ethics and Information Technology Journal.

Authors: Manel Rodriguez-Soto, Maite Lopez-Sanchez, Juan Antonio Rodriguez-Aguilar.

The files implement in Python3 our Ethical Environment Designing Algorithm, a novel algorithm from our paper.

Required libraries:

* SciPy 1.4 or a higher version
* NumPy 1.14 or a higher version


Now we quickly review each Python file:

- Executing Main.py applies the ethical environment designing algorithm to the Public Civility Example, and shows
the calculated partial convex hull and the ethical weight obtained that guarantees ethical behaviour. There you will
see the results shown in Section 5 of our paper.

- Environment.py has the logic of the Public Civility Game environment.
- ItemAndAgent.py has the logic of the agents and the piece of garbage.
- ValuesNorms.py saves the ethical knowledge that we want the agents to learn.
- convexhull.py implements an algorithm to calculate a 2-D positive convex hull.
- CHVI.py implements the (partial) convex hull value iteration algorithm.

- Learning.py implements the q-learning algorithm for comparison. In this file you can see the policy that the agent
  ultimately learns after the ethical environment has been designed. As expected, the agent learns to behave ethically.
