# AutomatedDriving

The files implement in Python3 the Autonomous Car Environment, a Multi-Objective Markov Decision Process.

## Required libraries:

* SciPy 1.4 or a higher version
* NumPy 1.14 or a higher version
* PyGame 2.0 or a higher version (only necessary for animated visualisation of the learnt policies)
* mip 

## A brief summary of the files provided

### Environment itself
- ADS_Environment.py has the logic of the environment.
- ItemAndAgent.py has the logic of the agents
- ValuesNorms.py saves the ethical knowledge that we want the agents to learn.
- window.py has the internal logic for visualizing the environment.

### Algorithmic
- Learning.py implements the q-learning algorithm. 
- VI.py implements value iteration algorithm.
- CHVI.py implements convex hull value iteration algoritm (the multi-objective version of VI).
- convexhull.py has some auxiliary functions for CHVI.py
- Lex.py implements auxiliary functions for computing a lexicographic ordering.
- Main.py implements the full multi-valued ethical embedding algorithm.

### Documentation
- ALA_AAMAS_Paper15 contains the workshop paper in which this environment as well as the multi-valued ethical embedding algorithm are presented.
Please read this paper for details about how the algorithm works.

## Testing it

If everything is correctly installed, running "python Main.py" should yield a result: both the convex hull of the environment and the necessary weights
for guaranteeing that the ethical policy is optimal.
