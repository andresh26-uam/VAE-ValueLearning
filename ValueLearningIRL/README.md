# Value Learning through Inverse Reinforcement Learning (dev branch)

In this repository we approach the value learning problem, as learning computational specifications from human values. 

In previous work, we introduce a technique-agnostic framework for the problem of **Value System Learning**, consisting of learning a computational specification of the alignment with a given set of values in a certain context (a task we name **Value Grounding Learning**) and the value preferences (value systems) of possibly heterogeneous agents (what we name **Value System Identification**). We will learn such ethical abstractions from demonstrations of the behavior of different kinds of agents.

 The problem is approached here through a MDP formulation and the value alignment and value preferences are learned through Inverse Reinforcement Learning. Specifically, we use versions of "Deep Maximum Entropy Inverse Reinforcement Learning" (Wulfmeier, 2015) and assume routes are chosen by maximizing an initially unknown value alignment MDP reward function.

 Additionally, we are exploring the use of preference-based (inverse) reinforcement learning algorithms [Christiano et al., 2017] (PbRL) that can learn from agent preferences regarding values.

## Use cases

* RoadWorld Environment. A route choice modeling use case where the goal is the routes that some simulated agents will choose according to their preferences over the alignment with three human values: *sustainability*, *security* and *efficiency*.
* Firefighters Environment. A simulated firefighting scenario where agents must choose the correct sequence of action to perform according to the maximization of two different values, namely *proffesionalism* and *proximity*. Future work will be done with real firefighter simulations.

## Installation

Create a new Python 3.9.6 environment with the built-in `venv` method, e.g.:
``python -m venv vslearning``

Activate the environment by executing the following command:
``source vslearning/bin/activate``

Install the requirements with:
``pip install -r requirements.txt``

## Reproduce results:
In all scripts, the `-sh` option is solely for showing the results when finished. The results will be under the `results/` folder in any case. The `-df` flag indicates the discount factor used to calculate value alignment of trajectories and for (inverse) reinforcement learning. The `-dfp` flag indicates the discount factor used to calculate value alignment of trajectories to compare them when using the preference comparisons algorithm. The `-a` flag indicates the IRL algorithm (or PbRL algorithm) to use for the tasks (`pc` stands for the PbRL algorithm or "preference comparisons", `me`for the Maximum Entropy IRL algorithm).

- **Value Grounding Learning**: learns a reward vector that tries to implements the ground truth grounding (and the ground truth reward vector).
It uses a custom version of the preference comparisons algorithm with quantified preferences [Christiano et al., 2017]. Use `-t 'vgl'` to perform this task:
    * Roadworld: `python train_vsl.py -t 'vgl' -e roadworld -a 'pc' -qp -n=10 -df=1.0 -dfp=1.0 -ename='reproduce_paper_vgl' -sh`
    * Firefighters: `python train_vsl.py -t 'vgl' -e firefighters -a 'pc' -qp -n=10 -df=1.0 -dfp=1.0 -ename='reproduce_paper_vgl' -sh`
- **Value System Identification**: learns a value system from the real environment grounding (given by a ground-truth value-related reward vector):
We employ Maximum Entropy IRL [Wulfmeier et al., 2015] by default. Use `-t 'vsi'` to perform this task:
    * Roadworld: `python train_vsl.py -t 'vsi' -e roadworld -a 'me' -n=10 -df=1.0 -dfp=1.0 -ename='reproduce_paper_vsi' -sh`
    * Firefighters: `python train_vsl.py -t 'vsi' -e firefighters -a 'me' -qp -n=10 -df=1.0 -dfp=1.0 -ename='reproduce_paper_vsi' -sh`
- **Value System Learning**: value grounding learning, then value system identification from the learned reward vector. Use `-t 'all'` to perform this pipeline altogether:
    * Roadworld: `python train_vsl.py -t 'all' -e roadworld -a 'pc-me' -qp -n=10 -df=1.0 -dfp=1.0 -ename='reproduce_paper_vsl' -sh`
    * Firefighters: `python train_vsl.py -t 'all' -e firefighters -a 'pc-me' -qp -n=10 -df=1.0 -dfp=1.0 -ename='reproduce_paper_vsl' -sh`

### Published work
---
* Under [Branch "TFM"](https://github.com/andresh26-uam/VAE-ValueLearning/tree/TFM/ValueLearningIRL): Master's thesis (July 2024) (in Spanish: TFM Trabajo Fin de Máster) in Artificial Intelligence from Universidad Politécnica de Madrid:

    **Learning Alignment with Human Values: A Case on Route Choice Modeling via Inverse Reinforcement Learning**
    It is also the same source code for the to-be-published paper:

    Holgado-Sánchez, A., Bajo, J., Billhardt, H., Os- sowski, S., and Arias, J. (2024a). **Value Learning for Value-Aligned Route Choice Modeling via Inverse Reinforcement Learning**. Submitted to Value Engineering in AI (VALE 2024) track of the International Workshop on AI Value Engineering and AI Compliance Mechanisms (VECOMP 2024), affiliated with the 27th European Conference on Artificial Intelligence (ECAI 2024).
    https://hal.science/hal-0462779

### References
---
* [Wulfmeier et al., 2015] Wulfmeier, M., Ondrúška, P., Ondrúška, O., and Posner, I. (2015). Maximum entropy deep inverse reinforcement learning. arXiv preprint arXiv:1507.04888. 
* [Christiano et al., 2017] Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei (2017). Deep reinforcement learning from human preferences. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). Curran Associates Inc., Red Hook, NY, USA, 4302–4310.