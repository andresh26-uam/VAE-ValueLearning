# Learning Alignment with Human Values: A Case on Route Choice Modeling via Inverse Reinforcement Learning

This is the source code for the Master's thesis *Learning Alignment with Human Values: A Case on Route Choice Modeling via Inverse Reinforcement Learning* presented at UPM (Universidad Politécnica de Madrid) in July 2024. It is also the same source code for the to-be-published paper:

Holgado-Sánchez, A., Bajo, J., Billhardt, H., Os- sowski, S., and Arias, J. (2024a). Value Learning for Value-Aligned Route Choice Modeling via Inverse Reinforcement Learning. Submitted to Value Engineering in AI (VALE 2024) track of the International Workshop on AI Value Engineering and AI Compliance Mechanisms (VECOMP 2024), affiliated with the 27th European Conference on Artificial Intelligence (ECAI 2024).
https://hal.science/hal-0462779

In this code, we approach a first view on the general value learning problem. In the paper, we introduce a technique-agnostic framework for the problem of *Value System Learning*, consisting of learning a computational specification of the alignment with a given set of values in a certain context (a task we name *Value Grounding Learning*) and the value preferences (value systems) of possibly heterogeneous agents (what we name *Value System Identification*). We will learn such ethical abstractions from demonstrations of the behavior of different kinds of agents.

As a use case of the idea, we delve into a route choice modeling use case where the goal is the routes that some simulated agents will choose according to their preferences over the alignment with three human values: sustainability, security and efficiency. The problem is approached through a MDP formulation and the value alignment and value preferences are learned through Inverse Reinforcement Learning. Specifically, we use versions of Deep Maximum Entropy Inverse Reinforcement Learning (Wulfmeier, 2008) and assume routes are chosen by maximizing an initially unknown value alignment MDP reward function.




## Installation

Create a new Python 3.9.6 environment with the built-in `venv` method, e.g.:
``python -m venv vslearning_route_choice``

Activate the environment by executing the following command:
``source vslearning_route_choice/bin/activate``

Install the requirements with:
``pip install -r requirements.txt``

## Usage


A series of Python programs are included that provide the results seen in the papers. These appear under the folder `results/`. The programs and the results are divided into two main groups, the ones for *Value Grounding Learning* and the ones about *Value System Identification*. All the results are divided into ``train`` and ``test`` instances, where the items compared are sampled from the OD pairs observed during training and the OD pairs used for testing, respectively. The dataset and learning procedures are described in the papers.

* **Value Grounding Learning**: To replicate the results of the paper use:

``python experiments_value_grounding_learning.py``


Results will be under the folder `results/value_grounding_learning`.

This program executes 50 times the value grounding learning algorithm (Algorithm 1 in the papers) with different random initial parameters of the to-be-learned reward function network. The results present different learning curves (with standard deviations over the 50 executions) depicting comparison metrics (explained in the papers) between the observed training set routes of agents that maximize alignment with the given values and routes sampled by policies optimal with our learned rewards (one for each of the values) at each iteration. Also, we represent a matrix of the learned relation between state-action properties and the learned values (explained in the paper).

* **Value System Identification**: To replicate the results of the paper use:
``python experiments_value_system_identification.py``
Results will be under the folder `results/value_system_identification`.<br/>
This program learns the profiles of agents of different kinds (different scenarios) using Algorithm 2 from the papers. There are two kinds of agents, individuals (or "expert agents" because they act by perfectly minimizing a value-related alignment cost) and societies (groups of diverse individual/expert agents). We learn from the two kinds of agents separately, treating them as different learning scenarios. Both individuals and societies are identified by their *profile*, a tuple of three values between 0 and 1 which add up to 1 that represents the normalized agent's preference for each of the three values. Refer to the paper for more details.<br/>
There is a third scenario where no learning is carried out and we provide the system with the original profile but using the learned alignment functions from the grounding learning task. This is called in the files as ``given_profile``or `for_unseen_profile`. This scenario is explained only in the Master's thesis under the value grounding learning part of the evaluation<br/>
The results are grouped into different series of profiles to be learned with our algorithm. There are 4 of these series. The first one consists of agents that value different degrees of sustainability and security (`sus_sec`) while neglecting efficiency (11 different profiles); the second, agents that value different degrees of security and efficiency (`sec_eff`) while neglecting sustainability (11 different profiles); the third, agents that value different degrees of sustainability and efficiency (`sus_eff`) while neglecting security (11 different profiles); the fourth and last, agents that value the three of them in different degrees (covering the space of all possible agents, 15 different profiles) (`all`).<br/>
For each series of profiles, the results show (I) the learned value alignment cost distribution (in boxplots) of the routes sampled with policies optimal w.r.t. to the learned profiles (the learned profile routes), routes from the individual agents, and routes from the societies; (II) bar plots presenting, for each learning scenario, the average alignment costs of the learned profile routes among all OD pairs (train/test) and those of the original profile agents; (III) tables presenting the average and standard deviation of the value alignment costs and other specific combinations; and (IV) for each route comparison metric and for each learning scenario, a table that, for each profile, show the expected comparison metric value between the routes sampled with a policy optimal with our learned reward function (the learned profile routes) and the original demonstrations adapted to each individual value, between the learned profile routes and the individual agents that are act by maximizing alignment (minimizing alignment cost) with each of the three values (sustainability, security, and efficiency "experts") and between the learned profile routes and the original profile individual and the original profile society.<br/>
The four types of results are identifiable by their file name. We provide one example of each type for reference:
 - (I) `sus_eff_11_given_profile_boxplot_train.pdf`. Boxplots of the route alignment cost distribution learning from 11 original agents that value sustainability and efficiency alone in different degrees. There is one boxplot per value (in three different colors), grouped by profile and the kind of agent/policy that took/sampled the routes (the individual agent with the original profile, the society with that profile, and the policy optimal for the learned profile rewards). The results are for routes in the `train` set.
 - (II) `sus_sec_11__sus_values_comparison_society_train.pdf`. A set of bar plots representing the average sustainability costs conveying the demotion of the sustainability of the routes sampled with our learned policy and with the original profile agents, in this case, societies that have a profile varying among 11 profiles that value sustainability and security alone.
 - (III) `sec_eff_11__statistics_learning_from_expert_test.csv`. A table representing the average alignment costs with each value (with standard deviations) of the route cost distributions of the learned routes and the original ones of individuals and societies. In this case, the results are for routes in the test set for 11 different profiles valuing only security and efficiency. The learned routes are from route demonstrations of individual agents (experts). There is another file that presents the results for the learned profiles taking the route demonstrations of societies with the same profiles as training data: `sec_eff_11__statistics_learning_from_society_test.csv`.
 - (IV) `all_5__similarities_visitation_count_learning_from_society_train.csv`. A table representing, for each profile out of a series of 15 profiles evenly distributed in the profile space, the visitation count comparison metric (TVC, total absolute difference in visitation counts) among different sets of routes (from the train set) where the learned routes are obtained by fitting a certain reward function observing the routes of a society. The other metrics are written in the files with 'agou' (profiled cost similarity metric, PS), and 'jaccard' (Profiled Jaccard similarity, JAC). This metric does not adapt to the value alignments that we want to evaluate, which is why not all columns are used unlike with the other two metrics.<br/>
 There are versions of some files that have a particle `def_` at the beginning. These are last-moment manually corrected fiiles of the ones that do not have that particle.


* **Other files**. The snippets `final_results_paper.py` and `final_results_tfm.py` adapt the raw results from the last programs to present the most relevant results in the paper and the Master's thesis, respectfully. Other intermediate results obtained during the training processes and some route network visuals can be seen under the `plots/` folder. You can contact me if you need clarification on any specific result or figure.

Lastly, to stop using the Python environment execute the following command:
``deactivate vslearning_route_choice``

## References

* [Wulfmeier et al., 2015] Wulfmeier, M., Ondrúška, P., Ondrúška, O., and Posner, I.
(2015). Maximum entropy deep inverse reinforcement learning. arXiv preprint
arXiv:1507.04888.
