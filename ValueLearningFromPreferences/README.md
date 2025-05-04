# Learning the value systems of societies from preferences - submitted for ECAI 2025 (paper id 6755)
This repository is the source code for the ECAI paper titled "Learning the value systems of societies from preferences". The paper presents a novel approach to learning value systems (value-based preferences) and value groundings (domain-specific value alignment measures) of a society of agents or stakeholders from examples of pairwise preferences between alternatives in a decision-making problem domain.

In the paper we utilize the Apollo dataset from [https://rdrr.io/cran/apollo/man/apollo_swissRouteChoiceData.html](https://rdrr.io/cran/apollo/man/apollo_swissRouteChoiceData.html), about train choice in Switzerland. The dataset includes features such as cost, time, headway, and interchanges, which are used to model agent preferences based on values. Although it also works for sequential decision making, in the paper we focus on the non-sequential decision making use case that the Apollo Dataset is about. 

Refer to the paper for more details on the methodology and experiments. Then read the [notebook](execute_experiments.ipynb) for an explanation about the code and how to replicate the experiments. for an explanation about the code and how to replicate the experiments.

## INSTALLATION
To set up the environment and run the code, follow these steps:

1. **Install Python**: Ensure you have Python 3.13 or higher installed on your system. You can download it from [python.org](https://www.python.org/).

2. **Create a Virtual Environment**:
    ```bash
    python3 -m venv .venv
    ```

3. **Activate the Virtual Environment**:
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```

4. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

After installation, you can open the `execute_experiments.ipynb` notebook in your preferred Jupyter environment to explore the code and replicate the experiments.
