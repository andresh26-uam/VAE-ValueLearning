import itertools
import numpy as np
from envs.tabularVAenv import TabularVAMDP
from typing import Any, List

from src.data import TrajectoryWithValueSystemRews
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

DATASET_FILE = 'use_cases/routechoice_datasets/apolloDataset.csv'

class RouteChoiceEnvironmentApollo(TabularVAMDP):
    def __init__(self, value_columns: List[int] = (2,3,4,5), test_size = 0.2, random_state=42):
        # Extract the number of states from the dataset
        
        with open(DATASET_FILE, 'r') as f:
            data = pd.read_csv(f)
            header = data.head(0)
            # Split the dataset into training and test sets
            train_data = []
            test_data = []

            for agent_id, group in data.groupby('ID'):  # Assuming 'ID' is the column for agent IDs
                train_group, test_group = train_test_split(group, test_size=test_size, random_state=random_state)
                train_data.append(train_group)
                test_data.append(test_group)

            train_data = pd.concat(train_data).reset_index(drop=True)
            test_data = pd.concat(test_data).reset_index(drop=True)

            # Preprocess the training dataset for classification using Scikit-learn scalers

            # household income is not modified, just for later identifiability.
            #hh_inc_scaler = Normalizer('max')
            #train_data['hh_inc_abs'] = hh_inc_scaler.fit_transform(train_data[['hh_inc_abs']])

            # Normalize numeric columns for alternatives
            alt_scaler = Normalizer(norm='max')
            alt_columns = ['tt1', 'tc1', 'hw1', 'ch1', 'tt2', 'tc2', 'hw2', 'ch2']
            train_data[alt_columns] = alt_scaler.fit_transform(train_data[alt_columns])

            # Convert categorical columns to one-hot encoding
            for col in ['car_availability', 'commute', 'shopping', 'business', 'leisure']:
                train_data[col] = train_data[col].astype(int)  # Ensure binary columns are integers

            # Apply the same transformations to the test set
            # TODO: maybe change this test_data['hh_inc_abs'] = hh_inc_scaler.transform(test_data[['hh_inc_abs']])
            test_data[alt_columns] = alt_scaler.transform(test_data[alt_columns])

            self.train_val_data = train_data.to_numpy()
            self.test_data = test_data.to_numpy()  # Store the test set for later use

            self.full_dataset = np.vstack([train_data, test_data])
        
        self.value_columns = value_columns
        self.n_values = len(self.value_columns)

        self.basic_profiles = [tuple(pr) for pr in np.eye(self.n_values, dtype=np.float32)]
        
        self.routes_train, self.routes_test, observation_matrix, self._reward_matrix_per_va, agent_context_matrix, self.preferences_per_agent_id = self._parse_trajectory_data_from_pd_dataset(self.train_val_data, self.test_data)
        self.all_routes = self.routes_train + self.routes_test
        self.agent_context_matrix = agent_context_matrix

        
        self.agent_ids_train = list({a.agent for a in self.routes_train})
        self.agent_ids_test =  list({a.agent for a in self.routes_test})
        self.agent_ids = list({*self.agent_ids_train, *self.agent_ids_test})

        n_trajs = len(self.routes_train) + len( self.routes_test)
        n_states = len(observation_matrix)
        transition_matrix = np.eye(n_states).reshape((n_states, 1, n_states))
        print(n_states)
        assert n_states >= len(observation_matrix)
        # Create an observation matrix (identity matrix)
        #observation_matrix = np.delete(dataset, value_columns, axis=1)
        
        # Extract the reward matrix from the dataset

        # Initialize the environment with the given matrices
        super().__init__(
            n_values=4,
            transition_matrix=transition_matrix,
            observation_matrix=observation_matrix,
            reward_matrix_per_va=self._get_reward_matrix_per_va,
            default_reward_matrix=self._reward_matrix_per_va[self.basic_profiles[0]],
            initial_state_dist=np.ones(n_states) / n_states,
            horizon=1,
            done_when_horizon_is_met=True,
            trunc_when_horizon_is_met=True
        )
    def _get_reward_matrix_per_va(self, align_func, custom_grounding=None):
        return self._reward_matrix_per_va[align_func] if align_func in self._reward_matrix_per_va.keys() else np.sum([align_func[i]*self._reward_matrix_per_va[self.basic_profiles[i]] for i in range(len(align_func))], axis=0)

    def valid_actions(self, state, align_func=None):
        # No actions available
        return np.array([0])

    def get_state_actions_with_known_reward(self, align_func):
        return None


    def real_reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[Any, dict[str, Any]]:
        # Reset the environment to a random initial state
        s,i = super().real_reset(seed=seed,options=options)
        i.update({'agent_context': self.agent_context_matrix[self.state]})
        return s, i

    def real_step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # Since there are no actions, just return the current state and reward
        ns,r,done,t,i = super().real_step(action=0)
        done = True  # Since horizon is 1, we are always done after one step
        i.update({'agent_context': self.agent_context_matrix[self.state]})
        return ns,r,done,done, i
    
    def _parse_trajectory_data_from_pd_dataset(self, train_dataset: np.ndarray, test_dataset: np.ndarray) -> List[TrajectoryWithValueSystemRews]:
        trajectories_train = []
        trajectories_test = []

        preferences_per_agent_id = {}
        unique_observations = {}
        new_id = 0

        for idat, dataset in enumerate((train_dataset, test_dataset)):
            for row in dataset:
                agent_id = int(row[0])  # Assuming the first column is the ID
                choice = int(row[1])
                if agent_id not in preferences_per_agent_id:
                    preferences_per_agent_id[agent_id] = {}
                observations_of_both = np.delete(row, (0,1))
                observations1 = tuple(observations_of_both[0:4])
                observations2 = tuple(observations_of_both[4:8])
                context_info = tuple(observations_of_both[-6:])
                
                observation_size = len(observations1)
                context_size= len(context_info)

                second_pair_id = None
                first_pair_id = None
                for first_or_second, obs in enumerate([observations1, observations2]):
                    cur_id = new_id
                    new_id+=1
                    
                    if first_or_second == 0:
                        first_pair_id = cur_id
                    else:
                        second_pair_id = cur_id
                        preferences_per_agent_id[agent_id][(first_pair_id, second_pair_id)] = 1.0-(choice-1)
                    trajectory = TrajectoryWithValueSystemRews(
                            obs=[obs,[0.0]*self.n_values],
                            acts=[0],
                            infos=[{'agent_context': context_info, 'state': cur_id}],
                            terminal=True,
                            n_vals=self.n_values,
                            # The rewards are always lower the better, the 4 values are costs-like 
                            v_rews=-np.array([obs]).transpose(),
                            rews=np.array([1.0 if (choice == 1 and first_or_second == 0) or (choice == 2 and first_or_second == 1) else 0.0]),
                            agent=str(int(agent_id))
                        )
                    if first_or_second == 0 and choice == 1:
                        assert trajectory.rews[0] == 1.0
                        assert trajectory.infos[0]['state'] == first_pair_id
                    elif first_or_second == 0 and choice == 2:
                        assert trajectory.rews [0] == 0.0
                        assert trajectory.infos[0]['state'] == first_pair_id
                    elif first_or_second == 1 and choice == 1:
                        assert trajectory.rews [0] == 0.0
                        assert trajectory.infos[0]['state'] == second_pair_id
                        assert preferences_per_agent_id[agent_id][(first_pair_id, second_pair_id)] == 1.0
                    elif first_or_second == 1 and choice == 2:
                        assert trajectory.rews [0] == 1.0
                        assert trajectory.infos[0]['state'] == second_pair_id
                        assert preferences_per_agent_id[agent_id][(first_pair_id, second_pair_id)] == 0.0
                    
                    if idat == 0:
                        trajectories_train.append(trajectory)
                    else:
                        trajectories_test.append(trajectory)

        observation_matrix = np.zeros((new_id + 1, observation_size))
        _reward_matrix_per_va = {tuple(prof): np.zeros((new_id +1, 1))  for prof in self.basic_profiles}
        agent_context_matrix = np.zeros((new_id + 1, context_size ))
        for traj in itertools.chain(trajectories_train, trajectories_test):
            observation_matrix[traj.infos[0]['state']] = traj.obs[0]
            for ipr, prof in enumerate(self.basic_profiles):
                # The rewards are always lower the better, the 4 values are costs-like 
                _reward_matrix_per_va[prof][traj.infos[0]['state'], 0] = -traj.obs[0][ipr]
            agent_context_matrix[traj.infos[0]['state']] = traj.infos[0]['agent_context']
            
        return trajectories_train, trajectories_test, observation_matrix, _reward_matrix_per_va, agent_context_matrix, preferences_per_agent_id