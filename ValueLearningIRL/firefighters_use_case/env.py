import numpy as np
import gymnasium as gym
from gymnasium import spaces
from firefighters_use_case.constants import *


class HighRiseFireEnv(gym.Env):
    """
    A simplified two-objective MDP environment for an urban high-rise fire scenario.
    Objectives: Professionalism and Proximity
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(HighRiseFireEnv, self).__init__()

        # Define the state and action spaces, and the objective space
        self.action_space = spaces.Discrete(N_ACTIONS)  # 7 actions
        self.objective_space = spaces.Discrete(N_OBJECTIVES)
        self.state_space = spaces.MultiDiscrete(
            [1, 5, 5, 2, 2, 4], dtype=np.int64)  # Floor, Fire Intensity, Occupancy, Equipment, Visibility, Medical Condition
        """
        1. Floor Level (Low, Medium, High): Represents different segments of the high-rise building.
        2. Fire Intensity (None, Low, Moderate, High, Severe): Indicates the severity of the fire at the current state.
        3. Occupancy (0 to 4): Indicates whether the area is sparsely or densely populated.
        4. Equipment Readiness (Not Ready, Ready): Reflects the availability and readiness of necessary firefighting equipment.
        5. Visibility (Poor, Good): Represents the environmental condition affecting firefighting efforts.
        6. Firefighter condition (Perfect health, Slightly Injured, Moderately Injured, incapacitated) 
        Total number of states: 3*5*5*2*2*4 = 1200
        """
        self.states = {
            STATE_FLOOR_LEVEL: "Floor level",
            STATE_FIRE_INTENSITY: "Fire intensity",
            STATE_OCCUPANCY: "Occupancy",
            STATE_EQUIPMENT: "Equipment readiness",
            STATE_VISIBILITY: "Visibility",
            STATE_MEDICAL: "Firefighter condition"
        }
        self.n_states = np.prod(self.state_space.nvec)
        # Map actions to numbers
        """
        1. Evacuate Occupants: Prioritize evacuating people from the building.
        2. Contain Fire: Focus efforts on containing the fire to prevent it from spreading.
        3. Aggressive Fire Suppression: Engage in direct firefighting to quickly reduce fire intensity.
        4. Coordinate with Other Agencies: Request additional resources or coordination with other emergency services.
        5. Assess and Plan: Conduct a thorough assessment of the building and fire to plan subsequent actions.
        6. Go upstairs: go one floor up
        7. Go downstairs: go one floor down
        """

        self.actions = {
            ACTION_EVACUATE_OCCUPANTS: "Evacuate Occupants",
            ACTION_CONTAIN_FIRE: "Contain Fire",
            ACTION_AGGRESSIVE_FIRE_SUPPRESSION: "Aggressive Fire Suppression",
            ACTION_COORDINATE_WITH_OTHER_AGENCIES: "Coordinate with Other Agencies",
            ACTION_ASSESS_AND_PLAN: "Assess and Plan",
            ACTION_GO_UPSTAIRS: "Move one floor up",
            ACTION_GO_DOWNSTAIRS: "Move one floor down"
        }

        self.objectives = {
            OBJECTIVE_PROFESSIONALISM: "Professionalism",
            OBJECTIVE_PROXIMITY: "Proximity"
        }

        # Initial state: lower floor, high fire, high occupancy, equipment not ready, poor visibility, perfect health
        self.state = np.array([0, 3, 4, 0, 0, 3])



    def step(self, action):
        # Simulate state transitions and compute rewards
        next_state = self.transition(self.state, action)
        rewards = self.calculate_rewards(self.state, action, next_state)
        self.state = next_state

        # Check if the episode is done
        done = self.is_done(self.state)

        # In case
        info = None

        return self.state, rewards, done, info

    def reset(self, force_new_state=None):
        # Reset the environment state

        if force_new_state is not None:
            self.state = force_new_state
        else:
            self.state = np.array([0, 3, 4, 0, 0, 3])
        return self.state

    def render(self, mode='human'):
        pass

    def transition(self, state, action):
        # Simplified transition logic
        floor, fire_intensity, occupancy, equipment, visibility, medical_condition = state

        # Effect of actions on state
        if action == ACTION_EVACUATE_OCCUPANTS:  # Evacuate Occupants
            occupancy = max(0, occupancy - 1)

            if fire_intensity >= 3:
                if not equipment and not visibility:
                    medical_condition = max(0, medical_condition - 1)

                if fire_intensity == 5:
                    equipment = 0

        elif action == ACTION_CONTAIN_FIRE:  # Contain Fire
            fire_intensity = max(0, fire_intensity - 1)
        elif action == ACTION_AGGRESSIVE_FIRE_SUPPRESSION:  # Aggressive Fire Suppression

            if fire_intensity >= 3:
                if not equipment:
                    medical_condition = max(0, medical_condition - 1)
                if not visibility:
                    medical_condition = max(0, medical_condition - 1)

                if fire_intensity == 5:
                    equipment = 0

            #elif fire_intensity >= 1:
            #    if not equipment:
            #        medical_condition = max(0, medical_condition - 1)

            fire_intensity = max(0, fire_intensity - 2)

        elif action == ACTION_COORDINATE_WITH_OTHER_AGENCIES:  # Coordinate with Other Agencies
            equipment = 1  # Improved equipment readiness
        elif action == ACTION_ASSESS_AND_PLAN:  # Assess and Plan
            visibility = 1  # Improve visibility
        elif action == ACTION_GO_DOWNSTAIRS: # Go downstairs
            floor = max(0, floor - 1)
        elif action == ACTION_GO_UPSTAIRS: # Go upstairs
            floor = min(0, floor + 1)
        else:
            print("Warning, incorrect action specified!!!")

        return np.array([floor, fire_intensity, occupancy, equipment, visibility, medical_condition])

    def calculate_rewards(self, state, action, next_state):
        # Define rewards for professionalism and proximity
        professionalism_reward = 0
        proximity_reward = 0

        # Rewards based on action and state transition
        if action == ACTION_EVACUATE_OCCUPANTS:  # Evacuate Occupants
            professionalism_reward = max(0, 1-0.2*state[STATE_FIRE_INTENSITY]-0.1*state[STATE_VISIBILITY])
            proximity_reward = 1.0

            if state[STATE_OCCUPANCY] == 0:
                professionalism_reward = -1.0
                proximity_reward = -1.0

        elif action == ACTION_CONTAIN_FIRE:  # Contain Fire
            professionalism_reward = 0.8
            proximity_reward = 0.2

            if state[STATE_FIRE_INTENSITY] == 0:
                professionalism_reward = -1
                proximity_reward = -1

        elif action == ACTION_AGGRESSIVE_FIRE_SUPPRESSION:  # Aggressive Fire Suppression
            professionalism_reward = 0.3 if state[STATE_EQUIPMENT] == 0 else 0.6  # Less if equipment degraded
            proximity_reward = 0.5

            if state[STATE_FIRE_INTENSITY] == 0:
                professionalism_reward = -1.0
                proximity_reward = -1.0

        elif action == ACTION_COORDINATE_WITH_OTHER_AGENCIES:  # Coordinate with Other Agencies
            professionalism_reward = 0.5 if state[STATE_EQUIPMENT] == 0 else -1
            proximity_reward = -0.1 if state[STATE_EQUIPMENT] == 0 else -1
        elif action == ACTION_ASSESS_AND_PLAN:  # Assess and Plan
            professionalism_reward = 1.0 if state[STATE_VISIBILITY] == 0 else -1.0
            proximity_reward = -0.5 if state[STATE_EQUIPMENT] == 0 else -1.0

        if next_state[STATE_MEDICAL] == 0:
            professionalism_reward = -1.0
            proximity_reward = -1.0

        return [professionalism_reward, proximity_reward]

    def allowed_action(self, state, action):
        """

        Checks if the action does not make any sense
        :param state: a numpy array with the details of the state
        :param action: an int number specifying an action
        :return: True if it is allowed, False if it is nonsensical.
        """
        if action == ACTION_EVACUATE_OCCUPANTS:
            if state[STATE_OCCUPANCY] == 0:
                return False
        elif action == ACTION_CONTAIN_FIRE:
            if state[STATE_FIRE_INTENSITY] == 0:
                return False
        elif action == ACTION_AGGRESSIVE_FIRE_SUPPRESSION:
            if state[STATE_FIRE_INTENSITY] <= 1:
                return False
        elif action == ACTION_ASSESS_AND_PLAN:
            if state[STATE_VISIBILITY] == 1:
                return False
        elif action == ACTION_COORDINATE_WITH_OTHER_AGENCIES:
            if state[STATE_EQUIPMENT] == 1:
                return False
        elif action == ACTION_GO_DOWNSTAIRS:
            if state[STATE_FLOOR_LEVEL] == 0:
                return False
        elif action == ACTION_GO_UPSTAIRS:
            if state[STATE_FLOOR_LEVEL] == self.state_space[STATE_FLOOR_LEVEL].n - 1:
                return False

        return True

    def allowed_actions(self, state, return_mask=False):

        allowed_mask = np.zeros(self.action_space.n)

        allowed = list()

        for action in range(len(allowed_mask)):
            allowed_mask[action] = self.allowed_action(state, action)

            if allowed_mask[action]:
                allowed.append(action)

        if return_mask:
            return allowed_mask
        else:
            return allowed

    def is_done(self, state):
        # Check if fire intensity is 0 (fire is out) and no occupants left
        fire_intensity = state[STATE_FIRE_INTENSITY]
        occupancy = state[STATE_OCCUPANCY]

        good_ending = fire_intensity == 0 and occupancy == 0

        # or check if the firefighter is KO

        medical_condition = state[STATE_MEDICAL]

        bad_ending = medical_condition == 0

        return good_ending or bad_ending

    def encrypt(self, state):

        new_number = 0
        total_states = 1

        for i in range(self.state_space.shape[0]):
            new_number += state[i]*total_states
            total_states *= self.state_space[i].n

        return int(new_number)


    def translate(self, state):

        new_state = np.zeros(self.state_space.shape[0], dtype=np.int64)

        for i in range(len(new_state)):
            new_modulo = self.state_space[i].n
            new_state[i] = state % new_modulo

            state -= new_state[i]
            state /= new_modulo

        return new_state


if __name__ == "__main__":
    # Example usage
    env = HighRiseFireEnv()
    state = env.reset()
    print("Initial State:", state, env.encrypt(state))

    action = ACTION_AGGRESSIVE_FIRE_SUPPRESSION  # Example action: Aggressive Fire Suppression
    next_state, rewards, done, info = env.step(action)
    print("Next State:", next_state)
    print("Rewards:", rewards)
    print("Done:", done)

    #print(env.encrypt([0, 3, 4, 0, 0, 3]))

