"""
1. Evacuate Occupants: Prioritize evacuating people from the building.
2. Contain Fire: Focus efforts on containing the fire to prevent it from spreading.
3. Aggressive Fire Suppression: Engage in direct firefighting to quickly reduce fire intensity.
4. Coordinate with Other Agencies: Request additional resources or coordination with other emergency services.
5. Assess and Plan: Conduct a thorough assessment of the building and fire to plan subsequent actions.
6. Go upstairs: go one floor up
7. Go downstairs: go one floor down
"""
ACTION_EVACUATE_OCCUPANTS = 0
ACTION_CONTAIN_FIRE = 1
ACTION_AGGRESSIVE_FIRE_SUPPRESSION = 2
ACTION_COORDINATE_WITH_OTHER_AGENCIES = 3
ACTION_ASSESS_AND_PLAN = 4
ACTION_GO_UPSTAIRS = 5
ACTION_GO_DOWNSTAIRS = 6

N_ACTIONS = len([ACTION_EVACUATE_OCCUPANTS,ACTION_CONTAIN_FIRE,ACTION_AGGRESSIVE_FIRE_SUPPRESSION,ACTION_COORDINATE_WITH_OTHER_AGENCIES,ACTION_ASSESS_AND_PLAN])
N_OBJECTIVES = 2

"""
1. Floor Level (Low, Medium, High): Represents different segments of the high-rise building.
2. Fire Intensity (None, Low, Moderate, High, Severe): Indicates the severity of the fire at the current state.
3. Occupancy (1 to 5): Indicates whether the area is sparsely or densely populated.
4. Equipment Readiness (Not Ready, Ready): Reflects the availability and readiness of necessary firefighting equipment.
5. Visibility (Poor, Good): Represents the environmental condition affecting firefighting efforts.
6. Firefighter condition (Perfect health, Slightly Injured, Moderately Injured, incapacitated) 
Total number of states: 3*5*5*2*2*4 = 1200
"""

STATE_FLOOR_LEVEL = 0
STATE_FIRE_INTENSITY = 1
STATE_OCCUPANCY = 2
STATE_EQUIPMENT = 3
STATE_VISIBILITY = 4
STATE_MEDICAL = 5

OBJECTIVE_PROFESSIONALISM = 0
OBJECTIVE_PROXIMITY = 1
