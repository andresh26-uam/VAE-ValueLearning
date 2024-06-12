

from collections import OrderedDict
import numpy as np

BASIC_PROFILES = [(1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)]

BASIC_PROFILE_NAMES = {(1.0,0.0,0.0): 'sus', (0.0,1.0,0.0): 'sec', (0.0,0.0,1.0): 'eff'}
PROFILE_NAMES_TO_TUPLE = {'sus': (1.0,0.0,0.0), 'sec': (0.0,1.0,0.0), 'eff': (0.0,0.0,1.0)}

PROFILE_COLORS = {(1.0,0.0,0.0): 'green', (0.0,1.0,0.0): 'blue', (0.0,0.0,1.0): 'red'} 

FULL_NAME_VALUES = OrderedDict({'sus': 'Sustainability', 'sec': 'Security', 'eff': 'Efficiency'} )

def eco_cost(feature_vector, path_feature_vector=None):
    # feature_vector = residential, primary, unclassified, tertiary, living street, secondary
    # path_feature_vector = number of road segments, total length, number of left turns, number of right turns, number of U turns, number of road segment per type (6 dimensions)
    fuel_consumption_estimate = np.asarray([17, 12, 12, 10, 20, 10]) # esto está inventado, es difícil encontrar referencias consistentes...
    normalized_fuel = fuel_consumption_estimate/np.max(fuel_consumption_estimate)
    #print(np.dot(fuel_consumption_estimate, feature_vector[1:]))
    return feature_vector[0]*(np.dot(normalized_fuel, feature_vector[1:]))

def sec_cost(feature_vector, path_feature_vector=None):
    sec_cost_estimate = [0.1, 0.3, 0.4, 0.8, 0.9, 0.5] # INSEC(e)
    return feature_vector[0]*np.dot(sec_cost_estimate, feature_vector[1:])

def eff_cost(feature_vector, path_feature_vector=None):
    speed_estimate = [0.1, 0.9, 0.4, 0.6, 0.1, 0.7] # VEL(e)
    #normalized_speed = speed_estimate/np.max(speed_estimate)
    return feature_vector[0]/np.dot(speed_estimate, feature_vector[1:])

VALUE_COSTS = OrderedDict({'sus': eco_cost, 'sec': sec_cost, 'eff': eff_cost})

def eco_cost_precomputed(link_features, path_features=None):
    #print(link_features)
    
    return link_features[1]


def sec_cost_precomputed(link_features, path_features=None):
    #print(link_features)
    return link_features[2]


def eff_cost_precomputed(link_features, path_features=None):
    return link_features[3]

VALUE_COSTS_PRE_COMPUTED_FEATURES = OrderedDict({'sus': eco_cost_precomputed, 'sec': sec_cost_precomputed, 'eff': eff_cost_precomputed})
