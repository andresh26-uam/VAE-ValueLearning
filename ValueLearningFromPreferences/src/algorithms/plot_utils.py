import csv
import pprint
import matplotlib.pyplot as plt
import itertools
from matplotlib import cm, pyplot as plt
import numpy as np
import torch

from src.algorithms.base_vsl_algorithm import BaseVSLAlgorithm
from src.algorithms.utils import  mce_partition_fh, mce_occupancy_measures
from src.policies.vsl_policies import VAlignedDictDiscreteStateActionPolicyTabularMDP


def get_color_gradient(c1, c2, mix):
    """
    Given two hex colors, returns a color gradient corresponding to a given [0,1] value
    """
    c1_rgb = np.array(c1)
    c2_rgb = np.array(c2)
    mix = torch.softmax(torch.tensor(np.array(mix)),dim=0).detach().numpy()
    return (mix[0]*c1_rgb + ((1-mix[0])*c2_rgb))



def pad(array, length):
    new_arr = np.zeros((length,))
    new_arr[0:len(array)] = np.asarray(array)
    if len(new_arr) > len(array):
        new_arr[len(array):] = array[-1]
    return new_arr
