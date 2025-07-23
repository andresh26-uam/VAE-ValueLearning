from abc import abstractmethod
from typing import Callable, Optional
import torch as th

from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy

class MOCustomRewardVector():
    """
    A custom policy that uses a reward vector for multi-objective reinforcement learning.
    This class is a placeholder and should be implemented with specific methods for training and evaluation.
    """
    def __init__(self, env, **kwargs):
        self.reward_vector = None  # Placeholder for the reward vector

    @abstractmethod
    def train(self, **kwargs):
        # Implement training logic here
        pass

    @abstractmethod
    def set_reward_vector(self, reward_vector: Callable[[th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]], th.Tensor]) -> None:
        """
        Set the reward vector for the policy.
        :param reward_vector: A callable representing the reward vector.
        """
        pass


