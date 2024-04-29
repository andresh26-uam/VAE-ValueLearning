import numpy as np


from typing import Any, Dict

from src.src_rl.metrics import BaseMetric



class BaseAggregationMethod(BaseMetric):

    def __call__(self, sequence_of_equity: np.ndarray[float]) -> Any:
        eq = self.aggregation(sequence_of_equity)
        return eq
    def aggregation(self, sequence_of_equity: np.ndarray[float]) -> float:
        raise NotImplementedError(str(f"No provided implementation for {self.__class__}.aggregation() method"))


class SumAggregation(BaseAggregationMethod):
    def aggregation(self, sequence_of_equity: np.ndarray[float]) -> float:
        return np.sum(sequence_of_equity)


class MeanAggregation(BaseAggregationMethod):
    def aggregation(self, sequence_of_equity: np.ndarray[float]) -> float:
        return np.mean(sequence_of_equity)




class MinAggregation(BaseAggregationMethod):
    def aggregation(self, sequence_of_equity: np.ndarray[float]) -> float:
        return np.min(sequence_of_equity)


class DiscountedSumAggregation(BaseAggregationMethod):
    def __init__(self, params: Dict = {"gamma": 0.999}) -> None:
        super().__init__(params)
        self.gamma = self.params.get("gamma")
        self.gammas = [1, self.params.get("gamma")]
        self.ngammas = 2
    def aggregation(self, sequence_of_equity: np.ndarray[float]) -> float:
        length = len(sequence_of_equity) 
        if length > self.ngammas:
            
            for i in range(self.ngammas-1, length-1):
                self.gammas.append(self.gamma*self.gammas[i])
            self.ngammas = length
        return np.sum(np.multiply(self.gammas[0:length], sequence_of_equity))


class Score: # Scores are accumulated aggregation functions

    def __init__(self, params={}) -> None:
        self.params = params
        self.reset()

    def update_score(self, reward, prev_score):
        pass
    def reset(self):
        pass


class MeanScore(Score): # Equivalent to MeanAggregation

    def update_score(self, reward, prev_score):
        self.scs.append(reward)
        return np.mean(self.scs)
    def reset(self):
        self.n = 0
        self.scs = []
        return 0

class SumScore(Score): # Equivalent to MeanAggregation

    def update_score(self, reward, prev_score):
        return prev_score + reward
    def reset(self):
        return 0

class DiscountedScore(Score): # Equivalent to DiscountedSumAggregation
    def __init__(self, params: Dict={"gamma":0.9}) -> None:
        super().__init__(params)
        assert params.get("gamma", None) is not None

    def update_score(self, reward, prev_score):
        val = self.gamma_rec*reward + prev_score
        self.gamma_rec*=self.params["gamma"]
        return val
    def reset(self):
        self.gamma_rec = 1
        return 0