
from typing import Any, Dict, List, Optional

class BaseMetric():
    def __init__(self, params: Optional[Dict] = None) -> None:
        self.params = params

    def __str__(self) -> str:
        extra = ". Params: " + str(self.params) if self.params is not None else ""
        return f"{self.__class__.__name__ }{extra}"
    
    def __repr__(self) -> str:
        return self.__str__()
    

class BaseScoreMetric(BaseMetric):
    def __init__(self, params: Optional[Dict] = {}, plot_limits = [0.5,1.2]) -> None:
        super().__init__(params)
        self.plot_limits = plot_limits

    def __call__(self, features) -> Any:
        
        return self.feature_function(features)
    def feature_function(self, feature_vector) -> float:
        raise NotImplementedError(str(f"No provided implementation for {self.__class__}.equity() method"))
