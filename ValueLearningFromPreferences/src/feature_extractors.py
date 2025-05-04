from copy import deepcopy
from typing import Any, Dict
import numpy as np
import torch
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.tabularVAenv import ContextualEnv

def one_hot_encoding(a, n, dtype=np.float32):
    v = np.zeros(shape=(n,), dtype=dtype)
    v[a] = 1.0
    return v


class OneHotFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that performs one-hot encoding on integer observations.
    """

    def __init__(self, observation_space, n_categories, dtype):
        # The output of the extractor is the size of one-hot encoded vectors
        super(OneHotFeatureExtractor, self).__init__(
            observation_space, features_dim=n_categories)
        self.n_categories = n_categories
        self.dtype=dtype

    def forward(self, observations):
        # Convert observations to integers (if needed) and perform one-hot encoding
        with torch.no_grad():
            if len(observations.shape) > 2:
                observations = torch.squeeze(observations, dim=1)
            if observations.shape[-1] != int(self.features_dim) or len(observations.shape) == 1:
                ret = torch.functional.F.one_hot(
                    observations.long(), num_classes=int(self.features_dim)).to(self.dtype)
            else:
                ret = observations
            return ret


class ObservationMatrixFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that performs one-hot encoding on integer observations.
    """

    def __init__(self, observation_space, observation_matrix, dtype=np.float32):
        # The output of the extractor is the size of one-hot encoded vectors
        super(ObservationMatrixFeatureExtractor, self).__init__(
            observation_space, features_dim=observation_matrix.shape[1])
        self.observation_matrix = torch.tensor(
            observation_matrix, dtype=dtype, requires_grad=False)

    def forward(self, observations):
        # Convert observations to integers (if needed) and perform one-hot encoding
        
        with torch.no_grad():
            idx = observations
            
            if idx.shape[-1] > 1:
                # Convert idx to a boolean mask and use it to index the observation_matrix
                mask = idx.bool()
                # Get indices where mask is True
                selected_indices = mask.nonzero(as_tuple=True)[-1]
                assert len(selected_indices) == observations.shape[0]
                # Select the first 32 True indices to maintain the output shape
                ret = self.observation_matrix[selected_indices]
            else:
                # Directly index the observation_matrix using long indices
                ret = self.observation_matrix[idx.view(-1).long()]
        return ret


class FeatureExtractorFromVAEnv(BaseFeaturesExtractor):
    def __init__(self, observation_space, **kwargs) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))

        #self.env = kwargs['env']
        self.dtype = kwargs['dtype']
        self.torch_obs_mat = torch.tensor(
            kwargs['env'].observation_matrix, dtype=self.dtype )
    def adapt_info(self, info: Dict):
        pass

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.torch_obs_mat[observations.long()]
    

class ContextualFeatureExtractorFromVAEnv(FeatureExtractorFromVAEnv):
    def __init__(self, observation_space, **kwargs):
        super().__init__(observation_space, **kwargs)
        self.env: ContextualEnv
        self.env = kwargs['env']
        self.obs_per_context = dict()
        self.obs_per_context[self.env.context] = self.torch_obs_mat
        
    def contextualize(self, context: Any):
        self.env.contextualize(context)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        a = self.obs_per_context.get(self.env.context, torch.tensor(
            self.env.observation_matrix, dtype=self.dtype, device=self.torch_obs_mat.device ))
        if self.env.context not in self.obs_per_context.keys():
            self.obs_per_context[self.env.context] = a
        return a[observations.long()]
        
    

    def adapt_info(self, info: Dict):
        if info is None:
            print("WARNING: COULD NOT ADAPT TO THIS INFO: NONE")
            return
        self.contextualize(info.get('context', None))