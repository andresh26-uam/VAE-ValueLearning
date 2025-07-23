from typing import Any, Dict, Union
import gymnasium
import numpy as np
import torch
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.tabularVAenv import ContextualEnv

def one_hot_encoding(a, n, dtype=np.float32):
    v = np.zeros(shape=(n,), dtype=dtype)
    v[a] = 1.0
    return v
import gymnasium.spaces as spaces
class BaseRewardFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space,
                 action_space: spaces.Space, 
                 use_obs: bool, use_act: bool, use_next_obs: bool, use_done: bool, device: Union[str, torch.device] = 'cpu', dtype=torch.float32):
        self.use_obs = use_obs
        self.use_act = use_act
        self.use_next_obs = use_next_obs
        self.use_done = use_done
        self.device = device
        self.dtype = dtype

        features_dim = get_flattened_obs_dim(observation_space) + (get_flattened_obs_dim(action_space) if self.use_act else 0) + (get_flattened_obs_dim(observation_space) if self.use_next_obs else 0) + (1 if self.use_done else 0)
        super().__init__(observation_space, features_dim)
    
    def modify_observations(self, observations: torch.Tensor) -> torch.Tensor:
        return observations
    def modify_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions
    def adapt_info(self, info: Dict) -> None:
        return
    
    def forward(self, observations: torch.Tensor, actions: torch.Tensor = None, next_observations: torch.Tensor = None, dones: torch.Tensor = None, info: Dict=None) -> torch.Tensor:
        
        features = []
        self.adapt_info(info)
        if self.use_obs:
            features.append(torch.flatten(self.modify_observations(observations), 1))
        if self.use_act and actions is not None:
            features.append(torch.flatten(self.modify_actions(actions), 1))
        if self.use_next_obs and next_observations is not None:
            features.append(torch.flatten(self.modify_observations(next_observations), 1))
        if self.use_done and dones is not None:
            features.append(torch.reshape(dones.float(), [-1, 1]))
        ret = torch.cat(features, dim=1)
        assert ret.dtype == self.dtype, f"Expected dtype {self.dtype}, but got {ret.dtype}"
        
        return ret
    
    def __call__(self, state=None, action=None, next_state=None, done=None, info=None) -> torch.Tensor:
        # convert inputs to tensors if they are not already
        assert state is not None or action is not None or next_state is not None or done is not None, "At least one input must be provided"
        
        if state is not None and not isinstance(state, torch.Tensor):
            assert isinstance(state, (torch.Tensor, np.ndarray))
            state = torch.tensor(state, dtype=self.dtype, device=self.device)
        if action is not None and not isinstance(action, torch.Tensor):
            assert isinstance(state, (torch.Tensor, np.ndarray))
            action = torch.tensor(action, dtype=self.dtype, device=self.device)
        if next_state is not None and not isinstance(next_state, torch.Tensor):
            assert isinstance(state, (torch.Tensor, np.ndarray))
            next_state = torch.tensor(next_state, dtype=self.dtype, device=self.device)
        if done is not None and not isinstance(done, torch.Tensor):
            assert isinstance(state, (torch.Tensor, np.ndarray))
            done = torch.tensor(done, dtype=self.dtype, device=self.device)

        if state is not None and len(state.shape) == 1:
            state = state.unsqueeze(0)
        if action is not None and len(action.shape) == 1:
            action = action.unsqueeze(0)
        if next_state is not None and len(next_state.shape) == 1:
            next_state = next_state.unsqueeze(0)
        if done is not None and len(done.shape) == 1:
            done = done.unsqueeze(0)

        return self.forward(state, action, next_state, done, info)
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
        """batch_size = observations.shape[0]
        one_hot = torch.zeros((batch_size, self.n_categories), device=observations.device)

        one_hot.scatter_(1, observations.long(), 1)"""
        with torch.no_grad():
            if len(observations.shape) > 2:
                observations = torch.squeeze(observations, dim=1)
            if observations.shape[-1] != int(self.features_dim) or len(observations.shape) == 1:
                ret = torch.functional.F.one_hot(
                    observations.long(), num_classes=int(self.features_dim)).to(self.dtype)
            else:
                ret = observations
            return ret

class RewardOneHotFeatureExtractor(BaseRewardFeatureExtractor):
    """
    A custom feature extractor that performs one-hot encoding on integer observations.
    """

    def __init__(self,  n_categories: int, observation_space, action_space, use_obs, use_act, use_next_obs, use_done, device = 'cpu', dtype=torch.float32):
        super().__init__(observation_space, action_space, use_obs, use_act, use_next_obs, use_done, device, dtype)
        self.n_categories = n_categories
        self.encoder = OneHotFeatureExtractor(observation_space, self.n_categories, self.dtype)
    def modify_observations(self, observations):
        return self.encoder.forward(observations)
    
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
        """batch_size = observations.shape[0]
        one_hot = torch.zeros((batch_size, self.n_categories), device=observations.device)

        one_hot.scatter_(1, observations.long(), 1)"""
        with torch.no_grad():
            idx = observations
            """if idx.shape[-1] > 1:
                ret =  torch.vstack([self.observation_matrix[id] for id in idx.bool()])
            else:
                ret =  torch.vstack([self.observation_matrix[id] for id in idx.long()])"""
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
    
class RewardFeatureExtractorFromLongObservations(BaseRewardFeatureExtractor):
    def __init__(self, observation_matrix, **kwargs) -> None:
        super().__init__(**kwargs)
        #self.env = kwargs['env']
        self.torch_obs_mat = torch.tensor(observation_matrix, dtype=self.dtype, device=self.device )
    def modify_observations(self, observations: torch.Tensor) -> torch.Tensor:
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