from typing import Any, Dict, Iterable, Optional, Sequence, cast
from sklearn.model_selection import KFold
from imitation.algorithms import preference_comparisons
from imitation.data import types
import dataclasses
import dill
from typing import List, Sequence, Tuple, TypeVar, overload

import logging
import os
import warnings
from typing import Mapping, Sequence, cast

import datasets
import jsonpickle
import numpy as np

from imitation.data import huggingface_utils
from imitation.data.types import AnyPath, Trajectory, TrajectoryWithRew
from imitation.util import util

import torch


def _vs_traj_validation(vs_rews: np.ndarray | torch.Tensor, value_rews: np.ndarray | torch.Tensor, n_values: int, acts: np.ndarray | torch.Tensor):
    if vs_rews.shape != (len(acts),):
        raise ValueError(
            "Value system rewards must be 1D array, one entry for each action: "
            f"{vs_rews.shape} != ({len(acts)},)",
        )
    if not isinstance(vs_rews.dtype, np.dtype) and not isinstance(vs_rews.dtype, torch.dtype):
        raise ValueError(f"rewards dtype {vs_rews.dtype} not a float")

    if value_rews.shape != (n_values, len(acts)):
        raise ValueError(
            "Individual value rewards must each be 1D array, one entry for each action: "
            f"{value_rews.shape} != ({n_values}, {len(acts)})",
        )
    if not isinstance(value_rews.dtype, np.dtype) and not isinstance(value_rews.dtype, torch.dtype):
        raise ValueError(f"rewards dtype {value_rews.dtype} not a float")


@dataclasses.dataclass(frozen=True, eq=False)
class TrajectoryWithValueSystemRews(Trajectory):
    """A `Trajectory` that additionally includes reward information of the value system rewards (for the value system alignment and each individual value alignment)."""

    """Reward, shape (trajectory_len, ). dtype float."""
    n_vals: int
    v_rews: np.ndarray
    rews: np.ndarray
    agent: str

    @property
    def vs_rews(self):
        return self.rews

    @property
    def value_rews(self):
        return self.v_rews

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()
        _vs_traj_validation(self.vs_rews, self.value_rews,
                            self.n_vals, self.acts)


class TrajectoryValueSystemDatasetSequence(huggingface_utils.TrajectoryDatasetSequence):
    """A wrapper to present an HF dataset as a sequence of trajectories.

    Converts the dataset to a sequence of trajectories on the fly.
    """

    def __init__(self, dataset: datasets.Dataset, dtype=np.float32):
        super().__init__(dataset)
        self._trajectory_class = TrajectoryWithValueSystemRews if 'v_rews' in dataset.features else self._trajectory_class


T = TypeVar("T")
Pair = Tuple[T, T]
TrajectoryWithValueSystemRewsPair = Pair[TrajectoryWithValueSystemRews]


def vs_trajectories_to_dict(trajectories, use_infos=False, dtype=np.float32):
    has_reward = [isinstance(traj, TrajectoryWithValueSystemRews)
                  for traj in trajectories]
    all_trajectories_have_reward = all(has_reward)
    if not all_trajectories_have_reward and any(has_reward):
        raise ValueError("Some trajectories have VS structure but not all")

    # Convert to dict
    trajectory_dict: Dict[str, Sequence[Any]] = dict(
        obs=[traj.obs for traj in trajectories],
        acts=[traj.acts for traj in trajectories],
        # Replace 'None' values for `infos`` with array of empty dicts
        infos=[
            traj.infos if traj.infos is not None and use_infos else [
                {}] * len(traj)
            for traj in trajectories
        ],
        terminal=[traj.terminal for traj in trajectories],
    )
    if any(isinstance(traj.obs, types.DictObs) for traj in trajectories):
        raise ValueError("DictObs are not currently supported")

    # Encode infos as jsondilld strings
    trajectory_dict["infos"] = [
        [jsonpickle.encode(info) for info in traj_infos]
        for traj_infos in cast(Iterable[Iterable[Dict]], trajectory_dict["infos"])
    ]

    # Add rewards if applicable
    if all_trajectories_have_reward:
        trajectory_dict["rews"] = [
            np.astype(cast(TrajectoryWithValueSystemRews, traj).rews, dtype) for traj in trajectories
        ]
        trajectory_dict["v_rews"] = [
            np.astype(cast(TrajectoryWithValueSystemRews, traj).v_rews, dtype) for traj in trajectories
        ]
        trajectory_dict["n_vals"] = [
            cast(TrajectoryWithValueSystemRews, traj).n_vals for traj in trajectories
        ]
        trajectory_dict["agent"] = [
            cast(TrajectoryWithValueSystemRews, traj).agent for traj in trajectories
        ]

    return trajectory_dict


def vs_trajectories_to_dataset(
    trajectories: Sequence[types.Trajectory],
    info: Optional[datasets.DatasetInfo] = None,
    dtype=np.float32,
    use_infos=False,
) -> datasets.Dataset:
    """Convert a sequence of trajectories to a HuggingFace dataset."""
    if isinstance(trajectories, TrajectoryValueSystemDatasetSequence):
        return trajectories.dataset
    else:
        dataset = datasets.Dataset.from_dict(vs_trajectories_to_dict(
            trajectories, dtype=dtype, use_infos=use_infos), info=info)
        return dataset


def save_vs_trajectories(path: AnyPath, trajectories: Sequence[TrajectoryWithValueSystemRews], dtype=np.float32, use_infos=False) -> None:
    """Save a sequence of Trajectories to disk using HuggingFace's datasets library.

    Args:
        path: Trajectories are saved to this path.
        trajectories: The trajectories to save.
    """
    p = util.parse_path(path)
    vs_trajectories_to_dataset(
        trajectories, dtype=dtype, use_infos=use_infos).save_to_disk(str(p))
    logging.info(f"Dumped demonstrations to {p}.")


def load_vs_trajectories(path: AnyPath) -> Sequence[Trajectory]:
    """Loads a sequence of trajectories saved by `save()` from `path`."""
    # Interestingly, np.load will just silently load a normal dill file when you
    # set `allow_dill=True`. So this call should succeed for both the new compressed
    # .npz format and the old dill based format. To tell the difference, we need to
    # look at the type of the resulting object. If it's the new compressed format,
    # it should be a Mapping that we need to decode, whereas if it's the old format,
    # it's just the sequence of trajectories, and we can return it directly.

    if os.path.isdir(path):  # huggingface datasets format
        dataset = datasets.load_from_disk(str(path))
        if not isinstance(dataset, datasets.Dataset):  # pragma: no cover
            raise ValueError(
                f"Expected to load a `datasets.Dataset` but got {type(dataset)}",
            )

        return TrajectoryValueSystemDatasetSequence(dataset)
    raise NotImplementedError("Only huggingface datasets format is supported")

    data = np.load(path, allow_dill=True)  # works for both .npz and .pkl

    if isinstance(data, Sequence):  # dill format
        warnings.warn("Loading old dill version of Trajectories",
                      DeprecationWarning)
        return data
    if isinstance(data, Mapping):  # .npz format
        warnings.warn("Loading old npz version of Trajectories",
                      DeprecationWarning)
        num_trajs = len(data["indices"])
        fields = [
            # Account for the extra obs in each trajectory
            np.split(data["obs"], data["indices"] + np.arange(num_trajs) + 1),
            np.split(data["acts"], data["indices"]),
            np.split(data["infos"], data["indices"]),
            data["terminal"],
        ]
        if 'vs_rews' in data:
            fields = [
                *fields,
                np.split(data["vs_rews"], data["indices"]),
            ]
            for k in data["value_rews"].keys():
                fields = [
                    *fields,
                    np.split(data["value_rews"][k], data["indices"]),
                ]
            return [TrajectoryWithRew(*args) for args in zip(*fields)]
        elif "rews" in data:
            fields = [
                *fields,
                np.split(data["rews"], data["indices"]),
            ]
            return [TrajectoryWithRew(*args) for args in zip(*fields)]
        else:
            return [Trajectory(*args) for args in zip(*fields)]  # pragma: no cover
    else:  # pragma: no cover
        raise ValueError(
            f"Expected either an .npz file or a dilld sequence of trajectories; "
            f"got a dilld object of type {type(data).__name__}",
        )


class VSLPreferenceDataset(preference_comparisons.PreferenceDataset):
    """A PyTorch Dataset for preference comparisons.

    Each item is a tuple consisting of two trajectory fragments
    and a probability that fragment 1 is preferred over fragment 2.

    This dataset is meant to be generated piece by piece during the
    training process, which is why data can be added via the .push()
    method.
    """

    def __init__(self, n_values: int, single_agent=False) -> None:
        """Builds an empty PreferenceDataset for Value System Learning
        """
        self.l_fragments1: List[TrajectoryWithValueSystemRews] = []
        self.l_fragments2: List[TrajectoryWithValueSystemRews] = []
        self.preferences: np.ndarray = np.array([])
        self.list_preferences_with_grounding: list = [np.array([])] * n_values
        self.n_values = n_values
        self.l_agent_ids = []
        self.agent_data = {}

        if not single_agent:
            self.data_per_agent = {}

    @property
    def preferences_with_grounding(self):
        return np.asarray(self.list_preferences_with_grounding).T

    @property
    def agent_ids(self):
        return np.asarray(self.l_agent_ids)

    @property
    def fragments1(self):
        return np.asarray(self.l_fragments1)

    @property
    def fragments2(self):
        return np.asarray(self.l_fragments2)

    def push(
        self,
        fragments: Sequence[TrajectoryWithValueSystemRewsPair],
        preferences: np.ndarray,
        preferences_with_grounding: np.ndarray,
        agent_ids=None,
        agent_data=None,
    ) -> None:
        """Add more samples to the dataset.

        Args:
            fragments: list of pairs of trajectory fragments to add
            preferences: corresponding preference probabilities (probability
                that fragment 1 is preferred over fragment 2)

        Raises:
            ValueError: `preferences` shape does not match `fragments` or
                has non-float32 dtype.
        """
        assert len(preferences_with_grounding.shape) == 2 and preferences_with_grounding.shape == (
            len(preferences), self.n_values)
        if agent_ids is not None:
            self.l_agent_ids.extend(agent_ids)
        if agent_data is not None:
            self.agent_data.update(agent_data)
            for agent_id in set(agent_ids):
                if agent_id not in self.data_per_agent.keys():
                    self.data_per_agent[agent_id] = VSLPreferenceDataset(
                        self.n_values, single_agent=True)
                idxs_agent_id = np.where(np.asarray(agent_ids) == agent_id)[0]
                self.data_per_agent[agent_id].push(fragments[idxs_agent_id], preferences[idxs_agent_id], preferences_with_grounding[idxs_agent_id, :], agent_ids=[
                                                   agent_id]*len(idxs_agent_id), agent_data=None)
        if len(fragments) == 0:
            return
        fragments1, fragments2 = zip(*fragments)
        if preferences.shape != (len(fragments),):
            raise ValueError(
                f"Unexpected preferences shape {preferences.shape}, "
                f"expected {(len(fragments),)}",
            )
            # raise ValueError("preferences should have dtype float32")

        self.l_fragments1.extend(fragments1)
        self.l_fragments2.extend(fragments2)
        self.preferences = np.concatenate((self.preferences, preferences))
        for i in range(self.n_values):
            self.list_preferences_with_grounding[i] = np.concatenate(
                (self.list_preferences_with_grounding[i], preferences_with_grounding[:, i]))

    @overload
    def __getitem__(self, key: int) -> Tuple[TrajectoryWithValueSystemRewsPair, float]:
        pass

    @overload
    def __getitem__(
        self,
        key: slice,
    ) -> Tuple[types.Pair[Sequence[TrajectoryWithValueSystemRews]], Sequence[float], Sequence[Sequence[float]]]:
        pass

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, np.integer):
            return (self.l_fragments1[key], self.l_fragments2[key]), self.preferences[key], self.preferences_with_grounding[key], self.agent_ids[key]
        else:

            return np.asarray(list(zip(self.fragments1[key], self.fragments2[key]))), self.preferences[key], self.preferences_with_grounding[key], self.agent_ids[key]

    def __len__(self) -> int:
        assert len(self.fragments1) == len(
            self.fragments2) == len(self.preferences)
        return len(self.fragments1)

    def save(self, path: AnyPath) -> None:
        with open(path, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load(path: AnyPath) -> "VSLPreferenceDataset":
        with open(path, "rb") as file:
            return dill.load(file)

    def k_fold_split(self, k: int):
        """Generates k-fold train and validation datasets, ensuring equal agent representation.

        Args:
            k: Number of folds.

        Returns:
            A list of tuples, where each tuple contains a train and validation VSLPreferenceDataset.
        """

        unique_agents = np.unique(self.agent_ids)
        agent_indices = {agent: np.where(self.agent_ids == agent)[
            0] for agent in unique_agents}

        if k >= 2:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)

            # Create agent-specific splits
            agent_splits = {agent: list(
                kf.split(agent_indices[agent])) for agent in unique_agents}
        else:
            # If k is 1, just use a single random split
            agent_splits = {agent: [np.array_split(np.random.permutation(
                len(agent_indices[agent])), 2)] for agent in unique_agents}

        folds = []
        for fold_idx in range(k):
            train_indices = []
            val_indices = []

            # Collect train and val indices for each agent
            for agent in unique_agents:
                train_idx, val_idx = agent_splits[agent][fold_idx]
                train_indices.extend(agent_indices[agent][train_idx])
                val_indices.extend(agent_indices[agent][val_idx])

                # Create train and validation datasets
                train_dataset = VSLPreferenceDataset(self.n_values)
                val_dataset = VSLPreferenceDataset(self.n_values)

                # Populate train dataset
                train_dataset.push(
                    [(self.l_fragments1[i], self.l_fragments2[i])
                     for i in train_indices],
                    self.preferences[train_indices],
                    self.preferences_with_grounding[train_indices, :],
                    agent_ids=self.agent_ids[train_indices],
                )

            # Populate validation dataset
            val_dataset.push(
                [(self.l_fragments1[i], self.l_fragments2[i])
                 for i in val_indices],
                self.preferences[val_indices],
                self.preferences_with_grounding[val_indices, :],
                agent_ids=self.agent_ids[val_indices],
            )
            folds.append((train_dataset, val_dataset))

        # Validation script to check splits
        for fold_idx, (train_dataset, val_dataset) in enumerate(folds):
            train_agents, train_counts = np.unique(
                train_dataset.agent_ids, return_counts=True)
            val_agents, val_counts = np.unique(
                val_dataset.agent_ids, return_counts=True)

            assert set(train_agents) == set(
                unique_agents), f"Fold {fold_idx}: Missing agents in train set"
            assert set(val_agents) == set(
                unique_agents), f"Fold {fold_idx}: Missing agents in val set"

            for agent in unique_agents:
                train_count = train_counts[np.where(
                    train_agents == agent)[0][0]]
                val_count = val_counts[np.where(val_agents == agent)[0][0]]
                total_count = len(agent_indices[agent])
                assert train_count + val_count == total_count, (
                    f"Fold {fold_idx}: Incorrect split for agent {agent}. "
                    f"Train: {train_count}, Val: {val_count}, Total: {total_count}"
                )

        return folds
