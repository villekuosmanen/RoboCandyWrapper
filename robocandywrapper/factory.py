import logging
from typing import List, Optional

import torch
import packaging.version

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.backward_compatibility import BackwardCompatibilityError
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.utils.constants import ACTION, REWARD

from robocandywrapper.wrapper import WrappedRobotDataset
from robocandywrapper import DatasetPlugin
from robocandywrapper.datasets.legacy_dataset import LegacyLeRobotDataset, LegacyLeRobotDatasetMetadata

# Version threshold for legacy dataset detection
LEGACY_VERSION_THRESHOLD = packaging.version.parse("2.1")

def _indices_to_times(indices: List, fps: float) -> List[float]:
    """Helper to convert frame indices to time offsets."""
    return [i / fps for i in indices]

def resolve_delta_timestamps(
    ds_meta: LeRobotDatasetMetadata,
    cfg: Optional[PreTrainedConfig] = None,
    action_delta_indices: Optional[List] = None,
    observation_delta_indices: Optional[List] = None,
    reward_delta_indices: Optional[List] = None,
) -> dict[str, list] | None:
    """Converts frame indices into temporal offsets using dataset FPS.
    
    Accepts either a config object OR direct index arrays.
    
    Args:
        ds_meta: Dataset metadata providing the FPS for conversion.
        cfg: Optional config containing delta_indices. If provided, takes precedence.
        action_delta_indices: Direct frame indices for actions (used if cfg is None).
        observation_delta_indices: Direct frame indices for observations (used if cfg is None).
        reward_delta_indices: Direct frame indices for rewards (used if cfg is None).
    
    Returns:
        Dictionary mapping features to time offsets (in seconds), or None if no deltas configured.
    """
    # Extract indices from config if provided, otherwise use direct parameters
    if cfg is not None:
        action_indices = cfg.action_delta_indices
        observation_indices = cfg.observation_delta_indices
        reward_indices = cfg.reward_delta_indices
    else:
        action_indices = action_delta_indices
        observation_indices = observation_delta_indices
        reward_indices = reward_delta_indices
    
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and reward_indices is not None:
            delta_timestamps[key] = _indices_to_times(reward_indices, ds_meta.fps)
        if key == ACTION and action_indices is not None:
            delta_timestamps[key] = _indices_to_times(action_indices, ds_meta.fps)
        if key.startswith("observation.") and observation_indices is not None:
            delta_timestamps[key] = _indices_to_times(observation_indices, ds_meta.fps)
    
    return delta_timestamps if delta_timestamps else None

def _create_datasets(
    repo_ids: List[str],
    root: Optional[str],
    revision: Optional[str],
    episodes: Optional[list[int]],
    video_backend: str,
    action_delta_indices: Optional[List] = None,
    observation_delta_indices: Optional[List] = None,
    reward_delta_indices: Optional[List] = None,
    use_imagenet_stats: bool = True,
) -> List[LeRobotDataset | LegacyLeRobotDataset]:
    """Private helper to create dataset instances from a list of repo IDs.
    
    Args:
        repo_ids: List of repository IDs to load.
        root: Root directory for datasets.
        revision: Dataset revision.
        episodes: Specific episodes to load.
        video_backend: Video backend to use.
        action_delta_indices: Frame indices for actions.
        observation_delta_indices: Frame indices for observations.
        reward_delta_indices: Frame indices for rewards.
        use_imagenet_stats: Whether to apply ImageNet normalization stats.
    
    Returns:
        List of dataset instances.
    """
    datasets = []
    
    for repo_id in repo_ids:
        # Try loading as v3.0 first
        try:
            ds_meta = LeRobotDatasetMetadata(
                repo_id, root=root, revision=revision
            )
            dataset_cls = LeRobotDataset
        except (BackwardCompatibilityError, NotImplementedError):
            # use legacy loader
            ds_meta = LegacyLeRobotDatasetMetadata(
                repo_id, root=root, revision=revision
            )
            dataset_cls = LegacyLeRobotDataset

        delta_timestamps = resolve_delta_timestamps(
            ds_meta,
            action_delta_indices=action_delta_indices,
            observation_delta_indices=observation_delta_indices,
            reward_delta_indices=reward_delta_indices,
        )

        dataset = dataset_cls(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=None,  # Will be applied by WrappedRobotDataset
            revision=revision,
            video_backend=video_backend,
        )
        
        # Apply ImageNet stats if needed
        if use_imagenet_stats:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32).numpy()
        
        datasets.append(dataset)
    
    if len(repo_ids) > 1:
        logging.info(f"Multiple datasets were provided: {repo_ids}")
    
    return datasets


def make_dataset(
    cfg: TrainPipelineConfig,
    plugins: Optional[list[DatasetPlugin]] = None,
) -> WrappedRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.
        plugins (Optional[list[DatasetPlugin]]): Optional list of plugins to attach to the dataset(s).

    Returns:
        WrappedRobotDataset: A wrapped dataset with plugin support.
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    # Handle single or multiple datasets
    if cfg.dataset.repo_id.startswith('['):
        repo_ids = cfg.dataset.repo_id.strip('[]').split(',')
        repo_ids = [x.strip(' \'') for x in repo_ids]
    else:
        repo_ids = [cfg.dataset.repo_id]
    
    # Create datasets using the helper
    datasets = _create_datasets(
        repo_ids=repo_ids,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
        episodes=cfg.dataset.episodes,
        video_backend=cfg.dataset.video_backend,
        action_delta_indices=cfg.policy.action_delta_indices,
        observation_delta_indices=cfg.policy.observation_delta_indices,
        reward_delta_indices=cfg.policy.reward_delta_indices,
        use_imagenet_stats=cfg.dataset.use_imagenet_stats,
    )
    
    # Wrap in WrappedRobotDataset with plugins
    wrapped_dataset = WrappedRobotDataset(
        datasets=datasets,
        plugins=plugins,
        image_transforms=image_transforms,
    )
    
    return wrapped_dataset


def make_dataset_without_config(
    repo_id: str | list[str],
    action_delta_indices: List = None,
    observation_delta_indices: List = None,
    root: str = None,
    video_backend: str = "pyav",
    episodes: list[int] | None = None,
    revision: str | None = None,
    use_imagenet_stats: bool = True,
    plugins: Optional[list[DatasetPlugin]] = None,
) -> WrappedRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        repo_id (str | list[str]): Single repo ID string, list of repo IDs, or bracket-enclosed string "[repo1, repo2]"
        action_delta_indices (List): Delta indices for actions
        observation_delta_indices (List, optional): Delta indices for observations
        root (str, optional): Root directory for datasets
        video_backend (str): Video backend to use (default: "pyav")
        episodes (list[int], optional): Specific episodes to load
        revision (str, optional): Dataset revision
        use_imagenet_stats (bool): Whether to use ImageNet normalization stats (default: True)
        plugins (Optional[list[DatasetPlugin]]): Optional list of plugins to attach to the dataset(s)

    Returns:
        WrappedRobotDataset: A wrapped dataset with plugin support.
    """
    # Parse repo_id into a list
    if isinstance(repo_id, str) and repo_id.startswith('['):
        # Handle bracket-enclosed string format: "[repo1, repo2]"
        repo_ids = repo_id.strip('[]').split(',')
        repo_ids = [x.strip() for x in repo_ids]
    elif isinstance(repo_id, list):
        repo_ids = repo_id
    else:
        repo_ids = [repo_id]
    
    # Create datasets using the helper
    datasets = _create_datasets(
        repo_ids=repo_ids,
        root=root,
        revision=revision,
        episodes=episodes,
        video_backend=video_backend,
        action_delta_indices=action_delta_indices,
        observation_delta_indices=observation_delta_indices,
        use_imagenet_stats=use_imagenet_stats,
    )
    
    # Wrap in WrappedRobotDataset with plugins
    wrapped_dataset = WrappedRobotDataset(
        datasets=datasets,
        plugins=plugins,
    )
    
    return wrapped_dataset

