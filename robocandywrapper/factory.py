import logging
from pprint import pformat
from typing import List, Optional

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.constants import ACTION, REWARD

from robocandywrapper.wrapper import WrappedRobotDataset
from robocandywrapper import DatasetPlugin


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps

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
    datasets = []
    
    for repo_id in repo_ids:
        ds_meta = LeRobotDatasetMetadata(
            repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=None,  # Will be applied by WrappedRobotDataset
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
        
        # Apply ImageNet stats if needed
        if cfg.dataset.use_imagenet_stats:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32).numpy()
        
        datasets.append(dataset)
    
    if len(repo_ids) > 1:
        logging.info(
            f"Multiple datasets were provided: {repo_ids}"
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
    action_delta_indices: List,
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
    
    # Create datasets
    datasets = []
    for repo_id_str in repo_ids:
        ds_meta = LeRobotDatasetMetadata(
            repo_id_str,
            root=root if root else HF_LEROBOT_HOME / repo_id_str,
        )
        delta_timestamps = resolve_delta_timestamps_without_config(
            ds_meta, action_delta_indices, observation_delta_indices
        )
        
        dataset = LeRobotDataset(
            repo_id_str,
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
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
        logging.info(
            f"Multiple datasets were provided: {repo_ids}"
        )
    
    # Wrap in WrappedRobotDataset with plugins
    wrapped_dataset = WrappedRobotDataset(
        datasets=datasets,
        plugins=plugins,
    )
    
    return wrapped_dataset

def resolve_delta_timestamps_without_config(
    ds_meta: LeRobotDatasetMetadata, action_delta_indices: List, observation_delta_indices: List = None
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "action" and action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in action_delta_indices]
        if key.startswith("observation.state") and observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps

