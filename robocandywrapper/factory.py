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
from robocandywrapper.dataformats.lerobot_21 import LeRobot21Dataset, LeRobot21DatasetMetadata


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
    episodes: Optional[list[int] | dict[str, list[int]]],
    video_backend: str,
    action_delta_indices: Optional[List] = None,
    observation_delta_indices: Optional[List] = None,
    reward_delta_indices: Optional[List] = None,
    use_imagenet_stats: bool = True,
) -> List[LeRobotDataset | LeRobot21Dataset]:
    """Private helper to create dataset instances from a list of repo IDs.
    
    Args:
        repo_ids: List of repository IDs to load.
        root: Root directory for datasets.
        revision: Dataset revision.
        episodes: Specific episodes to load. Can be:
            - None: Load all episodes from all datasets
            - list[int]: Load these episodes from ALL datasets
            - dict[str, list[int]]: Load specific episodes per dataset (key: repo_id, value: episode list)
        video_backend: Video backend to use.
        action_delta_indices: Frame indices for actions.
        observation_delta_indices: Frame indices for observations.
        reward_delta_indices: Frame indices for rewards.
        use_imagenet_stats: Whether to apply ImageNet normalization stats.
    
    Returns:
        List of dataset instances.
    """
    datasets = []

    if isinstance(episodes, dict):
        # Check for keys that don't match repo_ids
        invalid_keys = set(episodes.keys()) - set(repo_ids)
        if invalid_keys:
            logging.warning(
                f"Episode selection dictionary contains keys that do not match any dataset repo_ids: {invalid_keys}. "
                f"These keys will be ignored. Available repo_ids: {repo_ids}"
            )
        # Check for repo_ids that don't have episodes specified
        missing_repo_ids = set(repo_ids) - set(episodes.keys())
        if missing_repo_ids:
            logging.warning(
                f"Repo IDs without episode selection (will load all episodes): {missing_repo_ids}"
            )
    
    for repo_id in repo_ids:
        # Determine which episodes to load for this dataset
        dataset_episodes = None
        if isinstance(episodes, dict):
            # Per-dataset episode selection
            dataset_episodes = episodes.get(repo_id)
        elif isinstance(episodes, list):
            # Same episodes for all datasets
            dataset_episodes = episodes
        
        # Log episode selection for this dataset
        if isinstance(episodes, dict) and dataset_episodes is not None:
            logging.info(f"Loading {len(dataset_episodes)} episodes for {repo_id}")
        elif isinstance(episodes, list):
            logging.info(f"Loading {len(episodes)} episodes for all datasets")
        else:
            logging.info(f"Loading all episodes for {repo_id} (no episode selection provided)")

        # Load metadata first to check version
        try:
            # Try loading metadata with the standard class first
            ds_meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
            # Check for version in multiple places
            version = getattr(ds_meta, "codebase_version", None)
            if version is None and hasattr(ds_meta, "info"):
                version = ds_meta.info.get("codebase_version")
            
            # If version is missing or less than 3.0, treat as legacy
            if version is None or packaging.version.parse(str(version)) < packaging.version.parse("3.0"):
                logging.info(f"Detected legacy dataset version {version} for {repo_id}. Using LeRobot21Dataset.")
                dataset_cls = LeRobot21Dataset
                # Reload metadata with legacy class to be safe
                ds_meta = LeRobot21DatasetMetadata(repo_id, root=root, revision=revision)
            else:
                dataset_cls = LeRobotDataset

        except (BackwardCompatibilityError, NotImplementedError):
            # Fallback for cases where standard metadata loading fails completely
            logging.info(f"Standard metadata loading failed for {repo_id}. Falling back to LeRobot21Dataset.")
            ds_meta = LeRobot21DatasetMetadata(repo_id, root=root, revision=revision)
            dataset_cls = LeRobot21Dataset

        delta_timestamps = resolve_delta_timestamps(
            ds_meta,
            action_delta_indices=action_delta_indices,
            observation_delta_indices=observation_delta_indices,
            reward_delta_indices=reward_delta_indices,
        )

        dataset = dataset_cls(
            repo_id,
            root=root,
            episodes=dataset_episodes,
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
    episodes: list[int] | dict[str, list[int]] | None = None,
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
        episodes (list[int], optional): Specific episodes to load Can be:
            - None: Load all episodes from all datasets
            - list[int]: Load these episodes from ALL datasets
            - dict[str, list[int]]: Load specific episodes per dataset (key: repo_id, value: episode list)
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

