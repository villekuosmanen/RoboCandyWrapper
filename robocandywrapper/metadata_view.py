"""
Metadata view for WrappedRobotDataset that collates metadata from multiple datasets.
"""
import logging
from typing import Optional

import numpy as np


def aggregate_stats_weighted(
    stats_list: list[dict[str, dict]],
    weights: list[float],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Aggregate stats from multiple datasets with custom weights.
    
    Uses the correct statistical formula for combining variances:
    new_var = weighted_avg(stds**2 + (means - new_mean)**2)
    
    This accounts for both within-group and between-group variance.
    
    Args:
        stats_list: List of stats dicts from each dataset
        weights: List of weight multipliers for each dataset (e.g., [1.0, 2.0])
    
    Returns:
        Aggregated stats dictionary
    """
    if len(stats_list) != len(weights):
        raise ValueError("stats_list and weights must have the same length")
    
    # Get union of all data keys
    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}
    
    for key in data_keys:
        # Get stats for this key from all datasets that have it
        stats_with_key = []
        weights_with_key = []
        
        for stats, weight in zip(stats_list, weights):
            if key in stats:
                stats_with_key.append(stats[key])
                weights_with_key.append(weight)
        
        if not stats_with_key:
            continue
        
        # Extract arrays
        means = np.stack([np.array(s["mean"]) for s in stats_with_key])
        stds = np.stack([np.array(s["std"]) for s in stats_with_key])
        mins = np.stack([np.array(s["min"]) for s in stats_with_key])
        maxs = np.stack([np.array(s["max"]) for s in stats_with_key])
        
        # Get counts and apply weight multipliers
        # Extract scalar count value (handle both scalar and array counts)
        counts = np.array([
            float(np.atleast_1d(s.get("count", 1))[0]) 
            for s in stats_with_key
        ], dtype=np.float64)
        
        # Effective counts = original counts * weight multipliers
        effective_counts = counts * np.array(weights_with_key, dtype=np.float64)
        total_count = effective_counts.sum()
        
        # Compute weighted mean
        new_mean = np.average(means, axis=0, weights=effective_counts)
        
        # Compute combined variance using the correct formula
        # Var = E[(X - μ)²] = E[X²] - μ²
        # When combining groups: σ²_total = Σ(w_i * (σ²_i + (μ_i - μ_total)²))
        new_var = np.average(
            stds**2 + (means - new_mean)**2,
            axis=0,
            weights=effective_counts
        )
        new_std = np.sqrt(new_var)
        
        # Min/max are elementwise across datasets (not weighted)
        aggregated_stats[key] = {
            "min": mins.min(axis=0),
            "max": maxs.max(axis=0),
            "mean": new_mean,
            "std": new_std,
            "count": int(total_count),
        }
    
    return aggregated_stats


class WrappedRobotDatasetMetadataView:
    """
    Provides a collated view of metadata across multiple datasets.
    
    This class acts as a facade that exposes the necessary metadata fields
    (features and stats) needed by policy factories, while handling the
    complexity of multiple datasets and optional weighted sampling.
    """
    
    def __init__(
        self,
        datasets: list,
        plugin_instances: list[list],
        dataset_weights: Optional[dict[str, float]] = None,
        dataset_renames: Optional[list[dict[str, str]]] = None,
    ):
        """
        Initialize metadata view.
        
        Args:
            datasets: List of LeRobotDataset instances
            plugin_instances: List of plugin instances for each dataset
            dataset_weights: Optional weights for each dataset (for weighted stats)
            dataset_renames: Optional list of rename dicts for each dataset,
                mapping source_key -> target_key. Used to unify keys across
                datasets with different naming conventions.
        """
        self._datasets = datasets
        self._plugin_instances = plugin_instances
        self._dataset_weights = dataset_weights or {}
        self._dataset_renames = dataset_renames or [{} for _ in datasets]
        
        # Cache computed properties
        self._features = None
        self._stats = None
        self._tasks = None
    
    def _get_renamed_features(self, dataset_idx: int) -> dict[str, dict]:
        """Get features from a dataset with key renames applied."""
        dataset = self._datasets[dataset_idx]
        renames = self._dataset_renames[dataset_idx]
        
        renamed_features = {}
        for key, value in dataset.meta.features.items():
            # Apply rename if applicable
            effective_key = renames.get(key, key)
            renamed_features[effective_key] = value
        
        return renamed_features

    @property
    def features(self) -> dict[str, dict]:
        """
        Collated features across all datasets and plugins.
        
        Returns intersection of:
        1. Features from all datasets (taking intersection, not union)
        2. Features provided by plugins (added to intersection)
        
        Key renames are applied before computing the intersection, allowing
        datasets with different naming conventions to be unified.
        """
        if self._features is not None:
            return self._features
        
        # Start with features from first dataset, then intersect with others
        if not self._datasets:
            all_features = {}
        else:
            # Start with all features from first dataset (with renames applied)
            all_features = self._get_renamed_features(0)
            
            # Intersect with features from other datasets
            for i in range(1, len(self._datasets)):
                dataset_features = self._get_renamed_features(i)
                dataset_feature_keys = set(dataset_features.keys())
                all_feature_keys = set(all_features.keys())
                
                # Keep only features that exist in both
                keys_to_keep = all_feature_keys & dataset_feature_keys
                all_features = {k: all_features[k] for k in keys_to_keep}
        
        # Add plugin-provided features (these are always added)
        for dataset_plugins in self._plugin_instances:
            for plugin_instance in dataset_plugins:
                # Get data keys from plugins
                plugin_keys = plugin_instance.get_data_keys()
                # Note: We don't have type information for plugin features
                # They will be inferred by the policy factory from actual data
                for key in plugin_keys:
                    if key not in all_features:
                        # Mark as plugin-provided (policy will infer shape/type)
                        all_features[key] = {
                            "dtype": "unknown",  # Will be inferred
                            "shape": [],
                            "names": None,
                        }
        
        self._features = all_features
        return self._features
    
    def _get_renamed_stats(self, dataset_idx: int) -> dict[str, dict]:
        """Get stats from a dataset with key renames applied."""
        dataset = self._datasets[dataset_idx]
        renames = self._dataset_renames[dataset_idx]
        
        renamed_stats = {}
        for key, value in dataset.meta.stats.items():
            # Apply rename if applicable
            effective_key = renames.get(key, key)
            renamed_stats[effective_key] = value
        
        return renamed_stats

    @property
    def stats(self) -> dict:
        """
        Collated statistics across datasets, optionally weighted.
        
        If dataset_weights are provided, stats are computed as a weighted
        average based on effective dataset sizes (size * weight).
        Uses the correct statistical formula for combining variances.
        
        Key renames are applied before aggregation, so different source keys
        (e.g., "action.pos" and "trajectory") that map to the same target key
        (e.g., "action") will have their stats combined as if they were the
        same key across all datasets.
        """
        if self._stats is not None:
            return self._stats
        
        # Collect stats (with renames applied) and weights for each dataset
        stats_list = [self._get_renamed_stats(i) for i in range(len(self._datasets))]
        
        # Get weight multiplier for each dataset
        weights = []
        for dataset in self._datasets:
            weight_multiplier = 1.0
            if self._dataset_weights and dataset.repo_id in self._dataset_weights:
                weight_multiplier = self._dataset_weights[dataset.repo_id]
            weights.append(weight_multiplier)
        
        # Use our custom weighted aggregation
        self._stats = aggregate_stats_weighted(stats_list, weights)
        
        return self._stats

    @property
    def tasks(self) -> dict[int, str]:
        """
        Collated task index -> task name mapping across all datasets.

        Collects all unique task strings from every dataset that has a `tasks`
        attribute on its meta (e.g. LeRobot v2.1 datasets). Assigns a unified
        task_index to each unique task name (sorted for determinism), so
        consumers get a single dict[int, str] suitable for _coerce_task_mapping
        and similar use.
        """
        if self._tasks is not None:
            return self._tasks

        all_task_names: set[str] = set()
        for dataset in self._datasets:
            meta_tasks = getattr(dataset.meta, "tasks", None)
            if meta_tasks is not None and isinstance(meta_tasks, dict):
                for _idx, name in meta_tasks.items():
                    if isinstance(name, str):
                        all_task_names.add(name)

        sorted_names = sorted(all_task_names)
        self._tasks = {idx: name for idx, name in enumerate(sorted_names)}
        return self._tasks

    @property
    def task_to_task_index(self) -> dict[str, int]:
        """
        Reverse mapping of tasks: task name -> unified task_index.

        Matches the single-dataset meta API so code can use either
        meta.tasks or meta.task_to_task_index.
        """
        return {name: idx for idx, name in self.tasks.items()}

    @property
    def repo_id(self) -> str:
        """Return a combined repo_id representing all datasets."""
        if len(self._datasets) == 1:
            return self._datasets[0].repo_id
        return f"multi_dataset[{len(self._datasets)}]"
    
    @property
    def fps(self) -> int:
        """
        Return FPS, warning if datasets have different values.
        
        Returns the first dataset's FPS and logs a warning if others differ.
        """
        if not self._datasets:
            raise ValueError("No datasets available")
        
        first_fps = self._datasets[0].meta.fps
        
        # Check if all datasets have the same FPS
        different_fps = []
        for dataset in self._datasets[1:]:
            if dataset.meta.fps != first_fps:
                different_fps.append((dataset.repo_id, dataset.meta.fps))
        
        if different_fps:
            logging.warning(
                f"Datasets have different FPS values! Using FPS={first_fps} from {self._datasets[0].repo_id}. "
                f"Other values: {different_fps}. "
                f"This may cause synchronization issues if not handled carefully."
            )
        
        return first_fps
    
    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities across all datasets."""
        all_camera_keys = set()
        for dataset in self._datasets:
            all_camera_keys.update(dataset.meta.camera_keys)
        return sorted(list(all_camera_keys))
    
    @property
    def image_keys(self) -> list[str]:
        """Keys to access image modalities across all datasets."""
        all_image_keys = set()
        for dataset in self._datasets:
            all_image_keys.update(dataset.meta.image_keys)
        return sorted(list(all_image_keys))
    
    @property
    def video_keys(self) -> list[str]:
        """Keys to access video modalities across all datasets."""
        all_video_keys = set()
        for dataset in self._datasets:
            all_video_keys.update(dataset.meta.video_keys)
        return sorted(list(all_video_keys))
    
    def update_dataset_weights(self, dataset_weights: dict[str, float]) -> None:
        """
        Update dataset weights and invalidate cached stats.
        
        This allows weights to be set after the metadata view is created,
        which is useful when weights come from sampler configuration.
        
        Args:
            dataset_weights: Dict mapping dataset repo IDs to weight multipliers
        """
        self._dataset_weights = dataset_weights
        # Invalidate cached stats so they'll be recomputed with new weights
        self._stats = None
        
        logging.info(f"Updated dataset weights in metadata view: {dataset_weights}")
    
    def __repr__(self):
        dataset_ids = [d.repo_id for d in self._datasets]
        return (
            f"{self.__class__.__name__}(\n"
            f"  Datasets: {dataset_ids},\n"
            f"  Total features: {len(self.features)},\n"
            f"  FPS: {self.fps},\n"
            f")"
        )

