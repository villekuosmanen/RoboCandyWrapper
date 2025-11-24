import datasets
import logging
from typing import Any, Callable, Optional, Sequence, Union
import warnings

from lerobot.datasets.video_utils import VideoFrame
import torch

from robocandywrapper import DatasetPlugin, PluginConflictError, PluginInstance
from robocandywrapper.metadata_view import WrappedRobotDatasetMetadataView


class WrappedRobotDataset(torch.utils.data.Dataset):
    """Extended dataset wrapper with isolated plugin support."""
    
    def __init__(
        self,
        datasets: Union[Any, Sequence[Any]],
        plugins: Optional[list[DatasetPlugin]] = None,
        image_transforms: Optional[Callable] = None,
        warn_on_key_conflicts: bool = True,
        error_on_key_conflicts: bool = True,
        dataset_weights: Optional[dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize extended dataset with plugins.
        
        Args:
            datasets: Single dataset or list of datasets
            plugins: List of plugin classes to attach
            image_transforms: Optional image transforms
            warn_on_key_conflicts: Warn when plugins have overlapping keys (if not raising errors)
            error_on_key_conflicts: Raise error on key conflicts (default: True)
            dataset_weights: Optional weights for computing weighted stats (e.g., {"dataset_id": 2.0})
        """
        super().__init__()
        
        # Normalize to list
        self._datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
        self.image_transforms = image_transforms
        self.warn_on_key_conflicts = warn_on_key_conflicts
        self.error_on_key_conflicts = error_on_key_conflicts
        
        # Calculate dataset boundaries for flat index space
        self._dataset_lengths = [len(dataset) for dataset in self._datasets]
        self._cumulative_lengths = [0]
        for length in self._dataset_lengths:
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + length)
        self._total_length = self._cumulative_lengths[-1]
        
        # Plugin management: one plugin class, many instances (one per dataset)
        self._plugins: list[DatasetPlugin] = plugins or []
        self._plugin_instances: list[list[PluginInstance]] = []
        
        # Attach plugins to each dataset
        for dataset in self._datasets:
            dataset_plugins = []
            for plugin in self._plugins:
                instance = plugin.attach(dataset)
                dataset_plugins.append(instance)
            self._plugin_instances.append(dataset_plugins)
        
        # Validate plugin keys and warn about conflicts
        self._validate_plugin_keys()
        
        # Create metadata view that collates features and stats across all datasets
        self._meta = WrappedRobotDatasetMetadataView(
            datasets=self._datasets,
            plugin_instances=self._plugin_instances,
            dataset_weights=dataset_weights,
        )

        # ** MATCHING LeRobot MULTI-DATASET API DESIGN **
        
        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)

        # Validate that common features have compatible shapes
        for key in intersection_features:
            shapes = []
            for ds in self._datasets:
                if key in ds.meta.features:
                    feature_shape = ds.meta.features[key].get('shape', [])
                    shapes.append(tuple(feature_shape))
            
            # Check if all shapes are the same
            unique_shapes = set(shapes)
            if len(unique_shapes) > 1:
                shape_details = []
                for ds in self._datasets:
                    if key in ds.meta.features:
                        shape = ds.meta.features[key].get('shape', [])
                        shape_details.append(f"{ds.repo_id}: {shape}")
                
                raise ValueError(
                    f"Incompatible shapes for feature '{key}' across datasets:\n" +
                    "\n".join(f"  - {detail}" for detail in shape_details) +
                    f"\n\nCannot mix datasets with different {key} dimensions. "
                    f"This typically happens when mixing datasets from different robot configurations."
                )

        # Keep backward compatible stats property
        self.stats = self._meta.stats
    
    @property
    def repo_ids(self) -> list[str]:
        """Return a list of dataset repo_ids."""
        return [dataset.repo_id for dataset in self._datasets]

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping of repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index.items()}

    @property
    def fps(self) -> int:
        """
        Frames per second used during data collection.
        
        Warns if datasets have different FPS values.
        """
        return self._meta.fps

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files. Assumes all datasets have the same video flag.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        """
        Features available across all datasets and plugins.
        
        Returns union of dataset features (minus disabled ones) and plugin features.
        """
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        
        # Add plugin features (from metadata view)
        # Note: These may not have full type info, but they'll be available
        for key in self._meta.features:
            if key not in features:
                # Plugin feature - will be inferred from actual data
                features[key] = self._meta.features[key]
        
        return features
    
    @property
    def meta(self) -> WrappedRobotDatasetMetadataView:
        """
        Collated metadata view across all datasets and plugins.
        
        Provides features and stats aggregated across all component datasets,
        optionally weighted if dataset_weights were provided during initialization.
        """
        return self._meta

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras (union across all datasets)."""
        return self._meta.camera_keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self._total_length
    
    def get_dataset_ranges(self) -> tuple[list[tuple[int, int]], list[str]]:
        """
        Get dataset boundaries and IDs for use with samplers.
        
        Returns:
            Tuple of:
                - List of (start_idx, end_idx) tuples for each dataset
                - List of dataset repo_id strings
        """
        ranges = []
        ids = []
        for i, dataset in enumerate(self._datasets):
            start_idx = self._cumulative_lengths[i]
            end_idx = self._cumulative_lengths[i + 1]
            ranges.append((start_idx, end_idx))
            ids.append(dataset.repo_id)
        return ranges, ids
    
    def update_dataset_weights(self, dataset_weights: dict[str, float]) -> None:
        """
        Update dataset weights for weighted stats computation.
        
        This is typically called after sampler creation, when weights are
        extracted from the sampler configuration.
        
        Args:
            dataset_weights: Dict mapping dataset repo IDs to weight multipliers
                           (e.g., {"dataset1": 2.0, "dataset2": 0.5})
        """
        self._meta.update_dataset_weights(dataset_weights)
        # Also update the cached stats property
        self.stats = self._meta.stats

    def _validate_plugin_keys(self):
        """
        Check for key conflicts between plugins.
        
        With sequential execution, later plugins override earlier ones.
        """
        for dataset_idx, dataset_plugins in enumerate(self._plugin_instances):
            key_to_plugins: dict[str, list[tuple[int, PluginInstance]]] = {}
            
            # Collect all keys from all plugins
            for plugin_idx, plugin_instance in enumerate(dataset_plugins):
                for key in plugin_instance.get_data_keys():
                    if key not in key_to_plugins:
                        key_to_plugins[key] = []
                    key_to_plugins[key].append((plugin_idx, plugin_instance))
            
            # Check for conflicts
            for key, plugins_with_key in key_to_plugins.items():
                if len(plugins_with_key) > 1:
                    # With sequential execution, last plugin wins
                    plugin_names = [
                        f"{type(p).__name__} (position {idx})"
                        for idx, p in plugins_with_key
                    ]
                    winner_idx, winner_instance = plugins_with_key[-1]
                    winner = type(winner_instance).__name__
                    
                    msg = (
                        f"Key conflict for '{key}' in dataset {dataset_idx} "
                        f"({self._datasets[dataset_idx].repo_id}): "
                        f"multiple plugins provide this key: {plugin_names}. "
                        f"Plugin '{winner}' (last in order) will be used."
                    )
                    
                    if self.error_on_key_conflicts:
                        raise PluginConflictError(msg)
                    elif self.warn_on_key_conflicts:
                        warnings.warn(msg, UserWarning)
    
    def get_plugin_instance(
        self, 
        plugin_type: type[DatasetPlugin], 
        dataset_idx: int = 0
    ) -> Optional[PluginInstance]:
        """
        Get a specific plugin instance for a dataset.
        
        Args:
            plugin_type: Type/class of the plugin
            dataset_idx: Index of the dataset
            
        Returns:
            The plugin instance, or None if not found
        """
        if dataset_idx >= len(self._plugin_instances):
            return None
        
        for instance in self._plugin_instances[dataset_idx]:
            if isinstance(instance, plugin_type.attach(self._datasets[0]).__class__):
                # Check if the instance came from the right plugin type
                # This is a bit hacky - better to store plugin class reference
                pass
        
        # Better approach: store plugin->instance mapping
        for i, plugin in enumerate(self._plugins):
            if isinstance(plugin, plugin_type):
                return self._plugin_instances[dataset_idx][i]
        
        return None
    
    def add_plugin(self, plugin: DatasetPlugin):
        """
        Add a plugin to all datasets dynamically.
        
        Args:
            plugin: The plugin to add
        """
        self._plugins.append(plugin)
        
        # Attach to all datasets
        for i, dataset in enumerate(self._datasets):
            instance = plugin.attach(dataset)
            self._plugin_instances[i].append(instance)
        
        # Re-validate keys
        self._validate_plugin_keys()
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get item with plugin transformations applied sequentially.
        
        Plugins execute in order, and each plugin can access data added
        by previous plugins in the chain.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        
        # Determine which dataset to get an item from based on the index
        dataset_idx = 0
        for i, cumulative_length in enumerate(self._cumulative_lengths[1:]):
            if idx >= cumulative_length:
                dataset_idx = i + 1
            else:
                break
        
        # Get local index within the dataset
        local_idx = idx - self._cumulative_lengths[dataset_idx]
        dataset = self._datasets[dataset_idx]
        
        # Get base item from dataset
        item = dict(dataset[local_idx])
        episode_idx = item["episode_index"].item()
        
        # Add dataset index
        item["dataset_index"] = torch.tensor(dataset_idx)
        
        # Remove disabled features
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]
        
        # Execute plugins sequentially, passing accumulated data
        for plugin_instance in self._plugin_instances[dataset_idx]:
            try:
                # Pass current accumulated data to the plugin
                plugin_data = plugin_instance.get_item_data(local_idx, episode_idx, item)
                
                # Merge plugin data into accumulated item
                if plugin_data:
                    item.update(plugin_data)
                    
            except Exception as e:
                warnings.warn(
                    f"Plugin {type(plugin_instance).__name__} failed for idx {idx}: {e}",
                    UserWarning
                )
        

        # Apply image transforms if provided
        if self.image_transforms is not None:
            for cam_key in dataset.meta.camera_keys:
                if cam_key in item:
                    item[cam_key] = self.image_transforms(item[cam_key])
        
        return item
    
    def __repr__(self):
        plugin_names = [type(p).__name__ for p in self._plugins]
        return (
            f"{self.__class__.__name__}(\n"
            f"  Datasets: {[d.repo_id for d in self._datasets]},\n"
            f"  Plugins: {plugin_names},\n"
            f"  Total samples: {len(self)},\n"
            f")"
        )
    
    def __del__(self):
        """Cleanup plugin instances."""
        for dataset_plugins in self._plugin_instances:
            for plugin_instance in dataset_plugins:
                try:
                    plugin_instance.detach()
                except Exception:
                    pass
