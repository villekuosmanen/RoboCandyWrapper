import datasets
import bisect
import logging
from typing import Any, Callable, Optional, Sequence, Union
import warnings

from lerobot.datasets.video_utils import VideoFrame
from lerobot.configs.types import FeatureType, PolicyFeature
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
        key_rename_map: Optional[dict[str, str]] = None,
        pad_to_max_dim: bool = False,
        fill_missing_images: str = "disable",
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
            key_rename_map: Optional mapping from source keys to target keys for unifying
                datasets with different naming conventions. Keys are renamed before the
                intersection logic runs, allowing datasets with different key names to be
                combined. Example: {"action.pos": "action", "trajectory": "action"}
                
                Note: When a key is renamed, any corresponding "_is_pad" key (added by
                LeRobot when using delta_timestamps) is automatically renamed as well.
                E.g., "action.pos" -> "action" also renames "action.pos_is_pad" -> "action_is_pad".
            pad_to_max_dim: If True, features with different shapes across datasets
                (e.g. 7-dim vs 14-dim actions) are zero-padded to the max dim instead
                of raising an error. Adds ``action_dim_mask`` (bool tensor, True for
                real dims) to each item so downstream loss functions can ignore the
                padded dimensions.
            fill_missing_images: How to handle image keys not present in all datasets.
                - "disable" (default): remove the key entirely (original behaviour)
                - "zeros": fill with a zero tensor of the same shape as other datasets
                - "noise": fill with random noise (uniform [0, 255] uint8)
                When set to "zeros" or "noise", an ``image_mask`` dict entry is also
                added (True = real image, False = filled placeholder).
        """
        super().__init__()
        
        # Normalize to list
        self._datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
        self.image_transforms = image_transforms
        self.warn_on_key_conflicts = warn_on_key_conflicts
        self.error_on_key_conflicts = error_on_key_conflicts
        self.pad_to_max_dim = pad_to_max_dim
        self.fill_missing_images = fill_missing_images
        
        # Calculate dataset boundaries for flat index space
        self._dataset_lengths = []
        self._index_maps = [] # List of (list[int] | None)
        
        for dataset in self._datasets:
            # Get valid indices for the dataset
            valid_indices = self._get_valid_indices(dataset)
            self._index_maps.append(valid_indices)
            
            if valid_indices is not None:
                self._dataset_lengths.append(len(valid_indices))
            else:
                self._dataset_lengths.append(len(dataset))

        # Calculate cumulative lengths
        self._cumulative_lengths = [0]
        for length in self._dataset_lengths:
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + length)
        self._total_length = self._cumulative_lengths[-1]
        
        # Key rename mapping: unify differently-named keys across datasets
        self.key_rename_map = key_rename_map or {}
        self._dataset_renames = self._compute_dataset_renames()
        
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
            dataset_renames=self._dataset_renames,
        )

        # ** MATCHING LeRobot MULTI-DATASET API DESIGN **
        
        # Compute feature intersection across datasets (post-rename).
        # Image keys that are missing from some datasets can optionally be
        # filled with zeros/noise instead of being disabled.
        union_features = set()
        all_ds_features = []
        for i in range(len(self._datasets)):
            ef = self._get_effective_features(i)
            all_ds_features.append(ef)
            union_features.update(ef)

        intersection_features = set.intersection(*all_ds_features) if all_ds_features else set()

        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )

        # Determine which non-common features to disable vs fill
        self.disabled_features = set()
        self._filled_image_keys: set[str] = set()  # image keys that need filling per-item

        for i, repo_id in enumerate(self.repo_ids):
            extra_keys = all_ds_features[i].difference(intersection_features)
            for key in extra_keys:
                is_image = "image" in key.lower() or "cam" in key.lower()
                if is_image and self.fill_missing_images != "disable":
                    self._filled_image_keys.add(key)
                else:
                    if key not in self._filled_image_keys:
                        self.disabled_features.add(key)

        # Promote filled image keys into the effective feature set
        for key in self._filled_image_keys:
            if key in self.disabled_features:
                self.disabled_features.discard(key)

        if self.disabled_features:
            logging.warning(
                f"Non-common features disabled: {self.disabled_features} "
                f"(not in all datasets and not eligible for filling)"
            )

        active_features = intersection_features | self._filled_image_keys

        # Validate shapes and compute padding info for active features
        self._feature_max_dims: dict[str, int] = {}  # key -> max last-dim across datasets
        self._per_dataset_dims: dict[str, dict[int, int]] = {}  # key -> {ds_idx: dim}

        for key in active_features:
            shapes = []
            per_ds = {}
            for i, ds in enumerate(self._datasets):
                renames = self._dataset_renames[i]
                reverse_renames = {v: k for k, v in renames.items()}
                original_key = reverse_renames.get(key, key)
                if original_key in ds.meta.features:
                    feature_shape = ds.meta.features[original_key].get('shape', [])
                    shapes.append(tuple(feature_shape))
                    if feature_shape:
                        per_ds[i] = feature_shape[-1]

            unique_shapes = set(shapes)
            if len(unique_shapes) > 1:
                if self.pad_to_max_dim:
                    max_dim = max(s[-1] for s in unique_shapes if s)
                    self._feature_max_dims[key] = max_dim
                    self._per_dataset_dims[key] = per_ds
                    logging.info(
                        f"Feature '{key}' has mixed dims {unique_shapes}; "
                        f"padding to {max_dim} (pad_to_max_dim=True)"
                    )
                else:
                    shape_details = []
                    for i, ds in enumerate(self._datasets):
                        renames = self._dataset_renames[i]
                        reverse_renames = {v: k for k, v in renames.items()}
                        original_key = reverse_renames.get(key, key)
                        if original_key in ds.meta.features:
                            feature_shape = ds.meta.features[original_key].get('shape', [])
                            shape_details.append(f"{ds.repo_id}: {feature_shape}")
                    raise ValueError(
                        f"Incompatible shapes for feature '{key}' across datasets:\n" +
                        "\n".join(f"  - {detail}" for detail in shape_details) +
                        f"\n\nCannot mix datasets with different {key} dimensions. "
                        f"Use pad_to_max_dim=True to zero-pad smaller dims."
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
    def plugin_features(self) -> datasets.Features:
        """
        Features added only by plugins.
        
        Returns only the features that come from plugins, excluding any
        features that are part of the base datasets.
        """
        # Get all base dataset features (excluding disabled ones)
        base_features = set()
        for dataset in self._datasets:
            base_features.update(k for k in dataset.hf_features.keys() if k not in self.disabled_features)
        
        # Return only plugin features that are not in base datasets
        plugin_only_features = {}
        for key, value in self._meta.features.items():
            if key not in base_features:
                if 'action' in key:
                    plugin_only_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=value['shape'])
                else:
                    plugin_only_features[key] = PolicyFeature(type=FeatureType.STATE, shape=value['shape'])

        return plugin_only_features
    
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
        return sum(len(d) for d in self._datasets)

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

    def _compute_dataset_renames(self) -> list[dict[str, str]]:
        """
        Pre-compute which key renames apply to each dataset.
        
        For each dataset, determines which source keys from key_rename_map exist
        and can be renamed (i.e., target key doesn't already exist).
        
        Also automatically handles derived _is_pad keys that LeRobot adds when
        delta_timestamps are used. For example, if renaming "action.pos" -> "action",
        this will also rename "action.pos_is_pad" -> "action_is_pad".
        
        Returns:
            List of dicts mapping source_key -> target_key for each dataset
        """
        dataset_renames = []
        for dataset in self._datasets:
            ds_renames = {}
            ds_keys = set(dataset.features)
            
            for source, target in self.key_rename_map.items():
                if source in ds_keys:
                    if target in ds_keys:
                        # Target already exists in this dataset - skip rename to avoid conflict
                        logging.warning(
                            f"Skipping rename '{source}' -> '{target}' for {dataset.repo_id}: "
                            f"target key already exists in dataset"
                        )
                    else:
                        ds_renames[source] = target
                        
                        # Also handle the _is_pad suffix that LeRobot adds for delta_timestamps
                        # These keys are dynamically added during __getitem__ and may not be in
                        # dataset.features, but we still want to rename them consistently
                        is_pad_source = f"{source}_is_pad"
                        is_pad_target = f"{target}_is_pad"
                        
                        # Check for conflicts on the _is_pad key as well
                        if is_pad_target in ds_keys:
                            logging.warning(
                                f"Skipping derived rename '{is_pad_source}' -> '{is_pad_target}' "
                                f"for {dataset.repo_id}: target key already exists in dataset"
                            )
                        else:
                            ds_renames[is_pad_source] = is_pad_target
            
            dataset_renames.append(ds_renames)
        
        return dataset_renames
    
    def _get_effective_features(self, dataset_idx: int) -> set[str]:
        """
        Get the effective feature keys for a dataset after applying renames.
        
        Args:
            dataset_idx: Index of the dataset
            
        Returns:
            Set of feature keys that would exist after renaming
        """
        ds = self._datasets[dataset_idx]
        renames = self._dataset_renames[dataset_idx]
        
        effective = set()
        for key in ds.features:
            if key in renames:
                effective.add(renames[key])
            else:
                effective.add(key)
        
        return effective

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
            if isinstance(instance, plugin_type.attach(self._datasets[dataset_idx]).__class__):
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
    
    def _get_valid_indices(self, dataset) -> list[int] | None:
        """
        Returns a list of valid frame indices for the given dataset based on its selected episodes.
        Returns None if no filtering is needed (all indices valid).
        """
        if not hasattr(dataset, 'episodes') or dataset.episodes is None:
            return None
            
        # If the dataset already correctly reports filtered length (like LegacyLeRobotDataset),
        # we don't need to do manual mapping (assuming it handles getitem correctly too)
        total_frames_meta = dataset.meta.total_frames
        if len(dataset) < total_frames_meta:
            return None
            
        # Calculate valid indices for LeRobotDataset (v3.0) which loads everything
        valid_indices = []
        episodes_meta = dataset.meta.episodes
        
        # Handle both dict and list structures
        if isinstance(episodes_meta, dict):
            # Dict structure: episodes_meta[ep_idx] gives episode info
            for ep_idx in dataset.episodes:
                if ep_idx not in episodes_meta:
                    logging.warning(f"Episode {ep_idx} not found in metadata, skipping")
                    continue
                ep_info = episodes_meta[ep_idx]
                    
                # Get range of frames for this episode
                start = ep_info.get("dataset_from_index")
                end = ep_info.get("dataset_to_index")
                
                if start is None or end is None:
                     continue
                     
                valid_indices.extend(range(start, end))
        else:
            # List structure: could be full list or filtered
            # Check if it's a full list (length matches total episodes) or filtered (matches selected episodes)
            if len(episodes_meta) == dataset.meta.total_episodes:
                # Full list: use episode index directly
                for ep_idx in dataset.episodes:
                    if ep_idx >= len(episodes_meta):
                        logging.warning(f"Episode index {ep_idx} out of range, skipping")
                        continue
                    ep_info = episodes_meta[ep_idx]
                    
                    start = ep_info.get("dataset_from_index")
                    end = ep_info.get("dataset_to_index")
                    
                    if start is None or end is None:
                        continue
                        
                    valid_indices.extend(range(start, end))
            else:
                # Filtered list: episodes_meta is already filtered, iterate sequentially
                # Assumes episodes_meta and dataset.episodes are in same order
                for ep_meta in episodes_meta:
                    start = ep_meta.get("dataset_from_index")
                    end = ep_meta.get("dataset_to_index")
                    
                    if start is None or end is None:
                        continue
                        
                    valid_indices.extend(range(start, end))
            
        return valid_indices

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
        # Use bisect to find the right interval for efficiency
        dataset_idx = bisect.bisect_right(self._cumulative_lengths, idx) - 1
        if dataset_idx < 0: dataset_idx = 0 # Should not happen if idx >= 0

        # Get local index within the dataset
        local_idx = idx - self._cumulative_lengths[dataset_idx]
        dataset = self._datasets[dataset_idx]
        
        # Get base item from dataset (with episode filtering support)
        if self._index_maps[dataset_idx] is not None:
            # Map virtual index to real dataset index
            real_idx = self._index_maps[dataset_idx][local_idx]
            item = dataset[real_idx]
        else:
            item = dataset[local_idx]

        episode_idx = item["episode_index"].item()
        
        # Add dataset index
        item["dataset_index"] = torch.tensor(dataset_idx)
        
        # Apply key renaming for this dataset (before filtering disabled features)
        renames = self._dataset_renames[dataset_idx]
        for source, target in renames.items():
            if source in item:
                item[target] = item.pop(source)
        
        # Remove disabled features (now operates on effective/renamed key names)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        # Fill missing image keys with zeros/noise
        if self._filled_image_keys:
            # Find a reference image shape from an existing image key in this item
            ref_shape = None
            for k, v in item.items():
                if hasattr(v, 'shape') and len(getattr(v, 'shape', ())) >= 3:
                    ref_shape = v.shape
                    ref_dtype = v.dtype if hasattr(v, 'dtype') else torch.uint8
                    break
            for key in self._filled_image_keys:
                if key not in item and ref_shape is not None:
                    if self.fill_missing_images == "noise":
                        item[key] = torch.randint(0, 256, ref_shape, dtype=torch.uint8)
                    else:
                        item[key] = torch.zeros(ref_shape, dtype=ref_dtype)

        # Pad features with mismatched dims to the max dim across datasets
        if self._feature_max_dims:
            import numpy as np
            for key, max_dim in self._feature_max_dims.items():
                if key not in item:
                    continue
                val = item[key]
                if not hasattr(val, 'shape'):
                    continue
                current_dim = val.shape[-1]
                if current_dim < max_dim:
                    pad_size = max_dim - current_dim
                    if isinstance(val, torch.Tensor):
                        pad = torch.zeros(*val.shape[:-1], pad_size, dtype=val.dtype)
                        item[key] = torch.cat([val, pad], dim=-1)
                    else:
                        pad_widths = [(0, 0)] * (len(val.shape) - 1) + [(0, pad_size)]
                        item[key] = np.pad(val, pad_widths, constant_values=0)

                    # Add a dimension mask for action/state features
                    if "action" in key.lower() or "state" in key.lower():
                        mask_key = f"{key}_dim_mask"
                        mask = torch.zeros(max_dim, dtype=torch.bool)
                        mask[:current_dim] = True
                        item[mask_key] = mask
        
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
