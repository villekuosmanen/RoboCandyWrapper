from typing import Any, Dict, Optional
import warnings
from pathlib import Path

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from robocandywrapper.plugin import DatasetPlugin, PluginInstance
from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, AFFORDANCE_PLUGIN_NAME


class LabelledAffordancePlugin(DatasetPlugin):
    """
    Plugin for adding affordance labels to frames.
    
    Affordances are specified as (x, y) coordinates in the frame,
    normalized to [0, 1] for both axes.
    
    Data is stored in parquet files:
        {dataset_root}/candywrapper_plugins/affordance/episode_{idx}.parquet
    
    Each row in the parquet file contains:
        - frame_idx: Frame index within the episode
        - x: Normalized x coordinate [0, 1]
        - y: Normalized y coordinate [0, 1]
    """
    
    def __init__(self):
        """Initialize the affordance plugin."""
        pass
    
    def attach(self, dataset: 'LeRobotDataset') -> 'AffordanceInstance':
        return AffordanceInstance(dataset, self)


class AffordanceInstance(PluginInstance):
    """
    Dataset-specific affordance labels.
    
    Stores (x, y) coordinates for each frame, normalized to [0, 1].
    """
    
    def __init__(self, dataset: 'LeRobotDataset', config: LabelledAffordancePlugin):
        super().__init__(dataset)
        self.config = config
        self.affordances: Dict[int, np.ndarray] = {}  # episode_idx -> (n_frames, 2) array of (x, y) coords
        self._load_affordances()
    
    def get_data_keys(self) -> list[str]:
        """This plugin adds affordance-related keys."""
        return ["affordance", "affordance_mask"]
    
    def get_item_data(
        self,
        idx: int,
        episode_idx: int,
        accumulated_data: Optional[Dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Get affordance coordinates for this frame.
        
        Args:
            idx: Global index in the dataset
            episode_idx: Episode index
            accumulated_data: Data from previous plugins (not used by this plugin)
        
        Returns:
            dict with:
                - affordance: torch.Tensor of shape (2,) with [x, y] coordinates in [0, 1]
                - affordance_mask: torch.bool indicating if affordance is available
        """
        ep_start = self.dataset.meta.episodes[episode_idx]["dataset_from_index"]
        frame_index_in_episode = idx - ep_start
        
        if episode_idx in self.affordances:
            # Get (x, y) coordinates for this frame
            affordance_xy = self.affordances[episode_idx][frame_index_in_episode]
            has_affordance = True
        else:
            # No affordances for this episode, return zeros
            affordance_xy = np.zeros(2, dtype=np.float32)
            has_affordance = False
        
        return {
            "affordance": torch.from_numpy(affordance_xy).float(),
            "affordance_mask": torch.tensor(has_affordance, dtype=torch.bool)
        }
    
    def _get_plugin_dir(self) -> Path:
        """Get the plugin storage directory for this dataset."""
        plugin_dir = Path(self.dataset.root) / CANDYWRAPPER_PLUGINS_DIR / AFFORDANCE_PLUGIN_NAME
        plugin_dir.mkdir(parents=True, exist_ok=True)
        return plugin_dir
    
    def _load_affordances(self):
        """
        Load affordances from parquet files.
        
        Expected parquet format per episode:
            - frame_idx: int (frame index within episode)
            - x: float (normalized x coordinate [0, 1])
            - y: float (normalized y coordinate [0, 1])
        """
        import pandas as pd
        
        plugin_dir = self._get_plugin_dir()
        
        # Load all parquet files
        for parquet_file in plugin_dir.glob("episode_*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                episode_idx = int(parquet_file.stem.split('_')[1])
                
                # Validate columns
                if not all(col in df.columns for col in ['frame_idx', 'x', 'y']):
                    warnings.warn(
                        f"Parquet file {parquet_file} missing required columns (frame_idx, x, y). "
                        "Skipping this episode."
                    )
                    continue
                
                # Sort by frame_idx to ensure correct ordering
                df = df.sort_values('frame_idx')
                
                # Extract (x, y) coordinates as (n_frames, 2) array
                affordance_coords = df[['x', 'y']].values.astype(np.float32)
                
                # Validate coordinates are in [0, 1]
                if not (np.all(affordance_coords >= 0) and np.all(affordance_coords <= 1)):
                    warnings.warn(
                        f"Affordance coordinates in {parquet_file} are not normalized to [0, 1]. "
                        "Clipping to valid range."
                    )
                    affordance_coords = np.clip(affordance_coords, 0, 1)
                
                self.affordances[episode_idx] = affordance_coords
                
            except Exception as e:
                warnings.warn(f"Could not load affordances from {parquet_file}: {e}")
    
    def add_episode_affordances(self, episode_index: int, affordances: np.ndarray):
        """
        Add affordances for an episode and save to parquet.
        
        Args:
            episode_index: Episode index in the dataset
            affordances: np.ndarray of shape (n_frames, 2) with (x, y) coordinates in [0, 1]
        
        Raises:
            ValueError: If affordances shape is invalid or values out of range
        """
        import pandas as pd
        
        # Validate input
        if affordances.ndim != 2 or affordances.shape[1] != 2:
            raise ValueError(
                f"Expected affordances shape (n_frames, 2), got {affordances.shape}"
            )
        
        if not (np.all(affordances >= 0) and np.all(affordances <= 1)):
            raise ValueError(
                "Affordance coordinates must be normalized to [0, 1]. "
                f"Got min={affordances.min()}, max={affordances.max()}"
            )
        
        self.affordances[episode_index] = affordances.astype(np.float32)
        
        # Save to parquet with explicit frame indices
        n_frames = len(affordances)
        df = pd.DataFrame({
            'frame_idx': np.arange(n_frames),
            'x': affordances[:, 0],
            'y': affordances[:, 1]
        })
        
        output_file = self._get_plugin_dir() / f"episode_{episode_index}.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"Saved {n_frames} affordance labels for episode {episode_index} to {output_file}")
