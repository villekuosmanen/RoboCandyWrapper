from typing import Any, Dict
import warnings
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robocandywrapper.plugin import DatasetPlugin, PluginInstance


# TODO delete maybe?

class LabelledAffordancesPlugin(DatasetPlugin):
    """Plugin for adding affordance labels to frames."""
    
    def __init__(self, affordance_types: list[str]):
        self.affordance_types = affordance_types
    
    def attach(self, dataset: 'LeRobotDataset') -> 'AffordancesInstance':
        return AffordancesInstance(dataset, self)


class AffordancesInstance(PluginInstance):
    """Dataset-specific affordance labels."""
    
    def __init__(self, dataset: 'LeRobotDataset', config: LabelledAffordancesPlugin):
        super().__init__(dataset)
        self.config = config
        self.affordances: Dict[int, np.ndarray] = {}  # episode_idx -> affordance array
        self._load_affordances()
    
    def get_data_keys(self) -> list[str]:
        """This plugin adds affordance-related keys."""
        return ["affordances", "affordance_mask"]
    
    def get_item_data(self, idx: int, episode_idx: int) -> dict[str, Any]:
        """Get affordances for this frame - operates independently."""
        ep_start = self.dataset.meta.episodes[episode_idx]["dataset_from_index"]
        frame_index_in_episode = idx - ep_start
        
        if episode_idx in self.affordances:
            affordance_vector = self.affordances[episode_idx][frame_index_in_episode]
            has_affordances = True
        else:
            # No affordances for this episode, return zeros
            affordance_vector = np.zeros(len(self.config.affordance_types))
            has_affordances = False
        
        return {
            "affordances": torch.from_numpy(affordance_vector).float(),
            "affordance_mask": torch.tensor(has_affordances, dtype=torch.bool)
        }
    
    def _get_plugin_dir(self) -> Path:
        plugin_dir = Path(self.dataset.root) / "plugins" / "affordances"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        return plugin_dir
    
    def _load_affordances(self):
        """Load affordances - using parquet for dense data."""
        import pandas as pd
        
        plugin_dir = self._get_plugin_dir()
        
        # Load all parquet files
        for parquet_file in plugin_dir.glob("episode_*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                episode_idx = int(parquet_file.stem.split('_')[1])
                
                # Convert to numpy array
                self.affordances[episode_idx] = df.values
            except Exception as e:
                warnings.warn(f"Could not load affordances from {parquet_file}: {e}")
    
    def add_episode_affordances(self, episode_index: int, affordances: np.ndarray):
        """Add affordances for an episode."""
        import pandas as pd
        
        self.affordances[episode_index] = affordances
        
        # Save to parquet
        df = pd.DataFrame(affordances, columns=self.config.affordance_types)
        output_file = self._get_plugin_dir() / f"episode_{episode_index}.parquet"
        df.to_parquet(output_file)
