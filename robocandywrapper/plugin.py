from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class DatasetPlugin(ABC):
    """
    Lightweight plugin interface for extending LeRobotDataset.
    
    Design principles:
    - Plugins execute in the order they're defined (left-to-right)
    - Each plugin can access data from previous plugins in the chain
    - Minimal interface - plugins decide their own storage/caching
    - Each dataset gets its own plugin instance
    
    Plugin Order Example:
        plugins = [EpisodeOutcomePlugin(), MyPlugin()]
        # MyPlugin can access 'episode_outcome' data added by EpisodeOutcomePlugin
    """
    
    @abstractmethod
    def attach(self, dataset: 'LeRobotDataset') -> 'PluginInstance':
        """
        Create a dataset-specific plugin instance.
        
        Args:
            dataset: The LeRobotDataset to attach to
            
        Returns:
            PluginInstance for this specific dataset
        """
        pass


class PluginInstance(ABC):
    """
    Dataset-specific instance of a plugin.
    
    Each dataset gets its own instance, which manages data loading,
    caching, and transformations for that specific dataset.
    """
    
    def __init__(self, dataset: 'LeRobotDataset'):
        self.dataset = dataset
    
    @abstractmethod
    def get_data_keys(self) -> list[str]:
        """
        Return the keys this plugin will add to items.
        
        Used for validation and warning about conflicts.
        """
        pass
    
    @abstractmethod
    def get_item_data(
        self,
        idx: int,
        episode_idx: int,
        accumulated_data: Optional[Dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Get plugin data for a specific item.
        
        Plugins execute in order, and each plugin can access data added
        by previous plugins via the accumulated_data dict.
        
        Args:
            idx: Global index in the dataset
            episode_idx: Episode index
            accumulated_data: Dict containing base dataset data + data from
                            previous plugins in the chain. None for first plugin.
                            
        Returns:
            Dict with plugin-specific data to add to the item
            
        Example:
            # MyPlugin depends on EpisodeOutcomePlugin
            def get_item_data(self, idx, episode_idx, accumulated_data=None):
                if accumulated_data and 'episode_outcome' in accumulated_data:
                    outcome = accumulated_data['episode_outcome']
                    # Use outcome to compute my_feature...
                return {"my_feature": ...}
        """
        pass
    
    def detach(self):
        """Optional cleanup when dataset is destroyed."""
        pass


class PluginConflictError(Exception):
    """Raised when plugins have conflicting data keys."""
    pass
