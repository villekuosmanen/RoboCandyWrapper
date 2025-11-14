from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pathlib import Path
import hashlib

from lerobot.datasets.lerobot_dataset import LeRobotDataset

class DatasetPlugin(ABC):
    """
    Lightweight plugin interface for extending LeRobotDataset.
    
    Design principles:
    - Plugins operate independently and don't see each other's outputs
    - Minimal interface - plugins decide their own storage/caching
    - Each dataset gets its own plugin instance
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
    def get_item_data(self, idx: int, episode_idx: int) -> dict[str, Any]:
        """
        Get plugin data for a specific item.
        
        This is called independently for each plugin. The plugin should return
        a dict with only the keys it wants to add (from get_data_keys()).
        
        Args:
            idx: Global index in the dataset
            episode_idx: Episode index
            
        Returns:
            Dict with plugin-specific data to add to the item
        """
        pass
    
    def priority(self) -> int:
        """
        Calculate priority based on hash of class name and module.
        
        This provides a consistent, deterministic ordering across all plugins
        without allowing developers to arbitrarily set themselves as highest priority.
        The full class path (module + class name) is hashed to create the priority.
        
        Returns:
            Integer priority derived from hash (lower values have higher priority)
        """
        # Get fully qualified class name (module + class name)
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        full_name = f"{module_name}.{class_name}"
        
        # Create deterministic hash and convert to integer
        hash_value = hashlib.sha256(full_name.encode('utf-8')).hexdigest()
        # Use first 8 hex chars to get a reasonable integer range
        priority_int = int(hash_value[:8], 16)
        
        return priority_int
    
    def detach(self):
        """Optional cleanup when dataset is destroyed."""
        pass


class PluginConflictError(Exception):
    """Raised when plugins have conflicting data keys."""
    pass
