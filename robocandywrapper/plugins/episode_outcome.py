"""
Episode Outcome Plugin for labeling episode success/failure.
"""
from typing import Any, Dict, Literal, Optional
import json
import warnings
from pathlib import Path

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from robocandywrapper.plugin import DatasetPlugin, PluginInstance
from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, EPISODE_OUTCOME_PLUGIN_NAME


OutcomeType = Literal["success", "failure", "unknown"]


class EpisodeOutcomePlugin(DatasetPlugin):
    """
    Plugin for labeling episode outcomes (success/failure/unknown).
    
    Data is stored in a single JSON file:
        {dataset_root}/candywrapper_plugins/episode_outcome/outcomes.json
    
    JSON format (sorted by episode_index):
    [
        {"episode_index": 0, "outcome": "success"},
        {"episode_index": 1, "outcome": "failure"},
        {"episode_index": 2, "outcome": "unknown"}
    ]
    
    Returns:
        - episode_outcome: torch.bool (True=success, False=failure)
        - episode_outcome_mask: torch.bool (True=labeled, False=unlabeled)
        
    When mask is False, the episode is unlabeled. When mask is True,
    outcome indicates success (True) or failure (False). "unknown"
    labels are treated as unlabeled (mask=False).
    """
    
    def __init__(self):
        """Initialize the episode outcome plugin."""
        pass
    
    def attach(self, dataset: 'LeRobotDataset') -> 'EpisodeOutcomeInstance':
        return EpisodeOutcomeInstance(dataset, self)


class EpisodeOutcomeInstance(PluginInstance):
    """
    Dataset-specific episode outcome labels.
    
    Stores success/failure/unknown outcome for each episode.
    """
    
    def __init__(self, dataset: 'LeRobotDataset', config: EpisodeOutcomePlugin):
        super().__init__(dataset)
        self.config = config
        self.outcomes: Dict[int, OutcomeType] = {}  # episode_idx -> outcome
        self._load_outcomes()
    
    def get_data_keys(self) -> list[str]:
        """This plugin adds outcome-related keys."""
        return ["episode_outcome", "episode_outcome_mask"]
    
    def get_item_data(
        self,
        idx: int,
        episode_idx: int,
        accumulated_data: Optional[Dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Get episode outcome for this frame.
        
        Note: All frames in an episode share the same outcome.
        
        Args:
            idx: Global index in the dataset
            episode_idx: Episode index
            accumulated_data: Data from previous plugins (not used by this plugin)
        
        Returns:
            dict with:
                - episode_outcome: torch.bool (True = success, False = failure)
                - episode_outcome_mask: torch.bool indicating if outcome is labeled
                
            When mask is False, the episode is unlabeled (unknown).
            When mask is True, outcome indicates success (True) or failure (False).
        """
        if episode_idx in self.outcomes:
            outcome = self.outcomes[episode_idx]
            
            if outcome == "unknown":
                # Labeled as unknown - treat as unlabeled
                is_success = False
                has_outcome = False
            else:
                # Labeled as success or failure
                is_success = (outcome == "success")
                has_outcome = True
        else:
            # No outcome for this episode (unlabeled)
            is_success = False
            has_outcome = False
        
        return {
            "episode_outcome": torch.tensor(is_success, dtype=torch.bool),
            "episode_outcome_mask": torch.tensor(has_outcome, dtype=torch.bool)
        }
    
    def _get_plugin_dir(self) -> Path:
        """Get the plugin storage directory for this dataset."""
        plugin_dir = Path(self.dataset.root) / CANDYWRAPPER_PLUGINS_DIR / EPISODE_OUTCOME_PLUGIN_NAME
        plugin_dir.mkdir(parents=True, exist_ok=True)
        return plugin_dir
    
    def _get_outcomes_file(self) -> Path:
        """Get path to the outcomes JSON file."""
        return self._get_plugin_dir() / "outcomes.json"
    
    def _load_outcomes(self):
        """
        Load episode outcomes from JSON file.
        
        Expected JSON format:
        [
            {"episode_index": 0, "outcome": "success"},
            {"episode_index": 1, "outcome": "failure"},
            ...
        ]
        """
        outcomes_file = self._get_outcomes_file()
        
        if not outcomes_file.exists():
            return
        
        try:
            with open(outcomes_file, 'r') as f:
                data = json.load(f)
            
            # Validate format
            if not isinstance(data, list):
                warnings.warn(
                    f"Outcomes file {outcomes_file} should contain a JSON array. "
                    "Found non-array format. Skipping."
                )
                return
            
            # Load outcomes
            for entry in data:
                if not isinstance(entry, dict):
                    warnings.warn(f"Skipping invalid entry in outcomes file: {entry}")
                    continue
                
                if "episode_index" not in entry or "outcome" not in entry:
                    warnings.warn(f"Skipping entry missing required fields: {entry}")
                    continue
                
                episode_idx = entry["episode_index"]
                outcome = entry["outcome"]
                
                # Validate outcome value
                if outcome not in ["success", "failure", "unknown"]:
                    warnings.warn(
                        f"Invalid outcome '{outcome}' for episode {episode_idx}. "
                        "Must be 'success', 'failure', or 'unknown'. Skipping."
                    )
                    continue
                
                self.outcomes[episode_idx] = outcome
            
            if self.outcomes:
                print(f"Loaded {len(self.outcomes)} episode outcomes from {outcomes_file}")
                
        except json.JSONDecodeError as e:
            warnings.warn(f"Could not parse outcomes file {outcomes_file}: {e}")
        except Exception as e:
            warnings.warn(f"Could not load outcomes from {outcomes_file}: {e}")
    
    def _save_outcomes(self):
        """Save all outcomes to JSON file, sorted by episode index."""
        outcomes_file = self._get_outcomes_file()
        
        # Convert to list of dicts, sorted by episode_index
        data = [
            {"episode_index": episode_idx, "outcome": outcome}
            for episode_idx, outcome in sorted(self.outcomes.items())
        ]
        
        # Save with pretty formatting for readability
        with open(outcomes_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} episode outcomes to {outcomes_file}")
    
    def set_episode_outcome(
        self,
        episode_index: int,
        outcome: OutcomeType,
        save: bool = True
    ):
        """
        Set outcome for a single episode.
        
        Args:
            episode_index: Episode index in the dataset
            outcome: "success", "failure", or "unknown"
            save: Whether to immediately save to file (default: True)
        
        Raises:
            ValueError: If outcome is invalid or episode_index out of range
        """
        # Validate outcome
        if outcome not in ["success", "failure", "unknown"]:
            raise ValueError(
                f"Invalid outcome '{outcome}'. Must be 'success', 'failure', or 'unknown'."
            )
        
        # Validate episode exists
        if episode_index < 0 or episode_index >= len(self.dataset.meta.episodes):
            raise ValueError(
                f"Episode index {episode_index} out of range. "
                f"Dataset has {len(self.dataset.meta.episodes)} episodes."
            )
        
        self.outcomes[episode_index] = outcome
        
        if save:
            self._save_outcomes()
    
    def set_episode_outcomes_batch(
        self,
        outcomes_dict: Dict[int, OutcomeType]
    ):
        """
        Set outcomes for multiple episodes at once.
        
        More efficient than calling set_episode_outcome multiple times
        as it only saves once at the end.
        
        Args:
            outcomes_dict: Dict mapping episode indices to outcomes
        
        Raises:
            ValueError: If any outcome is invalid or episode index out of range
        """
        num_episodes = len(self.dataset.meta.episodes)
        
        # Validate all entries first
        for episode_idx, outcome in outcomes_dict.items():
            if outcome not in ["success", "failure", "unknown"]:
                raise ValueError(
                    f"Invalid outcome '{outcome}' for episode {episode_idx}. "
                    "Must be 'success', 'failure', or 'unknown'."
                )
            
            if episode_idx < 0 or episode_idx >= num_episodes:
                raise ValueError(
                    f"Episode index {episode_idx} out of range. "
                    f"Dataset has {num_episodes} episodes."
                )
        
        # Apply all changes
        self.outcomes.update(outcomes_dict)
        
        # Save once
        self._save_outcomes()
    
    def get_outcome(self, episode_index: int) -> OutcomeType | None:
        """
        Get outcome for an episode.
        
        Args:
            episode_index: Episode index
            
        Returns:
            Outcome string or None if not labeled
        """
        return self.outcomes.get(episode_index)
    
    def get_all_outcomes(self) -> Dict[int, OutcomeType]:
        """Get all episode outcomes as a dict."""
        return self.outcomes.copy()
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about labeled outcomes.
        
        Returns:
            Dict with counts for each outcome type and total episodes
        """
        total_episodes = len(self.dataset.meta.episodes)
        labeled_episodes = len(self.outcomes)
        
        success_count = sum(1 for o in self.outcomes.values() if o == "success")
        failure_count = sum(1 for o in self.outcomes.values() if o == "failure")
        unknown_count = sum(1 for o in self.outcomes.values() if o == "unknown")
        unlabeled_count = total_episodes - labeled_episodes
        
        return {
            "total_episodes": total_episodes,
            "labeled": labeled_episodes,
            "unlabeled": unlabeled_count,
            "success": success_count,
            "failure": failure_count,
            "unknown": unknown_count,
        }


