from typing import Any, Dict, Union
import warnings
import json
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robocandywrapper.plugin import DatasetPlugin, PluginInstance


# TODO delete and refactor in RewACT repo

class DenseRewardsPlugin(DatasetPlugin):
    """Plugin for adding dense reward signals to episodes."""
    
    def __init__(
        self,
        reward_start_pct: float = 0.05,
        reward_end_pct: float = 0.95,
        interpolation_method: str = "smooth"
    ):
        """
        Initialize plugin with shared configuration.
        
        Args:
            reward_start_pct: Fallback start percentage
            reward_end_pct: Fallback end percentage
            interpolation_method: "smooth" or "linear"
        """
        self.reward_start_pct = reward_start_pct
        self.reward_end_pct = reward_end_pct
        self.interpolation_method = interpolation_method
    
    def attach(self, dataset: 'LeRobotDataset') -> 'DenseRewardsInstance':
        """Create dataset-specific instance."""
        return DenseRewardsInstance(dataset, self)


class DenseRewardsInstance(PluginInstance):
    """Dataset-specific instance for dense rewards."""
    
    def __init__(self, dataset: 'LeRobotDataset', config: DenseRewardsPlugin):
        super().__init__(dataset)
        self.config = config
        self.keypoint_rewards: Dict[int, Dict[int, float]] = {}
        self._episode_reward_cache: Dict[int, list[float]] = {}
        self._load_rewards()
    
    def get_data_keys(self) -> list[str]:
        """This plugin adds 'reward' key."""
        return ["reward"]
    
    def get_item_data(self, idx: int, episode_idx: int) -> dict[str, Any]:
        """
        Get reward for this item.
        
        This method is called in isolation - it doesn't see other plugins' data.
        """
        episode_length = self.dataset.meta.episodes[episode_idx]["length"]
        ep_start = self.dataset.meta.episodes[episode_idx]["dataset_from_index"]
        frame_index_in_episode = idx - ep_start
        
        # Calculate reward
        if episode_idx in self.keypoint_rewards and self.keypoint_rewards[episode_idx]:
            # Use cached interpolated rewards
            if episode_idx not in self._episode_reward_cache:
                keypoints = self.keypoint_rewards[episode_idx]
                self._episode_reward_cache[episode_idx] = self._interpolate_rewards(
                    keypoints, episode_length
                )
            
            if frame_index_in_episode < len(self._episode_reward_cache[episode_idx]):
                reward = self._episode_reward_cache[episode_idx][frame_index_in_episode]
            else:
                reward = self._episode_reward_cache[episode_idx][-1]
        else:
            # Fallback linear interpolation
            progress = frame_index_in_episode / (episode_length - 1) if episode_length > 1 else 0.0
            if progress <= self.config.reward_start_pct:
                reward = 0.0
            elif progress >= self.config.reward_end_pct:
                reward = 1.0
            else:
                interpolation_progress = (progress - self.config.reward_start_pct) / (
                    self.config.reward_end_pct - self.config.reward_start_pct
                )
                reward = interpolation_progress
        
        # Return only the data this plugin provides
        return {"reward": torch.tensor(reward, dtype=torch.float32)}
    
    def _get_plugin_dir(self) -> Path:
        """Get plugin data directory."""
        plugin_dir = Path(self.dataset.root) / "plugins" / "dense_rewards"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        return plugin_dir
    
    def _load_rewards(self):
        """Load rewards - plugin decides its own storage format."""
        reward_file = self._get_plugin_dir() / "reward_keypoints.json"
        if reward_file.exists():
            try:
                with open(reward_file, 'r') as f:
                    data = json.load(f)
                
                for ep_idx_str, ep_data in data.items():
                    episode_index = int(ep_idx_str)
                    self.keypoint_rewards[episode_index] = {}
                    
                    for kp_data in ep_data.get("keypoints", []):
                        frame_idx = kp_data["frame_index"]
                        reward = kp_data["reward"]
                        self.keypoint_rewards[episode_index][frame_idx] = float(reward)
            except Exception as e:
                warnings.warn(f"Could not load rewards for {self.dataset.repo_id}: {e}")
    
    def _save_rewards(self):
        """Save rewards - plugin decides its own storage format."""
        reward_file = self._get_plugin_dir() / "reward_keypoints.json"
        
        data = {}
        for ep_idx, keypoints in self.keypoint_rewards.items():
            keypoint_list = []
            for frame_idx, reward in keypoints.items():
                timestamp = frame_idx / self.dataset.fps
                keypoint_list.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "reward": float(reward)
                })
            data[str(ep_idx)] = {
                "keypoints": keypoint_list,
                "fps": self.dataset.fps
            }
        
        with open(reward_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_episode_rewards(self, episode_index: int, 
                          keypoints: Union[list[KeypointReward], Dict[int, float]]):
        """Add rewards for an episode."""
        if isinstance(keypoints, dict):
            frame_rewards = keypoints
        else:
            frame_rewards = self._normalize_keypoints(keypoints, episode_index)
        
        if episode_index not in self.keypoint_rewards:
            self.keypoint_rewards[episode_index] = {}
        
        self.keypoint_rewards[episode_index].update(frame_rewards)
        
        if episode_index in self._episode_reward_cache:
            del self._episode_reward_cache[episode_index]
        
        self._save_rewards()
    
    def _interpolate_rewards(self, keypoints: Dict[int, float], 
                           episode_length: int) -> list[float]:
        """Your existing interpolation logic."""
        # ... (same as before)
        pass
    
    def _normalize_keypoints(self, keypoints: list[KeypointReward], 
                            episode_index: int) -> Dict[int, float]:
        """Convert KeypointReward objects to frame_index -> reward mapping."""
        # ... (same as before)
        pass
