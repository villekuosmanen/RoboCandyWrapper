"""
Subtask Plugin for per-frame subtask labels within episodes.

Storage: {dataset_root}/candywrapper_plugins/subtask/subtasks.json

JSON format:
{
    "0": {  // episode index as string key
        "subtasks": [
            {
                "start_frame": 0,
                "end_frame": 221,
                "subtask_id": "pick_object",
                "description": "pick up the red coffee cup",
                "variant_descriptions": ["grab the red mug", ...]
            },
            ...
        ]
    }
}

Frame ranges are continuous: (0, 221), (221, 447), etc.
"""
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, SUBTASK_PLUGIN_NAME
from robocandywrapper.plugin import DatasetPlugin, PluginInstance


def _calculate_episode_data_index(hf_dataset) -> dict[str, torch.Tensor]:
    """Calculate episode data index from hf_dataset's episode_index column."""
    episode_data_index: dict[str, list[int]] = {"from": [], "to": []}
    if len(hf_dataset) == 0:
        return {"from": torch.tensor([]), "to": torch.tensor([])}
    current_episode = None
    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            episode_data_index["from"].append(idx)
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            current_episode = episode_idx
    episode_data_index["to"].append(idx + 1)
    return {k: torch.tensor(v) for k, v in episode_data_index.items()}


class SubtaskPlugin(DatasetPlugin):
    """Plugin that provides per-frame subtask labels for episodes."""

    def attach(self, dataset: "LeRobotDataset") -> "SubtaskInstance":
        return SubtaskInstance(dataset, self)


class SubtaskInstance(PluginInstance):
    """Dataset-specific subtask label data."""

    def __init__(self, dataset: "LeRobotDataset", config: SubtaskPlugin):
        super().__init__(dataset)
        self.config = config
        self._data: Dict[int, list[dict]] = {}
        self._cached_episode_data_index: dict[str, torch.Tensor] | None = None
        self._load()

    def _get_episode_data_index(self) -> dict[str, torch.Tensor]:
        """Get episode data index, supporting both newer and older dataset formats."""
        if hasattr(self.dataset, 'episode_data_index'):
            return self.dataset.episode_data_index
        if self._cached_episode_data_index is None:
            self._cached_episode_data_index = _calculate_episode_data_index(self.dataset.hf_dataset)
        return self._cached_episode_data_index

    def get_data_keys(self) -> list[str]:
        return ["subtask", "subtask_id", "subtask_variants"]

    def get_item_data(
        self,
        idx: int,
        episode_idx: int,
        accumulated_data: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if episode_idx not in self._data:
            return {
                "subtask": "",
                "subtask_id": "",
                "subtask_variants": [],
            }

        ep_data_index = self._get_episode_data_index()
        ep_start = ep_data_index["from"][episode_idx].item()
        frame_in_episode = idx - ep_start

        for seg in self._data[episode_idx]:
            if seg["start_frame"] <= frame_in_episode < seg["end_frame"]:
                return {
                    "subtask": seg["description"],
                    "subtask_id": seg.get("subtask_id", ""),
                    "subtask_variants": seg.get("variant_descriptions", []),
                }

        return {
            "subtask": "",
            "subtask_id": "",
            "subtask_variants": [],
        }

    def _get_plugin_dir(self) -> Path:
        plugin_dir = Path(self.dataset.root) / CANDYWRAPPER_PLUGINS_DIR / SUBTASK_PLUGIN_NAME
        plugin_dir.mkdir(parents=True, exist_ok=True)
        return plugin_dir

    def _get_data_file(self) -> Path:
        return self._get_plugin_dir() / "subtasks.json"

    def _load(self) -> None:
        data_file = self._get_data_file()
        if not data_file.exists():
            return

        try:
            with open(data_file) as f:
                raw = json.load(f)

            for ep_str, ep_data in raw.items():
                ep_idx = int(ep_str)
                subtasks = ep_data if isinstance(ep_data, list) else ep_data.get("subtasks", [])
                self._data[ep_idx] = subtasks

            if self._data:
                print(f"Loaded subtask labels for {len(self._data)} episodes from {data_file}")

        except (json.JSONDecodeError, Exception) as e:
            warnings.warn(f"Could not load subtask data from {data_file}: {e}")

    def save(self, data: Dict[int, list[dict]]) -> None:
        """Save subtask data. Keys are episode indices, values are lists of subtask segments."""
        self._data = data
        data_file = self._get_data_file()
        serializable = {str(k): v for k, v in sorted(data.items())}
        with open(data_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved subtask labels for {len(data)} episodes to {data_file}")

    def get_episode_subtasks(self, episode_idx: int) -> list[dict]:
        return self._data.get(episode_idx, [])
