"""
MolmoPoint Plugin for per-frame visual affordance point labels.

Storage: {dataset_root}/candywrapper_plugins/molmopoint/episode_{idx:06d}.parquet

Parquet schema per episode:
    frame_index  : int     — episode-relative frame index (0-based)
    camera       : str     — camera key, e.g. "observation.images.front"
    point_x      : float   — normalised x coordinate (-1 to 1, left to right)
    point_y      : float   — normalised y coordinate (-1 to 1, top to bottom)
    is_keyframe  : bool    — True for raw MolmoPoint predictions, False for interpolated

Coordinates are in [-1, 1] where (-1,-1) is top-left and (1,1) is bottom-right.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, MOLMOPOINT_PLUGIN_NAME
from robocandywrapper.plugin import DatasetPlugin, PluginInstance


def _calculate_episode_data_index(hf_dataset) -> dict[str, torch.Tensor]:
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


class MolmoPointPlugin(DatasetPlugin):
    """Plugin that provides per-frame MolmoPoint affordance labels."""

    def attach(self, dataset: "LeRobotDataset") -> "MolmoPointInstance":
        return MolmoPointInstance(dataset, self)


class MolmoPointInstance(PluginInstance):
    """Dataset-specific MolmoPoint label data."""

    def __init__(self, dataset: "LeRobotDataset", config: MolmoPointPlugin):
        super().__init__(dataset)
        self.config = config
        self._data: Dict[int, "pd.DataFrame"] = {}
        self._cached_episode_data_index: dict[str, torch.Tensor] | None = None

        self._camera_keys: list[str] = list(dataset.meta.video_keys)
        self._camera_to_idx: dict[str, int] = {
            k: i for i, k in enumerate(self._camera_keys)
        }
        self._load()

    def _get_episode_data_index(self) -> dict[str, torch.Tensor]:
        if hasattr(self.dataset, "episode_data_index"):
            return self.dataset.episode_data_index
        if self._cached_episode_data_index is None:
            self._cached_episode_data_index = _calculate_episode_data_index(
                self.dataset.hf_dataset
            )
        return self._cached_episode_data_index

    def get_data_keys(self) -> list[str]:
        return ["pointing", "pointing_mask"]

    def get_item_data(
        self,
        idx: int,
        episode_idx: int,
        accumulated_data: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Return per-camera pointing vector and mask for this frame.

        Returns:
            pointing: tensor of shape (num_cameras * 2,) with [x0, y0, x1, y1, ...]
                in [-1, 1]. One (x, y) pair per camera in dataset.meta.video_keys
                order. Unlabelled cameras get (0, 0).
            pointing_mask: bool tensor of shape (num_cameras,), True where
                a real label exists for that camera.
        """
        num_cameras = len(self._camera_keys)
        pointing = torch.zeros(num_cameras * 2)
        pointing_mask = torch.zeros(num_cameras, dtype=torch.bool)

        if episode_idx not in self._data:
            return {"pointing": pointing, "pointing_mask": pointing_mask}

        ep_data_index = self._get_episode_data_index()
        ep_start = ep_data_index["from"][episode_idx].item()
        frame_in_episode = idx - ep_start

        df = self._data[episode_idx]
        rows = df[df["frame_index"] == frame_in_episode]

        for _, row in rows.iterrows():
            cam_idx = self._camera_to_idx.get(row["camera"])
            if cam_idx is None:
                continue
            pointing[cam_idx * 2] = float(row["point_x"])
            pointing[cam_idx * 2 + 1] = float(row["point_y"])
            pointing_mask[cam_idx] = True

        return {"pointing": pointing, "pointing_mask": pointing_mask}

    def _get_plugin_dir(self) -> Path:
        plugin_dir = (
            Path(self.dataset.root) / CANDYWRAPPER_PLUGINS_DIR / MOLMOPOINT_PLUGIN_NAME
        )
        plugin_dir.mkdir(parents=True, exist_ok=True)
        return plugin_dir

    def _episode_path(self, episode_idx: int) -> Path:
        return self._get_plugin_dir() / f"episode_{episode_idx:06d}.parquet"

    def _load(self) -> None:
        plugin_dir = self._get_plugin_dir()
        import pandas as pd

        loaded = 0
        for pq_file in sorted(plugin_dir.glob("episode_*.parquet")):
            try:
                ep_idx = int(pq_file.stem.split("_")[1])
                self._data[ep_idx] = pd.read_parquet(pq_file)
                loaded += 1
            except Exception as e:
                warnings.warn(f"Could not load {pq_file}: {e}")

        if loaded:
            print(f"Loaded MolmoPoint labels for {loaded} episodes from {plugin_dir}")

    def save_episode(self, episode_idx: int, df: "pd.DataFrame") -> Path:
        """Save MolmoPoint labels for one episode. Returns the saved path."""
        self._data[episode_idx] = df
        path = self._episode_path(episode_idx)
        df.to_parquet(path, index=False)
        return path

    def get_episode_labels(self, episode_idx: int) -> "pd.DataFrame | None":
        return self._data.get(episode_idx)

    def get_episode_labels_by_camera(
        self, episode_idx: int, camera: str,
    ) -> "pd.DataFrame | None":
        df = self._data.get(episode_idx)
        if df is None:
            return None
        filtered = df[df["camera"] == camera]
        return filtered if not filtered.empty else None

    @property
    def labeled_episodes(self) -> list[int]:
        return sorted(self._data.keys())
