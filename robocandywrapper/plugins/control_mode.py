"""Control Mode Plugin for labeling human vs autonomous (policy) frames.

Reads episode_modes.json from candywrapper_plugins and adds per-frame control
mode labels. Handles both legacy and current storage formats transparently:

Path variants (checked in order):
  1. candywrapper_plugins/dagger_data_source/episode_modes.json  (legacy)
  2. candywrapper_plugins/control_mode/episode_modes.json         (current)

JSON format variants (both handled):
  Flat list:   {"0": [{"start_index": 0, "end_index": 57, "mode": "policy"}, ...]}
  Wrapped:     {"0": {"segments": [{"start_index": 0, "end_index": 57, "mode": "policy"}, ...]}}

Keys added to each item:
  - control_mode: str ("human", "policy", or "unknown")
  - control_mode_is_human: torch.bool (True if human-controlled)
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from robocandywrapper.plugin import DatasetPlugin, PluginInstance
from robocandywrapper.constants import CANDYWRAPPER_PLUGINS_DIR, CONTROL_MODE_PLUGIN_NAME


@dataclass
class ControlModeSegment:
    start_index: int
    end_index: int
    mode: str


class ControlModePlugin(DatasetPlugin):
    """Plugin for loading per-frame control mode (human vs policy) from episode_modes.json."""

    def attach(self, dataset) -> ControlModeInstance:
        return ControlModeInstance(dataset, self)


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


class ControlModeInstance(PluginInstance):
    """Dataset-specific control mode labels."""

    def __init__(self, dataset, config: ControlModePlugin):
        super().__init__(dataset)
        self.config = config
        self.episode_modes: dict[int, list[ControlModeSegment]] = {}
        self._cached_episode_data_index: dict[str, torch.Tensor] | None = None
        self._load()

    def _get_episode_data_index(self) -> dict[str, torch.Tensor]:
        if hasattr(self.dataset, 'episode_data_index'):
            return self.dataset.episode_data_index
        if self._cached_episode_data_index is None:
            self._cached_episode_data_index = _calculate_episode_data_index(self.dataset.hf_dataset)
        return self._cached_episode_data_index

    def get_data_keys(self) -> list[str]:
        return ["control_mode", "control_mode_autonomous"]

    def get_item_data(
        self,
        idx: int,
        episode_idx: int,
        accumulated_data: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        mode = self._get_mode_for_frame(idx, episode_idx)
        return {
            "control_mode": mode,
            "control_mode_autonomous": torch.tensor(mode == "policy", dtype=torch.bool),
        }

    # ── loading ────────────────────────────────────────────────────────

    def _find_file(self) -> Path | None:
        dataset_root = Path(self.dataset.root)
        cw = dataset_root / CANDYWRAPPER_PLUGINS_DIR
        legacy = cw / "dagger_data_source" / "episode_modes.json"
        legacy_2 = dataset_root / "dagger_data_source" / "episode_modes.json"
        current = cw / CONTROL_MODE_PLUGIN_NAME / "episode_modes.json"
        if legacy.exists():
            return legacy
        if legacy_2.exists():
            return legacy_2
        if current.exists():
            return current
        return None

    def _load(self):
        path = self._find_file()
        if path is None:
            return

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"Could not load control mode data from {path}: {e}")
            return

        for ep_str, segments_data in data.items():
            ep_idx = int(ep_str)
            if isinstance(segments_data, dict) and "segments" in segments_data:
                segments_data = segments_data["segments"]
            if not isinstance(segments_data, list):
                warnings.warn(f"Unexpected format for episode {ep_idx} in {path}, skipping")
                continue
            segs = []
            for seg in segments_data:
                segs.append(ControlModeSegment(
                    start_index=seg["start_index"],
                    end_index=seg["end_index"],
                    mode=seg.get("mode", "unknown"),
                ))
            self.episode_modes[ep_idx] = segs

    # ── per-frame lookup ───────────────────────────────────────────────

    def _get_mode_for_frame(self, idx: int, episode_idx: int) -> str:
        segs = self.episode_modes.get(episode_idx)
        if segs is None:
            return "unknown"

        ep_data_index = self._get_episode_data_index()
        ep_start = ep_data_index["from"][episode_idx].item()
        frame_in_ep = idx - ep_start

        for seg in segs:
            if seg.start_index <= frame_in_ep <= seg.end_index:
                return seg.mode
        return "unknown"

    # ── direct access (for index generation without frame iteration) ──

    def get_episode_segments(self, episode_idx: int) -> list[ControlModeSegment]:
        """Get raw segments for an episode. Empty list if no data."""
        return self.episode_modes.get(episode_idx, [])

    def has_data(self) -> bool:
        return len(self.episode_modes) > 0

    @staticmethod
    def load_from_path(path: Path) -> dict[int, list[ControlModeSegment]]:
        """Load episode modes from a file path without needing a dataset.

        Useful for index generation scripts that want to read control mode
        data directly without loading the full dataset.
        """
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

        result: dict[int, list[ControlModeSegment]] = {}
        for ep_str, segments_data in data.items():
            ep_idx = int(ep_str)
            if isinstance(segments_data, dict) and "segments" in segments_data:
                segments_data = segments_data["segments"]
            if not isinstance(segments_data, list):
                continue
            result[ep_idx] = [
                ControlModeSegment(
                    start_index=seg["start_index"],
                    end_index=seg["end_index"],
                    mode=seg.get("mode", "unknown"),
                )
                for seg in segments_data
            ]
        return result

    @staticmethod
    def find_episode_modes_file(dataset_root: Path) -> Path | None:
        """Find the episode_modes.json file for a dataset root, or None."""
        cw = dataset_root / CANDYWRAPPER_PLUGINS_DIR
        legacy = cw / "dagger_data_source" / "episode_modes.json"
        current = cw / CONTROL_MODE_PLUGIN_NAME / "episode_modes.json"
        if legacy.exists():
            return legacy
        if current.exists():
            return current
        return None
