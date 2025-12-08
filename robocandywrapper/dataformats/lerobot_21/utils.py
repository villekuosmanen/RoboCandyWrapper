#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Legacy utility functions from lerobot v0.3.3 (v2.1 dataset format).

These functions were removed in lerobot v0.4.1 but are required for LegacyLeRobotDataset support.
Copied from: https://github.com/huge ggingface/lerobot/blob/v0.3.3/lerobot/datasets/utils.py
"""

import json
import logging
from itertools import accumulate
from pathlib import Path
from pprint import pformat

import jsonlines
import numpy as np
import torch

# Constants for v2.1 file paths
TASKS_PATH = "meta/tasks.jsonl"
EPISODES_PATH = "meta/episodes.jsonl"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"


def load_tasks(local_dir: Path) -> tuple[dict, dict]:
    """Load tasks from v2.1 jsonl format."""
    tasks = load_jsonlines(local_dir / TASKS_PATH)
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index


def load_episodes(local_dir: Path) -> dict:
    """Load episodes from v2.1 jsonl format."""
    episodes = load_jsonlines(local_dir / EPISODES_PATH)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def append_jsonlines(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "a") as writer:
        writer.write(data)


def load_jsonlines(fpath: Path) -> list:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    """Convert tensors/arrays to lists for JSON serialization."""
    from lerobot.common.datasets.utils import flatten_dict, unflatten_dict
    
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int, float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def cast_stats_to_numpy(stats) -> dict[str, dict[str, np.ndarray]]:
    from lerobot.common.datasets.utils import flatten_dict, unflatten_dict
    
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def write_episode(episode: dict, local_dir: Path):
    append_jsonlines(episode, local_dir / EPISODES_PATH)


def write_episode_stats(episode_index: int, episode_stats: dict, local_dir: Path):
    # We wrap episode_stats in a dictionary since `episode_stats["episode_index"]`
    # is a dictionary of stats and not an integer.
    episode_stats = {"episode_index": episode_index, "stats": serialize_dict(episode_stats)}
    append_jsonlines(episode_stats, local_dir / EPISODES_STATS_PATH)


def load_episodes_stats(local_dir: Path) -> dict:
    episodes_stats = load_jsonlines(local_dir / EPISODES_STATS_PATH)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }


def backward_compatible_episodes_stats(
    stats: dict[str, dict[str, np.ndarray]], episodes: list[int]
) -> dict[str, dict[str, np.ndarray]]:
    return dict.fromkeys(episodes, stats)


def get_episode_data_index(
    episode_dicts: dict[dict], episodes: list[int] | None = None
) -> dict[str, torch.Tensor]:
    episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in episode_dicts.items()}
    if episodes is not None:
        episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in episodes}

    cumulative_lengths = list(accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
        "to": torch.LongTensor(cumulative_lengths),
    }


def check_timestamps_sync(
    timestamps: np.ndarray,
    episode_indices: np.ndarray,
    episode_data_index: dict[str, np.ndarray],
    fps: int,
    tolerance_s: float,
    raise_value_error: bool = True,
) -> bool:
    """
    This check is to make sure that each timestamp is separated from the next by (1/fps) +/- tolerance
    to account for possible numerical error.

    Args:
        timestamps (np.ndarray): Array of timestamps in seconds.
        episode_indices (np.ndarray): Array indicating the episode index for each timestamp.
        episode_data_index (dict[str, np.ndarray]): A dictionary that includes 'to',
            which identifies indices for the end of each episode.
        fps (int): Frames per second. Used to check the expected difference between consecutive timestamps.
        tolerance_s (float): Allowed deviation from the expected (1/fps) difference.
        raise_value_error (bool): Whether to raise a ValueError if the check fails.

    Returns:
        bool: True if all checked timestamp differences lie within tolerance, False otherwise.

    Raises:
        ValueError: If the check fails and `raise_value_error` is True.
    """
    if timestamps.shape != episode_indices.shape:
        raise ValueError(
            "timestamps and episode_indices should have the same shape. "
            f"Found {timestamps.shape=} and {episode_indices.shape=}."
        )

    # Consecutive differences
    diffs = np.diff(timestamps)
    within_tolerance = np.abs(diffs - (1.0 / fps)) <= tolerance_s

    # Mask to ignore differences at the boundaries between episodes
    mask = np.ones(len(diffs), dtype=bool)
    ignored_diffs = episode_data_index["to"][:-1] - 1  # indices at the end of each episode
    mask[ignored_diffs] = False
    filtered_within_tolerance = within_tolerance[mask]

    # Check if all remaining diffs are within tolerance
    if not np.all(filtered_within_tolerance):
        # Track original indices before masking
        original_indices = np.arange(len(diffs))
        filtered_indices = original_indices[mask]
        outside_tolerance_filtered_indices = np.nonzero(~filtered_within_tolerance)[0]
        outside_tolerance_indices = filtered_indices[outside_tolerance_filtered_indices]

        outside_tolerances = []
        for idx in outside_tolerance_indices:
            entry = {
                "timestamps": [timestamps[idx], timestamps[idx + 1]],
                "diff": diffs[idx],
                "episode_index": episode_indices[idx].item()
                if hasattr(episode_indices[idx], "item")
                else episode_indices[idx],
            }
            outside_tolerances.append(entry)

        if raise_value_error:
            raise ValueError(
                f"""One or several timestamps unexpectedly violate the tolerance inside episode range.
                This might be due to synchronization issues during data collection.
                \n{pformat(outside_tolerances)}"""
            )
        else:
            # Log warning instead of raising error
            import logging
            logging.warning(
                f"Timestamp sync check failed: {len(outside_tolerances)} violations found. "
                f"This might be due to synchronization issues during data collection. "
                f"Continuing anyway, but results may be affected. "
                f"First few violations:\n{pformat(outside_tolerances[:5])}"
            )
        return False

    return True
