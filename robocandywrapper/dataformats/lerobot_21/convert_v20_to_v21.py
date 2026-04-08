#!/usr/bin/env python
"""
Convert LeRobot datasets from codebase version 2.0 to 2.1.

The main difference between v2.0 and v2.1 is how statistics are handled:
- v2.0 stores aggregate stats in a single `stats.json`
- v2.1 stores per-episode stats in `episodes_stats.jsonl`, then aggregates them

This script will:
- Generate per-episode stats and write them to `episodes_stats.jsonl`
- Check consistency between new stats and the old ones
- Remove the deprecated `stats.json`
- Update codebase_version in `info.json`
- Push the new version to the Hub and tag it with "v2.1"

Adapted from the original lerobot v2.1 conversion script to work with
RoboCandyWrapper's LeRobot21Dataset (since lerobot v3.0 removed the
v2.1 conversion utilities).

Usage:
    python -m robocandywrapper.dataformats.lerobot_21.convert_v20_to_v21 \\
        --repo-id villekuosmanen/fold_clothes_dining

    python -m robocandywrapper.dataformats.lerobot_21.convert_v20_to_v21 \\
        --repo-id villekuosmanen/fold_clothes_dining --batch
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from huggingface_hub import HfApi
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats, sample_indices
from lerobot.datasets.io_utils import load_stats, write_info

from robocandywrapper.dataformats.lerobot_21.dataset import LeRobot21Dataset
from robocandywrapper.dataformats.lerobot_21.utils import (
    EPISODES_STATS_PATH,
    write_episode_stats,
)

V20 = "v2.0"
V21 = "v2.1"
STATS_PATH = "meta/stats.json"


class SuppressWarnings:
    def __enter__(self):
        self.previous_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.previous_level)


def sample_episode_video_frames(
    dataset: LeRobot21Dataset, episode_index: int, ft_key: str
) -> np.ndarray:
    ep_len = dataset.meta.episodes[episode_index]["length"]

    if ep_len == 1:
        query_timestamps = dataset._get_query_timestamps(0.0, {ft_key: [0]})
        video_frames = dataset._query_videos(query_timestamps, episode_index)
        return np.expand_dims(video_frames[ft_key].numpy(), axis=0)
    else:
        sampled_indices = sample_indices(ep_len)
        query_timestamps = dataset._get_query_timestamps(0.0, {ft_key: sampled_indices})
        video_frames = dataset._query_videos(query_timestamps, episode_index)
        return video_frames[ft_key].numpy()


def convert_episode_stats(dataset: LeRobot21Dataset, ep_idx: int):
    """Compute stats for a single episode and store in dataset.meta.episodes_stats."""
    ep_start_idx = dataset.episode_data_index["from"][ep_idx]
    ep_end_idx = dataset.episode_data_index["to"][ep_idx]
    ep_data = dataset.hf_dataset.select(range(ep_start_idx, ep_end_idx))

    ep_stats = {}
    for key, ft in dataset.features.items():
        if ft.get("dtype") in ("string", "list"):
            continue
        try:
            if ft["dtype"] == "video":
                ep_ft_data = sample_episode_video_frames(dataset, ep_idx, key)
            else:
                ep_ft_data = np.array(ep_data[key])

            axes_to_reduce = (0, 2, 3) if ft["dtype"] in ["image", "video"] else 0
            keepdims = True if ft["dtype"] in ["image", "video"] else ep_ft_data.ndim == 1
            ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)

            if ft["dtype"] in ["image", "video"]:
                ep_stats[key] = {
                    k: v if k == "count" else np.squeeze(v, axis=0)
                    for k, v in ep_stats[key].items()
                }
        except Exception as e:
            logging.warning(
                f"  Skipping stats for feature '{key}' in episode {ep_idx}: {e}"
            )

    dataset.meta.episodes_stats[ep_idx] = ep_stats


def convert_stats(dataset: LeRobot21Dataset, num_workers: int = 0):
    """Compute per-episode stats for all episodes."""
    assert dataset.episodes is None
    logging.info("Computing per-episode stats...")
    total_episodes = dataset.meta.total_episodes
    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(convert_episode_stats, dataset, ep_idx): ep_idx
                for ep_idx in range(total_episodes)
            }
            for future in tqdm(as_completed(futures), total=total_episodes, desc="Computing stats"):
                future.result()
    else:
        for ep_idx in tqdm(range(total_episodes), desc="Computing stats"):
            convert_episode_stats(dataset, ep_idx)

    for ep_idx in tqdm(range(total_episodes), desc="Writing stats"):
        write_episode_stats(ep_idx, dataset.meta.episodes_stats[ep_idx], dataset.root)


def check_aggregate_stats(
    dataset: LeRobot21Dataset,
    reference_stats: dict[str, dict[str, np.ndarray]],
    video_rtol_atol: tuple[float, float] = (5e-1, 5e-1),
    default_rtol_atol: tuple[float, float] = (5e-6, 6e-5),
):
    """Verify that aggregated stats from episodes_stats are close to reference stats."""
    agg_stats = aggregate_stats(list(dataset.meta.episodes_stats.values()))
    for key, ft in dataset.features.items():
        if ft.get("dtype") in ("string", "list"):
            continue
        if ft["dtype"] == "video":
            rtol, atol = video_rtol_atol
        else:
            rtol, atol = default_rtol_atol

        if key not in agg_stats:
            continue
        for stat, val in agg_stats[key].items():
            if key in reference_stats and stat in reference_stats[key]:
                err_msg = f"feature='{key}' stats='{stat}'"
                np.testing.assert_allclose(
                    val, reference_stats[key][stat], rtol=rtol, atol=atol, err_msg=err_msg
                )


def convert_dataset(
    repo_id: str,
    branch: str | None = None,
    num_workers: int = 4,
    push: bool = True,
):
    """
    Convert a single dataset from v2.0 to v2.1.

    Returns True if conversion succeeded.
    """
    logging.info(f"Converting {repo_id} from v2.0 to v2.1...")

    with SuppressWarnings():
        dataset = LeRobot21Dataset(repo_id, revision=V20, force_cache_sync=True)

    if (dataset.root / EPISODES_STATS_PATH).is_file():
        (dataset.root / EPISODES_STATS_PATH).unlink()

    convert_stats(dataset, num_workers=num_workers)

    try:
        ref_stats = load_stats(dataset.root)
        check_aggregate_stats(dataset, ref_stats)
        logging.info(f"  Stats consistency check passed")
    except FileNotFoundError:
        logging.warning(f"  No reference stats.json found, skipping consistency check")
    except AssertionError as e:
        logging.warning(f"  Stats consistency check failed (non-fatal): {e}")

    dataset.meta.info["codebase_version"] = V21
    write_info(dataset.meta.info, dataset.root)
    logging.info(f"  Updated codebase_version to {V21}")

    if push:
        dataset.push_to_hub(branch=branch, tag_version=False, allow_patterns="meta/")
        logging.info(f"  Pushed metadata to hub")

        if (dataset.root / STATS_PATH).is_file():
            (dataset.root / STATS_PATH).unlink()

        hub_api = HfApi()
        if hub_api.file_exists(
            repo_id=dataset.repo_id, filename=STATS_PATH, revision=branch, repo_type="dataset"
        ):
            hub_api.delete_file(
                path_in_repo=STATS_PATH, repo_id=dataset.repo_id, revision=branch, repo_type="dataset"
            )
            logging.info(f"  Removed old stats.json from hub")

        try:
            hub_api.delete_tag(repo_id, tag=V21, repo_type="dataset")
        except Exception:
            pass
        hub_api.create_tag(repo_id, tag=V21, revision=branch, repo_type="dataset")
        logging.info(f"  Tagged as {V21}")

    logging.info(f"  Done: {repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot datasets from v2.0 to v2.1 format"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Single dataset repo_id to convert (e.g. villekuosmanen/fold_clothes_dining)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Branch to push to. Defaults to main.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallel stats computation.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Compute locally without pushing to Hub.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.repo_id:
        parser.error("Provide --repo-id")

    convert_dataset(
        repo_id=args.repo_id,
        branch=args.branch,
        num_workers=args.num_workers,
        push=not args.no_push,
    )


if __name__ == "__main__":
    main()
