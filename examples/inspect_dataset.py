"""
Inspect dataset loading, sampling, and distribution.

Sample commands:
  # Mixed v2.1 + v3.0 loading (episode selection defined in config file)
  python examples/inspect_dataset.py --config-path examples/configs/sampler_config.json
"""
import logging
from pprint import pformat
import argparse
import json
from collections import Counter

import torch
from termcolor import colored
from torch.utils.data import DataLoader

from lerobot.utils.utils import init_logging
from robocandywrapper.factory import make_dataset_without_config
from robocandywrapper.samplers import make_sampler


def inspect(config_path: str):
    """
    Loads datasets, applies sampling logic from a config file,
    and inspects the results. Also provides a placeholder for plugins.

    Args:
        config_path: Path to the sampler config file
    """
    logging.info(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Parse episodes from config
    episodes_parsed = config.get("episodes")

    if episodes_parsed is not None:
        logging.info(f"Using episode selection: {episodes_parsed}")
    else:
        logging.info("Loading all episodes (no episode selection)")

    # --- 1. Dataset Creation ---
    # The config file for the sampler is now the source of truth
    sampler_config = config
    repo_ids = list(sampler_config.get("dataset_weights", {}).keys())

    if not repo_ids:
        logging.error("Config must contain 'dataset_weights' mapping repo IDs to weights.")
        return

    dataset = make_dataset_without_config(repo_ids, episodes=episodes_parsed)
    logging.info("Dataset object created successfully.")

    # --- 2. Plugin Loading (Placeholder) ---
    if "plugins" in config:
        logging.warning(
            "Plugin configuration found, but dynamic loading is not yet implemented in this script "
            "due to potentially complex dependencies (e.g., pre-loaded models)."
        )
        # Here you would typically have a plugin factory to instantiate plugins
        # and pass them to the dataset constructor.

    # --- 3. Raw Dataset Info ---
    print("\n" + colored("=" * 80, "cyan"))
    print(colored("1. Raw Dataset Information (Before Sampling)", "cyan", attrs=["bold"]))
    print(colored("=" * 80, "cyan"))

    total_episodes = 0
    total_frames = 0
    repo_id_map = {}
    for i, d in enumerate(dataset._datasets):
        repo_id_map[i] = d.repo_id
        # Use filtered length from wrapper if available
        real_num_frames = dataset._dataset_lengths[i]
        
        print(f"\n  - Dataset {i}: {d.repo_id}")
        print(f"    - Type:     {type(d).__name__}")
        print(f"    - Episodes: {d.num_episodes}")
        print(f"    - Frames:   {real_num_frames}")
        total_episodes += d.num_episodes
        total_frames += real_num_frames

    print("\n" + colored("Unweighted Totals:", "yellow"))
    print(f"  - Total Episodes: {total_episodes}")
    print(f"  - Total Frames:   {total_frames}")

    # --- 4. Sampler Creation & Info ---
    print("\n" + colored("=" * 80, "cyan"))
    print(colored("2. Sampler Configuration", "cyan", attrs=["bold"]))
    print(colored("=" * 80, "cyan") + "\n")

    sampler, shuffle, _ = make_sampler(dataset, sampler_config=sampler_config)
    print(pformat(sampler_config))
    if sampler:
        print(f"\nSampler object created: {type(sampler).__name__}")
    else:
        print("\nNo sampler created, using standard DataLoader shuffling.")

    # --- 5. Dataloader & Sampling Verification ---
    print("\n" + colored("=" * 80, "cyan"))
    print(colored("3. Sampling Verification", "cyan", attrs=["bold"]))
    print(colored("=" * 80, "cyan") + "\n")

    # Use a dataloader to see the sampler in action
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, shuffle=shuffle)

    num_samples_to_draw = 1000  # Draw a representative number of samples
    print(f"Drawing {num_samples_to_draw} samples to verify distribution...")

    distribution_counter = Counter()
    for i, batch in enumerate(dataloader):
        if i >= num_samples_to_draw:
            break
        # The WrappedRobotDataset adds this key to identify the source dataset
        dataset_idx = batch["dataset_index"].item()
        distribution_counter[dataset_idx] += 1

    print("\n" + colored("Observed Sample Distribution:", "yellow"))
    total_drawn = sum(distribution_counter.values())
    if total_drawn > 0:
        for idx, count in sorted(distribution_counter.items()):
            repo_id = repo_id_map.get(idx, "Unknown")
            percentage = (count / total_drawn) * 100
            print(f"  - Dataset {idx} ({repo_id}):")
            print(f"    - Samples: {count} (~{percentage:.2f}%)")

    print("\n" + colored("Expected Distribution (from config weights):", "yellow"))
    
    # Get the weights from the sampler config
    dataset_weights = sampler_config.get("dataset_weights", {})
    
    # Calculate the "effective size" of each dataset
    effective_sizes = {}
    total_effective_size = 0
    for i, d in enumerate(dataset._datasets):
        weight = dataset_weights.get(d.repo_id, 1.0)
        # Use filtered length from wrapper
        real_num_frames = dataset._dataset_lengths[i]
        effective_size = real_num_frames * weight
        effective_sizes[i] = effective_size
        total_effective_size += effective_size
        
    if total_effective_size > 0:
        for i, d in enumerate(dataset._datasets):
            repo_id = d.repo_id
            weight = dataset_weights.get(repo_id, 1.0)
            # Use filtered length from wrapper
            real_num_frames = dataset._dataset_lengths[i]
            effective_size = effective_sizes.get(i, 0)
            percentage = (effective_size / total_effective_size) * 100
            print(f"  - Dataset {i} ({repo_id}):")
            print(f"    - Frames: {real_num_frames}, Weight: {weight} -> Effective Size: {effective_size:.0f} (~{percentage:.2f}%)")



def main():
    init_logging()
    parser = argparse.ArgumentParser(
        description="Inspect a LeRobot dataset, sampler, and plugins from a config file."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="examples/configs/sampler_config.json",
        help="The path to the JSON config file (default: examples/configs/sampler_config.json).",
    )
    args = parser.parse_args()
    inspect(args.config_path)


if __name__ == "__main__":
    main()
