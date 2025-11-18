"""
Example: Adding affordance labels to a dataset.

This example shows how to:
1. Load a dataset
2. Add affordance labels (x, y coordinates in [0, 1])
3. Save them to parquet files
4. Load and use them during training
"""

import numpy as np
from robocandywrapper.plugins import LabelledAffordancePlugin
from robocandywrapper import WrappedRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def create_mock_affordances(n_frames: int) -> np.ndarray:
    """
    Create mock affordance labels for demonstration.
    
    In practice, these would come from human annotation or a vision model.
    
    Args:
        n_frames: Number of frames in the episode
        
    Returns:
        Array of shape (n_frames, 2) with (x, y) coordinates in [0, 1]
    """
    # Example: Gradually moving affordance from top-left to bottom-right
    x = np.linspace(0.2, 0.8, n_frames)
    y = np.linspace(0.3, 0.7, n_frames)
    
    affordances = np.stack([x, y], axis=1).astype(np.float32)
    
    # Ensure values are in [0, 1]
    assert np.all(affordances >= 0) and np.all(affordances <= 1)
    
    return affordances


def add_affordances_to_dataset(
    dataset_repo_id: str,
    dataset_root: str = None,
    episodes_to_label: list[int] = None,
):
    """
    Add affordance labels to specific episodes in a dataset.
    
    Args:
        dataset_repo_id: Dataset repository ID (e.g., "lerobot/pusht")
        dataset_root: Root directory for datasets
        episodes_to_label: List of episode indices to add labels for
    """
    print(f"Loading dataset: {dataset_repo_id}")
    
    # Load base dataset
    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    
    # Create affordance plugin
    affordance_plugin = LabelledAffordancePlugin()
    
    # Wrap dataset with plugin
    wrapped_dataset = WrappedRobotDataset(
        datasets=dataset,
        plugins=[affordance_plugin],
    )
    
    # Get the affordance plugin instance
    affordance_instance = wrapped_dataset._plugin_instances[0][0]
    
    # Add affordances for specified episodes
    if episodes_to_label is None:
        episodes_to_label = [0, 1, 2]  # Default: label first 3 episodes
    
    print(f"\nLabeling episodes: {episodes_to_label}")
    
    for episode_idx in episodes_to_label:
        if episode_idx >= len(dataset.meta.episodes):
            print(f"  ⚠️  Skipping episode {episode_idx}: not in dataset")
            continue
        
        # Get episode info
        episode_info = dataset.meta.episodes[episode_idx]
        n_frames = episode_info["length"]
        
        print(f"\n  Episode {episode_idx}:")
        print(f"    Frames: {n_frames}")
        
        # Create affordance labels
        # In practice, you would:
        # 1. Load the episode frames
        # 2. Run a vision model or manual annotation tool
        # 3. Extract (x, y) coordinates
        affordances = create_mock_affordances(n_frames)
        
        print(f"    Affordances shape: {affordances.shape}")
        print(f"    Sample affordances (first 3 frames):")
        for i in range(min(3, n_frames)):
            x, y = affordances[i]
            print(f"      Frame {i}: x={x:.3f}, y={y:.3f}")
        
        # Save affordances
        try:
            affordance_instance.add_episode_affordances(episode_idx, affordances)
            print(f"    ✅ Saved affordances")
        except Exception as e:
            print(f"    ❌ Error saving affordances: {e}")
    
    print(f"\n✅ Affordance labeling complete!")
    print(f"   Data saved to: {affordance_instance._get_plugin_dir()}")


def use_affordances_in_training(
    dataset_repo_id: str,
    dataset_root: str = None,
):
    """
    Example of using affordances during training.
    
    Args:
        dataset_repo_id: Dataset repository ID
        dataset_root: Root directory for datasets
    """
    print(f"\nLoading dataset with affordances: {dataset_repo_id}")
    
    # Load dataset
    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    
    # Wrap with affordance plugin
    wrapped_dataset = WrappedRobotDataset(
        datasets=dataset,
        plugins=[LabelledAffordancePlugin()],
    )
    
    # Get a sample
    print("\nSampling data with affordances:")
    sample = wrapped_dataset[0]
    
    print(f"  Keys in sample: {list(sample.keys())}")
    
    if "affordance" in sample:
        affordance = sample["affordance"]
        affordance_mask = sample["affordance_mask"]
        
        print(f"  Affordance: {affordance}")
        print(f"  Affordance shape: {affordance.shape}")
        print(f"  Has affordance: {affordance_mask.item()}")
        
        if affordance_mask.item():
            x, y = affordance
            print(f"  Coordinates: x={x:.3f}, y={y:.3f}")
    else:
        print("  ⚠️  No affordance data found")
    
    return wrapped_dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add affordance labels to a dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lerobot/pusht",
        help="Dataset repository ID"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Dataset root directory"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Episode indices to label"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["label", "use"],
        default="label",
        help="Mode: 'label' to add labels, 'use' to demonstrate usage"
    )
    
    args = parser.parse_args()
    
    if args.mode == "label":
        add_affordances_to_dataset(
            dataset_repo_id=args.dataset,
            dataset_root=args.root,
            episodes_to_label=args.episodes,
        )
    else:
        use_affordances_in_training(
            dataset_repo_id=args.dataset,
            dataset_root=args.root,
        )

