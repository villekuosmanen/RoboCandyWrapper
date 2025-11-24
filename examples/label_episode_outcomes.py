"""
Script for labeling episode outcomes (success/failure/unknown).

Supports multiple input methods:
1. Command-line ranges: --ranges "0-42:success" "43:failure" "44-:success"
2. Input file: --file labels.txt
3. Interactive mode (default)

Range Format:
- Single episode:    5:success
- Episode range:     0-42:success
- Open-ended range:  44-:success  (from episode 44 to end)
"""

import argparse
from pathlib import Path
from typing import Dict

from robocandywrapper.plugins import EpisodeOutcomePlugin
from robocandywrapper import WrappedRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_range_spec(spec: str) -> tuple[int | None, int | None, str]:
    """
    Parse a range specification string.
    
    Args:
        spec: Range specification like "0-42:success", "43:failure", or "44-:success"
        
    Returns:
        Tuple of (start, end, outcome) where:
            - start: Starting episode index (inclusive)
            - end: Ending episode index (inclusive), or None for open-ended
            - outcome: "success", "failure", or "unknown"
    
    Examples:
        "5:success" -> (5, 5, "success")
        "0-42:failure" -> (0, 42, "failure")
        "44-:success" -> (44, None, "success")
    
    Raises:
        ValueError: If format is invalid
    """
    if ':' not in spec:
        raise ValueError(f"Invalid format '{spec}'. Expected format: 'episode(s):outcome'")
    
    range_part, outcome = spec.rsplit(':', 1)
    
    # Validate outcome
    outcome = outcome.strip().lower()
    if outcome not in ['success', 'failure', 'unknown']:
        raise ValueError(
            f"Invalid outcome '{outcome}'. Must be 'success', 'failure', or 'unknown'."
        )
    
    # Parse range
    range_part = range_part.strip()
    
    if '-' in range_part:
        # Range format: "0-42" or "44-"
        parts = range_part.split('-', 1)
        
        start_str = parts[0].strip()
        end_str = parts[1].strip()
        
        # Parse start
        if not start_str:
            raise ValueError(f"Invalid range '{range_part}'. Start index required.")
        try:
            start = int(start_str)
        except ValueError:
            raise ValueError(f"Invalid start index '{start_str}'. Must be an integer.")
        
        # Parse end (may be empty for open-ended)
        if end_str:
            try:
                end = int(end_str)
            except ValueError:
                raise ValueError(f"Invalid end index '{end_str}'. Must be an integer.")
            
            if end < start:
                raise ValueError(f"End index {end} < start index {start}")
        else:
            end = None  # Open-ended
    else:
        # Single episode
        try:
            start = end = int(range_part)
        except ValueError:
            raise ValueError(f"Invalid episode index '{range_part}'. Must be an integer.")
    
    return start, end, outcome


def expand_range_specs(
    specs: list[str],
    max_episode: int
) -> Dict[int, str]:
    """
    Expand range specifications into a dict of episode_idx -> outcome.
    
    Args:
        specs: List of range specifications
        max_episode: Maximum episode index in dataset (for open-ended ranges)
        
    Returns:
        Dict mapping episode indices to outcomes
        
    Raises:
        ValueError: If any spec is invalid or conflicts occur
    """
    outcomes = {}
    
    for spec in specs:
        start, end, outcome = parse_range_spec(spec)
        
        # Handle open-ended ranges
        if end is None:
            end = max_episode
        
        # Expand range
        for episode_idx in range(start, end + 1):
            if episode_idx in outcomes:
                if outcomes[episode_idx] != outcome:
                    raise ValueError(
                        f"Conflict: Episode {episode_idx} specified multiple times "
                        f"with different outcomes ('{outcomes[episode_idx]}' and '{outcome}')"
                    )
            outcomes[episode_idx] = outcome
    
    return outcomes


def load_ranges_from_file(file_path: Path) -> list[str]:
    """
    Load range specifications from a text file.
    
    File format (one range per line, # for comments):
        # Training episodes
        0-42: success
        43: failure
        44-: success
        
        # Validation episodes
        100-105: unknown
    
    Args:
        file_path: Path to file
        
    Returns:
        List of range specifications
    """
    ranges = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Strip whitespace and comments
            line = line.strip()
            if '#' in line:
                line = line[:line.index('#')].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Validate format
            try:
                parse_range_spec(line)  # Validate
                ranges.append(line)
            except ValueError as e:
                print(f"Warning: Skipping invalid line {line_num}: {e}")
    
    return ranges


def interactive_labeling(
    dataset: LeRobotDataset,
    outcome_instance,
    start_episode: int = 0
):
    """
    Interactive labeling mode with episode preview.
    
    Args:
        dataset: Dataset to label
        outcome_instance: Episode outcome plugin instance
        start_episode: Episode to start from
    """
    print("\n" + "="*60)
    print("Interactive Episode Outcome Labeling")
    print("="*60)
    print("\nCommands:")
    print("  s / success  - Mark as success")
    print("  f / failure  - Mark as failure")
    print("  u / unknown  - Mark as unknown")
    print("  skip         - Skip this episode")
    print("  goto N       - Jump to episode N")
    print("  stats        - Show statistics")
    print("  quit / q     - Save and exit")
    print("="*60 + "\n")
    
    episode_idx = start_episode
    num_episodes = len(dataset.meta.episodes)
    
    while episode_idx < num_episodes:
        episode_info = dataset.meta.episodes[episode_idx]
        current_outcome = outcome_instance.get_outcome(episode_idx)
        
        print(f"\nEpisode {episode_idx}/{num_episodes-1}:")
        print(f"  Frames: {episode_info['length']}")
        print(f"  Current label: {current_outcome or '(none)'}")
        
        # Prompt
        response = input(f"\nLabel episode {episode_idx} (s/f/u/skip/goto/stats/quit): ").strip().lower()
        
        if response in ['q', 'quit']:
            print("\nSaving and exiting...")
            break
        elif response in ['s', 'success']:
            outcome_instance.set_episode_outcome(episode_idx, 'success')
            print(f"✓ Episode {episode_idx} marked as SUCCESS")
            episode_idx += 1
        elif response in ['f', 'failure']:
            outcome_instance.set_episode_outcome(episode_idx, 'failure')
            print(f"✓ Episode {episode_idx} marked as FAILURE")
            episode_idx += 1
        elif response in ['u', 'unknown']:
            outcome_instance.set_episode_outcome(episode_idx, 'unknown')
            print(f"✓ Episode {episode_idx} marked as UNKNOWN")
            episode_idx += 1
        elif response == 'skip':
            print(f"Skipped episode {episode_idx}")
            episode_idx += 1
        elif response.startswith('goto'):
            try:
                target = int(response.split()[1])
                if 0 <= target < num_episodes:
                    episode_idx = target
                    print(f"Jumped to episode {episode_idx}")
                else:
                    print(f"Invalid episode {target}. Must be 0-{num_episodes-1}")
            except (IndexError, ValueError):
                print("Invalid goto command. Usage: goto N")
        elif response == 'stats':
            stats = outcome_instance.get_statistics()
            print("\nStatistics:")
            print(f"  Total episodes: {stats['total_episodes']}")
            print(f"  Labeled: {stats['labeled']} ({100*stats['labeled']/stats['total_episodes']:.1f}%)")
            print(f"  Success: {stats['success']}")
            print(f"  Failure: {stats['failure']}")
            print(f"  Unknown: {stats['unknown']}")
            print(f"  Unlabeled: {stats['unlabeled']}")
        else:
            print("Invalid command. Try: s, f, u, skip, goto N, stats, or quit")
    
    print("\n✓ Labeling complete")


def main():
    parser = argparse.ArgumentParser(
        description="Label episode outcomes for a robot dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Range Format Examples:
  Single episode:     5:success
  Episode range:      0-42:failure
  Open-ended range:   44-:success (from 44 to end)

Usage Examples:
  # Interactive mode
  python label_episode_outcomes.py --dataset lerobot/pusht
  
  # Command-line ranges
  python label_episode_outcomes.py --dataset lerobot/pusht \\
      --ranges "0-42:success" "43:failure" "44-:success"
  
  # From file
  python label_episode_outcomes.py --dataset lerobot/pusht \\
      --file labels.txt
  
  # Show current labels
  python label_episode_outcomes.py --dataset lerobot/pusht --show

labels.txt format:
  # One range per line, # for comments
  0-42: success
  43: failure
  44-: success
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset repository ID (e.g., lerobot/pusht)'
    )
    parser.add_argument(
        '--root',
        type=str,
        default=None,
        help='Dataset root directory'
    )
    parser.add_argument(
        '--ranges',
        type=str,
        nargs='+',
        help='Range specifications (e.g., "0-42:success" "43:failure")'
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Load ranges from file'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show current labels and statistics'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive labeling mode'
    )
    parser.add_argument(
        '--start-episode',
        type=int,
        default=0,
        help='Starting episode for interactive mode (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(args.dataset, root=args.root)
    print(f"  Episodes: {len(dataset.meta.episodes)}")
    print(f"  Total frames: {dataset.num_frames}")
    
    # Create plugin
    plugin = EpisodeOutcomePlugin()
    wrapped = WrappedRobotDataset(datasets=dataset, plugins=[plugin])
    outcome_instance = wrapped._plugin_instances[0][0]
    
    # Show mode
    if args.show:
        stats = outcome_instance.get_statistics()
        print("\n" + "="*60)
        print("Current Labels:")
        print("="*60)
        
        print(f"\nStatistics:")
        print(f"  Total episodes: {stats['total_episodes']}")
        print(f"  Labeled: {stats['labeled']} ({100*stats['labeled']/stats['total_episodes']:.1f}%)")
        print(f"  Success: {stats['success']}")
        print(f"  Failure: {stats['failure']}")
        print(f"  Unknown: {stats['unknown']}")
        print(f"  Unlabeled: {stats['unlabeled']}")
        
        # Show all outcomes
        if stats['labeled'] > 0:
            print(f"\nLabeled Episodes:")
            all_outcomes = outcome_instance.get_all_outcomes()
            for episode_idx in sorted(all_outcomes.keys()):
                outcome = all_outcomes[episode_idx]
                print(f"  Episode {episode_idx:4d}: {outcome}")
        
        return
    
    # Determine mode
    if args.ranges or args.file:
        # Batch mode from ranges or file
        if args.file:
            print(f"\nLoading ranges from: {args.file}")
            range_specs = load_ranges_from_file(args.file)
        else:
            range_specs = args.ranges
        
        print(f"\nParsing {len(range_specs)} range specification(s)...")
        for spec in range_specs:
            print(f"  {spec}")
        
        try:
            max_episode = len(dataset.meta.episodes) - 1
            outcomes_dict = expand_range_specs(range_specs, max_episode)
            
            print(f"\nApplying labels to {len(outcomes_dict)} episodes...")
            outcome_instance.set_episode_outcomes_batch(outcomes_dict)
            
            print("\n✓ Labels applied successfully!")
            
            # Show statistics
            stats = outcome_instance.get_statistics()
            print(f"\nStatistics:")
            print(f"  Total episodes: {stats['total_episodes']}")
            print(f"  Labeled: {stats['labeled']} ({100*stats['labeled']/stats['total_episodes']:.1f}%)")
            print(f"  Success: {stats['success']}")
            print(f"  Failure: {stats['failure']}")
            print(f"  Unknown: {stats['unknown']}")
            
        except ValueError as e:
            print(f"\n❌ Error: {e}")
            return 1
    
    else:
        # Interactive mode (default)
        interactive_labeling(dataset, outcome_instance, args.start_episode)
    
    return 0


if __name__ == "__main__":
    exit(main())


