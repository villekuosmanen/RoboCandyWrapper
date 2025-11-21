"""
Exploration script to compare v2.1 and v3.0 dataset formats.

This script loads both versions and systematically compares:
- Directory structure
- Metadata properties
- Sample data structure
- Stats format
- API compatibility

Usage:
    python examples/explore_v2_v3_compatibility.py \
        --v2-repo=username/dataset_v21 \
        --v3-repo=username/dataset_v30 
"""

import argparse
from pathlib import Path
import pprint

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robocandywrapper.datasets.legacy_dataset import LegacyLeRobotDataset


def explore_metadata(dataset, version_label):
    """Explore metadata properties of a dataset."""
    print(f"\n{'='*80}")
    print(f"METADATA EXPLORATION: {version_label}")
    print(f"{'='*80}\n")
    
    meta = dataset.meta
    
    print("Available properties on meta:")
    print("-" * 40)
    meta_attrs = [attr for attr in dir(meta) if not attr.startswith('_')]
    for attr in sorted(meta_attrs):
        try:
            value = getattr(meta, attr)
            if callable(value):
                print(f"  {attr}() - method")
            else:
                value_type = type(value).__name__
                print(f"  {attr}: {value_type}")
        except Exception as e:
            print(f"  {attr}: ERROR - {e}")
    
    print("\nKey properties:")
    print("-" * 40)
    try:
        print(f"fps: {meta.fps}")
    except Exception as e:
        print(f"fps: ERROR - {e}")
    
    try:
        print(f"total_episodes: {meta.total_episodes}")
    except Exception as e:
        print(f"total_episodes: ERROR - {e}")
    
    try:
        print(f"total_frames: {meta.total_frames}")
    except Exception as e:
        print(f"total_frames: ERROR - {e}")
    
    try:
        print(f"camera_keys: {meta.camera_keys}")
    except Exception as e:
        print(f"camera_keys: ERROR - {e}")
    
    print("\nFeatures:")
    print("-" * 40)
    try:
        for key, feature in meta.features.items():
            print(f"  {key}: {feature}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\nStats structure:")
    print("-" * 40)
    try:
        # Show first feature's stats as example
        first_feature = list(meta.stats.keys())[0]
        print(f"Example ({first_feature}):")
        pprint.pprint(meta.stats[first_feature], indent=4)
        print(f"\nAll feature keys: {list(meta.stats.keys())}")
    except Exception as e:
        print(f"ERROR: {e}")


def explore_sample_data(dataset, version_label, index=0):
    """Explore sample data structure."""
    print(f"\n{'='*80}")
    print(f"SAMPLE DATA EXPLORATION: {version_label} [index={index}]")
    print(f"{'='*80}\n")
    
    try:
        sample = dataset[index]
        
        print("Sample keys:")
        print("-" * 40)
        for key in sorted(sample.keys()):
            value = sample[key]
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value).__name__}, value={value}")
    
    except Exception as e:
        print(f"ERROR loading sample: {e}")
        import traceback
        traceback.print_exc()


def explore_temporal_windowing(dataset, version_label):
    """Test temporal windowing (delta_timestamps)."""
    print(f"\n{'='*80}")
    print(f"TEMPORAL WINDOWING TEST: {version_label}")
    print(f"{'='*80}\n")
    
    print("Testing if delta_timestamps parameter is supported...")
    print("(This requires reloading the dataset)")
    print("Status: ⏳ MANUAL TEST REQUIRED")
    print("\nTo test, modify this script to reload with:")
    print("  delta_timestamps={'observation.state': [-0.1, 0.0]}")
    print("And check if returned tensors are stacked along time dimension.")


def explore_directory_structure(dataset, version_label):
    """Explore the actual directory structure."""
    print(f"\n{'='*80}")
    print(f"DIRECTORY STRUCTURE: {version_label}")
    print(f"{'='*80}\n")
    
    root = Path(dataset.root)
    
    if not root.exists():
        print(f"Dataset root does not exist locally: {root}")
        return
    
    print(f"Root: {root}\n")
    
    # Check meta/ directory
    meta_dir = root / "meta"
    if meta_dir.exists():
        print("meta/")
        for item in sorted(meta_dir.iterdir()):
            if item.is_dir():
                num_files = len(list(item.iterdir()))
                print(f"  {item.name}/ ({num_files} files)")
            else:
                size_kb = item.stat().st_size / 1024
                print(f"  {item.name} ({size_kb:.1f} KB)")
    
    # Check data/ directory
    data_dir = root / "data"
    if data_dir.exists():
        data_files = list(data_dir.glob("**/*.parquet"))  # Recursive to find files in chunks
        print(f"\ndata/ ({len(data_files)} parquet files)")
        if data_files:
            # Show first few
            for f in sorted(data_files)[:3]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.relative_to(data_dir)} ({size_mb:.1f} MB)")
            if len(data_files) > 3:
                print(f"  ... and {len(data_files) - 3} more")
    
    # Check videos/ directory
    videos_dir = root / "videos"
    if videos_dir.exists():
        video_files = list(videos_dir.glob("**/*.mp4"))
        print(f"\nvideos/ ({len(video_files)} mp4 files)")
        
        # Check structure
        subdirs = [d for d in videos_dir.iterdir() if d.is_dir()]
        if subdirs:
            print("  Organized by camera/chunk:")
            for subdir in sorted(subdirs):
                vids = sorted(subdir.glob("**/*.mp4"))
                print(f"    {subdir.name}/ ({len(vids)} videos)")
                # Show first few videos with sizes
                for vid in vids[:2]:
                    size_mb = vid.stat().st_size / (1024 * 1024)
                    rel_path = vid.relative_to(subdir)
                    print(f"      {rel_path} ({size_mb:.1f} MB)")
                if len(vids) > 2:
                    print(f"      ... and {len(vids) - 2} more")
        elif video_files:
            # Flat structure
            for f in sorted(video_files)[:3]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name} ({size_mb:.1f} MB)")
            if len(video_files) > 3:
                print(f"  ... and {len(video_files) - 3} more")


def compare_datasets(v2_dataset, v3_dataset):
    """Direct comparison between v2 and v3."""
    print(f"\n{'='*80}")
    print(f"DIRECT COMPARISON")
    print(f"{'='*80}\n")
    
    print("Dataset lengths:")
    print(f"  v2.1: {len(v2_dataset)} frames")
    print(f"  v3.0: {len(v3_dataset)} frames")
    
    print("\nSample keys comparison:")
    try:
        v2_sample = v2_dataset[0]
        v3_sample = v3_dataset[0]
        
        v2_keys = set(v2_sample.keys())
        v3_keys = set(v3_sample.keys())
        
        common = v2_keys & v3_keys
        v2_only = v2_keys - v3_keys
        v3_only = v3_keys - v2_keys
        
        print(f"  Common keys: {len(common)}")
        for key in sorted(common):
            print(f"    {key}")
        
        if v2_only:
            print(f"\n  v2.1 only: {v2_only}")
        if v3_only:
            print(f"\n  v3.0 only: {v3_only}")
    
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\nMetadata properties comparison:")
    v2_meta_attrs = set(attr for attr in dir(v2_dataset.meta) if not attr.startswith('_'))
    v3_meta_attrs = set(attr for attr in dir(v3_dataset.meta) if not attr.startswith('_'))
    
    common_attrs = v2_meta_attrs & v3_meta_attrs
    v2_only_attrs = v2_meta_attrs - v3_meta_attrs
    v3_only_attrs = v3_meta_attrs - v2_meta_attrs
    
    print(f"  Common metadata properties: {len(common_attrs)}")
    if v2_only_attrs:
        print(f"  v2.1 only: {v2_only_attrs}")
    if v3_only_attrs:
        print(f"  v3.0 only: {v3_only_attrs}")


def main():
    parser = argparse.ArgumentParser(description="Explore v2.1 and v3.0 dataset compatibility")
    parser.add_argument("--v2-repo", required=True, help="v2.1 dataset repo ID")
    parser.add_argument("--v3-repo", required=True, help="v3.0 dataset repo ID")
    parser.add_argument("--root", default=None, help="Dataset cache root directory")
    parser.add_argument("--index", type=int, default=0, help="Sample index to inspect")
    
    args = parser.parse_args()
    
    print("="*80)
    print("v2.1 vs v3.0 Dataset Format Exploration")
    print("="*80)
    
    # Load v2.1 dataset
    print(f"\nLoading v2.1 dataset: {args.v2_repo}")
    print("-" * 40)
    try:
        v2_dataset = LegacyLeRobotDataset(
            repo_id=args.v2_repo,
            root=args.root,
        )
        print("✅ Successfully loaded v2.1 dataset")
    except Exception as e:
        print(f"❌ Failed to load v2.1 dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load v3.0 dataset
    print(f"\nLoading v3.0 dataset: {args.v3_repo}")
    print("-" * 40)
    try:
        v3_dataset = LeRobotDataset(
            repo_id=args.v3_repo,
            root=args.root,
        )
        print("✅ Successfully loaded v3.0 dataset")
    except Exception as e:
        print(f"❌ Failed to load v3.0 dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Explore both datasets
    explore_directory_structure(v2_dataset, "v2.1")
    explore_directory_structure(v3_dataset, "v3.0")
    
    explore_metadata(v2_dataset, "v2.1")
    explore_metadata(v3_dataset, "v3.0")
    
    explore_sample_data(v2_dataset, "v2.1", args.index)
    explore_sample_data(v3_dataset, "v3.0", args.index)
    
    explore_temporal_windowing(v2_dataset, "v2.1")
    explore_temporal_windowing(v3_dataset, "v3.0")
    
    compare_datasets(v2_dataset, v3_dataset)
    
    print("\n" + "="*80)
    print("Exploration complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the output above")
    print("2. Update docs/v2_v3_format_comparison.md with findings")
    print("3. Test temporal windowing by modifying this script")
    print("4. Try mixed loading with make_dataset_without_config()")


if __name__ == "__main__":
    main()
