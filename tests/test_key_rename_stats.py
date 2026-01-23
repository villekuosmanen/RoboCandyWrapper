"""
Test for key_rename_map functionality in stats aggregation.

Tests that when datasets have differently-named keys that map to the same 
target key, their stats are properly combined as if they were the same key.
"""

import sys
from pathlib import Path

# Add the local package to path before imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from robocandywrapper.wrapper import WrappedRobotDataset


class MockLeRobotDataset:
    """Mock dataset for testing."""
    
    def __init__(self, repo_id, fps, features, num_frames, stats=None):
        self.repo_id = repo_id
        self._fps = fps
        self._features = features
        self._num_frames = num_frames
        
        # Create mock metadata
        self.meta = MockMetadata(repo_id, fps, features, stats)
        self.hf_features = features
        self.features = features
    
    def __len__(self):
        return self._num_frames
    
    def __getitem__(self, idx):
        # Minimal mock for dataset access
        return {"action": np.array([0.0, 0.0])}


class MockMetadata:
    """Mock metadata object."""
    
    def __init__(self, repo_id, fps, features, stats=None):
        self.repo_id = repo_id
        self._fps = fps
        self._features = features
        self.info = {"fps": fps}
        
        # Use provided stats or default
        if stats is None:
            self.stats = {
                "action": {
                    "mean": np.array([0.0, 0.0]),
                    "std": np.array([1.0, 1.0]),
                    "min": np.array([-1.0, -1.0]),
                    "max": np.array([1.0, 1.0]),
                    "count": np.array([1000]),
                }
            }
        else:
            self.stats = stats
            
        self.camera_keys = [k for k in features if "image" in k or "video" in k]
        self.image_keys = [k for k in features if "image" in k]
        self.video_keys = [k for k in features if "video" in k]
    
    @property
    def fps(self):
        return self._fps
    
    @property
    def features(self):
        return self._features


def test_key_rename_stats_aggregation():
    """
    Test that keys are properly renamed in stats aggregation.
    
    Scenario:
    - Dataset 1 has "action.pos" key with certain stats
    - Dataset 2 has "trajectory" key with certain stats
    - key_rename_map maps both to "action"
    - Result should have "action" stats that combine both sources
    """
    print("\n" + "="*60)
    print("Test: Key Rename Stats Aggregation")
    print("="*60)
    
    # Dataset 1: has "action.pos" key
    stats1 = {
        "action.pos": {
            "mean": np.array([1.0, 2.0]),
            "std": np.array([0.5, 0.5]),
            "min": np.array([-1.0, -1.0]),
            "max": np.array([3.0, 4.0]),
            "count": np.array([1000]),  # 1000 samples
        }
    }
    
    # Dataset 2: has "trajectory" key  
    stats2 = {
        "trajectory": {
            "mean": np.array([5.0, 6.0]),
            "std": np.array([1.0, 1.0]),
            "min": np.array([0.0, 0.0]),
            "max": np.array([10.0, 12.0]),
            "count": np.array([1000]),  # Same count for simpler math
        }
    }
    
    dataset1 = MockLeRobotDataset(
        repo_id="dataset_with_action_pos",
        fps=20,
        features={"action.pos": {"shape": [2]}},
        num_frames=1000,
        stats=stats1
    )
    
    dataset2 = MockLeRobotDataset(
        repo_id="dataset_with_trajectory",
        fps=20,
        features={"trajectory": {"shape": [2]}},
        num_frames=1000,
        stats=stats2
    )
    
    # Create wrapped dataset with key rename map
    key_rename_map = {
        "action.pos": "action",
        "trajectory": "action",
    }
    
    print(f"\n1. Creating wrapped dataset with key_rename_map: {key_rename_map}")
    
    wrapped_dataset = WrappedRobotDataset(
        datasets=[dataset1, dataset2],
        plugins=None,
        key_rename_map=key_rename_map,
    )
    
    # Check that features were renamed
    print("\n2. Checking features")
    assert "action" in wrapped_dataset.meta.features, \
        "Renamed 'action' key should be in features"
    assert "action.pos" not in wrapped_dataset.meta.features, \
        "Original 'action.pos' key should not be in features"
    assert "trajectory" not in wrapped_dataset.meta.features, \
        "Original 'trajectory' key should not be in features"
    print("   ✅ Features correctly renamed")
    
    # Check that stats were combined
    print("\n3. Checking stats aggregation")
    combined_stats = wrapped_dataset.meta.stats
    
    assert "action" in combined_stats, \
        "Combined stats should have 'action' key"
    assert "action.pos" not in combined_stats, \
        "Original 'action.pos' should not be in combined stats"
    assert "trajectory" not in combined_stats, \
        "Original 'trajectory' should not be in combined stats"
    
    # With equal counts (1000 each), the combined mean should be the average
    # mean = (1000 * [1.0, 2.0] + 1000 * [5.0, 6.0]) / 2000 = [3.0, 4.0]
    expected_mean = np.array([3.0, 4.0])
    
    np.testing.assert_allclose(
        combined_stats["action"]["mean"],
        expected_mean,
        rtol=1e-5,
        err_msg="Combined mean should be average of both datasets' means"
    )
    print(f"   Combined mean: {combined_stats['action']['mean']}")
    print(f"   Expected mean: {expected_mean}")
    print("   ✅ Stats correctly combined")
    
    # Check min/max
    expected_min = np.array([-1.0, -1.0])  # min of both datasets
    expected_max = np.array([10.0, 12.0])  # max of both datasets
    
    np.testing.assert_allclose(
        combined_stats["action"]["min"],
        expected_min,
        rtol=1e-5,
        err_msg="Combined min should be minimum across both datasets"
    )
    np.testing.assert_allclose(
        combined_stats["action"]["max"],
        expected_max,
        rtol=1e-5,
        err_msg="Combined max should be maximum across both datasets"
    )
    print(f"   Combined min: {combined_stats['action']['min']}, expected: {expected_min}")
    print(f"   Combined max: {combined_stats['action']['max']}, expected: {expected_max}")
    print("   ✅ Min/max correctly combined")
    
    # Check count
    assert combined_stats["action"]["count"] == 2000, \
        f"Combined count should be 2000, got {combined_stats['action']['count']}"
    print(f"   Combined count: {combined_stats['action']['count']}")
    print("   ✅ Count correctly combined")
    
    print("\n" + "="*60)
    print("✅ KEY RENAME STATS TEST PASSED!")
    print("="*60 + "\n")


def test_key_rename_with_different_counts():
    """
    Test key rename with datasets having different sample counts.
    
    This ensures weighted aggregation works correctly with renamed keys.
    """
    print("\n" + "="*60)
    print("Test: Key Rename Stats with Different Counts")
    print("="*60)
    
    # Dataset 1: 1000 samples with "pos"
    stats1 = {
        "pos": {
            "mean": np.array([0.0]),
            "std": np.array([1.0]),
            "min": np.array([-3.0]),
            "max": np.array([3.0]),
            "count": np.array([1000]),
        }
    }
    
    # Dataset 2: 3000 samples with "position"
    stats2 = {
        "position": {
            "mean": np.array([4.0]),
            "std": np.array([2.0]),
            "min": np.array([-2.0]),
            "max": np.array([10.0]),
            "count": np.array([3000]),
        }
    }
    
    dataset1 = MockLeRobotDataset(
        repo_id="ds1",
        fps=20,
        features={"pos": {"shape": [1]}},
        num_frames=1000,
        stats=stats1
    )
    
    dataset2 = MockLeRobotDataset(
        repo_id="ds2",
        fps=20,
        features={"position": {"shape": [1]}},
        num_frames=3000,
        stats=stats2
    )
    
    key_rename_map = {
        "pos": "state",
        "position": "state",
    }
    
    print(f"\n1. Dataset 1: 1000 samples, mean=0.0")
    print(f"   Dataset 2: 3000 samples, mean=4.0")
    print(f"   key_rename_map: {key_rename_map}")
    
    wrapped_dataset = WrappedRobotDataset(
        datasets=[dataset1, dataset2],
        plugins=None,
        key_rename_map=key_rename_map,
    )
    
    combined_stats = wrapped_dataset.meta.stats
    
    # Expected weighted mean: (1000 * 0.0 + 3000 * 4.0) / 4000 = 3.0
    expected_mean = np.array([3.0])
    
    print(f"\n2. Combined stats for 'state':")
    print(f"   Mean: {combined_stats['state']['mean']} (expected: {expected_mean})")
    
    np.testing.assert_allclose(
        combined_stats["state"]["mean"],
        expected_mean,
        rtol=1e-5,
        err_msg="Weighted mean should account for different counts"
    )
    
    assert combined_stats["state"]["count"] == 4000, \
        f"Total count should be 4000, got {combined_stats['state']['count']}"
    print(f"   Count: {combined_stats['state']['count']} (expected: 4000)")
    
    print("\n" + "="*60)
    print("✅ KEY RENAME WITH DIFFERENT COUNTS TEST PASSED!")
    print("="*60 + "\n")


def test_key_rename_partial_rename():
    """
    Test that keys that don't need renaming are preserved.
    
    Only keys in key_rename_map should be renamed; others should pass through.
    """
    print("\n" + "="*60)
    print("Test: Partial Key Rename")
    print("="*60)
    
    # Both datasets have "action" (no rename needed) but different secondary keys
    stats1 = {
        "action": {
            "mean": np.array([1.0]),
            "std": np.array([1.0]),
            "min": np.array([0.0]),
            "max": np.array([2.0]),
            "count": np.array([1000]),
        },
        "observation.state": {
            "mean": np.array([0.5]),
            "std": np.array([0.1]),
            "min": np.array([0.0]),
            "max": np.array([1.0]),
            "count": np.array([1000]),
        }
    }
    
    stats2 = {
        "action": {
            "mean": np.array([3.0]),
            "std": np.array([1.0]),
            "min": np.array([2.0]),
            "max": np.array([4.0]),
            "count": np.array([1000]),
        },
        "observation.state": {
            "mean": np.array([0.5]),
            "std": np.array([0.1]),
            "min": np.array([0.0]),
            "max": np.array([1.0]),
            "count": np.array([1000]),
        }
    }
    
    dataset1 = MockLeRobotDataset(
        repo_id="ds1",
        fps=20,
        features={"action": {"shape": [1]}, "observation.state": {"shape": [1]}},
        num_frames=1000,
        stats=stats1
    )
    
    dataset2 = MockLeRobotDataset(
        repo_id="ds2",
        fps=20,
        features={"action": {"shape": [1]}, "observation.state": {"shape": [1]}},
        num_frames=1000,
        stats=stats2
    )
    
    # No key rename map - should work normally
    print("\n1. Creating wrapped dataset without key_rename_map")
    
    wrapped_dataset = WrappedRobotDataset(
        datasets=[dataset1, dataset2],
        plugins=None,
        key_rename_map=None,
    )
    
    combined_stats = wrapped_dataset.meta.stats
    
    # "action" should be combined normally
    assert "action" in combined_stats, "action key should be present"
    expected_action_mean = np.array([2.0])  # (1.0 + 3.0) / 2
    
    np.testing.assert_allclose(
        combined_stats["action"]["mean"],
        expected_action_mean,
        rtol=1e-5
    )
    print(f"   action mean: {combined_stats['action']['mean']} (expected: {expected_action_mean})")
    
    # "observation.state" should also be present  
    assert "observation.state" in combined_stats, "observation.state should be present"
    print(f"   observation.state mean: {combined_stats['observation.state']['mean']}")
    
    print("\n" + "="*60)
    print("✅ PARTIAL KEY RENAME TEST PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_key_rename_stats_aggregation()
    test_key_rename_with_different_counts()
    test_key_rename_partial_rename()
    
    print("\n" + "="*60)
    print("ALL KEY RENAME STATS TESTS PASSED!")
    print("="*60 + "\n")
