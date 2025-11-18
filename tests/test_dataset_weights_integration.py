"""
Integration test for dataset weights flow:
1. Create dataset without weights
2. Create sampler (which extracts weights from config)
3. Update dataset metadata with weights
4. Verify stats are computed with correct weights
"""

import numpy as np
from robocandywrapper.wrapper import WrappedRobotDataset
from robocandywrapper.samplers import make_sampler


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


def test_weights_integration():
    """Test the full workflow of creating dataset, sampler, and updating weights."""
    print("\n" + "="*60)
    print("Integration Test: Dataset Weights Flow")
    print("="*60)
    
    # Create datasets with different stats
    stats1 = {
        "action": {
            "mean": np.array([1.0, 2.0]),
            "std": np.array([0.5, 0.5]),
            "min": np.array([-1.0, -1.0]),
            "max": np.array([3.0, 4.0]),
            "count": np.array([10000]),
        }
    }
    
    stats2 = {
        "action": {
            "mean": np.array([4.0, 5.0]),
            "std": np.array([1.0, 1.0]),
            "min": np.array([0.0, 0.0]),
            "max": np.array([8.0, 10.0]),
            "count": np.array([1000]),
        }
    }
    
    dataset1 = MockLeRobotDataset(
        repo_id="large_dataset",
        fps=20,
        features={"action": {}},
        num_frames=10000,
        stats=stats1
    )
    
    dataset2 = MockLeRobotDataset(
        repo_id="small_dataset",
        fps=20,
        features={"action": {}},
        num_frames=1000,
        stats=stats2
    )
    
    print("\n1. Creating wrapped dataset (without weights initially)")
    wrapped_dataset = WrappedRobotDataset(
        datasets=[dataset1, dataset2],
        plugins=None,
    )
    
    # Get initial stats (should be unweighted)
    initial_stats = wrapped_dataset.meta.stats
    print(f"   Initial stats mean (unweighted): {initial_stats['action']['mean']}")
    
    # Expected unweighted mean
    w1_unweighted = 10000 / 11000
    w2_unweighted = 1000 / 11000
    expected_unweighted = stats1['action']['mean'] * w1_unweighted + stats2['action']['mean'] * w2_unweighted
    print(f"   Expected: {expected_unweighted}")
    
    print("\n2. Creating sampler with weights")
    sampler_config = {
        "type": "weighted",
        "dataset_weights": {
            "large_dataset": 1.0,
            "small_dataset": 2.0,  # 2x weight
        },
        "shuffle": True,
        "seed": 42,
    }
    
    sampler, shuffle, dataset_weights = make_sampler(
        wrapped_dataset,
        sampler_config=sampler_config
    )
    
    print(f"   Sampler created: {sampler is not None}")
    print(f"   Dataset weights extracted: {dataset_weights}")
    
    print("\n3. Updating dataset metadata with weights")
    if dataset_weights is not None:
        wrapped_dataset.update_dataset_weights(dataset_weights)
    
    # Get updated stats (should be weighted)
    updated_stats = wrapped_dataset.meta.stats
    print(f"   Updated stats mean (weighted): {updated_stats['action']['mean']}")
    
    # Expected weighted mean
    effective_count1 = 10000 * 1.0
    effective_count2 = 1000 * 2.0
    total_effective = effective_count1 + effective_count2
    w1_weighted = effective_count1 / total_effective  # 10000/12000 = 0.833
    w2_weighted = effective_count2 / total_effective  # 2000/12000 = 0.167
    expected_weighted = stats1['action']['mean'] * w1_weighted + stats2['action']['mean'] * w2_weighted
    print(f"   Expected: {expected_weighted}")
    
    # Verify the change
    print("\n4. Verification")
    np.testing.assert_allclose(
        updated_stats['action']['mean'],
        expected_weighted,
        rtol=1e-5,
        err_msg="Weighted mean doesn't match expected value"
    )
    
    # Verify the stats have changed from initial to updated
    assert not np.allclose(initial_stats['action']['mean'], updated_stats['action']['mean']), \
        "Stats should have changed after updating weights"
    
    print("   ✅ Stats correctly updated with weights!")
    print(f"   Mean shifted from {initial_stats['action']['mean']} to {updated_stats['action']['mean']}")
    print(f"   Small dataset (2x weight) pulls mean toward [4.0, 5.0] as expected")
    
    print("\n" + "="*60)
    print("✅ INTEGRATION TEST PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_weights_integration()

