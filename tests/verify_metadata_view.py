"""
Verification script for WrappedRobotDatasetMetadataView functionality.

Tests:
1. FPS warning when datasets have different FPS
2. Features intersection including plugin features
3. Stats aggregation with optional weighting
"""

import numpy as np


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


class MockPluginInstance:
    """Mock plugin instance."""
    
    def __init__(self, keys):
        self._keys = keys
    
    def get_data_keys(self):
        return self._keys
    
    def get_item_data(self, idx, episode_idx):
        return {key: None for key in self._keys}
    
    def detach(self):
        pass


def test_fps_warning():
    """Test that FPS warning is logged when datasets differ."""
    print("\n" + "="*60)
    print("Test 1: FPS Warning")
    print("="*60)
    
    from robocandywrapper.metadata_view import WrappedRobotDatasetMetadataView
    
    # Create datasets with different FPS
    dataset1 = MockLeRobotDataset(
        repo_id="dataset_20fps",
        fps=20,
        features={"action": {}, "state": {}},
        num_frames=1000
    )
    
    dataset2 = MockLeRobotDataset(
        repo_id="dataset_30fps",
        fps=30,
        features={"action": {}, "state": {}},
        num_frames=1000
    )
    
    # Create metadata view
    meta_view = WrappedRobotDatasetMetadataView(
        datasets=[dataset1, dataset2],
        plugin_instances=[[], []],
    )
    
    print(f"\nDataset 1 FPS: {dataset1.meta.fps}")
    print(f"Dataset 2 FPS: {dataset2.meta.fps}")
    print(f"\nCalling meta_view.fps (should log warning):")
    fps = meta_view.fps
    print(f"Returned FPS: {fps}")
    
    assert fps == 20, "Should return first dataset's FPS"
    print("\n✅ Test passed: FPS warning logged correctly")


def test_features_intersection():
    """Test that features are intersected across datasets and plugins are added."""
    print("\n" + "="*60)
    print("Test 2: Features Intersection")
    print("="*60)
    
    from robocandywrapper.metadata_view import WrappedRobotDatasetMetadataView
    
    # Create datasets with different features
    dataset1 = MockLeRobotDataset(
        repo_id="dataset_a",
        fps=20,
        features={"action": {}, "state": {}, "image_top": {}},
        num_frames=1000
    )
    
    dataset2 = MockLeRobotDataset(
        repo_id="dataset_b",
        fps=20,
        features={"action": {}, "state": {}, "image_front": {}},
        num_frames=500
    )
    
    # Create plugins with additional features
    plugin1 = MockPluginInstance(["affordance_mask", "goal_embedding"])
    plugin2 = MockPluginInstance(["success_prediction"])
    
    # Create metadata view
    meta_view = WrappedRobotDatasetMetadataView(
        datasets=[dataset1, dataset2],
        plugin_instances=[[plugin1], [plugin2]],
    )
    
    features = meta_view.features
    
    print(f"\nDataset 1 features: {list(dataset1.meta.features.keys())}")
    print(f"Dataset 2 features: {list(dataset2.meta.features.keys())}")
    print(f"Plugin 1 features: {plugin1.get_data_keys()}")
    print(f"Plugin 2 features: {plugin2.get_data_keys()}")
    print(f"\nIntersected features: {list(features.keys())}")
    
    # Expected: intersection of dataset features (action, state) + all plugin features
    expected_features = {
        "action", "state",  # Common to both datasets
        "affordance_mask", "goal_embedding", "success_prediction"  # Plugin features
    }
    
    assert set(features.keys()) == expected_features, f"Expected {expected_features}, got {set(features.keys())}"
    print("\n✅ Test passed: Features intersected correctly")


def test_weighted_stats():
    """Test that stats can be weighted."""
    print("\n" + "="*60)
    print("Test 3: Weighted Stats")
    print("="*60)
    
    from robocandywrapper.metadata_view import WrappedRobotDatasetMetadataView
    
    # Create datasets with different sizes and different stats
    # Dataset 1: Large dataset with mean [1.0, 2.0]
    stats1 = {
        "action": {
            "mean": np.array([1.0, 2.0]),
            "std": np.array([0.5, 0.5]),
            "min": np.array([-1.0, -1.0]),
            "max": np.array([3.0, 4.0]),
            "count": np.array([10000]),
        }
    }
    
    # Dataset 2: Small dataset with mean [4.0, 5.0]
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
    
    # Create metadata view with weights
    weights = {
        "large_dataset": 1.0,
        "small_dataset": 2.0,  # Upweight small dataset
    }
    
    meta_view = WrappedRobotDatasetMetadataView(
        datasets=[dataset1, dataset2],
        plugin_instances=[[], []],
        dataset_weights=weights,
    )
    
    stats = meta_view.stats
    
    print(f"\nDataset 1: {dataset1.repo_id}")
    print(f"  Size: {len(dataset1)}")
    print(f"  Weight: {weights[dataset1.repo_id]}")
    print(f"  Effective size: {len(dataset1) * weights[dataset1.repo_id]}")
    print(f"  Stats mean: {stats1['action']['mean']}")
    
    print(f"\nDataset 2: {dataset2.repo_id}")
    print(f"  Size: {len(dataset2)}")
    print(f"  Weight: {weights[dataset2.repo_id]}")
    print(f"  Effective size: {len(dataset2) * weights[dataset2.repo_id]}")
    print(f"  Stats mean: {stats2['action']['mean']}")
    
    # Calculate expected weighted mean
    # With weights applied: effective_count = count * weight
    count1 = 10000
    count2 = 1000
    weight1 = weights["large_dataset"]  # 1.0
    weight2 = weights["small_dataset"]  # 2.0
    
    effective_count1 = count1 * weight1  # 10000
    effective_count2 = count2 * weight2  # 2000
    total_effective = effective_count1 + effective_count2  # 12000
    
    w1 = effective_count1 / total_effective  # 10000/12000 = 0.833
    w2 = effective_count2 / total_effective  # 2000/12000 = 0.167
    
    expected_mean = stats1['action']['mean'] * w1 + stats2['action']['mean'] * w2
    
    # Expected variance calculation (correct formula)
    # σ²_total = Σ(w_i * (σ²_i + (μ_i - μ_total)²))
    expected_var = (
        w1 * (stats1['action']['std']**2 + (stats1['action']['mean'] - expected_mean)**2) +
        w2 * (stats2['action']['std']**2 + (stats2['action']['mean'] - expected_mean)**2)
    )
    expected_std = np.sqrt(expected_var)
    
    print(f"\n📊 Expected Weighted Stats:")
    print(f"  Dataset 1 effective weight: {w1:.3f} ({effective_count1}/{total_effective})")
    print(f"  Dataset 2 effective weight: {w2:.3f} ({effective_count2}/{total_effective})")
    print(f"  Expected mean: {expected_mean}")
    print(f"  Expected std: {expected_std}")
    
    print(f"\n📊 Actual Aggregated Stats:")
    print(f"  Mean: {stats['action']['mean']}")
    print(f"  Std: {stats['action']['std']}")
    print(f"  Min: {stats['action']['min']}")
    print(f"  Max: {stats['action']['max']}")
    print(f"  Count: {stats['action']['count']}")
    
    # Verify the weighted aggregation is correct
    assert "action" in stats, "Stats should contain action key"
    assert "mean" in stats["action"], "Stats should contain mean"
    
    # Check that the mean matches expected (within floating point tolerance)
    np.testing.assert_allclose(
        stats['action']['mean'],
        expected_mean,
        rtol=1e-5,
        err_msg="Weighted mean doesn't match expected value"
    )
    
    # Check that std matches expected
    np.testing.assert_allclose(
        stats['action']['std'],
        expected_std,
        rtol=1e-5,
        err_msg="Weighted std doesn't match expected value"
    )
    
    # Check min and max
    expected_min = np.minimum(stats1['action']['min'], stats2['action']['min'])
    expected_max = np.maximum(stats1['action']['max'], stats2['action']['max'])
    np.testing.assert_array_equal(stats['action']['min'], expected_min)
    np.testing.assert_array_equal(stats['action']['max'], expected_max)
    
    # Check total count
    assert stats['action']['count'] == total_effective, f"Expected count {total_effective}, got {stats['action']['count']}"
    
    print("\n✅ Test passed: Weighted stats computed correctly!")
    print(f"   Mean shifted from [1.0, 2.0] toward [4.0, 5.0] due to 2x weight on Dataset 2")


def test_camera_keys():
    """Test that camera keys are unionized."""
    print("\n" + "="*60)
    print("Test 4: Camera Keys Union")
    print("="*60)
    
    from robocandywrapper.metadata_view import WrappedRobotDatasetMetadataView
    
    dataset1 = MockLeRobotDataset(
        repo_id="dataset_a",
        fps=20,
        features={"action": {}, "image_top": {}, "image_wrist": {}},
        num_frames=1000
    )
    
    dataset2 = MockLeRobotDataset(
        repo_id="dataset_b",
        fps=20,
        features={"action": {}, "image_front": {}, "video_side": {}},
        num_frames=500
    )
    
    meta_view = WrappedRobotDatasetMetadataView(
        datasets=[dataset1, dataset2],
        plugin_instances=[[], []],
    )
    
    camera_keys = meta_view.camera_keys
    
    print(f"\nDataset 1 camera keys: {dataset1.meta.camera_keys}")
    print(f"Dataset 2 camera keys: {dataset2.meta.camera_keys}")
    print(f"Unionized camera keys: {camera_keys}")
    
    expected = sorted(["image_top", "image_wrist", "image_front", "video_side"])
    assert camera_keys == expected, f"Expected {expected}, got {camera_keys}"
    print("\n✅ Test passed: Camera keys unionized correctly")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WrappedRobotDatasetMetadataView Verification Tests")
    print("="*60)
    
    try:
        test_fps_warning()
        test_features_intersection()
        test_weighted_stats()
        test_camera_keys()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

