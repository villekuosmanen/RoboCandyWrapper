"""
Quick verification that weights work as multipliers on dataset sizes.
"""

from robocandywrapper.samplers import WeightedSampler


def test_weight_multipliers():
    """Verify weights multiply dataset sizes correctly."""
    
    # Scenario from the user's example:
    # - Large dataset: 10,000 frames
    # - Small corrections dataset: 1,000 frames
    # - Want to upsample corrections by 2.0x
    
    dataset_ranges = [(0, 10000), (10000, 11000)]
    dataset_ids = ["large_dataset", "corrections_dataset"]
    dataset_weights = {
        "large_dataset": 1.0,
        "corrections_dataset": 2.0,  # Sample as if 2x larger
    }
    
    sampler = WeightedSampler(
        dataset_ranges=dataset_ranges,
        dataset_ids=dataset_ids,
        dataset_weights=dataset_weights,
        samples_per_epoch=12000,
        shuffle=True,
        seed=42,
        replacement=True,
    )
    
    # Collect samples
    samples = list(sampler)
    
    # Count samples from each dataset
    large_count = sum(1 for idx in samples if idx < 10000)
    corrections_count = sum(1 for idx in samples if idx >= 10000)
    
    print("=" * 60)
    print("Weight Multiplier Verification")
    print("=" * 60)
    print(f"\nDataset Sizes:")
    print(f"  Large dataset: 10,000 frames")
    print(f"  Corrections dataset: 1,000 frames")
    print(f"\nWeights (multipliers on size):")
    print(f"  Large dataset: 1.0 → effective size 10,000")
    print(f"  Corrections dataset: 2.0 → effective size 2,000")
    print(f"\nTotal effective size: 12,000 frames")
    print(f"\nExpected sampling distribution:")
    print(f"  Large dataset: 10,000/12,000 = 83.3%")
    print(f"  Corrections dataset: 2,000/12,000 = 16.7%")
    print(f"\nActual sampling distribution:")
    print(f"  Large dataset: {large_count:,} / {len(samples):,} = {large_count/len(samples)*100:.1f}%")
    print(f"  Corrections dataset: {corrections_count:,} / {len(samples):,} = {corrections_count/len(samples)*100:.1f}%")
    
    # Verify results
    expected_large = 10000
    expected_corrections = 2000
    
    success = (
        abs(large_count - expected_large) < 100 and
        abs(corrections_count - expected_corrections) < 100
    )
    
    if success:
        print(f"\n✅ SUCCESS: Weights work as multipliers on dataset sizes!")
        print(f"   Corrections dataset is upsampled 2x but remains minority of samples")
    else:
        print(f"\n❌ FAILURE: Distribution doesn't match expected values")
        print(f"   Expected: large={expected_large}, corrections={expected_corrections}")
        print(f"   Got: large={large_count}, corrections={corrections_count}")
    
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    test_weight_multipliers()

