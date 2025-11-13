from typing import Iterator
import numpy as np
import torch.utils.data


class UniformSampler(torch.utils.data.Sampler):
    """Uniform sampling across all datasets (concatenated, flat index space)."""
    
    def __init__(
        self,
        dataset_ranges: list[tuple[int, int]],
        dataset_ids: list[str] | None = None,
        shuffle: bool = True,
        seed: int | None = None
    ):
        """
        Initialize uniform sampler.
        
        Args:
            dataset_ranges: List of (start_idx, end_idx) tuples for each dataset
            dataset_ids: Optional list of dataset IDs (for compatibility, not used)
            shuffle: Whether to shuffle indices
            seed: Random seed for reproducibility
        """
        self.dataset_ranges = dataset_ranges
        self.dataset_ids = dataset_ids
        self.shuffle = shuffle
        self.seed = seed
        
        # Calculate total length
        self.total_length = sum(end - start for start, end in dataset_ranges)
    
    def __iter__(self) -> Iterator[int]:
        """Iterate through all samples uniformly."""
        indices = list(range(self.total_length))
        
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)
        
        for idx in indices:
            yield idx
    
    def __len__(self) -> int:
        return self.total_length
