from typing import Iterator
import numpy as np
import torch.utils.data


class WeightedSampler(torch.utils.data.Sampler):
    """
    Sample from datasets with specified weights/ratios.
    
    This allows you to oversample/undersample specific datasets by their IDs.
    """
    
    def __init__(
        self,
        dataset_ranges: list[tuple[int, int]],
        dataset_ids: list[str],
        dataset_weights: dict[str, float] | None = None,
        samples_per_epoch: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        replacement: bool = True,
    ):
        """
        Initialize weighted sampler.
        
        Args:
            dataset_ranges: List of (start_idx, end_idx) tuples for each dataset
            dataset_ids: List of dataset repo IDs corresponding to ranges
            dataset_weights: Dict mapping dataset IDs to weight multipliers.
                           Datasets not in dict get weight 1.0. Example:
                           {"janedoe/rare-data": 3.0, "janedoe/boring-data": 0.5}
            samples_per_epoch: Total samples per epoch. If None, uses sum of dataset lengths.
            shuffle: Whether to shuffle the combined samples
            seed: Random seed for reproducibility
            replacement: If True, sample with replacement (can oversample small datasets)
        """
        if len(dataset_ranges) != len(dataset_ids):
            raise ValueError("dataset_ranges and dataset_ids must have same length")
        
        self.dataset_ranges = dataset_ranges
        self.dataset_ids = dataset_ids
        self.shuffle = shuffle
        self.seed = seed
        self.replacement = replacement
        
        # Calculate dataset lengths
        self.dataset_lengths = [end - start for start, end in dataset_ranges]
        self.total_length = sum(self.dataset_lengths)
        
        # Build weights array: use custom weights if provided, otherwise 1.0
        weights = []
        for dataset_id, length in zip(dataset_ids, self.dataset_lengths):
            if dataset_weights and dataset_id in dataset_weights:
                weights.append(dataset_weights[dataset_id])
            else:
                weights.append(1.0)
        
        # Normalize weights
        self.weights = np.array(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        
        # Determine total samples per epoch
        self.samples_per_epoch = samples_per_epoch or self.total_length
        
        # Pre-compute how many samples from each dataset
        self._samples_per_dataset = (self.weights * self.samples_per_epoch).astype(int)
        
        # Distribute remaining samples due to rounding
        remainder = self.samples_per_epoch - self._samples_per_dataset.sum()
        if remainder > 0:
            # Give extra samples to datasets with highest fractional parts
            fractional_parts = (self.weights * self.samples_per_epoch) - self._samples_per_dataset
            extra_indices = np.argsort(fractional_parts)[-remainder:]
            self._samples_per_dataset[extra_indices] += 1
    
    def __iter__(self) -> Iterator[int]:
        """Iterate with weighted sampling, yielding flat indices."""
        rng = np.random.RandomState(self.seed)
        
        # Generate indices for each dataset
        all_samples = []
        for dataset_idx, (n_samples, (start_idx, end_idx)) in enumerate(
            zip(self._samples_per_dataset, self.dataset_ranges, strict=True)
        ):
            if n_samples == 0:
                continue
            
            dataset_length = end_idx - start_idx
            
            if self.replacement or n_samples <= dataset_length:
                # Sample local indices within this dataset
                local_indices = rng.choice(
                    dataset_length,
                    size=n_samples,
                    replace=self.replacement
                )
            else:
                # Need more samples than dataset size, must use replacement
                local_indices = rng.choice(
                    dataset_length,
                    size=n_samples,
                    replace=True
                )
            
            # Convert to global indices
            global_indices = [start_idx + int(idx) for idx in local_indices]
            all_samples.extend(global_indices)
        
        # Shuffle the combined samples
        if self.shuffle:
            rng.shuffle(all_samples)
        
        for sample in all_samples:
            yield sample
    
    def __len__(self) -> int:
        return self.samples_per_epoch
