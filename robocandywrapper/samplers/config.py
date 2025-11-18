"""Configuration classes for samplers."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplerConfig:
    """
    Configuration for dataset samplers.
    
    Attributes:
        type: Type of sampler (currently only "weighted" is supported)
        dataset_weights: Dictionary mapping dataset repo IDs to weight multipliers.
                        Weights multiply the effective dataset size for sampling.
                        Weight 2.0 = sample as if dataset were 2x its actual size.
                        Example: Dataset with 1000 frames and weight 2.0 has effective size 2000.
                        Datasets not in dict get weight 1.0 (original size).
                        Example: {"lerobot/pusht": 2.0, "lerobot/aloha": 0.5}
        samples_per_epoch: Total samples per epoch. If None, uses sum of dataset lengths.
        shuffle: Whether to shuffle the combined samples after weighted sampling
        seed: Random seed for reproducibility. If None, sampling is non-deterministic.
        replacement: If True, sample with replacement (can oversample small datasets)
    """
    type: str = "weighted"
    dataset_weights: Optional[dict[str, float]] = None
    samples_per_epoch: Optional[int] = None
    shuffle: bool = True
    seed: Optional[int] = None
    replacement: bool = True
    
    @classmethod
    def from_dict(cls, data: dict) -> "SamplerConfig":
        """
        Create SamplerConfig from a dictionary.
        
        Args:
            data: Dictionary containing sampler configuration
            
        Returns:
            SamplerConfig instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate sampler type
        sampler_type = data.get("type", "weighted")
        if sampler_type != "weighted":
            raise ValueError(f"Unsupported sampler type: {sampler_type}. Only 'weighted' is supported.")
        
        # Extract and validate fields
        dataset_weights = data.get("dataset_weights", None)
        if dataset_weights is not None:
            if not isinstance(dataset_weights, dict):
                raise ValueError(f"dataset_weights must be a dict, got {type(dataset_weights)}")
            # Validate all weights are numeric
            for repo_id, weight in dataset_weights.items():
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Weight for {repo_id} must be numeric, got {type(weight)}")
                if weight <= 0:
                    raise ValueError(f"Weight for {repo_id} must be positive, got {weight}")
        
        samples_per_epoch = data.get("samples_per_epoch", None)
        if samples_per_epoch is not None:
            if not isinstance(samples_per_epoch, int):
                raise ValueError(f"samples_per_epoch must be an int, got {type(samples_per_epoch)}")
            if samples_per_epoch <= 0:
                raise ValueError(f"samples_per_epoch must be positive, got {samples_per_epoch}")
        
        shuffle = data.get("shuffle", True)
        if not isinstance(shuffle, bool):
            raise ValueError(f"shuffle must be a bool, got {type(shuffle)}")
        
        seed = data.get("seed", None)
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError(f"seed must be an int, got {type(seed)}")
        
        replacement = data.get("replacement", True)
        if not isinstance(replacement, bool):
            raise ValueError(f"replacement must be a bool, got {type(replacement)}")
        
        return cls(
            type=sampler_type,
            dataset_weights=dataset_weights,
            samples_per_epoch=samples_per_epoch,
            shuffle=shuffle,
            seed=seed,
            replacement=replacement,
        )

