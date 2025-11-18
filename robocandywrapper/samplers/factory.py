"""Factory functions for creating data samplers."""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch.utils.data

from robocandywrapper.samplers.config import SamplerConfig
from robocandywrapper.samplers.weighted import WeightedSampler


def load_sampler_config(config_path: Optional[str] = None) -> Optional[SamplerConfig]:
    """
    Load sampler configuration from a JSON file or environment variable.
    
    Args:
        config_path: Path to JSON config file. If None, checks SAMPLER_CONFIG_PATH env var.
        
    Returns:
        SamplerConfig instance, or None if no config found.
        
    Example JSON format:
        {
            "type": "weighted",
            "dataset_weights": {
                "repo/dataset1": 2.0,
                "repo/dataset2": 0.5
            },
            "samples_per_epoch": 10000,
            "shuffle": true,
            "seed": 42,
            "replacement": true
        }
    """
    # Try provided path or environment variable
    if config_path is None:
        config_path = os.environ.get("SAMPLER_CONFIG_PATH")
    
    if config_path is None:
        return None
    
    config_path = Path(config_path)
    if not config_path.exists():
        logging.warning(f"Sampler config file not found: {config_path}")
        return None
    
    # Only JSON is supported
    if config_path.suffix != '.json':
        logging.warning(f"Unsupported config file format: {config_path.suffix}. Only .json is supported.")
        return None
    
    # Load and parse JSON into SamplerConfig
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        config = SamplerConfig.from_dict(data)
        logging.info(f"Loaded sampler config from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {config_path}: {e}")
        return None
    except ValueError as e:
        logging.error(f"Invalid sampler configuration in {config_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to load sampler config from {config_path}: {e}")
        return None


def make_sampler(
    dataset,
    sampler_config: Optional[SamplerConfig | dict] = None,
    config_path: Optional[str] = None,
) -> tuple[Optional[torch.utils.data.Sampler], bool, Optional[dict[str, float]]]:
    """
    Create a sampler for the dataset based on configuration.
    
    This function provides a clean interface for creating samplers without
    tightly coupling the dataset wrapper and sampler implementations.
    
    Args:
        dataset: A WrappedRobotDataset instance with get_dataset_ranges() method
        sampler_config: SamplerConfig instance or dict. If dict, will be parsed into SamplerConfig.
                       If None, will try to load from config_path.
        config_path: Path to JSON config file. Only used if sampler_config is None.
        
    Returns:
        Tuple of (sampler, shuffle, dataset_weights):
            - sampler: The created sampler instance, or None if no sampler config provided
            - shuffle: Whether to shuffle (False if using sampler, True otherwise)
            - dataset_weights: Dict of dataset weights from config, or None if no weights specified
            
    Example sampler_config dict:
        {
            "type": "weighted",
            "dataset_weights": {
                "lerobot/pusht": 2.0,
                "lerobot/aloha_sim_insertion": 0.5,
            },
            "samples_per_epoch": 10000,
            "shuffle": true,
            "seed": 42,
            "replacement": true
        }
    """
    # Try to load config if not provided
    if sampler_config is None:
        sampler_config = load_sampler_config(config_path)
    
    # No config means use default PyTorch shuffling
    if sampler_config is None:
        return None, True, None
    
    # Convert dict to SamplerConfig if needed
    if isinstance(sampler_config, dict):
        try:
            sampler_config = SamplerConfig.from_dict(sampler_config)
        except ValueError as e:
            logging.error(f"Invalid sampler configuration: {e}. Using default shuffling.")
            return None, True, None
    
    # Validate sampler type
    if sampler_config.type != "weighted":
        logging.warning(f"Unsupported sampler type: {sampler_config.type}. Using default shuffling.")
        return None, True, None
    
    # Check if dataset has the required method
    if not hasattr(dataset, 'get_dataset_ranges'):
        logging.warning(
            "Dataset does not have get_dataset_ranges() method. "
            "Sampler can only be used with WrappedRobotDataset. "
            "Using default shuffling."
        )
        return None, True, None
    
    # Get dataset information
    try:
        dataset_ranges, dataset_ids = dataset.get_dataset_ranges()
    except Exception as e:
        logging.error(f"Failed to get dataset ranges: {e}. Using default shuffling.")
        return None, True, None
    
    # If only one dataset, no need for weighted sampling
    if len(dataset_ranges) == 1:
        logging.info("Single dataset detected. Using default shuffling instead of weighted sampler.")
        return None, True, None
    
    # Log sampler configuration
    logging.info("Creating WeightedSampler:")
    logging.info(f"  Datasets: {dataset_ids}")
    logging.info(f"  Dataset ranges: {dataset_ranges}")
    if sampler_config.dataset_weights:
        logging.info(f"  Custom weights: {sampler_config.dataset_weights}")
    else:
        logging.info("  Using uniform weights (based on dataset sizes)")
    if sampler_config.samples_per_epoch:
        logging.info(f"  Samples per epoch: {sampler_config.samples_per_epoch}")
    else:
        logging.info(f"  Samples per epoch: {sum(end - start for start, end in dataset_ranges)} (dataset total)")
    
    try:
        sampler = WeightedSampler(
            dataset_ranges=dataset_ranges,
            dataset_ids=dataset_ids,
            dataset_weights=sampler_config.dataset_weights,
            samples_per_epoch=sampler_config.samples_per_epoch,
            shuffle=sampler_config.shuffle,
            seed=sampler_config.seed,
            replacement=sampler_config.replacement,
        )
        
        # When using a sampler, DataLoader shuffle must be False
        # Return the dataset weights so they can be used for metadata stats
        return sampler, False, sampler_config.dataset_weights
        
    except Exception as e:
        logging.error(f"Failed to create WeightedSampler: {e}. Using default shuffling.")
        return None, True, None

