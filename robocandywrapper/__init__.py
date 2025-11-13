"""RoboCandyWrapper - Extensible dataset wrapper for LeRobot datasets."""

from robocandywrapper.plugin import (
    DatasetPlugin,
    PluginInstance,
    PluginConflictError,
)
from robocandywrapper.wrapper import WrappedRobotDataset
from robocandywrapper.samplers.uniform import UniformSampler
from robocandywrapper.samplers.weighted import WeightedSampler
from robocandywrapper.factory import make_dataset_without_config, make_dataset

__version__ = "0.1.0"

__all__ = [
    "DatasetPlugin",
    "PluginInstance",
    "PluginConflictError",
    "WrappedRobotDataset",
    "UniformSampler",
    "WeightedSampler",
    "make_dataset_without_config",
    "make_dataset",
]
