"""RoboCandyWrapper - Extensible dataset wrapper for LeRobot datasets."""

from robocandywrapper.plugin import (
    DatasetPlugin,
    PluginInstance,
    PluginConflictError,
)
from robocandywrapper.wrapper import WrappedRobotDataset
from robocandywrapper.metadata_view import WrappedRobotDatasetMetadataView
from robocandywrapper.samplers.weighted import WeightedSampler
from robocandywrapper.samplers.factory import make_sampler
from robocandywrapper.factory import make_dataset_without_config, make_dataset
from robocandywrapper.utils import WandBLogger
from robocandywrapper.constants import (
    CANDYWRAPPER_PLUGINS_DIR,
    AFFORDANCE_PLUGIN_NAME,
    EPISODE_OUTCOME_PLUGIN_NAME,
)

__version__ = "0.2.3"

__all__ = [
    "DatasetPlugin",
    "PluginInstance",
    "PluginConflictError",
    "WrappedRobotDataset",
    "WrappedRobotDatasetMetadataView",
    "WeightedSampler",
    "make_sampler",
    "make_dataset_without_config",
    "make_dataset",
    "WandBLogger",
    "CANDYWRAPPER_PLUGINS_DIR",
    "AFFORDANCE_PLUGIN_NAME",
    "EPISODE_OUTCOME_PLUGIN_NAME",
]
