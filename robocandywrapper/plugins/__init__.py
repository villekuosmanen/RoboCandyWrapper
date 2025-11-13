"""Plugin implementations for RoboCandyWrapper."""

from robocandywrapper.plugins.affordance import (
    LabelledAffordancesPlugin,
    AffordancesInstance,
)
from robocandywrapper.plugins.denserewards import (
    DenseRewardsPlugin,
    DenseRewardsInstance,
)

__all__ = [
    "LabelledAffordancesPlugin",
    "AffordancesInstance",
    "DenseRewardsPlugin",
    "DenseRewardsInstance",
]

