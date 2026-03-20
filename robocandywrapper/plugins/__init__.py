"""Plugin implementations for RoboCandyWrapper."""

from robocandywrapper.plugins.affordance import (
    LabelledAffordancePlugin,
    AffordanceInstance,
)
from robocandywrapper.plugins.control_mode import (
    ControlModePlugin,
    ControlModeInstance,
)
from robocandywrapper.plugins.episode_outcome import (
    EpisodeOutcomePlugin,
    EpisodeOutcomeInstance,
)

__all__ = [
    "ControlModePlugin",
    "ControlModeInstance",
    "LabelledAffordancePlugin",
    "AffordanceInstance",
    "EpisodeOutcomePlugin",
    "EpisodeOutcomeInstance",
]

