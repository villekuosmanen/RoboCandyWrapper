"""Plugin implementations for RoboCandyWrapper."""

from robocandywrapper.plugins.affordance import (
    LabelledAffordancePlugin,
    AffordanceInstance,
)
from robocandywrapper.plugins.episode_outcome import (
    EpisodeOutcomePlugin,
    EpisodeOutcomeInstance,
)

__all__ = [
    "LabelledAffordancePlugin",
    "AffordanceInstance",
    "EpisodeOutcomePlugin",
    "EpisodeOutcomeInstance",
]

