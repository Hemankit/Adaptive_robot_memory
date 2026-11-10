
from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    """Represents an observation from a robot simulator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rgb: dict[str, np.ndarray] = Field(
        ...,
        description="Mapping of camera names to RGB image observations.",
    )

    depth: dict[str, np.ndarray] | None = Field(
        default=None,
        description="Mapping of camera names to depth observations.",
    )

    robot_state: dict[str, Any] | np.ndarray | None = Field(
        default=None,
        description="Structured or vectorized robot state information.",
    )

    object_state: dict[str, Any] | None = Field(
    default=None,
    description=(
        "Structured object information available to the agent under the "
        "configured observation mode. Must not contain privileged simulator "
        "state unless that state is explicitly part of the agent's observation."
    ),
)

    timestamp: float | None = Field(
    default=None,
    description="Simulation or episode time associated with the observation, in seconds.",
)

    metadata: dict[str, Any] | None = Field(
    default=None,
    description="Optional auxiliary information associated with the observation.",
)


class ObservationRecord(BaseModel):
    episode_id: str = Field(
        ...,
        description="The identifier of the episode containing this observation.",
    )

    observation_index: int = Field(
        ...,
        description="The zero-based position of this observation within the episode.",
    )

    rgb_artifacts: dict[str, str] = Field(
        ...,
        description="Mapping of camera names to artifact keys containing persisted RGB arrays.",
    )

    depth_artifacts: dict[str, str] | None = Field(
        default=None,
        description="Mapping of camera names to artifact keys containing persisted depth arrays.",
    )

    robot_state: dict[str, Any] | list[Any] | None = Field(
        default=None,
        description="Storage-friendly representation of the robot state associated with this observation.",
    )

    object_state: dict[str, Any] | None = Field(
        default=None,
        description="Storage-friendly structured object information available to the agent.",
    )

    timestamp: float | None = Field(
        default=None,
        description="The simulation or episode time associated with the observation, in seconds.",
    )