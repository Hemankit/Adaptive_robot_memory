from typing import Any

from pydantic import BaseModel, Field


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