from pydantic import BaseModel, Field
from typing import Any

class StepRecord(BaseModel):
    episode_id: str = Field(
        ...,
        description="The identifier of the episode containing this step.",
    )

    step_index: int = Field(
        ...,
        description="The zero-based position of this step within the episode.",
    )

    action_type: str = Field(
        ...,
        description="The canonical type of action executed at this step.",
    )

    action_values: dict[str, float] = Field(
        ...,
        description="The canonical numeric action values executed at this step.",
    )

    reward: float = Field(
        ...,
        description="The scalar reward produced by the environment for this step.",
    )

    terminated: bool = Field(
        ...,
        description="Whether the environment reported termination after this step.",
    )

    truncated: bool = Field(
        ...,
        description="Whether the environment reported truncation after this step.",
    )

    success: bool | None = Field(
        default=None,
        description="Whether the task was successful at this step, when a success signal is available.",
    )

    timestamp: float | None = Field(
        default=None,
        description="The simulation or episode time associated with the pre-action observation.",
    )

    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional auxiliary information associated with the recorded step.",
    )