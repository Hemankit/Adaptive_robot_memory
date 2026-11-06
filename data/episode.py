from pydantic import BaseModel, Field
from typing import Any

class EpisodeRecord(BaseModel):
    episode_id: str = Field(
        ...,
        description="A unique identifier for the recorded episode.",
    )

    task_id: str = Field(
        ...,
        description="The stable task identifier associated with the episode.",
    )

    instruction: str = Field(
        ...,
        description="The natural-language task instruction associated with the episode.",
    )

    seed: int | None = Field(
        default=None,
        description="The optional environment seed used for the episode.",
    )

    num_steps: int = Field(
        ...,
        description="The number of recorded action steps in the episode.",
    )

    success: bool | None = Field(
        default=None,
        description="Whether the task was successfully completed, when a success signal is available.",
    )

    total_reward: float = Field(
        ...,
        description="The cumulative environment reward obtained during the episode.",
    )

    terminated: bool = Field(
        ...,
        description="Whether the episode ended because the environment reported termination.",
    )

    truncated: bool = Field(
        ...,
        description="Whether the environment reported that the episode was truncated.",
    )

    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional auxiliary information associated with the recorded episode.",
    )