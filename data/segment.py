from pydantic import BaseModel, Field


class TrajectorySegment(BaseModel):
    segment_id: str = Field(
        ...,
        description="A unique identifier for this trajectory segment.",
    )

    episode_id: str = Field(
        ...,
        description="The identifier of the source episode from which this segment was created.",
    )

    start_step: int = Field(
        ...,
        description="The inclusive index of the first action step contained in the segment.",
    )

    end_step: int = Field(
        ...,
        description="The inclusive index of the final action step contained in the segment.",
    )

    task_id: str = Field(
        ...,
        description="The stable identifier of the task associated with the source episode.",
    )

    instruction: str = Field(
        ...,
        description="The natural-language task instruction associated with the source episode.",
    )