from pydantic import BaseModel, Field
from data.step import StepRecord
from data.obs import ObservationRecord


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

class SegmentData(BaseModel):
    segment: TrajectorySegment = Field(
        ...,
        description="The trajectory segment describing the source episode and step range.",
    )

    steps: list[StepRecord] = Field(
        ...,
        description="The ordered persisted action-step records contained in the segment.",
    )

    observations: list[ObservationRecord] = Field(
        ...,
        description="The ordered persisted observations spanning the segment, including the observation before the first action and after the final action.",
    )


class SegmentEmbedding(BaseModel):
    segment_id: str = Field(
        ...,
        description="The unique identifier of the trajectory segment represented by this embedding.",
    )

    vector: list[float] = Field(
        ...,
        description="The fixed-dimensional numeric embedding representing the trajectory segment.",
    )

    embedding_id: str = Field(
        ...,
        description="A stable identifier describing the embedding method or model that produced the vector.",
    )

    dimension: int = Field(
        ...,
        description="The dimensionality of the embedding vector.",
    )