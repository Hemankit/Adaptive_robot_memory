from pydantic import BaseModel, Field

from data.segment import TrajectorySegment
from data.step import StepRecord
from data.record import ObservationRecord

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