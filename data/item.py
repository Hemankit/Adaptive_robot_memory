from pydantic import BaseModel, Field

from data.segment import SegmentEmbedding, TrajectorySegment



class MemoryItem(BaseModel):
    segment: TrajectorySegment = Field(
        ...,
        description="The trajectory segment identifying the source episode and contiguous step range represented by this memory item.",
    )

    embedding: SegmentEmbedding = Field(
        ...,
        description="The fixed-dimensional embedding used to represent this trajectory segment for retrieval.",
    )

class MemoryConfig(BaseModel):
    source_episode_ids: list[str] = Field(
        ...,
        description="The persisted successful episode identifiers used to construct episodic memory.",
    )