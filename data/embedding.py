from pydantic import BaseModel, Field


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