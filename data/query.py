from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    instruction: str = Field(
        ...,
        description="The natural-language task instruction used to query episodic memory.",
    )

class QueryEmbedding(BaseModel):
    vector: list[float] = Field(
        ...,
        description="The fixed-dimensional numeric embedding representing the retrieval query.",
    )

    embedding_id: str = Field(
        ...,
        description="A stable identifier describing the embedding method or model that produced the query vector.",
    )

    dimension: int = Field(
        ...,
        description="The dimensionality of the query embedding vector.",
    )

