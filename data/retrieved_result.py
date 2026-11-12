from pydantic import BaseModel, Field
from data.item import MemoryItem
class RetrievalResult(BaseModel):
    item: MemoryItem = Field(
        ...,
        description="The episodic memory item returned by retrieval.",
    )

    score: float = Field(
        ...,
        description="The similarity score between the retrieval query and the memory item.",
    )

class RetrievalControllerConfig(BaseModel):
    type: str = Field(
        ...,
        description="The retrieval timing strategy to use during rollout.",
    )

    interval: int | None = Field(
        default=None,
        description="The retrieval interval used by periodic retrieval controllers.",
    )

class RetrievalConfig(BaseModel):
    k: int = Field(
        ...,
        gt=0,
        description="The maximum number of memory items returned for each retrieval.",
    )