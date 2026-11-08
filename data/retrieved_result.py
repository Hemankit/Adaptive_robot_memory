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