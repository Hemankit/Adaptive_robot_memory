from pydantic import BaseModel, Field
from data.query import RetrievalQuery

class RetrievalEvent(BaseModel):
    step_index: int = Field(
        ...,
        description="The zero-based rollout step at which the retrieval decision was made.",
    )

    retrieval_performed: bool = Field(
        ...,
        description="Whether episodic-memory retrieval was performed at this step.",
    )

    query: RetrievalQuery | None = Field(
        default=None,
        description="The retrieval query used at this step when retrieval was performed.",
    )

    retrieved_segment_ids: list[str] = Field(
        default_factory=list,
        description="The ordered identifiers of trajectory segments returned by retrieval.",
    )

    scores: list[float] = Field(
        default_factory=list,
        description="The similarity scores corresponding to the retrieved trajectory segments.",
    )

    retrieval_latency_ms: float | None = Field(
        default=None,
        description="The elapsed retrieval time in milliseconds when retrieval was performed.",
    )

class RetrievalEventRecord(BaseModel):
    episode_id: str = Field(
        ...,
        description="The identifier of the episode during which this retrieval decision occurred.",
    )

    step_index: int = Field(
        ...,
        description="The zero-based rollout step at which the retrieval decision was made.",
    )

    retrieval_performed: bool = Field(
        ...,
        description="Whether episodic-memory retrieval was performed at this step.",
    )

    query_instruction: str | None = Field(
        default=None,
        description="The natural-language retrieval query used when retrieval was performed.",
    )

    retrieved_segment_ids: list[str] = Field(
        default_factory=list,
        description="The ordered identifiers of trajectory segments returned by retrieval.",
    )

    scores: list[float] = Field(
        default_factory=list,
        description="The similarity scores corresponding to the retrieved trajectory segments.",
    )

    retrieval_latency_ms: float | None = Field(
        default=None,
        description="The elapsed retrieval-search time in milliseconds when retrieval was performed.",
    )