from pydantic import BaseModel, Field


class EpisodeMetrics(BaseModel):
    episode_id: str = Field(
        ...,
        description="The identifier of the episode represented by these metrics.",
    )

    success: bool | None = Field(
        default=None,
        description="Whether the task was successfully completed, when a success signal is available.",
    )

    total_reward: float = Field(
        ...,
        description="The cumulative environment reward obtained during the episode.",
    )

    num_steps: int = Field(
        ...,
        description="The number of action steps executed during the episode.",
    )

    retrieval_count: int = Field(
        ...,
        description="The number of rollout steps at which episodic-memory retrieval was performed.",
    )

    retrieval_frequency: float = Field(
        ...,
        description="The fraction of episode steps at which episodic-memory retrieval was performed.",
    )

    average_retrieval_latency_ms: float | None = Field(
        default=None,
        description="The mean retrieval-search latency across steps where retrieval was performed.",
    )

    average_retrieval_score: float | None = Field(
        default=None,
        description="The mean similarity score across all retrieved memory items returned during the episode.",
    )

from pydantic import BaseModel, Field


class ExperimentMetrics(BaseModel):
    num_episodes: int = Field(
        ...,
        description="The total number of episodes included in the experiment metrics.",
    )

    num_success_evaluable_episodes: int = Field(
        ...,
        description="The number of episodes that expose a known task-success outcome.",
    )

    success_rate: float | None = Field(
        default=None,
        description="The fraction of success-evaluable episodes that completed successfully.",
    )

    average_total_reward: float = Field(
        ...,
        description="The mean cumulative reward across all episodes.",
    )

    average_num_steps: float = Field(
        ...,
        description="The mean number of action steps executed across all episodes.",
    )

    average_retrieval_count: float = Field(
        ...,
        description="The mean number of retrieval operations performed per episode.",
    )

    average_retrieval_frequency: float = Field(
        ...,
        description="The mean per-episode fraction of steps at which retrieval was performed.",
    )

    average_retrieval_latency_ms: float | None = Field(
        default=None,
        description="The mean per-episode retrieval-search latency among episodes with retrieval latency measurements.",
    )

    average_retrieval_score: float | None = Field(
        default=None,
        description="The mean per-episode retrieval similarity score among episodes with retrieved scores.",
    )

