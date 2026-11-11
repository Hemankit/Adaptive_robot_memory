from pydantic import BaseModel, Field

from data.metric import ExperimentMetrics, EpisodeMetrics
from data.task import TaskSpec


class EpisodeRunSpec(BaseModel):
    episode_id: str = Field(
        ...,
        description="A unique identifier for the episode produced by this experiment run.",
    )

    task: TaskSpec = Field(
        ...,
        description="The canonical task specification to execute during this episode.",
    )

    seed: int | None = Field(
        default=None,
        description="The optional environment seed used for this episode.",
    )

class ExperimentSpec(BaseModel):
    experiment_id: str = Field(
        ...,
        description="A unique identifier for this experiment execution.",
    )

    episodes: list[EpisodeRunSpec] = Field(
        ...,
        description="The ordered episode runs that should be executed as part of the experiment.",
    )

class ExperimentResult(BaseModel):
    experiment_id: str = Field(
        ...,
        description="The identifier of the experiment that produced this result.",
    )

    episode_metrics: list[EpisodeMetrics] = Field(
        ...,
        description="The ordered per-episode metrics produced during the experiment.",
    )

    metrics: ExperimentMetrics = Field(
        ...,
        description="The aggregate metrics calculated across all episodes in the experiment.",
    )