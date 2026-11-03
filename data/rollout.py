from pydantic import BaseModel, Field
from data.obs import Observation
from data.task import TaskSpec
from data.action import Action
from data.transition import Transition

class RolloutStep(BaseModel):
    step_index: int = Field(
        ...,
        description="The zero-based index of this step within the rollout.",
    )

    observation: Observation = Field(
        ...,
        description="The canonical observation available to the agent before selecting the action.",
    )

    action: Action = Field(
        ...,
        description="The canonical action selected by the agent for this step.",
    )

    transition: Transition = Field(
        ...,
        description="The canonical transition produced by the environment after applying the action.",
    )


class RolloutResult(BaseModel):
    task: TaskSpec = Field(
        ...,
        description="The canonical task specification used for this rollout.",
    )

    seed: int | None = Field(
        default=None,
        description="The optional seed used to initialize the environment for this rollout.",
    )

    initial_observation: Observation = Field(
        ...,
        description="The canonical observation returned when the environment was initially reset.",
    )

    steps: list[RolloutStep] = Field(
        ...,
        description="The ordered sequence of recorded rollout steps.",
    )