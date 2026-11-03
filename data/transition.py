from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from data.obs import Observation


class Transition(BaseModel):
    observation: Observation = Field(
        ...,
        description="The canonical observation produced after applying the action.",
    )

    reward: float = Field(
        ...,
        description="The scalar reward returned by the environment for this step.",
    )

    terminated: bool = Field(
        ...,
        description=(
            "Whether the episode ended because the environment reached a terminal state."
        ),
    )

    truncated: bool = Field(
        ...,
        description=(
            "Whether the episode ended because of a time limit or another external condition."
        ),
    )

    success: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the task objective has been successfully completed, when the "
            "environment exposes a success signal."
        ),
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional auxiliary diagnostics associated with the transition.",
    )