from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    task_id: str = Field(
        ...,
        description="A stable, simulator-independent identifier for the task used throughout the project",
    )

    instruction: str = Field(
        ...,
        description="A human-readable natural-language description of what the agent should accomplish",
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional task-specific information needed by an environment adapter",
    )