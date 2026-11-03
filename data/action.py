

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
class Action(BaseModel):
  values: dict[str, float] = Field(
    ...,
    description="The numeric action payload sent to the environment."
  )
  action_type: str = Field(
    ...,
    description="A simulator-independent label describing how to interpret the action"
  )
  metadata: Optional[Dict[str, Any]] = Field(
    None,
    description="Optional auxiliary information associated with the action."
  )
