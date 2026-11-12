from pydantic import BaseModel, Field

class PolicyConfig(BaseModel):
    type: str = Field(
        ...,
        description="The action policy implementation used during rollout.",
    )

    fallback_action_type: str = Field(
        default="eef_delta_pose",
        description="The canonical action type used by the fallback policy.",
    )

    fallback_action_values: dict[str, float] = Field(
        default_factory=dict,
        description="The canonical numeric action values used by the fallback policy.",
    )