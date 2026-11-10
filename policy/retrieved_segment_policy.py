
from data.task import TaskSpec
from policy.action_policy import ActionPolicy

from data.obs import Observation
from data.segment import SegmentData
from data.step import StepRecord
from data.action import Action
class RetrievedSegmentReplayPolicy(ActionPolicy):

    def __init__(
        self,
        fallback_policy: ActionPolicy,
    ):
        self._fallback_policy = fallback_policy

    def act(
        self,
        task: TaskSpec,
        observation: Observation,
        retrieved_segments: list[SegmentData],
    ) -> Action:
        if not retrieved_segments:
          return self._fallback_policy.act(
        task=task,
        observation=observation,
        retrieved_segments=[],
    )
        segment = retrieved_segments[0]
        if len(segment.steps) > 0:
            step = segment.steps[0]
            return self._action_from_step(step)
        else:
            raise ValueError("Retrieved segment has no steps.")
            

    def _action_from_step(
        self,
        step: StepRecord,
    ) -> Action:
        return Action(
    action_type=step.action_type,
    values=dict(step.action_values),
)