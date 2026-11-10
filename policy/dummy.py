
from data.task import TaskSpec
from policy.action_policy import ActionPolicy
from data.obs import Observation
from data.segment import SegmentData
from data.action import Action

class FixedActionPolicy(ActionPolicy):

    def __init__(self, action: Action | None = None):
        self._action = action
        self.retrieval_counts: list[int] = []

    def act(
        self,
        task: TaskSpec,
        observation: Observation,
        retrieved_segments: list[SegmentData],
    ) -> Action:
        self.retrieval_counts.append(len(retrieved_segments))
        if self._action is not None:
            return self._action
        return Action(
            action_type="eef_delta_pose",
            values={
                key: 0.0
                for key in ("dx", "dy", "dz", "dax", "day", "daz", "gripper")
            },
        )