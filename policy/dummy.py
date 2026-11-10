
from data.task import TaskSpec
from policy.action_policy import ActionPolicy
from data.obs import Observation
from data.segment import SegmentData
from data.action import Action

class RecordingActionPolicy(ActionPolicy):

    def __init__(self):
        self.retrieval_counts: list[int] = []

    def act(
        self,
        task: TaskSpec,
        observation: Observation,
        retrieved_segments: list[SegmentData],
    ) -> Action:
        self.retrieval_counts.append(len(retrieved_segments))
        return Action(
            action_type="eef_delta_pose",
            values={
                key: 0.0
                for key in ("dx", "dy", "dz", "dax", "day", "daz", "gripper")
            },
        )