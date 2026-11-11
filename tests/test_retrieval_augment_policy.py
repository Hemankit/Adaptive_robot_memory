from unittest.mock import MagicMock

import numpy as np
import pytest

from data.obs import Observation
from data.segment import SegmentData
from data.task import TaskSpec
from policy.action_policy import ActionPolicy
from policy.dummy import FixedActionPolicy
from policy.retrieved_segment_policy import RetrievedSegmentReplayPolicy
from retrieval.retrieval_interfaces.never_retrieve_controller import NeverRetrieveController
from retrieval.retrieval_interfaces.always_retrieve_controller import AlwaysRetrieveController
from retrieval.retrieval_interfaces.periodic_retrieve_controller import PeriodicRetrieveController
from action_selection.action_selection_interfaces.retrieval_augmented_selection import RetrievalAugmentedActionSelector
from data.action import Action
from data.step import StepRecord
from data.segment import TrajectorySegment

class RecordingPolicy(ActionPolicy):

    def __init__(self, action: Action):
        self._action = action
        self.received_segments = None
        self.calls: list[list[SegmentData]] = []

    def act(
        self,
        task: TaskSpec,
        observation: Observation,
        retrieved_segments: list[SegmentData],
    ) -> Action:
        self.received_segments = retrieved_segments
        self.calls.append(retrieved_segments)
        return self._action


def test_never_retrieve_passes_no_memory_to_policy():

    task = TaskSpec(
        task_id="test_task",
        instruction="Place the mug in the drawer.",
        metadata=None,
    )

    observation = Observation(
        rgb={
            "agentview": np.zeros(
                (2, 2, 3),
                dtype=np.uint8,
            )
        }
    )

    expected_action = Action(
        action_type="eef_delta_pose",
        values={
            "dx": 0.0,
        },
    )

    policy = RecordingPolicy(
        action=expected_action,
    )

    controller = NeverRetrieveController()

    retriever = MagicMock()
    segment_loader = MagicMock()

    selector = RetrievalAugmentedActionSelector(
        controller=controller,
        retriever=retriever,
        segment_loader=segment_loader,
        policy=policy,
        k=3,
        event_recorder=MagicMock(),
    )

    action = selector.select_action(
        task=task,
        observation=observation,
        step_index=0,
    )

    assert policy.received_segments == []

    retriever.retrieve.assert_not_called()
    segment_loader.load.assert_not_called()

    assert action.action_type == expected_action.action_type
    assert action.values == expected_action.values

def test_always_retrieve_passes_segments_to_policy():

    task = TaskSpec(
        task_id="test_task",
        instruction="Place the mug in the drawer.",
        metadata=None,
    )

    observation = Observation(
        rgb={
            "agentview": np.zeros(
                (2, 2, 3),
                dtype=np.uint8,
            )
        }
    )

    expected_action = Action(
        action_type="eef_delta_pose",
        values={
            "dx": 0.0,
        },
    )

    policy = RecordingPolicy(
        action=expected_action,
    )

    controller = AlwaysRetrieveController()

    retriever = MagicMock()
    segment_loader = MagicMock()

    retrieved_segment = MagicMock()
    retrieval_result = MagicMock()

    retrieval_result.item.segment = retrieved_segment
    retrieved_segment.segment_id = "segment-1"
    retrieval_result.score = 0.9
    loaded_segment = MagicMock(spec=SegmentData)

    retriever.retrieve.return_value = [
        retrieval_result
    ]

    segment_loader.load.return_value = loaded_segment

    selector = RetrievalAugmentedActionSelector(
        controller=controller,
        retriever=retriever,
        segment_loader=segment_loader,
        policy=policy,
        k=3,
        event_recorder=MagicMock(),
    )

    action = selector.select_action(
        task=task,
        observation=observation,
        step_index=0,
    )

    assert policy.received_segments == [
        loaded_segment
    ]

    retriever.retrieve.assert_called_once()

    segment_loader.load.assert_called_once_with(
        retrieved_segment
    )

    assert action.action_type == expected_action.action_type
    assert action.values == expected_action.values


def test_retrieved_segment_replay_policy_uses_retrieved_action():

    retrieved_step = StepRecord(
        episode_id="episode_1",
        step_index=0,
        action_type="eef_delta_pose",
        action_values={
            "dx": 0.123,
        },
        reward=0.0,
        terminated=False,
        truncated=False,
    )

    stored_segment = MagicMock(spec=SegmentData)
    stored_segment.steps = [
        retrieved_step
    ]

    retrieved_segment_reference = MagicMock(
        spec=TrajectorySegment
    )

    fallback_action = Action(
        action_type="eef_delta_pose",
        values={
            "dx": 0.456,
        },
    )

    fallback_policy = FixedActionPolicy(
        action=fallback_action,
    )

    policy = RetrievedSegmentReplayPolicy(
        fallback_policy=fallback_policy,
    )

    controller = AlwaysRetrieveController()

    retriever = MagicMock()
    segment_loader = MagicMock()

    retrieval_result = MagicMock()
    retrieval_result.item.segment = (
        retrieved_segment_reference
    )
    retrieved_segment_reference.segment_id = "segment-1"
    retrieval_result.score = 0.9

    retriever.retrieve.return_value = [
        retrieval_result
    ]

    segment_loader.load.return_value = (
        stored_segment
    )

    selector = RetrievalAugmentedActionSelector(
        controller=controller,
        retriever=retriever,
        segment_loader=segment_loader,
        policy=policy,
        k=1,
        event_recorder=MagicMock(),
    )

    task = TaskSpec(
        task_id="test_task",
        instruction="Place the mug in the drawer.",
        metadata=None,
    )

    observation = Observation(
        rgb={
            "agentview": np.zeros(
                (2, 2, 3),
                dtype=np.uint8,
            )
        }
    )

    action = selector.select_action(
        task=task,
        observation=observation,
        step_index=0,
    )

    assert action.action_type == (
        retrieved_step.action_type
    )

    assert action.values == (
        retrieved_step.action_values
    )

def test_replay_policy_uses_fallback_when_memory_not_retrieved():
    fallback_action = Action(
        action_type="eef_delta_pose",
        values={
            "dx": 0.456,
        },
    )

    fallback_policy = FixedActionPolicy(
        action=fallback_action,
    )

    policy = RetrievedSegmentReplayPolicy(
        fallback_policy=fallback_policy,
    )

    controller = NeverRetrieveController()

    retriever = MagicMock()
    segment_loader = MagicMock()

    selector = RetrievalAugmentedActionSelector(
        controller=controller,
        retriever=retriever,
        segment_loader=segment_loader,
        policy=policy,
        k=1,
        event_recorder=MagicMock(),
    )

    task = TaskSpec(
        task_id="test_task",
        instruction="Place the mug in the drawer.",
        metadata=None,
    )

    observation = Observation(
        rgb={
            "agentview": np.zeros(
                (2, 2, 3),
                dtype=np.uint8,
            )
        }
    )

    action = selector.select_action(
        task=task,
        observation=observation,
        step_index=0,
    )

    assert action.action_type == fallback_action.action_type
    assert action.values == fallback_action.values

    retriever.retrieve.assert_not_called()
    segment_loader.load.assert_not_called()

def test_periodic_retrieval_changes_policy_input_at_expected_steps():
    controller = PeriodicRetrieveController(
        interval=2,
    )

    retriever = MagicMock()
    segment_loader = MagicMock()

    expected_action = Action(
        action_type="eef_delta_pose",
        values={
            "dx": 0.456,
        },
    )

    policy = RecordingPolicy(
        action=expected_action,
    )

    retrieved_segment_reference = MagicMock(
        spec=TrajectorySegment
    )

    retrieval_result = MagicMock()
    retrieval_result.item.segment = (
        retrieved_segment_reference
    )
    retrieved_segment_reference.segment_id = "segment-1"
    retrieval_result.score = 0.9

    loaded_segment = MagicMock(
        spec=SegmentData
    )

    retriever.retrieve.return_value = [
        retrieval_result
    ]

    segment_loader.load.return_value = (
        loaded_segment
    )

    selector = RetrievalAugmentedActionSelector(
        controller=controller,
        retriever=retriever,
        segment_loader=segment_loader,
        policy=policy,
        k=1,
        event_recorder=MagicMock(),
    )

    task = TaskSpec(
        task_id="test_task",
        instruction="Place the mug in the drawer.",
        metadata=None,
    )

    observation = Observation(
        rgb={
            "agentview": np.zeros(
                (2, 2, 3),
                dtype=np.uint8,
            )
        }
    )

    selector.select_action(
        task=task,
        observation=observation,
        step_index=0,
    )

    selector.select_action(
        task=task,
        observation=observation,
        step_index=1,
    )

    selector.select_action(
        task=task,
        observation=observation,
        step_index=2,
    )

    selector.select_action(
        task=task,
        observation=observation,
        step_index=3,
    )

    assert retriever.retrieve.call_count == 2

    assert segment_loader.load.call_count == 2

    assert policy.calls[0] == [
        loaded_segment
    ]

    assert policy.calls[1] == []

    assert policy.calls[2] == [
        loaded_segment
    ]

    assert policy.calls[3] == []


def test_replay_policy_rejects_empty_segment():
    fallback_action = Action(
        action_type="eef_delta_pose",
        values={"dx": 0.0},
    )

    fallback_policy = FixedActionPolicy(
        action=fallback_action,
    )

    replay_policy = RetrievedSegmentReplayPolicy(
        fallback_policy=fallback_policy,
    )

    empty_segment = MagicMock(spec=SegmentData)
    empty_segment.steps = []

    task = TaskSpec(
        task_id="test_task",
        instruction="Place the mug in the drawer.",
        metadata=None,
    )

    observation = Observation(
        rgb={
            "agentview": np.zeros(
                (2, 2, 3),
                dtype=np.uint8,
            )
        }
    )

    with pytest.raises(ValueError):
        replay_policy.act(
            task=task,
            observation=observation,
            retrieved_segments=[empty_segment],
        )
    
