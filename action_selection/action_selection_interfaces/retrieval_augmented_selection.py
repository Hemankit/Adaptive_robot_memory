from time import perf_counter

from data.event import RetrievalEvent
from data.query import RetrievalQuery
from policy.action_policy import ActionPolicy
from recording.event_recorder import RetrievalEventRecorder
from retrieval.retrieval_interfaces.retrieval_controller import RetrievalController
from retrieval.retrieval_interfaces.retriever import Retriever
from trajectory_segmentation.segment_loader import SegmentLoader

from data.task import TaskSpec
from data.obs import Observation
from data.action import Action


class RetrievalAugmentedActionSelector:

    def __init__(
        self,
        controller: RetrievalController,
        retriever: Retriever,
        segment_loader: SegmentLoader,
        policy: ActionPolicy,
        k: int,
        event_recorder: RetrievalEventRecorder,
    ):
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        self._controller = controller
        self._retriever = retriever
        self._segment_loader = segment_loader
        self._policy = policy
        self._k = k
        self._event_recorder = event_recorder

    def select_action(
        self,
        task: TaskSpec,
        observation: Observation,
        step_index: int,
    ) -> Action:

        should_retrieve = self._controller.should_retrieve(
            task=task,
            observation=observation,
            step_index=step_index,
        )

        retrieved_segments = []

        if should_retrieve:
            query = RetrievalQuery(
                instruction=task.instruction,
            )

            start = perf_counter()

            results = self._retriever.retrieve(
                query=query,
                k=self._k,
            )

            elapsed_ms = (
                perf_counter() - start
            ) * 1000

            event = RetrievalEvent(
                step_index=step_index,
                retrieval_performed=True,
                query=query,
                retrieved_segment_ids=[
                    result.item.segment.segment_id
                    for result in results
                ],
                scores=[
                    result.score
                    for result in results
                ],
                retrieval_latency_ms=elapsed_ms,
            )

            self._event_recorder.record(event)

            for result in results:
                segment_data = self._segment_loader.load(
                    result.item.segment
                )

                retrieved_segments.append(
                    segment_data
                )

        else:
            event = RetrievalEvent(
                step_index=step_index,
                retrieval_performed=False,
            )

            self._event_recorder.record(event)

        return self._policy.act(
            task=task,
            observation=observation,
            retrieved_segments=retrieved_segments,
        )
