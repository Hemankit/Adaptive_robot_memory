
from unittest import result

from policy.action_policy import ActionPolicy
from data.query import RetrievalQuery
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
    ):
        self._controller = controller
        self._retriever = retriever
        self._segment_loader = segment_loader
        self._policy = policy
        self._k = k if k > 0 else 1

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
              results = self._retriever.retrieve(
                  query=query,
                  k=self._k,
              )
              for result in results:
                  segment_data = self._segment_loader.load(result.item.segment)
                  retrieved_segments.append(segment_data)
          return self._policy.act(
          task=task,
          observation=observation,
          retrieved_segments=retrieved_segments,
)
