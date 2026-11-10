from retrieval.retrieval_interfaces.retrieval_controller import RetrievalController
from data.task import TaskSpec
from data.obs import Observation

class PeriodicRetrieveController(RetrievalController):

    def __init__(
        self,
        interval: int,
    ):
        self.interval = interval if interval > 0 else 1

    def should_retrieve(
        self,
        task: TaskSpec,
        observation: Observation,
        step_index: int,
    ) -> bool:
        return step_index % self.interval == 0