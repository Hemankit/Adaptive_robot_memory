from retrieval.retrieval_interfaces.retrieval_controller import RetrievalController
from data.task import TaskSpec
from data.obs import Observation

class NeverRetrieveController(RetrievalController):

    def should_retrieve(
        self,
        task: TaskSpec,
        observation: Observation,
        step_index: int,
    ) -> bool:
        return False