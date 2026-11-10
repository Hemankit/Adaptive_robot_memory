from abc import ABC, abstractmethod

from data.obs import Observation
from data.task import TaskSpec

from data.task import TaskSpec
from data.obs import Observation


class RetrievalController(ABC):

    @abstractmethod
    def should_retrieve(
        self,
        task: TaskSpec,
        observation: Observation,
        step_index: int,
    ) -> bool:
          ...