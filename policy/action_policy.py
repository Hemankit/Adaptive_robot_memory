from abc import ABC, abstractmethod

from data.task import TaskSpec
from data.task import TaskSpec
from data.obs import Observation
from data.segment import SegmentData
from data.action import Action

class ActionPolicy(ABC):

    @abstractmethod
    def act(
        self,
        task: TaskSpec,
        observation: Observation,
        retrieved_segments: list[SegmentData],
    ) -> Action:
          ...