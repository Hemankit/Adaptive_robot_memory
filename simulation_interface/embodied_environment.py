from abc import ABC, abstractmethod

from data.action import Action
from data.obs import Observation
from data.task import TaskSpec
from data.transition import Transition


class EmbodiedEnvironment(ABC):

    @abstractmethod
    def reset(
        self,
        task: TaskSpec,
        seed: int | None = None,
    ) -> Observation:
        """Reset the environment for a task and return the initial observation."""
        pass

    @abstractmethod
    def step(
        self,
        action: Action,
    ) -> Transition:
        """Apply one canonical action and return the resulting transition."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""
        pass