from abc import ABC, abstractmethod

from data.episode import EpisodeRecord
from data.obs import ObservationRecord
from data.step import StepRecord


class TrajectoryStore(ABC):

    @abstractmethod
    def save_episode(
        self,
        episode: EpisodeRecord,
    ) -> None:
        ...

    @abstractmethod
    def save_steps(
        self,
        steps: list[StepRecord],
    ) -> None:
        ...

    @abstractmethod
    def save_observations(
        self,
        observations: list[ObservationRecord],
    ) -> None:
        ...

    @abstractmethod
    def load_episode(
        self,
        episode_id: str,
    ) -> EpisodeRecord:
        ...

    @abstractmethod
    def load_steps(
        self,
        episode_id: str,
    ) -> list[StepRecord]:
        ...

    @abstractmethod
    def load_observations(
        self,
        episode_id: str,
    ) -> list[ObservationRecord]:
        ...