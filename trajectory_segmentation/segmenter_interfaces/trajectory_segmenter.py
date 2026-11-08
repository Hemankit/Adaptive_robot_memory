from abc import ABC, abstractmethod

from data.episode import EpisodeRecord
from data.segment import TrajectorySegment
from data.step import StepRecord


class TrajectorySegmenter(ABC):

    @abstractmethod
    def segment(
        self,
        episode: EpisodeRecord,
        steps: list[StepRecord],
    ) -> list[TrajectorySegment]:
        ...