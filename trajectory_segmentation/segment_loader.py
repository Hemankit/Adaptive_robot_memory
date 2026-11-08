

from data.segment import TrajectorySegment
from data.segment_data import SegmentData
from trajectory_segmentation.storage_interfaces.trajectory_store import TrajectoryStore


class SegmentLoader:

    def __init__(
        self,
        trajectory_store: TrajectoryStore,
    ):
        self._trajectory_store = trajectory_store

    def load(
        self,
        segment: TrajectorySegment,
    ) -> SegmentData:
        steps = self._trajectory_store.load_steps(
    segment.episode_id
)
        observations = self._trajectory_store.load_observations(
    segment.episode_id
)       # verify that loaded records belong to segment.episode_id
        if any(step.episode_id != segment.episode_id for step in steps):
            raise ValueError("Loaded steps do not belong to the given segment's episode.")
        if any(obs.episode_id != segment.episode_id for obs in observations):
            raise ValueError("Loaded observations do not belong to the given segment's episode.")
        if segment.start_step >= 0 and segment.end_step >= segment.start_step:
          segment_steps = [step for step in steps if segment.start_step <= step.step_index <= segment.end_step] 
          # observations span from before the first action to after the last action
          segment_observations = [obs for obs in observations if segment.start_step <= obs.observation_index <= segment.end_step + 1]
        else:
          raise ValueError("Invalid segment start or end step.")

        # verify that all requested steps/observations actually exist
        if not segment_steps:
            raise ValueError("No steps found for the given segment.")
        if not segment_observations:
            raise ValueError("No observations found for the given segment.")

        if len(segment_observations) == len(segment_steps) + 1:
            return SegmentData(
                segment=segment,
                steps=segment_steps,
                observations=segment_observations,
            )
        else:
            raise ValueError("Mismatch between number of segment observations and steps.")

