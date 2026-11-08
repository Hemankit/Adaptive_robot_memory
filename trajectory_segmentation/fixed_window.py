
from trajectory_segmentation.segmenter_interfaces.trajectory_segmenter import TrajectorySegmenter

from data.episode import EpisodeRecord
from data.step import StepRecord
from data.segment import TrajectorySegment
class FixedWindowSegmenter(TrajectorySegmenter):

    def __init__(
        self,
        window_size: int,
        stride: int,
    ):
        self.window_size = max(1, window_size)
        self.stride = max(1, min(stride, self.window_size))


    def segment(
        self,
        episode: EpisodeRecord,
        steps: list[StepRecord],
    ) -> list[TrajectorySegment]:
        # Verify all StepRecords belong to episode.episode_id.
        for step in steps:
            if step.episode_id != episode.episode_id:
                raise ValueError(f"StepRecord with step_index '{step.step_index}' does not belong to episode with episode_id '{episode.episode_id}'.")
        # Verify step indices are ordered and contiguous.
        for i in range(1, len(steps)):
            if steps[i].step_index != steps[i-1].step_index + 1:
                raise ValueError(f"StepRecord with step_index '{steps[i].step_index}' is not contiguous with previous step.")
        # Produce contiguous windows of at most window_size steps.
        # Move forward by stride and Generate a unique segment_id for every produced segment.
        # Keep the final shorter window.
        segments = []
        for segment_id, start in enumerate(
        range(0, len(steps), self.stride)
    ):
             end = min(start + self.window_size, len(steps))
             segments.append(TrajectorySegment(
                segment_id=f"{episode.episode_id}_segment_{segment_id}",
                episode_id=episode.episode_id,
                start_step=start,
                end_step=end - 1,
                task_id=episode.task_id,
                instruction=episode.instruction,
            )
        )
        return segments
