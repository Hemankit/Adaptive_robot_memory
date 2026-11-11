
from action_selection.action_selection_interfaces.retrieval_augmented_selection import RetrievalAugmentedActionSelector
from data.experiment import ExperimentSpec, ExperimentResult
from evaluation.eps_metrics_calc import EpisodeMetricsCalculator
from evaluation.experiment_metrics_calc import ExperimentMetricsCalculator
from recording.event_recorder import RetrievalEventRecorder
from sim_evaluation.rollout_runner import RolloutRunner
from storage.storage_interfaces.event_store import RetrievalEventStore
from storage.storage_interfaces.retrieval_event_record import to_retrieval_event_record
from trajectory_segmentation.storage_interfaces.trajectory_store import TrajectoryStore
from trajectory_segmentation.trajectory_writer import TrajectoryWriter


class ExperimentRunner:

    def __init__(
        self,
        rollout_runner: RolloutRunner,
        action_selector: RetrievalAugmentedActionSelector,
        trajectory_writer: TrajectoryWriter,
        trajectory_store: TrajectoryStore,
        event_recorder: RetrievalEventRecorder,
        event_store: RetrievalEventStore,
        episode_metrics_calculator: EpisodeMetricsCalculator,
        experiment_metrics_calculator: ExperimentMetricsCalculator,
    ):
        self._rollout_runner = rollout_runner
        self._action_selector = action_selector
        self._trajectory_writer = trajectory_writer
        self._trajectory_store = trajectory_store
        self._event_recorder = event_recorder
        self._event_store = event_store
        self._episode_metrics_calculator = episode_metrics_calculator
        self._experiment_metrics_calculator = experiment_metrics_calculator

    def run(
        self,
        experiment: ExperimentSpec,
    ) -> ExperimentResult:
        if experiment.experiment_id is None:
            raise ValueError("Experiment must have a valid experiment_id.")
        if experiment.episodes == []:
            raise ValueError("Experiment must have at least one episode.")
        # verify episode_ids are unique
        episode_ids = [episode.episode_id for episode in experiment.episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("Episode IDs must be unique within an experiment.")

        # prepare episode metric collection
        episode_metrics = []
        for episode_spec in experiment.episodes:
            self._event_recorder.clear()
            rollout = self._rollout_runner.run(
            task=episode_spec.task,
            action_fn=self._action_selector.select_action,
            seed=episode_spec.seed,
          )
            self._trajectory_writer.write(
            rollout=rollout,
            episode_id=episode_spec.episode_id,
          )
            events = self._event_recorder.get_events()
            event_records = [
            to_retrieval_event_record(
            event=event,
            episode_id=episode_spec.episode_id,
            )
            for event in events
            ]
            self._event_store.save(
            episode_id=episode_spec.episode_id,
            events=event_records,
            )
            episode_record = (
            self._trajectory_store.load_episode(
            episode_spec.episode_id
            )
            )
            metrics = (
            self._episode_metrics_calculator.calculate(
            episode=episode_record,
            retrieval_events=event_records,
    )
)
            episode_metrics.append(metrics)

        experiment_metrics = (
            self._experiment_metrics_calculator.calculate(
                episodes=episode_metrics,
            )
        )
        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            episode_metrics=episode_metrics,
            metrics=experiment_metrics,
        )
