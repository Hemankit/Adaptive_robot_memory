from pathlib import Path

from data.obs import ObservationRecord
from pydantic.json import pydantic_encoder
from pydantic import TypeAdapter
from data.step import StepRecord
from data.episode import EpisodeRecord
from trajectory_segmentation.storage_interfaces.trajectory_store import TrajectoryStore
from pathlib import Path
import json
class LocalTrajectoryStore(TrajectoryStore):

    def __init__(
        self,
        root_dir: str | Path,
    ):
        self._root_dir = Path(root_dir)

    def save_episode(
        self,
        episode: EpisodeRecord,
    ) -> None:
        episode_dir = self._root_dir / "episodes" / episode.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        episode_file = episode_dir / "episode.json"

        with open(episode_file, "w", encoding="utf-8") as f:
         f.write(episode.model_dump_json(indent=2))

    def save_steps(
    self,
    steps: list[StepRecord],
) -> None:
    # 1. Validate non-empty list
      if not steps:
        raise ValueError("Steps list is empty")

    # 2. Validate episode IDs
      episode_id = steps[0].episode_id

      if any(step.episode_id != episode_id for step in steps):
        raise ValueError("All steps must belong to the same episode ID")

    # 3. Validate sequential step indices
      for i, step in enumerate(steps):
        if step.step_index != i:
            raise ValueError(
                f"Expected step index {i}, "
                f"got {step.step_index}"
            )

    # 4. Create episode directory
      episode_dir = self._root_dir / "episodes" / episode_id
      episode_dir.mkdir(parents=True, exist_ok=True)

    # 5. Serialize and save steps
      steps_file = episode_dir / "steps.json"

      adapter = TypeAdapter(list[StepRecord])

      with open(steps_file, "wb") as f:
        f.write(adapter.dump_json(steps, indent=2))

    def save_observations(
    self,
    observations: list[ObservationRecord],
) -> None:
    # 1. Validate non-empty list
      if not observations:
        raise ValueError("Observations list is empty")

    # 2. Validate same episode ID
      episode_id = observations[0].episode_id

      if any(obs.episode_id != episode_id for obs in observations):
        raise ValueError("All observations must belong to the same episode ID")

    # 3. Validate observation indices
      for i, obs in enumerate(observations):
        if obs.observation_index != i:
            raise ValueError(
                f"Expected observation index {i}, "
                f"got {obs.observation_index}"
            )

    # 4. Create episode directory
      episode_dir = self._root_dir / "episodes" / episode_id
      episode_dir.mkdir(parents=True, exist_ok=True)

    # 5. Serialize and save observations
      observations_file = episode_dir / "observations.json"

      adapter = TypeAdapter(list[ObservationRecord])

      with open(observations_file, "wb") as f:
        f.write(adapter.dump_json(observations, indent=2))

    def load_episode(
    self,
    episode_id: str,
) -> EpisodeRecord:
      try:
        episode_file = self._root_dir / "episodes" / episode_id / "episode.json"

        with open(episode_file, "r", encoding="utf-8") as f:
            return EpisodeRecord.model_validate_json(f.read())

      except FileNotFoundError:
        raise ValueError(
            f"No episode found for episode ID {episode_id}"
        )

    def load_steps(
        self,
        episode_id: str,
    ) -> list[StepRecord]:
          try:
            steps_file = self._root_dir / "episodes" / episode_id / "steps.json"
            adapter = TypeAdapter(list[StepRecord])
            with open(steps_file, "rb") as f:
                return adapter.validate_json(f.read())
          except FileNotFoundError:
            raise ValueError(f"No steps found for episode ID {episode_id}")

    def load_observations(
        self,
        episode_id: str,
    ) -> list[ObservationRecord]:
        try:
            observations_file = self._root_dir / "episodes" / episode_id / "observations.json"
            adapter = TypeAdapter(list[ObservationRecord])
            with open(observations_file, "rb") as f:
                return adapter.validate_json(f.read())
        except FileNotFoundError:
            raise ValueError(f"No observations found for episode ID {episode_id}")