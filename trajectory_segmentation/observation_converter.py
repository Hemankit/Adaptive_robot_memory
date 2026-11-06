from typing import Any
import numpy as np

from data.obs import Observation
from trajectory_segmentation.storage_interfaces.artifact_store import ArtifactStore
from data.record import ObservationRecord

class ObservationConverter:

    def __init__(
        self,
        artifact_store: ArtifactStore,
    ):
        self._artifact_store = artifact_store

    def convert(
        self,
        observation: Observation,
        episode_id: str,
        observation_index: int,
    ) -> ObservationRecord:
        # For every RGB camera, generate a deterministic key
        rgb_artifacts = {}
        for camera_name, rgb_array in observation.rgb.items():
            key = f"episodes/{episode_id}/observations/{observation_index}/rgb/{camera_name}"
            self._artifact_store.save_array(key, rgb_array)
            rgb_artifacts[camera_name] = key
        depth_artifacts = None
        if observation.depth is not None:
            depth_artifacts = {}
            for camera_name, depth_array in observation.depth.items():
                key = f"episodes/{episode_id}/observations/{observation_index}/depth/{camera_name}"
                self._artifact_store.save_array(key, depth_array)
                depth_artifacts[camera_name] = key
        return ObservationRecord(
            episode_id=episode_id,
            observation_index=observation_index,
            rgb_artifacts=rgb_artifacts,
            depth_artifacts=depth_artifacts,
            robot_state=self._serialize_state(observation.robot_state),
            object_state=self._serialize_state(observation.object_state),
            timestamp=observation.timestamp,
        )
    def _serialize_state(
        self,
        value: Any,
    ) -> Any:
        """None → None

np.ndarray → .tolist()

np scalar → .item()

dict → recursively convert values

list / tuple → recursively convert elements

ordinary Python scalar/string/bool → unchanged"""
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: self._serialize_state(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._serialize_state(v) for v in value)
        return value