from pathlib import Path

import numpy as np
from trajectory_segmentation.storage_interfaces.artifact_store import ArtifactStore


class LocalArtifactStore(ArtifactStore):

    def __init__(
        self,
        root_dir: str | Path,
    ):
        self._root_dir = Path(root_dir)

    def save_array(
        self,
        key: str,
        array: np.ndarray,
    ) -> str:
        file_path = self._root_dir / f"{key}.npy"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, array)
        return str(file_path)

    def load_array(
        self,
        key: str,
    ) -> np.ndarray:
        try:
            file_path = self._root_dir / f"{key}.npy"
            return np.load(file_path)
        except FileNotFoundError:
            raise KeyError(f"Artifact with key '{key}' not found.")