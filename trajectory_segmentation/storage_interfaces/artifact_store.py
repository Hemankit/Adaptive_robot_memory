from abc import ABC, abstractmethod

import numpy as np


class ArtifactStore(ABC):

    @abstractmethod
    def save_array(
        self,
        key: str,
        array: np.ndarray,
    ) -> str:
        ...

    @abstractmethod
    def load_array(
        self,
        key: str,
    ) -> np.ndarray:
        ...