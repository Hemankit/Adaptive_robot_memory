from pydantic import BaseModel, Field
from data.rollout import RolloutConfig
from data.segment import SegmentationConfig
from data.embedding import EmbeddingConfig
from data.retrieved_result import RetrievalConfig, RetrievalControllerConfig
from data.policy import PolicyConfig
from data.store import StorageConfig
from data.item import MemoryConfig

class PipelineConfig(BaseModel):
    rollout: RolloutConfig = Field(
        ...,
        description="Configuration controlling rollout execution.",
    )

    segmentation: SegmentationConfig = Field(
        ...,
        description="Configuration controlling trajectory segmentation.",
    )

    embedding: EmbeddingConfig = Field(
        ...,
        description="Configuration controlling segment and query embeddings.",
    )

    retrieval: RetrievalConfig = Field(
        ...,
        description="Configuration controlling memory retrieval.",
    )

    controller: RetrievalControllerConfig = Field(
        ...,
        description="Configuration controlling when retrieval is invoked.",
    )

    policy: PolicyConfig = Field(
        ...,
        description="Configuration controlling action selection.",
    )

    storage: StorageConfig = Field(
        ...,
        description="Configuration controlling local persistence locations.",
    )

    memory: MemoryConfig = Field(
        ...,
        description="Configuration controlling episodic memory construction.",
    )