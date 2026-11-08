

from embeddings.embedding_interfaces.segment_embedder import SegmentEmbedder
from data.embedding import SegmentEmbedding
from data.segment_data import SegmentData

from sklearn.feature_extraction.text import HashingVectorizer


class HashingInstructionEmbedder(SegmentEmbedder):

    def __init__(
        self,
        dimension: int = 256,
    ):
        self._dimension = dimension if dimension > 0 else 256
        self._vectorizer = HashingVectorizer(
        n_features=dimension,
        alternate_sign=False,
        norm="l2",
)

    @property
    def embedding_id(self) -> str:
        return f"hashing-instruction-v1-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(
        self,
        segment: SegmentData,
    ) -> SegmentEmbedding:
        # Read the instruction.
        instruction = segment.segment.instruction
        # Reject an empty instruction
        if not instruction:
            raise ValueError("Instruction cannot be empty.")
        # Transform it with the hashing vectorizer.
        # 
        vector = self._vectorizer.transform([instruction])
        # Convert the result into a one-dimensional dense vector
        dense_1d_vector = vector.toarray().ravel()
        # Convert it to ordinary Python floats
        float_vector = [float(x) for x in dense_1d_vector]

        # Verify its length equals self.dimension
        if len(float_vector) != self.dimension:
            raise ValueError(f"Embedding length {len(float_vector)} does not match expected dimension {self.dimension}.")

        return SegmentEmbedding(
            segment_id=segment.segment.segment_id,
            vector=float_vector,
            embedding_id=self.embedding_id,
            dimension=self.dimension
        )

    # NOTE: The hashingvectorizer will make it so that all the fixed-window segments coming from same instruction will recieve same embedding which is a limitation of this approach.
    # Therefore HashingInstructionEmbedder exists to validate embedding, indexing, storage, and retrieval infrastructure. It is not intended to demonstrate meaningful subtrajectory relevance.