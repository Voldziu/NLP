import threading
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from ..logger import get_logger
from .base import Doc, Retriever

logger = get_logger(__name__)

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "rag"


class FaissRetriever(Retriever):
    """Dense retriever: sentence-transformer embeddings in a flat FAISS index."""

    def __init__(
        self,
        docs: list[Doc],
        *,
        model_name: str = MODEL_NAME,
        cache_name: str = "squad_validation",
    ) -> None:
        """Embed the documents (or load cached embeddings) and build the index.

        Args:
            docs: Documents to index.
            model_name: Sentence-transformers model used for documents and queries.
            cache_name: Prefix of the embedding cache file in ``ARTIFACTS``.
        """
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        # LangChain .batch() calls retrieve from threads; encoder + FAISS crash concurrently.
        self._lock = threading.Lock()
        cache_path = ARTIFACTS / f"{cache_name}__{model_name.split('/')[-1]}.npy"

        embeddings = self._load_cache(cache_path, len(docs))
        if embeddings is None:
            logger.info("Encoding %d docs with %s", len(docs), model_name)
            embeddings = self._encode([d.text for d in docs], show_progress=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info("Saved embeddings to %s", cache_path)

        # Inner product: the model is trained for dot-product similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    @staticmethod
    def _load_cache(path: Path, n_docs: int) -> NDArray[np.float32] | None:
        if not path.exists():
            return None
        cached: NDArray[np.float32] = np.load(path)
        if cached.shape[0] != n_docs:
            logger.warning(
                "Cache %s has %d rows, expected %d; re-encoding",
                path,
                cached.shape[0],
                n_docs,
            )
            return None
        logger.info("Loaded %d cached embeddings from %s", n_docs, path)
        return cached

    def _encode(self, texts: list[str], *, show_progress: bool) -> NDArray[np.float32]:
        vectors = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def retrieve(self, query: str, k: int = 5) -> list[Doc]:
        """Return the top-k documents by inner-product similarity.

        Args:
            query: User question.
            k: Number of documents to return.

        Returns:
            Documents sorted from best to worst, with ``score`` filled in.
        """
        with self._lock:
            q = self._encode([query], show_progress=False)
            scores, indices = self.index.search(q, k)
        return [
            self.docs[i].model_copy(update={"score": float(s)})
            for s, i in zip(scores[0], indices[0], strict=True)
            if i != -1
        ]
