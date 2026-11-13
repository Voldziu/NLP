import os
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "multi-qa-mpnet-base-dot-v1"


def get_model(model_name: str = DEFAULT_MODEL, device: str = None) -> SentenceTransformer:
    """Ładuje pre-trenowany model SentenceTransformer."""
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    cache_path: str = None,
    batch_size: int = 64,
    normalize_embeddings: bool = False,
) -> np.ndarray:
    """
    Koduje listę tekstów do macierzy osadzeń numpy (N, D) i zapisuje w cache_path (jeśli podano).
    Uwaga: model 'multi-qa-mpnet-base-dot-v1' był trenowany pod iloczyn skalarny (dot-product),
    więc domyślnie normalize_embeddings=False.
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)

    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    embeddings = embeddings.astype(np.float32)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)

    return embeddings


def encode_corpus_and_queries(
    queries: List[Dict],
    docs: List[Dict],
    model_name: str = DEFAULT_MODEL,
    artifacts_dir: str = "../artifacts",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zwraca krotkę (query_embeddings, doc_embeddings) z automatycznym buforowaniem w artifacts/.
    """
    safe_name = model_name.replace("/", "_")
    q_cache = os.path.join(artifacts_dir, f"queries_{len(queries)}_{safe_name}.npy")
    d_cache = os.path.join(artifacts_dir, f"docs_{len(docs)}_{safe_name}.npy")

    q_texts = [q["text"] for q in queries]
    d_texts = [d["text"] for d in docs]

    q_emb = encode_texts(q_texts, model_name=model_name, cache_path=q_cache)
    d_emb = encode_texts(d_texts, model_name=model_name, cache_path=d_cache)

    return q_emb, d_emb