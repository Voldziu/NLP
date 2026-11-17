from typing import Dict, List, Tuple

from sentence_transformers import CrossEncoder

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"

Rankings = Dict[str, List[Tuple[str, float]]]

_MODEL_CACHE: Dict[str, CrossEncoder] = {}


def get_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER, device: str = None) -> CrossEncoder:
    """Ładuje (raz na sesję) pre-trenowany cross-enkoder."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = CrossEncoder(model_name, device=device)
    return _MODEL_CACHE[model_name]


def rerank(
    rankings: Rankings,
    queries: List[Dict],
    docs: List[Dict],
    model_name: str = DEFAULT_CROSS_ENCODER,
    top_n: int = 50,
    top_k: int = 5,
    batch_size: int = 128,
) -> Rankings:
    """
    Re-ranking cross-enkoderem: dla każdego zapytania bierze top_n kandydatów z `rankings`,
    ocenia pary (zapytanie, dokument) i zwraca top_k dokumentów według wyniku cross-enkodera.

    Wszystkie pary są oceniane jednym wywołaniem `predict` (batchowanie na GPU).
    Zwraca rankingi w formacie {query_id: [(doc_id, ce_score), ...]}.
    """
    model = get_cross_encoder(model_name)
    query_text = {q["query_id"]: q["text"] for q in queries}
    doc_text = {d["doc_id"]: d["text"] for d in docs}

    pair_owner: List[Tuple[str, str]] = []
    pairs: List[Tuple[str, str]] = []
    for q_id, ranked in rankings.items():
        for doc_id, _ in ranked[:top_n]:
            pair_owner.append((q_id, doc_id))
            pairs.append((query_text[q_id], doc_text[doc_id]))

    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False) if pairs else []

    per_query: Rankings = {q_id: [] for q_id in rankings}
    for (q_id, doc_id), score in zip(pair_owner, scores):
        per_query[q_id].append((doc_id, float(score)))

    return {
        q_id: sorted(candidates, key=lambda item: -item[1])[:top_k]
        for q_id, candidates in per_query.items()
    }
