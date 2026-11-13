import math
from typing import Dict, List, Tuple
import numpy as np


def recall_at_k(
    rankings: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, List[str]],
    k: int = 5,
) -> float:
    """Oblicza średni Recall@k po wszystkich zapytaniach."""
    recalls = []
    for q_id, rel_docs in qrels.items():
        if not rel_docs:
            continue
        retrieved = [doc_id for doc_id, _ in rankings.get(q_id, [])[:k]]
        hits = len(set(retrieved) & set(rel_docs))
        recalls.append(hits / len(rel_docs))
    return float(np.mean(recalls)) if recalls else 0.0


def mrr_at_k(
    rankings: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, List[str]],
    k: int = 5,
) -> float:
    """Oblicza Mean Reciprocal Rank (MRR@k)."""
    rr_scores = []
    for q_id, rel_docs in qrels.items():
        rel_set = set(rel_docs)
        retrieved = [doc_id for doc_id, _ in rankings.get(q_id, [])[:k]]
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in rel_set:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def ndcg_at_k(
    rankings: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, List[str]],
    k: int = 5,
) -> float:
    """Oblicza znormalizowany DCG (nDCG@k) dla binarnej trafności."""
    ndcg_scores = []
    for q_id, rel_docs in qrels.items():
        rel_set = set(rel_docs)
        if not rel_set:
            continue
        retrieved = [doc_id for doc_id, _ in rankings.get(q_id, [])[:k]]

        dcg = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in rel_set:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def evaluate_rankings(
    rankings: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, List[str]],
    k: int = 5,
) -> Dict[str, float]:
    """
    Zwraca kompletny słownik metryk dla zadanego k (domyślnie top_k = 5).
    Gotowa funkcja do porównania wyników w pkt 3, 4 oraz 5.
    """
    return {
        f"Recall@{k}": recall_at_k(rankings, qrels, k=k),
        f"MRR@{k}": mrr_at_k(rankings, qrels, k=k),
        f"nDCG@{k}": ndcg_at_k(rankings, qrels, k=k),
    }