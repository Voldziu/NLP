from collections import defaultdict
from typing import Dict, List, Tuple

Rankings = Dict[str, List[Tuple[str, float]]]


def reciprocal_rank_fusion(
    rankings_list: List[Rankings],
    k: int = 60,
    top_k: int = None,
) -> Rankings:
    """
    Łączy rankingi kilku systemów metodą Reciprocal Rank Fusion (Cormack i in., 2009):

        RRF(d) = sum_i 1 / (k + rank_i(d))

    gdzie rank_i(d) to pozycja dokumentu d na liście i-tego systemu, liczona od 1.
    Dokument nieobecny na liście systemu i nie dostaje od niego nic.
    Wartości score z wejściowych rankingów są ignorowane, liczy się tylko kolejność.

    Zwraca rankingi w formacie {query_id: [(doc_id, rrf_score), ...]},
    posortowane malejąco po wyniku RRF (remisy według doc_id), przycięte do top_k.
    """
    query_ids = set()
    for rankings in rankings_list:
        query_ids.update(rankings.keys())

    fused: Rankings = {}
    for q_id in query_ids:
        scores: Dict[str, float] = defaultdict(float)
        for rankings in rankings_list:
            for rank, (doc_id, _) in enumerate(rankings.get(q_id, []), start=1):
                scores[doc_id] += 1.0 / (k + rank)

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        fused[q_id] = ordered[:top_k] if top_k is not None else ordered

    return fused
