import os
from typing import Dict, List, Tuple

import bm25s
import Stemmer

STEMMER_LANGUAGE = "english"
STOPWORDS = "en"


def _tokenize(texts: List[str], stemmer: Stemmer.Stemmer, return_ids: bool):
    return bm25s.tokenize(
        texts,
        stopwords=STOPWORDS,
        stemmer=stemmer,
        return_ids=return_ids,
        show_progress=False,
    )


def build_bm25(
    docs: List[Dict],
    cache_path: str = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> bm25s.BM25:
    """
    Buduje indeks BM25 (biblioteka `bm25s`) na tym samym zbiorze dokumentów co bi-enkoder.
    Tokenizacja: małe litery, angielskie stopwords, stemmer Snowball.
    Kolejność dokumentów w indeksie odpowiada kolejności na liście `docs`.
    """
    if cache_path and os.path.exists(cache_path):
        return bm25s.BM25.load(cache_path)

    stemmer = Stemmer.Stemmer(STEMMER_LANGUAGE)
    corpus_tokens = _tokenize([d["text"] for d in docs], stemmer, return_ids=True)

    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(corpus_tokens, show_progress=False)

    if cache_path:
        os.makedirs(cache_path, exist_ok=True)
        retriever.save(cache_path, show_progress=False)

    return retriever


def search_bm25(
    retriever: bm25s.BM25,
    queries: List[Dict],
    docs: List[Dict],
    top_k: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Wyszukuje top_k dokumentów BM25 dla każdego zapytania.
    Zwraca rankingi w formacie {query_id: [(doc_id, score), ...]}.

    Dokumenty z wynikiem 0 (brak wspólnego termu z zapytaniem) są pomijane:
    bm25s dopełnia nimi listę do top_k, ale nie są to trafienia, a w RRF
    dostałyby fałszywe rangi. Lista może więc być krótsza niż top_k.
    """
    stemmer = Stemmer.Stemmer(STEMMER_LANGUAGE)
    query_tokens = _tokenize([q["text"] for q in queries], stemmer, return_ids=False)

    k = min(top_k, len(docs))
    indices, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)

    rankings: Dict[str, List[Tuple[str, float]]] = {}
    for q_idx, q in enumerate(queries):
        rankings[q["query_id"]] = [
            (docs[int(doc_pos)]["doc_id"], float(score))
            for doc_pos, score in zip(indices[q_idx], scores[q_idx])
            if score > 0
        ]
    return rankings
