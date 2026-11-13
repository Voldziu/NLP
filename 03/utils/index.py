import os
from typing import Dict, List, Tuple
import numpy as np
import faiss


def build_faiss_index(
    doc_embeddings: np.ndarray,
    index_type: str = "FlatIP",
    save_path: str = None,
) -> faiss.Index:
    """
    Buduje indeks FAISS na podstawie osadzeń dokumentów.
    Dla modelu 'multi-qa-mpnet-base-dot-v1' właściwą miarą jest iloczyn skalarny (IndexFlatIP).
    """
    if save_path and os.path.exists(save_path):
        return faiss.read_index(save_path)

    dim = doc_embeddings.shape[1]
    embeddings_f32 = np.ascontiguousarray(doc_embeddings, dtype=np.float32)

    if index_type == "FlatIP":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "FlatL2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Nieobsługiwany typ indeksu: {index_type}")

    index.add(embeddings_f32)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(index, save_path)

    return index


def search_faiss(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    queries: List[Dict],
    docs: List[Dict],
    top_k: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Przeszukuje indeks FAISS i zwraca słownik rankingów w ustandaryzowanym formacie:
    {
        query_id: [(doc_id, score), ...],
        ...
    }
    """
    q_emb_f32 = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    scores, indices = index.search(q_emb_f32, top_k)

    rankings: Dict[str, List[Tuple[str, float]]] = {}
    for q_idx, q in enumerate(queries):
        q_id = q["query_id"]
        q_results = []
        for rank_idx in range(top_k):
            doc_pos = int(indices[q_idx, rank_idx])
            score = float(scores[q_idx, rank_idx])
            if doc_pos >= 0:
                doc_id = docs[doc_pos]["doc_id"]
                q_results.append((doc_id, score))
        rankings[q_id] = q_results

    return rankings


def search_exact_no_index(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    queries: List[Dict],
    docs: List[Dict],
    top_k: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Wyszukiwanie bezpośrednio na macierzach (iloczyn skalarny / util.semantic_search bez FAISS)
    w tym samym formacie wyjściowym {query_id: [(doc_id, score), ...]}.
    """
    # Macierz podobieństwa: (liczba_zapytań, liczba_dokumentów)
    sim_matrix = np.matmul(query_embeddings, doc_embeddings.T)

    rankings: Dict[str, List[Tuple[str, float]]] = {}
    for q_idx, q in enumerate(queries):
        q_id = q["query_id"]
        row_scores = sim_matrix[q_idx]
        # Pobranie indeksów top_k największych wartości
        top_indices = np.argpartition(-row_scores, kth=min(top_k, len(row_scores) - 1))[:top_k]
        top_indices = top_indices[np.argsort(-row_scores[top_indices])]

        rankings[q_id] = [
            (docs[int(doc_pos)]["doc_id"], float(row_scores[doc_pos]))
            for doc_pos in top_indices
        ]

    return rankings