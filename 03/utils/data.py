import os
import json
import random
from typing import Dict, List, Tuple
from datasets import load_dataset


def load_squad_data(
    split: str = "train",
    sample_queries: int = 500,
    seed: int = 42,
    cache_dir: str = "../artifacts",
) -> Tuple[List[Dict], List[Dict], Dict[str, List[str]]]:
    """
    Ładuje zbiór 'sentence-transformers/squad', usuwa duplikaty dokumentów (kolumna 'answer')
    oraz przygotowuje zbiór zapytań ('question') i mapowanie qrels.

    Zwraca:
        queries: Lista słowników [{'query_id': 'q_0', 'text': '...'}, ...]
        docs: Lista unikalnych dokumentów [{'doc_id': 'd_0', 'text': '...'}, ...]
        qrels: Słownik {query_id: [doc_id, ...]} wskazujący relewantne dokumenty dla zapytania.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"squad_{split}_q{sample_queries}_s{seed}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return cached["queries"], cached["docs"], cached["qrels"]

    # Pobranie zbioru sentence-transformers/squad (zawiera kolumny 'question' oraz 'answer')
    dataset = load_dataset("sentence-transformers/squad", split=split)

    doc_text_to_id: Dict[str, str] = {}
    docs: List[Dict] = []
    all_queries: List[Dict] = []
    qrels: Dict[str, List[str]] = {}

    for row in dataset:
        q_text = row["question"].strip()
        d_text = row["answer"].strip()

        # Deduplikacja dokumentów (ten sam akapit/odpowiedź występuje przy wielu pytaniach)
        if d_text not in doc_text_to_id:
            doc_id = f"d_{len(docs)}"
            doc_text_to_id[d_text] = doc_id
            docs.append({"doc_id": doc_id, "text": d_text})
        else:
            doc_id = doc_text_to_id[d_text]

        query_id = f"q_{len(all_queries)}"
        all_queries.append({"query_id": query_id, "text": q_text})
        qrels[query_id] = [doc_id]

    # Losowanie powtarzalnej próbki zapytań (ziarno = 42) dla szybkiej i spójnej ewaluacji
    if sample_queries is not None and sample_queries < len(all_queries):
        rng = random.Random(seed)
        sampled_queries = rng.sample(all_queries, sample_queries)
        sampled_qrels = {q["query_id"]: qrels[q["query_id"]] for q in sampled_queries}
    else:
        sampled_queries = all_queries
        sampled_qrels = qrels

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(
            {"queries": sampled_queries, "docs": docs, "qrels": sampled_qrels},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return sampled_queries, docs, sampled_qrels