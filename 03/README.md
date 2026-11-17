# Zadanie 3: Wyszukiwanie semantyczne

Instrukcja: [Zadanie_3_zmodyfikowane.md](Zadanie_3_zmodyfikowane.md)

## Podział zadań

| Osoba | Zakres | Folder | Pkt |
|---|---|---|---|
| **Piotrek** | I.A–C (SentenceTransformers, artykuł o polyencoderach, metryki oceny) + notatki. Pkt 1: opis potoku retriever + re-ranker. Pkt 2: tutorial HF (chapter 5/6), analiza wyszukiwań i błędów. Pkt 3: bi-enkoder na `sentence-transformers/squad` (`multi-qa-mpnet-base-dot-v1`), deduplikacja dokumentów, indeks FAISS (top_k = 5), metryka oceny, porównanie bez indeksu vs FAISS. | `piotrek/` | 4 |
| **Dominik** | Pkt 4: BM25 (`bm25s` / `rank_bm25`), Reciprocal Rank Fusion (k = 60), porównanie BM25 / bi-enkoder / hybryda, analiza zapytań (leksykalne vs gęste), uzasadnienie użycia rang w RRF. Pkt 5: re-ranking cross-enkoderem na wyniku hybrydy, ocena i porównanie ze wszystkimi wcześniejszymi wynikami. | `dominik/` | 3 |
| **Mikołaj** (Część II) | II.A (artykuł o RAG). Pkt 1: API CLARIN (`/api/v1/oapi/models`, `/api/v1/oapi/chat/completions`). Pkt 2: wybór i indeksowanie zbioru (np. IMDB lub indeks FAISS Wikipedii). Pkt 3: moduł generacji RAG: prompt, własne zapytania, analiza i podsumowanie wyników. | `mikolaj/` | 3 |

## Zależności

- **Dominik zależy od Piotrka.** Pkt 4 i 5 działają na tym samym zbiorze dokumentów, tych samych zapytaniach i tych samych metrykach co pkt 3. Dominik **musi używać** kodu Piotrka z `utils/` (dane, osadzenia, indeks FAISS, metryki). Nie pisze własnych wersji, żeby wyniki były porównywalne.
- Piotrek wystawia w `utils/` wszystko, czego potrzebuje Dominik, i zapisuje to w tej tabeli, gdy moduł jest gotowy.
- Mikołaj (Część II) może zacząć od razu. Pkt 2.b instrukcji pozwala użyć retrievera hybrydowego z Części I, ale nie jest to konieczne: na start wystarczy własny indeks FAISS.

## Zasady współpracy

- Dane: `data/` (w `.gitignore`, każdy generuje/pobiera lokalnie).
- Wspólny kod trafia do `utils/` (małe, niezależne moduły). Kod własny zostaje w swoim folderze.
- Wspólny wariant ewaluacji: `top_k = 5`, ten sam zbiór dokumentów po deduplikacji i ta sama próbka zapytań (ziarno `42`).
- Artefakty (indeksy, osadzenia) idą do `artifacts/`.

## Środowisko

```bash
uv sync   # z korzenia repo
```

Notebooki leżą w podfolderach `03/`, więc na początku:

```python
import sys; sys.path.insert(0, "..")
from utils import (
    load_squad_data,
    encode_corpus_and_queries,
    build_faiss_index,
    search_faiss,
    evaluate_rankings,
)
```

## `utils/` (pisze Piotrek)

Wszystkie moduły są gotowe i przetestowane. Poniżej opis interfejsu dla Dominika:

| Moduł | Co robi | Status |
|---|---|---|
| `data.py` | `load_squad_data(split="train", sample_queries=500, seed=42, cache_dir="../artifacts")` -> zwraca `(queries, docs, qrels)`, gdzie `queries = [{'query_id': 'q_0', 'text': '...'}]`, `docs = [{'doc_id': 'd_0', 'text': '...'}]` (po deduplikacji), `qrels = {'q_0': ['d_0']}`. | gotowe |
| `encode.py` | `encode_corpus_and_queries(queries, docs, model_name="multi-qa-mpnet-base-dot-v1", artifacts_dir="../artifacts")` -> zwraca `(query_embeddings, doc_embeddings)` jako `np.ndarray` (`float32`), z cache w `artifacts/`. | gotowe |
| `index.py` | `build_faiss_index(doc_embeddings, index_type="FlatIP", save_path="../artifacts/squad_faiss_flatip.index")` oraz `search_faiss(index, query_embeddings, queries, docs, top_k=5)` -> zwraca słownik rankingów `{query_id: [(doc_id, score), ...]}`. | gotowe |
| `metrics.py` | `evaluate_rankings(rankings, qrels, k=5)` -> przyjmuje słownik rankingów `{query_id: [(doc_id, score), ...]}` oraz `qrels`, zwraca słownik `{'Recall@5': ..., 'MRR@5': ..., 'nDCG@5': ...}`. | gotowe |

## `utils/` (pisze Dominik)

Ten sam format rankingów `{query_id: [(doc_id, score), ...]}`, więc wszystko wchodzi wprost do `evaluate_rankings`.

| Moduł | Co robi | Status |
|---|---|---|
| `lexical.py` | `build_bm25(docs, cache_path=None, k1=1.5, b=0.75)` buduje indeks `bm25s` (lowercase, angielskie stopwords, stemmer Snowball). `search_bm25(retriever, queries, docs, top_k=5)` -> rankingi; dokumenty z wynikiem 0 są pomijane, więc lista może być krótsza niż `top_k`. | gotowe |
| `fusion.py` | `reciprocal_rank_fusion(rankings_list, k=60, top_k=None)` -> rankingi po fuzji RRF (rangi od 1, score = suma `1 / (k + rank)`). | gotowe |
| `rerank.py` | `rerank(rankings, queries, docs, model_name="cross-encoder/ms-marco-MiniLM-L6-v2", top_n=50, top_k=5)` -> ranking top-`top_n` kandydatów przestawiony przez cross-enkoder, przycięty do `top_k`. | gotowe |