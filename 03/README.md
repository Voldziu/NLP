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
from utils import ...
```

## `utils/` (pisze Piotrek)

Poniżej proponowany interfejs. Piotrek aktualizuje tabelę, gdy moduł jest gotowy.

| Moduł | Co robi | Status |
|---|---|---|
| `data.py` | Ładuje `sentence-transformers/squad`, usuwa duplikaty dokumentów, zwraca `queries`, `docs` i `qrels` (zapytanie → id relewantnego dokumentu). | do zrobienia |
| `encode.py` | Osadzenia zapytań i dokumentów bi-enkoderem (`multi-qa-mpnet-base-dot-v1`), z cache w `artifacts/`. | do zrobienia |
| `index.py` | Budowa i przeszukiwanie indeksu FAISS (`top_k`), zwraca rankingi `[(doc_id, score), ...]`. | do zrobienia |
| `metrics.py` | Metryki oceny (np. Recall@k, MRR, nDCG@k) dla rankingów i `qrels`. | do zrobienia |
