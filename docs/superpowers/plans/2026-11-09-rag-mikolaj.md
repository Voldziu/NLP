# RAG (lab 3, Część II) – plan wykonania

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Cel:** potok RAG (FAISS na akapitach SQuAD → prompt → model CLARIN) jako łańcuch LCEL, z porównaniem odpowiedzi z RAG i bez RAG.

**Architektura:** Kod wspólny w `03/utils/` (`logger.py`) i `03/utils/rag/` (konfiguracja, klasa abstrakcyjna `Retriever`, dane, FAISS, model, łańcuch). Notebook `03/mikolaj/mikolaj.ipynb` tylko wywołuje te moduły i pokazuje wyniki. `03/mikolaj/mikolaj.md` to ściąga do obrony.

**Technologie:** `datasets`, `sentence-transformers`, `faiss-cpu`, `langchain-core`, `langchain-openai`, `pydantic-settings`, `openai` (zależność `langchain-openai`).

Projekt: [docs/superpowers/specs/2026-11-09-rag-mikolaj-design.md](../specs/2026-11-09-rag-mikolaj-design.md)

## Ograniczenia globalne

- Zbiór: `rajpurkar/squad`, split `validation`.
- Bi-enkoder: `sentence-transformers/multi-qa-mpnet-base-dot-v1`, indeks `faiss.IndexFlatIP`, `k = 5`.
- Ziarno losowania pytań: `42`, próbka liczbowa: 50 pytań.
- Klucz tylko w `03/.env` (zmienne `CLARIN_API_KEY`, `CLARIN_BASE_URL`, `CLARIN_MODEL`); `.env` w `.gitignore`.
- Adres API: `https://services.clarin-pl.eu/api/v1/oapi`.
- Prompty po angielsku; notatki i opisy po polsku.
- Brak osobnych testów jednostkowych: każde zadanie kończy się poleceniem sprawdzającym w terminalu.
- Wszystkie polecenia uruchamiane z katalogu `03/` przez `uv run` (pyproject jest w korzeniu repo, `uv` go znajdzie).
- Commity tylko lokalnie na gałęzi `lab3/mikolaj`, bez `git push`.

## Struktura plików

| Plik | Akcja | Odpowiedzialność |
|---|---|---|
| `pyproject.toml`, `uv.lock` | modyfikacja | nowe zależności |
| `.gitignore` | modyfikacja | wpis `.env` |
| `03/.env.example` | nowy | wzór konfiguracji |
| `03/utils/logger.py` | nowy | `get_logger(name)` |
| `03/utils/rag/config.py` | nowy | `Settings`, `get_settings()` |
| `03/utils/rag/base.py` | nowy | `Doc`, `Retriever(ABC)` |
| `03/utils/rag/data.py` | nowy | `load_squad()` |
| `03/utils/rag/retriever.py` | nowy | `FaissRetriever` |
| `03/utils/rag/llm.py` | nowy | `list_models()`, `get_llm()` |
| `03/utils/rag/pipeline.py` | nowy | prompty, `format_docs`, łańcuchy LCEL, `contains_gold` |
| `03/mikolaj/mikolaj.ipynb` | modyfikacja | eksperymenty i analiza |
| `03/mikolaj/mikolaj.md` | nowy | ściąga do obrony |

---

### Zadanie 1: Środowisko, logger, konfiguracja

**Pliki:**
- Modyfikacja: `pyproject.toml`, `uv.lock`, `.gitignore`
- Nowe: `03/.env.example`, `03/utils/logger.py`, `03/utils/rag/config.py`

**Interfejsy:**
- Produkuje: `utils.logger.get_logger(name: str) -> logging.Logger`; `utils.rag.config.Settings` (pola `api_key: SecretStr`, `base_url: str`, `model: str`), `utils.rag.config.get_settings() -> Settings`, `utils.rag.config.ENV_FILE: Path`.

- [ ] **Krok 1: Zależności** (z korzenia repo)

```bash
uv add langchain-openai langchain-core pydantic-settings faiss-cpu
```

Oczekiwane: `pyproject.toml` zawiera cztery nowe pozycje, `uv.lock` zaktualizowany.

- [ ] **Krok 2: `.gitignore`** – dopisać na końcu:

```gitignore
# klucze API (lab03)
.env
```

- [ ] **Krok 3: `03/.env.example`**

```dotenv
CLARIN_API_KEY=
CLARIN_BASE_URL=https://services.clarin-pl.eu/api/v1/oapi
CLARIN_MODEL=
```

- [ ] **Krok 4: `03/utils/logger.py`**

```python
import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    # Jupyter re-imports modules; add the handler only once to avoid duplicated lines.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
```

- [ ] **Krok 5: `03/utils/rag/config.py`**

```python
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path, so notebooks in any subfolder of 03/ find the same file.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_prefix="CLARIN_", extra="ignore")

    api_key: SecretStr
    base_url: str = "https://services.clarin-pl.eu/api/v1/oapi"
    model: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Krok 6: Sprawdzenie** (z `03/`, bez pliku `.env`)

```bash
uv run python -c "from utils.logger import get_logger; get_logger('test').info('ok')"
uv run python -c "from utils.rag.config import get_settings, ENV_FILE; print(ENV_FILE); get_settings()"
```

Oczekiwane: pierwsza linia drukuje `HH:MM:SS | INFO    | test | ok`. Drugie polecenie drukuje ścieżkę kończącą się na `NLP/03/.env`, a potem `ValidationError` z `api_key Field required`. Następnie `git status --short` nie pokazuje `03/.env` nawet po jego utworzeniu (`cp .env.example .env`, potem usunąć lub zostawić z kluczem).

- [ ] **Krok 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore 03/.env.example 03/utils/logger.py 03/utils/rag/__init__.py 03/utils/rag/config.py
git commit -m "lab3/rag: zależności, logger, konfiguracja .env"
```

---

### Zadanie 2: Dane i retriever FAISS

**Pliki:**
- Nowe: `03/utils/rag/base.py`, `03/utils/rag/data.py`, `03/utils/rag/retriever.py`

**Interfejsy:**
- Pobiera: `get_logger` (zadanie 1).
- Produkuje:
  - `Doc(id: str, text: str, title: str, score: float = 0.0)` (dataclass, `frozen=True`),
  - `Retriever.retrieve(query: str, k: int = 5) -> list[Doc]` (abstrakcyjna),
  - `load_squad(split: str = "validation") -> tuple[list[Doc], list[dict]]`; element `qa`: `{"question": str, "answers": list[str], "doc_id": str}`,
  - `FaissRetriever(docs: list[Doc], model_name: str = MODEL_NAME, cache_name: str = "squad_validation")`, `MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"`.

- [ ] **Krok 1: `03/utils/rag/base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    title: str
    score: float = 0.0


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[Doc]:
        """Return the top-k documents, best first."""
```

- [ ] **Krok 2: `03/utils/rag/data.py`**

```python
from datasets import load_dataset

from ..logger import get_logger
from .base import Doc

logger = get_logger(__name__)


def load_squad(split: str = "validation") -> tuple[list[Doc], list[dict]]:
    ds = load_dataset("rajpurkar/squad", split=split)
    doc_ids: dict[str, str] = {}
    docs: list[Doc] = []
    qa: list[dict] = []
    for row in ds:
        context = row["context"]
        if context not in doc_ids:
            doc_ids[context] = str(len(docs))
            docs.append(Doc(id=doc_ids[context], text=context, title=row["title"].replace("_", " ")))
        qa.append({"question": row["question"], "answers": row["answers"]["text"], "doc_id": doc_ids[context]})
    logger.info("SQuAD %s: %d unique paragraphs, %d questions", split, len(docs), len(qa))
    return docs, qa
```

- [ ] **Krok 3: `03/utils/rag/retriever.py`**

```python
from dataclasses import replace
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..logger import get_logger
from .base import Doc, Retriever

logger = get_logger(__name__)

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "rag"


class FaissRetriever(Retriever):
    def __init__(self, docs: list[Doc], model_name: str = MODEL_NAME, cache_name: str = "squad_validation"):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        embeddings = self._embed_docs(ARTIFACTS / f"{cache_name}__{model_name.split('/')[-1]}.npy")
        # Inner product, because the model was trained for dot-product similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def _embed_docs(self, path: Path) -> np.ndarray:
        if path.exists():
            embeddings = np.load(path)
            if len(embeddings) == len(self.docs):
                logger.info("Loaded embeddings from %s", path)
                return embeddings
        logger.info("Encoding %d documents", len(self.docs))
        embeddings = self.model.encode(
            [d.text for d in self.docs], batch_size=64, convert_to_numpy=True, show_progress_bar=True
        ).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        return embeddings

    def retrieve(self, query: str, k: int = 5) -> list[Doc]:
        q = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        scores, idx = self.index.search(q, k)
        return [replace(self.docs[i], score=float(s)) for s, i in zip(scores[0], idx[0])]
```

- [ ] **Krok 4: Sprawdzenie** (z `03/`)

```bash
uv run python -c "
from utils.rag.data import load_squad
from utils.rag.retriever import FaissRetriever
docs, qa = load_squad()
r = FaissRetriever(docs)
hits = sum(q['doc_id'] in {d.id for d in r.retrieve(q['question'])} for q in qa[:200])
print(qa[0]['question'], '->', r.retrieve(qa[0]['question'])[0].title)
print('hit@5 on 200:', hits / 200)
"
```

Oczekiwane: log `SQuAD validation: 2067 unique paragraphs, 10570 questions` (liczby mogą się nieznacznie różnić), pasek postępu kodowania przy pierwszym uruchomieniu, przy drugim log `Loaded embeddings from ...`. `hit@5` powyżej `0.8`. Plik `03/artifacts/rag/squad_validation__multi-qa-mpnet-base-dot-v1.npy` istnieje.

Znany problem na macOS: jeśli proces kończy się błędem `OMP: Error #15` (konflikt bibliotek OpenMP w `torch` i `faiss`), dodać `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` na początku `retriever.py` przed `import faiss` i opisać to w `mikolaj.md`.

- [ ] **Krok 5: Commit**

```bash
git add 03/utils/rag/base.py 03/utils/rag/data.py 03/utils/rag/retriever.py
git commit -m "lab3/rag: dane SQuAD i retriever FAISS"
```

(`03/artifacts/rag/*.npy` nie wchodzi do commita; jeśli `git status` go pokazuje, dopisać `03/artifacts/rag/` do `.gitignore` w tym samym commicie.)

---

### Zadanie 3: Model CLARIN i łańcuch LCEL

**Pliki:**
- Nowe: `03/utils/rag/llm.py`, `03/utils/rag/pipeline.py`

**Interfejsy:**
- Pobiera: `get_settings()` (zadanie 1), `Doc`, `Retriever` (zadanie 2), `get_logger`.
- Produkuje:
  - `list_models() -> list[str]`, `get_llm(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI`,
  - `RAG_PROMPT`, `PLAIN_PROMPT: ChatPromptTemplate`, `format_docs(docs: list[Doc]) -> str`,
  - `build_rag_chain(retriever: Retriever, llm: BaseChatModel, k: int = 5) -> Runnable[str, dict]` (wynik: `{"question": str, "docs": list[Doc], "answer": str}`),
  - `build_plain_chain(llm: BaseChatModel) -> Runnable[dict, str]` (wejście: `{"question": str}`),
  - `contains_gold(pred: str, golds: list[str]) -> bool`.

**Wymaganie ręczne przed krokiem 4:** token CLARIN wpisany do `03/.env` (instrukcja w projekcie, sekcja „Konfiguracja poza kodem”), `curl` zwraca listę modeli.

- [ ] **Krok 1: `03/utils/rag/llm.py`**

```python
from langchain_openai import ChatOpenAI
from openai import OpenAI

from ..logger import get_logger
from .config import get_settings

logger = get_logger(__name__)


def list_models() -> list[str]:
    s = get_settings()
    client = OpenAI(base_url=s.base_url, api_key=s.api_key.get_secret_value())
    return sorted(m.id for m in client.models.list())


def get_llm(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    s = get_settings()
    model = model or s.model
    if not model:
        raise ValueError("No model: pass `model` or set CLARIN_MODEL in 03/.env")
    logger.info("LLM: %s (temperature=%s)", model, temperature)
    return ChatOpenAI(model=model, base_url=s.base_url, api_key=s.api_key, temperature=temperature)
```

- [ ] **Krok 2: `03/utils/rag/pipeline.py`**

```python
import re
import string
from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from .base import Doc, Retriever

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using only the numbered context passages below. "
     "Answer briefly (one sentence at most) and cite the passage number in square brackets, e.g. [2]. "
     "If the context does not contain the answer, reply exactly: I don't know.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

PLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer the question briefly (one sentence at most)."),
    ("human", "{question}"),
])


def format_docs(docs: list[Doc]) -> str:
    return "\n\n".join(f"[{i}] ({d.title}) {d.text}" for i, d in enumerate(docs, 1))


def build_rag_chain(retriever: Retriever, llm: BaseChatModel, k: int = 5) -> Runnable:
    generate = (
        {"context": itemgetter("docs") | RunnableLambda(format_docs), "question": itemgetter("question")}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(
        docs=RunnableLambda(lambda q: retriever.retrieve(q, k)),
        question=RunnablePassthrough(),
    ) | RunnablePassthrough.assign(answer=generate)


def build_plain_chain(llm: BaseChatModel) -> Runnable:
    return PLAIN_PROMPT | llm | StrOutputParser()


def _normalize(text: str) -> str:
    # Same normalization as the official SQuAD evaluation script.
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_gold(pred: str, golds: list[str]) -> bool:
    # Drop citations like [2], otherwise they would match numeric gold answers.
    p = f" {_normalize(re.sub(r'\[\d+\]', ' ', pred))} "
    return any(g and f" {g} " in p for g in map(_normalize, golds))
```

- [ ] **Krok 3: Sprawdzenie bez API** (z `03/`)

```bash
uv run python -c "
from utils.rag.base import Doc
from utils.rag.pipeline import contains_gold, format_docs
assert contains_gold('It was Denver Broncos [1].', ['Denver Broncos'])
assert contains_gold('The answer is 2015.', ['2015'])
assert not contains_gold('In 2015 [2].', ['2'])
assert not contains_gold(\"I don't know.\", ['Denver Broncos'])
print(format_docs([Doc('0', 'Text A', 'Title A'), Doc('1', 'Text B', 'Title B')]))
"
```

Oczekiwane: brak `AssertionError`, wydruk `[1] (Title A) Text A`, pusta linia, `[2] (Title B) Text B`.

- [ ] **Krok 4: Sprawdzenie z API** (wymaga `03/.env` z tokenem)

```bash
uv run python -c "from utils.rag.llm import list_models; print(list_models())"
```

Wybrać model (preferowany: najnowszy Bielik lub Llama z listy), wpisać do `CLARIN_MODEL` w `03/.env`, potem:

```bash
uv run python -c "
from utils.rag.data import load_squad
from utils.rag.retriever import FaissRetriever
from utils.rag.llm import get_llm
from utils.rag.pipeline import build_rag_chain, build_plain_chain
docs, qa = load_squad()
llm = get_llm()
out = build_rag_chain(FaissRetriever(docs), llm).invoke(qa[0]['question'])
print(out['question'], '|', out['answer'], '|', qa[0]['answers'])
print(build_plain_chain(llm).invoke({'question': qa[0]['question']}))
"
```

Oczekiwane: lista nazw modeli; odpowiedź RAG z numerem akapitu w nawiasach; odpowiedź bez RAG. Jeśli `list_models()` rzuca błąd formatu, a `curl` działa, CLARIN zwraca listę w innym kształcie niż OpenAI: zamienić ciało `list_models` na `httpx.get(f"{s.base_url}/models", headers={"Authorization": f"Bearer {key}"}).json()` i wyciągnąć nazwy z faktycznej struktury.

- [ ] **Krok 5: Commit**

```bash
git add 03/utils/rag/llm.py 03/utils/rag/pipeline.py
git commit -m "lab3/rag: klient CLARIN i łańcuch LCEL"
```

---

### Zadanie 4: Notebook z eksperymentami

**Pliki:**
- Modyfikacja: `03/mikolaj/mikolaj.ipynb` (zostawić istniejącą komórkę z opisem i `sys.path`)

**Interfejsy:**
- Pobiera wszystko z zadań 1–3.
- Produkuje: wykonany notebook z wynikami, liczby `rag_acc`, `plain_acc`, `hit_at_5` do `mikolaj.md`.

Komórki (markdown po polsku, kod jak niżej). Notebook uruchamiać po kolei; wyniki zostają w pliku.

- [ ] **Krok 1: II.A – notatka z artykułu** (markdown, 5–8 zdań): czym jest RAG, retriever i generator, po co (aktualna wiedza, mniej halucynacji, źródła), ograniczenia (jakość zależy od retrievera, długość kontekstu).

- [ ] **Krok 2: Importy i CLARIN**

```python
import random
import pandas as pd
from utils.rag.data import load_squad
from utils.rag.retriever import FaissRetriever
from utils.rag.llm import list_models, get_llm
from utils.rag.pipeline import build_rag_chain, build_plain_chain, contains_gold, format_docs

list_models()
```

```python
llm = get_llm()
llm.invoke("Say hello in Polish.").content
```

Markdown: dlaczego wybrany model (rozmiar, język, dostępność).

- [ ] **Krok 3: Indeks i przykładowe wyszukiwania**

```python
docs, qa = load_squad()
retriever = FaissRetriever(docs)
for q in ["Who won Super Bowl 50?", "What is the capital of Poland?", "How do plants make energy?"]:
    print(q)
    for d in retriever.retrieve(q, k=3):
        print(f"  {d.score:.1f}  {d.title}: {d.text[:100]}")
```

- [ ] **Krok 4: Łańcuchy i pytania przykładowe**

```python
rag_chain = build_rag_chain(retriever, llm, k=5)
plain_chain = build_plain_chain(llm)

questions = [
    qa[0]["question"], qa[100]["question"], qa[2000]["question"],  # z SQuAD
    "Which company developed the Xbox?",                         # własne, temat w bazie lub nie
    "Who was the first president of the United States?",
    "What is the population of Wrocław?",                        # spoza bazy -> oczekiwane "I don't know."
    "Why did the Normans conquer England in 1066?",              # wymaga syntezy
    "Who invented the steam engine in Poland?",                  # podchwytliwe / fałszywe założenie
]
rows = []
for q in questions:
    out = rag_chain.invoke(q)
    rows.append({
        "question": q,
        "plain": plain_chain.invoke({"question": q}),
        "rag": out["answer"],
        "sources": [d.title for d in out["docs"]],
    })
pd.set_option("display.max_colwidth", None)
pd.DataFrame(rows)
```

Przed uruchomieniem sprawdzić, czy pytania „własne” trafiają w tematy z bazy (`sorted({d.title for d in docs})`), i podmienić je tak, żeby były po 2 pytania w bazie i 2 spoza bazy.

- [ ] **Krok 5: Miara liczbowa (50 pytań, ziarno 42)**

```python
sample = random.Random(42).sample(qa, 50)
qs = [s["question"] for s in sample]
rag_out = rag_chain.batch(qs, config={"max_concurrency": 4})
plain_out = plain_chain.batch([{"question": q} for q in qs], config={"max_concurrency": 4})

hit_at_5 = sum(s["doc_id"] in {d.id for d in o["docs"]} for s, o in zip(sample, rag_out)) / 50
rag_acc = sum(contains_gold(o["answer"], s["answers"]) for s, o in zip(sample, rag_out)) / 50
plain_acc = sum(contains_gold(p, s["answers"]) for s, p in zip(sample, plain_out)) / 50
idk = sum(o["answer"].strip() == "I don't know." for o in rag_out)
pd.DataFrame({"metric": ["retriever hit@5", "RAG contains gold", "no-RAG contains gold", "RAG 'I don't know.'"],
              "value": [hit_at_5, rag_acc, plain_acc, idk]})
```

Jeśli API zwraca błędy limitu (HTTP 429), zmniejszyć `max_concurrency` do `1`.

- [ ] **Krok 6: Przykłady błędów**

```python
errors = [(s, o) for s, o in zip(sample, rag_out) if not contains_gold(o["answer"], s["answers"])]
for s, o in errors[:6]:
    found = s["doc_id"] in {d.id for d in o["docs"]}
    print(f"Q: {s['question']}\nGold: {s['answers']}\nRAG: {o['answer']}\nGood paragraph retrieved: {found}\n")
```

- [ ] **Krok 7: Analiza i podsumowanie** (markdown): podział błędów na (a) retriever nie znalazł akapitu, (b) akapit był, ale model odpowiedział inaczej lub szerzej niż wzorzec (fałszywy błąd miary), (c) halucynacja, (d) „I don't know.” mimo dobrego akapitu. Porównanie z RAG / bez RAG dla pytań spoza bazy. 3–5 zdań wniosków.

- [ ] **Krok 8: Commit**

```bash
git add 03/mikolaj/mikolaj.ipynb
git commit -m "lab3/rag: notebook z eksperymentami RAG"
```

---

### Zadanie 5: Ściąga do obrony `mikolaj.md`

**Pliki:**
- Nowy: `03/mikolaj/mikolaj.md`

**Interfejsy:**
- Pobiera: liczby i obserwacje z notebooka (zadanie 4).

- [ ] **Krok 1: Napisać `03/mikolaj/mikolaj.md`** z sekcjami (po polsku, zwięźle, bez skrótów bez rozwinięcia):
  1. **RAG w dwóch zdaniach** i po co (wiedza spoza wag modelu, źródła, mniej halucynacji).
  2. **Przepływ łańcucha LCEL krok po kroku**: `RunnableParallel` (retriever i przekazanie pytania równolegle) → `RunnablePassthrough.assign` (dokleja `answer`, zachowując `docs` i `question`) → `itemgetter` + `format_docs` → `RAG_PROMPT` → `ChatOpenAI` → `StrOutputParser`. Operator `|` to składanie `Runnable`; każdy krok ma `invoke` i `batch`.
  3. **Wybory i uzasadnienia**: SQuAD validation, `multi-qa-mpnet-base-dot-v1` (trenowany na parach pytanie–akapit, iloczyn skalarny), `IndexFlatIP` (2 tys. dokumentów, dokładne przeszukanie; indeksy przybliżone jak IVF czy HNSW mają sens dopiero przy dużych zbiorach), temperatura 0 (powtarzalność), prompt (tylko kontekst, cytowanie numeru, „I don't know.”), `pydantic-settings` (klucz poza repo, walidacja), klasa `Retriever` (podmiana na hybrydę Dominika).
  4. **Wyniki**: tabela z kroku 5 zadania 4 i 2–3 przykłady z kroku 4.
  5. **Błędy i ograniczenia**: kategorie z kroku 7 zadania 4, ograniczenia miary `contains_gold` (dłuższa poprawna odpowiedź może nie zawierać wzorca dosłownie), zależność od retrievera.
  6. **Pytania prowadzącego**: krótkie odpowiedzi na: „Co się stanie, gdy retriever zwróci złe akapity?”, „Czemu `k = 5`, a nie 1 albo 20?”, „Czemu iloczyn skalarny, a nie cosinus?”, „Jak wykryć halucynację?”, „Czym różni się RAG od dotrenowania modelu?”, „Jak podpiąć wyszukiwanie hybrydowe?”, „Po co `RunnablePassthrough.assign`?”.

- [ ] **Krok 2: Sprawdzenie** – każda liczba w `mikolaj.md` zgadza się z wynikami zapisanymi w notebooku.

- [ ] **Krok 3: Commit**

```bash
git add 03/mikolaj/mikolaj.md
git commit -m "lab3/rag: notatki do obrony"
```
