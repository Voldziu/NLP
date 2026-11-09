# Lab 3, Część II (RAG): projekt

Autor: Mikołaj. Zakres: Część II z [instrukcji](../../../03/Zadanie_3_zmodyfikowane.md) (3 pkt).

## Cel

Potok RAG: pytanie → retriever (FAISS na akapitach SQuAD) → prompt z kontekstem → model językowy CLARIN → odpowiedź. Porównanie odpowiedzi z RAG i bez RAG, jakościowo i prostą miarą liczbową.

## Decyzje

| Temat | Wybór | Powód |
|---|---|---|
| Zbiór | `rajpurkar/squad`, split `validation` (ok. 10,5 tys. pytań, ok. 2 tys. unikalnych akapitów) | Te same akapity Wikipedii co w Części I, są krótkie odpowiedzi wzorcowe, mały rozmiar (szybkie osadzenia na Macu). |
| Retriever | Własny `FaissRetriever` za klasą abstrakcyjną `Retriever` | Nie czekamy na `utils/` Piotrka; hybrydę Dominika podpina się jako kolejną podklasę `Retriever`. |
| Osadzenia | `multi-qa-mpnet-base-dot-v1`, indeks `faiss.IndexFlatIP` | Model trenowany pod iloczyn skalarny; 2 tys. dokumentów, więc dokładne przeszukanie bez kompromisów. |
| Model językowy | LangChain `ChatOpenAI` z `base_url` CLARIN | API CLARIN jest zgodne z OpenAI (`/models`, `/chat/completions`). |
| Potok | Pełny łańcuch LCEL | Czytelny przepływ danych do omówienia z prowadzącym. |
| Konfiguracja | `pydantic-settings`, plik `03/.env` | Klucz poza kodem i poza repo; walidacja i czytelny błąd przy braku klucza. |

## Konfiguracja poza kodem

1. Rejestracja i logowanie: <https://services.clarin-pl.eu/login>.
2. W panelu użytkownika wygenerować token API.
3. Skopiować `03/.env.example` do `03/.env` i wpisać token.
4. Sprawdzenie z terminala (z katalogu `03/`):

   ```bash
   source .env && curl -s -H "Authorization: Bearer $CLARIN_API_KEY" "$CLARIN_BASE_URL/models"
   ```

   Oczekiwany wynik: JSON z listą modeli. Nazwę wybranego modelu wpisać do `CLARIN_MODEL`.

`03/.env.example`:

```dotenv
CLARIN_API_KEY=
CLARIN_BASE_URL=https://services.clarin-pl.eu/api/v1/oapi
CLARIN_MODEL=
```

`.gitignore` dostaje wpis `.env` (pasuje w każdym katalogu, nie obejmuje `.env.example`).

Nowe zależności (`uv add`): `langchain-openai`, `langchain-core`, `pydantic-settings`, `faiss-cpu`.

## Moduły `03/utils/rag/`

| Plik | Zawartość |
|---|---|
| `config.py` | `Settings(BaseSettings)` z prefiksem `CLARIN_`: `api_key: SecretStr` (wymagany), `base_url: str` (domyślnie adres CLARIN), `model: str = ""`. `env_file` liczony jako ścieżka bezwzględna: `Path(__file__).parents[2] / ".env"` (czyli `03/.env`), więc działa z dowolnego katalogu roboczego. `get_settings()` zwraca jedną instancję (cache). |
| `base.py` | `Doc` (dataclass: `id: str`, `text: str`, `title: str`, `score: float`) oraz `Retriever(ABC)` z metodą abstrakcyjną `retrieve(query: str, k: int = 5) -> list[Doc]`. |
| `data.py` | `load_squad()` zwraca `docs: list[Doc]` (akapity po deduplikacji, id = indeks unikalnego akapitu) i `qa: list[dict]` (`question`, `answers`, `doc_id`). |
| `retriever.py` | `FaissRetriever(Retriever)`: koduje akapity modelem bi-enkoder, zapisuje osadzenia w `03/artifacts/rag/` (ponowne uruchomienie je wczytuje), buduje `IndexFlatIP`, `retrieve` zwraca top-k `Doc` z wynikiem. |
| `llm.py` | `list_models() -> list[str]` (endpoint `/models`), `get_llm(model: str \| None = None, temperature: float = 0) -> ChatOpenAI` (model z argumentu albo z `Settings.model`; brak obu → `ValueError`). |
| `pipeline.py` | `RAG_PROMPT`, `PLAIN_PROMPT` (`ChatPromptTemplate`), `format_docs(docs) -> str`, `build_rag_chain(retriever, llm, k=5)`, `build_plain_chain(llm)`, `contains_gold(pred, golds) -> bool`. |

### Łańcuch LCEL

```python
retrieve = RunnableLambda(lambda q: retriever.retrieve(q, k))

rag_chain = (
    RunnableParallel(docs=retrieve, question=RunnablePassthrough())
    | RunnablePassthrough.assign(
        answer=(
            {"context": itemgetter("docs") | RunnableLambda(format_docs),
             "question": itemgetter("question")}
            | RAG_PROMPT | llm | StrOutputParser()
        )
    )
)
# rag_chain.invoke("Who ...?") -> {"question", "docs", "answer"}

plain_chain = PLAIN_PROMPT | llm | StrOutputParser()
```

Łańcuch zwraca również dokumenty, więc w analizie widać, co trafiło do kontekstu. `rag_chain.batch(questions)` służy do miary liczbowej.

### Prompt RAG (po angielsku, bo SQuAD jest po angielsku)

- Rola systemowa: odpowiadaj wyłącznie na podstawie podanego kontekstu, krótko.
- Kontekst: akapity ponumerowane `[1]`–`[5]` z tytułem artykułu.
- Wymagane wskazanie numeru akapitu, z którego pochodzi odpowiedź.
- Gdy kontekst nie zawiera odpowiedzi: dokładnie `I don't know.`

`PLAIN_PROMPT`: to samo pytanie bez kontekstu, krótka odpowiedź.

### `contains_gold`

Normalizacja jak w ewaluacji SQuAD (małe litery, bez interpunkcji i przedimków `a/an/the`, scalone spacje). Wynik `True`, gdy któraś znormalizowana odpowiedź wzorcowa jest podciągiem znormalizowanej odpowiedzi modelu.

## Notebook `03/mikolaj/mikolaj.ipynb`

1. II.A: krótka notatka z artykułu o RAG.
2. CLARIN: `list_models()`, wybór modelu, jedno testowe wywołanie `get_llm().invoke(...)`.
3. Indeks: `load_squad()`, `FaissRetriever`, 2–3 przykładowe wyszukiwania.
4. RAG na ok. 8 pytaniach: z SQuAD, własnych, spoza bazy (oczekiwane `I don't know.`), podchwytliwych. Tabela: odpowiedź bez RAG / z RAG / tytuły znalezionych akapitów.
5. Miara liczbowa na 50 pytaniach z `qa` (ziarno 42): odsetek `contains_gold` z RAG i bez RAG oraz trafność retrievera (czy `doc_id` jest w top 5).
6. Analiza (błędy retrievera, halucynacje, odmowy) i podsumowanie.

## `03/mikolaj/mikolaj.md` (ściąga do obrony)

Czym jest RAG i po co. Przepływ łańcucha LCEL krok po kroku. Uzasadnienie wyborów (model, `IndexFlatIP`, iloczyn skalarny, temperatura 0, treść promptu). Wyniki liczbowe z notebooka. Typowe błędy i ograniczenia. Prawdopodobne pytania prowadzącego z krótkimi odpowiedziami.

## Obsługa błędów

- Brak `CLARIN_API_KEY`: `ValidationError` z `pydantic-settings` przy pierwszym `get_settings()`.
- Brak modelu: `ValueError` w `get_llm`.
- Błędy HTTP z API: przekazywane dalej bez opakowywania (widoczne w notebooku).

## Sprawdzenie działania

Bez osobnych testów jednostkowych. Weryfikacja w notebooku: `curl` z sekcji konfiguracji, `list_models()`, jedno pełne wywołanie `rag_chain.invoke`, przebieg miary na 50 pytaniach.

## Poza zakresem

Podpięcie retrievera hybrydowego Dominika (możliwe później jako podklasa `Retriever`), gotowy indeks FAISS Wikipedii z Kaggle, IMDB.
