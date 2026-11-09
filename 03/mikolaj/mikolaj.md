# Ściąga do obrony: RAG (Część II)

Kod: `03/utils/rag/`, eksperymenty: `03/mikolaj/mikolaj.ipynb`.

## 1. RAG w dwóch zdaniach

RAG (Retrieval-Augmented Generation) to potok z dwóch części: wyszukiwarka (retriever) znajduje akapity pasujące do pytania, a model językowy odpowiada, mając te akapity w prompcie. Po co: model dostaje wiedzę spoza swoich wag (aktualną lub prywatną), rzadziej zmyśla, a odpowiedź da się sprawdzić w źródle.

Konfiguracja poza kodem: konto na [services.clarin-pl.eu](https://services.clarin-pl.eu/login) → token API w panelu → `cp 03/.env.example 03/.env` i wpisanie `CLARIN_API_KEY` oraz `CLARIN_MODEL`. Plik `.env` jest w `.gitignore`, klucz nie trafia do repozytorium.

## 2. Przepływ łańcucha LCEL

LCEL (LangChain Expression Language) składa potok z małych kroków (`Runnable`) operatorem `|`. Każdy krok ma `invoke` (jedno wejście) i `batch` (wiele wejść, równolegle).

```python
rag_chain = (
    RunnableParallel(
        docs=RunnableLambda(lambda q: retriever.retrieve(q, k)),
        question=RunnablePassthrough(),
    )
    | RunnablePassthrough.assign(
        answer=(
            {"context": itemgetter("docs") | RunnableLambda(format_docs),
             "question": itemgetter("question")}
            | RAG_PROMPT | llm | StrOutputParser()
        )
    )
)
# rag_chain.invoke("Who ...?") -> {"question", "docs", "answer"}
```

```mermaid
flowchart LR
    Q[pytanie] --> P["RunnableParallel"]
    P -->|docs| R[retriever, k = 5]
    P -->|question| PT[RunnablePassthrough]
    R --> S["słownik: docs, question"]
    PT --> S
    S --> A["assign: answer"]
    subgraph gen [answer]
        F["format_docs (docs → context)"] --> PR[RAG_PROMPT] --> L[ChatOpenAI] --> O[StrOutputParser]
    end
    A --> gen
    gen --> W["wynik: question, docs, answer"]
```

Krok po kroku:

1. `RunnableParallel`: to samo pytanie idzie do dwóch gałęzi. Retriever daje `docs`, `RunnablePassthrough` przepisuje `question`. Wynik to słownik `{docs, question}`.
2. `RunnablePassthrough.assign(answer=...)`: dokleja klucz `answer` do istniejącego słownika, nie gubiąc `docs` i `question`.
3. Wewnątrz `answer`: `itemgetter("docs")` wyciąga listę akapitów, `format_docs` składa z niej tekst z numerami `[1]`–`[5]` i tytułami, `RAG_PROMPT` wstawia kontekst i pytanie, `ChatOpenAI` woła model CLARIN, `StrOutputParser` zamienia odpowiedź na zwykły tekst.
4. Łańcuch zwraca też `docs`, więc w analizie widać, co trafiło do kontekstu.

Łańcuch bez RAG: `plain_chain = PLAIN_PROMPT | llm | StrOutputParser()`, wejście `{"question": q}`.

## 3. Wybory i uzasadnienia

| Decyzja | Powód |
|---|---|
| SQuAD, split `validation` (ok. 10,5 tys. pytań, ok. 2 tys. unikalnych akapitów) | Są krótkie odpowiedzi wzorcowe, więc można policzyć trafność; zbiór mały, osadzenia liczą się szybko na laptopie. |
| `multi-qa-mpnet-base-dot-v1` | Model trenowany na parach pytanie–akapit, pod iloczyn skalarny. |
| `faiss.IndexFlatIP` | Iloczyn skalarny, dokładne przeszukiwanie. Przy 2 tys. akapitów jest szybkie; indeksy przybliżone (IVF, HNSW) mają sens dopiero przy milionach dokumentów. |
| `k = 5` | Kompromis między szansą trafienia a długością kontekstu i szumem. |
| Temperatura 0 | Powtarzalne odpowiedzi, łatwiejsze porównanie RAG z bez RAG. |
| Prompt RAG | Tylko podany kontekst, krótka odpowiedź, numer akapitu w nawiasach, a gdy brak odpowiedzi: dokładnie `I don't know.`. Prompty po angielsku, bo SQuAD jest po angielsku. |
| `pydantic-settings` | Klucz z `03/.env`, poza kodem; brak klucza daje czytelny błąd przy starcie. |
| Klasa abstrakcyjna `Retriever` | Retriever hybrydowy z Części I wystarczy zapisać jako podklasę, reszta potoku się nie zmienia. |
| Cache osadzeń w `03/artifacts/rag/` | Akapity koduje się raz, kolejne uruchomienia wczytują plik. |
| Pandas `DataFrame` zamiast wydruków | Wyniki w tabelach, łatwo porównać RAG z bez RAG. |

## 4. Wyniki

Próbka: 50 pytań z SQuAD, ziarno 42, `k = 5`, model: _(z notebooka)_.

| Miara | Liczba (z 50) | Udział |
|---|---|---|
| Retriever: właściwy akapit w top 5 (hit@5) | _(z notebooka)_ | _(z notebooka)_ |
| RAG: odpowiedź zawiera wzorzec | _(z notebooka)_ | _(z notebooka)_ |
| Bez RAG: odpowiedź zawiera wzorzec | _(z notebooka)_ | _(z notebooka)_ |
| RAG: odpowiedź `I don't know.` | _(z notebooka)_ | _(z notebooka)_ |

Przykłady do pokazania (z tabeli w sekcji 3 notebooka):

- pytanie z bazy: _(z notebooka: pytanie, odpowiedź bez RAG, odpowiedź z RAG, źródło)_,
- pytanie spoza bazy: _(z notebooka)_,
- pytanie z fałszywym założeniem: _(z notebooka)_.

## 5. Błędy i ograniczenia

Kategorie błędów RAG (liczby w notebooku, sekcja „Analiza”):

- **(a) Retriever nie znalazł akapitu**: model nie ma z czego odpowiedzieć.
- **(b) Akapit był, ale odpowiedź inna lub szersza niż wzorzec**: często fałszywy błąd miary.
- **(c) Halucynacja**: odpowiedź bez pokrycia w kontekście.
- **(d) `I don't know.` mimo dobrego akapitu**: zbyt ostrożny prompt albo długi kontekst.

Ograniczenia:

- `contains_gold` sprawdza, czy wzorzec występuje dosłownie w odpowiedzi (po normalizacji jak w SQuAD: małe litery, bez interpunkcji i przedimków). Poprawna parafraza lub dłuższa odpowiedź bez wzorca jest liczona jako błąd, więc miara zaniża wynik. Numery cytowań `[2]` są usuwane przed porównaniem.
- Jakość RAG nie przekroczy jakości retrievera.
- Próbka 50 pytań jest mała: różnica kilku punktów procentowych to nie jest wiarygodna różnica.
- Baza to tylko artykuły z SQuAD, więc pytania spoza nich nie mają odpowiedzi w indeksie.

## 6. Pytania prowadzącego

**Co się stanie, gdy retriever zwróci złe akapity?**
Model, zgodnie z promptem, powinien odpowiedzieć `I don't know.`; gorzej, gdy złe akapity brzmią wiarygodnie i model zmyśli odpowiedź na ich podstawie. Stąd analiza hit@5 osobno od jakości odpowiedzi.

**Czemu `k = 5`, a nie 1 albo 20?**
`k = 1` ryzykuje, że jedyny akapit jest zły. `k = 20` wydłuża prompt, dodaje szum i koszt, a model gorzej radzi sobie z długim kontekstem. Pięć to kompromis; można sprawdzić inne wartości, patrząc na hit@5.

**Czemu iloczyn skalarny, a nie cosinus?**
Model `multi-qa-mpnet-base-dot-v1` był trenowany pod iloczyn skalarny, więc `IndexFlatIP` pasuje do niego. Cosinus to iloczyn skalarny wektorów o długości 1. Ten model nie normalizuje wektorów, więc tu oba dają różne rankingi; długość wektora niesie część informacji o dopasowaniu, a model był uczony z jej uwzględnieniem.

**Jak wykryć halucynację?**
Sprawdzić, czy odpowiedź wynika z wskazanego akapitu (prompt wymusza numer w nawiasach). Można też porównać odpowiedź z akapitem ręcznie lub drugim wywołaniem modelu jako sędzią; w projekcie robię to ręcznie na przykładach błędów.

**Czym różni się RAG od dotrenowania modelu?**
Dotrenowanie zmienia wagi modelu: kosztowne, a wiedza starzeje się w wagach. RAG zostawia model bez zmian i dostarcza wiedzę w prompcie, więc zmiana bazy dokumentów nie wymaga treningu i da się wskazać źródło.

**Jak podpiąć wyszukiwanie hybrydowe?**
Napisać podklasę `Retriever` z metodą `retrieve(query, k) -> list[Doc]`, która łączy wyniki BM25 i bi-enkodera (np. z Części I), i przekazać ją do `build_rag_chain`. Reszta łańcucha się nie zmienia.

**Po co `RunnablePassthrough.assign`?**
Bez niego wynik miałby tylko odpowiedź. `assign` dokleja `answer` do słownika z `docs` i `question`, więc można ocenić, czy właściwy akapit trafił do kontekstu.
