# Notatki z realizacji: pkt 4 i 5 (Dominik)

Dziennik pracy, krok po kroku: co zrobiłem i dlaczego. Plan ustalony przed implementacją jest na końcu pliku.

## Krok 1: gałąź i zależności

- Gałąź `lab3/dominik` od `origin/main` (po merge'u pracy Piotrka, #10). Dzięki temu mam aktualne `utils/` i nie mieszam się z gałęzią Piotrka.
- `uv add bm25s PyStemmer matplotlib-venn`:
  - `bm25s`: BM25 na rzadkich macierzach (scipy), liczy wyniki dla wszystkich 500 zapytań naraz w ułamku sekundy. `rank_bm25` liczy każde zapytanie osobno w czystym Pythonie.
  - `PyStemmer`: stemmer Snowball dla angielskiego (np. `capital` → `capit`), więc „cats” i „cat” to ten sam term.
  - `matplotlib-venn`: diagram pokrycia trafień BM25 vs bi-enkoder (pkt 4.d).

## Krok 2: moduły w `utils/`

Wszystkie trzy moduły przyjmują i zwracają ten sam format co kod Piotrka: `{query_id: [(doc_id, score), ...]}`. Dzięki temu każdy wynik od razu wchodzi do `evaluate_rankings` i można go porównywać z bi-enkoderem.

### `utils/lexical.py` (BM25, pkt 4.a)

- Tokenizacja `bm25s.tokenize`: małe litery, angielskie stopwords, stemmer Snowball. Parametry `k1 = 1.5`, `b = 0.75` (standard, bez strojenia, żeby nie dopasowywać BM25 do zbioru testowego).
- **Odfiltrowuję dokumenty z wynikiem 0.** `bm25s` zawsze zwraca `k` dokumentów: jeśli zapytanie pasuje tylko do 30 akapitów, pozostałe 70 to przypadkowe dokumenty z wynikiem 0. W RRF dostałyby one rangi 31–100 i fałszywie podbijały swoje wyniki, a to nie są żadne trafienia. Lista BM25 może więc być krótsza niż `top_k`.
- `cache_path` pozwala zapisać indeks na dysk (`bm25s.BM25.save/load`). Przy 18 891 dokumentach indeksowanie trwa kilka sekund, więc to tylko wygoda.

### `utils/fusion.py` (RRF, pkt 4.b)

- `RRF(d) = Σ_i 1 / (k + rank_i(d))`, `k = 60`, rangi liczone **od 1** (jak u Cormacka i in., 2009).
- Dokument nieobecny na liście systemu `i` nic od niego nie dostaje (składnik 0).
- Remisy rozstrzygam deterministycznie według `doc_id`, żeby wynik nie zależał od kolejności w słowniku.

### `utils/rerank.py` (cross-enkoder, pkt 5)

- Bierze `top_n` pierwszych kandydatów z rankingu wejściowego (domyślnie hybrydy), ocenia każdą parę (zapytanie, dokument) cross-enkoderem i zwraca `top_k` najlepszych według jego wyniku.
- Wszystkie pary ze wszystkich zapytań idą do `CrossEncoder.predict` jednym wywołaniem z batchowaniem. Na GPU to wielokrotnie szybsze niż pętla po zapytaniach.
- Model domyślny: `cross-encoder/ms-marco-MiniLM-L6-v2` (decyzja Q6: dobry kompromis jakości i szybkości).

Szybki test na 3 zdaniach przed pisaniem notebooka: BM25 na zapytaniu z samych nieznanych słów zwraca pustą listę (a nie 3 przypadkowe dokumenty), RRF daje ręcznie policzoną kolejność, cross-enkoder działa na GPU.

## Krok 3: porządki w repo

- `.gitignore`: dodałem `03/artifacts/`. Osadzenia 18 891 akapitów to ~58 MB (`float32`, 768 wymiarów), a README i tak mówi, że każdy generuje je lokalnie. Wcześniej ignorowane było tylko `03/artifacts/rag/`, więc łatwo było je przypadkiem zacommitować.
- **Zmiana względem planu:** nie zapisuję rankingów w cache. Mierzę czas na zapytanie, więc i tak muszę je policzyć od nowa, a wszystko poza osadzeniami (które `utils` Piotrka już cache'uje) liczy się w sekundach. Cache tylko komplikowałby notebook.

## Krok 4: notebook `dominik.ipynb`

Struktura odpowiada instrukcji: 0 (dane + bi-enkoder) → 4.a → 4.b → 4.c → 4.d → 4.e → 5.a-b → 5.c → podsumowanie.

- **0. Dane:** `load_squad_data()` i `encode_corpus_and_queries()` bez zmian. Bi-enkoder zwraca top-100 (potrzebne do fuzji i re-rankingu). `evaluate_rankings(k=5)` i tak patrzy na 5 pierwszych pozycji, więc wynik @5 jest taki sam jak u Piotrka. **Assert** sprawdza zgodność z jego liczbami (tolerancja 0.005, czyli ~2 zapytania na 500, bo osadzenia liczone na innym GPU mogą minimalnie zmienić kolejność remisów).
- **Pomiar czasu:** średnia na zapytanie, po rozgrzewce, bez ładowania modeli. Bi-enkoder = kodowanie zapytań + FAISS; hybryda = BM25 + bi-enkoder + fuzja; CE = hybryda + re-ranking. Bez rozgrzewki pierwszy pomiar mierzyłby inicjalizację CUDA, a nie wyszukiwanie.
- **4.a:** dodatkowy **assert**: 15 pierwszych słów losowego akapitu jako zapytanie → ten akapit na pozycji 1 w ≥95% przypadków. Sprawdza, że tokenizacja korpusu i zapytań jest spójna (np. ten sam stemmer).
- **4.b:** **assert** RRF na ręcznie policzonym przykładzie, potem fuzja list top-100. Wykres wrażliwości na długość list N ∈ {5, 10, 20, 50, 100} pokazuje, ile daje fuzja głębszych list.
- **4.c:** wykres metryk + diagram Venna trafień w top-5 (BM25 vs bi-enkoder). Rozłączne części diagramu to dokładnie to, co hybryda może zyskać.
- **4.d:** koszyki „tylko BM25 / tylko bi-enkoder / oba / żaden” i cztery cechy zapytań (cyfry, nazwy własne, średnie IDF, pokrycie leksykalne ze złotym akapitem). Przykłady wybieram według największej różnicy pozycji, żeby pokazać najwyraźniejsze przypadki, a nie losowe.
- **4.e:** wyjaśnienie + histogramy wyników obu systemów + ablacja CombSUM (surowe / min-max) vs RRF. Mierzę też, jak często top-1 fuzji to top-1 BM25, a jak często bi-enkodera. To pokazuje, który system „dominuje” przy sumowaniu surowych wyników.
- **5.a-b:** CE na top-50 hybrydy (wariant główny), wariant kontrolny top-5 (**assert**: ten sam zbiór dokumentów i ta sama Recall@5) oraz L12 dla porównania. Ostatni ranking CE ma 50 pozycji zamiast 5, żeby dało się narysować krzywą Recall@k.
- **Pula re-rankingu:** n ∈ {5, 10, 20, 50, 100} z sufitem Recall@n hybrydy i czasem. Ten wykres to podstawa `rerank_pool.md`.
- **5.c:** 5 wykresów (metryki, Recall@k, rozkład pozycji, przejścia pozycji przed/po CE, jakość a czas) + tabela + przykłady naprawionych i zepsutych rankingów.
- Wykresy zapisują się do `03/dominik/figures/` (PNG, 150 dpi), każdy system ma jeden kolor we wszystkich wykresach.
- Komórki z interpretacją uzupełniam **po** uruchomieniu, żeby wnioski opierały się na faktycznych liczbach.

## Krok 5: pierwsze uruchomienie i poprawki

Notebook przeszedł od zera, wszystkie asserty zielone. Bi-enkoder odtwarza wynik Piotrka co do 4. miejsca po przecinku (0.882 / 0.7517 / 0.7845).

**Nieoczekiwany wynik:** CombSUM na surowych wynikach (Recall@5 0.934, nDCG@5 0.872) wypadł **lepiej** niż RRF (0.910 / 0.840). Nie ukrywam tego, tylko sprawdzam, skąd się bierze:
- Histogramy pokazują, że skale są różne (BM25 w top-100: głównie 2–10, top-1 do ~35; bi-enkoder: 12–31), ale w obrębie zapytania **rozpiętość** wyników obu systemów jest podobnego rzędu. Zwykłe dodawanie działa więc przypadkiem jak dobrze dobrana ważona suma.
- Dlatego dodałem test zmiany skali: `α·BM25 + bi-enkoder` dla α od 0.05 do 20. Przeskalowanie nie zmienia żadnego z pojedynczych rankingów, a odpowiada realnym zmianom (inny wariant BM25, model gęsty z kosinusem w [-1, 1]). Jeśli CombSUM mocno zależy od α, jego przewaga przy α = 1 to zbieg okoliczności, który trzeba by stroić (i to na zbiorze testowym), a RRF działa bez strojenia.

Poprawki wizualne po przejrzeniu wykresów:
- paski postępu HF/tqdm zaśmiecały wyjścia: `HF_HUB_DISABLE_PROGRESS_BARS=1`, `TQDM_DISABLE=1`,
- pozycje w przykładach wyświetlały się jako `1.0` / `nan` (pandas zamienia `None` na `NaN`): funkcja `fmt_rank`,
- legenda zasłaniała słupki (4.c) i linie (pula re-rankingu): przeniesiona nad/pod wykres,
- przez heatmapę przejść przebijała siatka seaborna: `grid(False)`,
- etykiety BM25 i bi-enkodera nachodziły na siebie na wykresie jakość–czas (prawie ten sam punkt): rozsunięte.

Analizy wpisuję po drugim uruchomieniu osobnym skryptem do oznaczonych komórek markdown (tagi `analiza-*`), żeby poprawki tekstu nie wymagały przeliczania notebooka.

## Krok 6: drugie uruchomienie, analizy i `rerank_pool.md`

Drugi przebieg od zera dał identyczne metryki, a czasy różnią się o mniej niż 2%, więc wyniki są powtarzalne.

**Wynik testu skali (4.e):** CombSUM ma szczyt dokładnie przy α = 1 i jest lepszy od RRF dla α ≈ 0.25–4. Przy rozjeździe skal ≥ 10× spada poniżej RRF. W notebooku piszę to wprost: na tym zbiorze fuzja wyników jest lepsza, ale tylko dzięki przypadkowej zgodności skal, a RRF daje odporność bez strojenia kosztem ~2.4 pp Recall@5. To uczciwsza odpowiedź na 4.e niż samo „rangi są lepsze”.

**Przed wpisaniem analiz sprawdziłem fakty w danych**, zamiast wnioskować z ucinanych podglądów:
- „Nepalese” → `nepales`, „Nepal” → `nepal` (stemmer ich nie łączy), a literówka „szlactha” ≠ „szlachta”,
- złoty akapit o Brazylii **zawiera** „Congress” („National Congress”), ale nie „Brazil”. BM25 przegrywa przez brak „Brazil”, a nie przez brak „Congress”. Poprawiłem opis,
- złoty akapit przy pytaniu o antydepresanty trójpierścieniowe podaje odpowiedź wprost w ostatnim zdaniu, więc to błąd cross-enkodera (dopasowanie „dekada + lek”), a nie problem z etykietą.

**Pula re-rankingu: zostaje 50.** Reguła z planu brzmiała: jeśli mniejsza pula wystarcza, zmniejszam. Między 20 a 50 różnica to ~3 zapytania (na granicy szumu), ale dopiero przy 50 wypłaszczają się i jakość, i sufit (Recall@50 hybrydy 0.992; 100 nie dodaje nic). Koszt ~80 ms/zapytanie jest w zadaniu ewaluacyjnym akceptowalny. W `rerank_pool.md` opisuję 10–20 jako lepszy wybór, gdyby liczyło się opóźnienie.

## Wyniki końcowe

| System | Recall@5 | MRR@5 | nDCG@5 | ms/zapytanie |
|---|---|---|---|---|
| BM25 | 0.852 | 0.761 | 0.784 | ~1.2 |
| Bi-enkoder (pkt 3) | 0.882 | 0.752 | 0.785 | ~0.8 |
| Hybryda RRF | 0.910 | 0.817 | 0.840 | ~2 |
| Hybryda + CE (top-5) | 0.910 | 0.858 | 0.871 | ~10 |
| **Hybryda + CE (top-50)** | **0.946** | **0.889** | **0.903** | ~79 |
| Hybryda + CE L12 (top-50) | 0.960 | 0.904 | 0.918 | ~156 |

---

## Plan ustalony przed implementacją

| # | Decyzja | Ustalenie |
|---|---|---|
| Q1 | Gałąź | `lab3/dominik` od `origin/main` |
| Q2 | Dane | `load_squad_data()` domyślnie (`train`, 500 zapytań, seed 42) + assert zgodności z pkt 3; poprawka `split` w README |
| Q3 | Kod | moduły w `utils/` (`lexical`, `fusion`, `rerank`), wspólny format rankingów |
| Q4 | BM25 | `bm25s` + `PyStemmer`, stopwords EN, `k1 = 1.5`, `b = 0.75` |
| Q5 | Fuzja | top-100 z każdego systemu → RRF (k = 60) → top-5; wrażliwość na N |
| Q6 | Cross-enkoder | `ms-marco-MiniLM-L6-v2`, L12 opcjonalnie |
| Q7 | Re-ranking | top-50 hybrydy, wszystkie 500 zapytań, wariant kontrolny top-5, krzywa puli; dokument z uzasadnieniem |
| Q8 | 4.d | koszyki trafień + cechy zapytań + ręczne przykłady |
| Q9 | 4.e | wyjaśnienie + histogramy + ablacja CombSUM / min-max / RRF |
| Q10 | API | sygnatury jak w `utils/`, wpis do README |
| Q11 | Weryfikacja | asserty w notebooku |
| Q12/14/15 | Porównanie | wykresy jako główne porównanie (7 typów), tabela dodatkowo, stałe kolory, PNG w `figures/` |
| Q13 | `rerank_pool.md` | pisany po pomiarach, z krzywą puli |
| Q16 | Notebook | po polsku, sekcje jak w instrukcji, podsumowanie na końcu |
| Q17 | Oddanie | commity na `lab3/dominik`, PR otwiera Dominik |
