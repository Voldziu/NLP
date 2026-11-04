# GPT-2 w Zadaniu 2: omówienie notebooka krok po kroku

Ten dokument przechodzi po kolei przez [`gpt2.ipynb`](gpt2.ipynb). Przy każdej sekcji podaje: **co mówi polecenie** z [Zadanie_2_zmodyfikowane.md](../Zadanie_2_zmodyfikowane.md), **co robimy**, **dlaczego właśnie tak** i **na co patrzeć w wynikach**.

- Wyniki pilota: [`gpt2_pilot.ipynb`](gpt2_pilot.ipynb) (2000 tekstów treningowych, 2 epoki, ok. 20 min).
- Wyniki pełnego runu: `gpt2_full.ipynb` (cały zbiór, 5 epok, ok. 2,5 h).

Liczby poniżej pochodzą **z pilota**, więc są orientacyjne. Po pełnym runie podmieniamy je i uzupełniamy komórki „Wnioski” w notebooku.

---

## 0. Mapa: polecenie → gdzie to jest zrobione

Część I jest podzielona na trzy osoby (patrz [README](../README.md)). Ten notebook to **osoba B (GPT-2)**.

| Polecenie | Pkt | Gdzie | Status |
|---|---|---|---|
| I.A: klasy `Bert*` | 0,25 | `bert/bert.ipynb` (osoba A) | zrobione |
| **I.B: biblioteka PEFT, quicktour, przykłady** | 0,25 | **`gpt2.ipynb` → I.B** + `notebooks/giga_zgoacone_intro.ipynb` (część B) | zrobione |
| **I.C: powtórzyć tutorial z Medium (PEFT BERT, diagnozy medyczne)** | 0,25 | `notebooks/giga_zgoacone_intro.ipynb` (część C), adapter w `artifacts/bert-lora-symptoms` | zrobione |
| **I.D: klasy `GPT2*`** | 0,25 | **`gpt2.ipynb` → I.D** (+ część D notatnika intro) | zrobione |
| Plan 1: HerBERT + PEFT, tekst i tokeny, test | 1,5 | `bert/bert.ipynb` (osoba A) | zrobione |
| Plan 2: wizualizacja `[CLS]` przed i po | 1 | `bert/bert.ipynb` (osoba A) | zrobione |
| **Plan 3: powtórzyć 1 i 2 dla GPT-2** (`polish-gpt2-medium`, PEFT, ostatni token, `-100` na padding i `<eos>`) | 1 | **`gpt2.ipynb` → 3a, 3b, 3c** | pilot ✓, pełny run w toku |
| Plan 4: augmentacja `[MASK]` | 0,5 | `bert/bert.ipynb` (osoba A) | zrobione |
| **Plan 5: augmentacja generacją GPT-2 (temperatura, top-p, top-k)** | 1 | **`gpt2.ipynb` → 5a, 5b, 5c** | pilot ✓, pełny run w toku |
| Część II | 4 | `part2/part2.ipynb` (osoba C) | zrobione |

---

## 1. Pięć pojęć, bez których reszta nie ma sensu

1. **Maska przyczynowa (causal mask).** GPT-2 czyta tekst od lewej do prawej: token na pozycji *t* widzi tylko tokeny *1..t*. BERT widzi całe zdanie naraz. Z tego wynika prawie każda różnica w wynikach.
2. **Ostatni token jako reprezentacja tekstu.** GPT-2 nie ma `[CLS]`. Jedyny token, który „przeczytał” cały tekst, to ostatni. Dlatego polecenie (pkt 3) każe go użyć zamiast `[CLS]`.
3. **LoRA.** Nie trenujemy 355 mln wag GPT-2. Do wybranych macierzy dokładamy małą poprawkę `W + B·A`, gdzie `A` i `B` mają rząd `r` (4, 8 lub 16), i trenujemy tylko ją. To ok. 0,2% parametrów.
4. **Etykieta `-100`.** Funkcja straty w PyTorch (`CrossEntropyLoss`) pomija pozycje z etykietą `-100`. Tak mówimy modelowi, że ten token się nie liczy: padding, `</s>`, drugi i kolejny kawałek słowa.
5. **`pad = eos`.** Tokenizer GPT-2 nie ma tokenu paddingu. Standardowo używamy w tej roli `</s>` (id 2). Ma to skutki, opisane w 3a i 3b.

---

## 2. Konfiguracja: tryby pilot i pełny

**Co robimy.** Notebook ma flagę `PILOT` (oraz `SMOKE` do testów technicznych). Papermill uruchamia go jako osobną usługę systemd ([`run.sh`](run.sh)), poza terminalem i z blokadą uśpienia laptopa.

| | pilot | pełny |
|---|---|---|
| train (tekst) | 2000 (warstwowo, proporcje klas zachowane) | 6573 (cały) |
| dev / test | pełne (823 / 820) | pełne |
| epoki: tekst / tokeny | 2 / 50 | 5 / 200 |
| prompty w siatce generacji | 64 | 200 |
| kopie augmentacji | 1 | 2 |

**Dlaczego.** Pilot w 20 min sprawdza, że wszystko działa end-to-end, i daje pierwsze liczby. Pełny run liczy się dłużej. Test jest **zawsze pełny i ten sam co u BERT-a**, żeby wyniki dało się porównać. Każdy eksperyment zapisuje wynik na dysk (`artifacts/gpt2-*/…/result.json`), więc przerwany run wznawia się od miejsca przerwania.

---

## 3. I.B: PEFT (notatka)

**Polecenie:** „zapoznać się z biblioteką PEFT oraz przykładami z repozytorium”.

**Co jest w notebooku.** Notatka o LoRA: wzór, parametry `LoraConfig` z wartościami, których używamy, i inne metody PEFT (prompt/prefix tuning, IA³). Do tego jedna rzecz ważna przy porównaniu z BERT-em:

- W GPT-2 Q, K i V liczy **jedna** macierz `c_attn` (1024 → 3072). LoRA na `c_attn` adaptuje więc Q, K i V naraz. U BERT-a adaptowane są tylko `query` i `value`.
- GPT-2 ma 24 warstwy (BERT 12). Przy tym samym `r` adapter GPT-2 jest ok. 2,7× większy: dla `r=8` to 790 tys. parametrów, u BERT-a 298 tys. Względem rozmiaru modelu udział jest podobny (ok. 0,2%).
- `fan_in_fan_out=True`: GPT-2 trzyma wagi w warstwie `Conv1D` w transpozycji względem `nn.Linear`. Bez tej flagi LoRA dostałaby macierze w złej orientacji.

**I.C (tutorial)** jest powtórzony w `notebooks/giga_zgoacone_intro.ipynb`, część C.

---

## 4. I.D: klasy GPT-2 (notatka)

**Polecenie:** „zapoznać się z klasami `GPT2Model`, `GPT2LMHeadModel`, `GPT2ForSequenceClassification`, `GPT2ForTokenClassification`”.

**Co jest w notebooku.** Tabela (głowica, kształt wyjścia, `labels` → `loss`) oraz komórka, która ładuje wszystkie cztery klasy i **sprawdza kształty assertami**. Najważniejsze:

| Klasa | Głowica | Wyjście |
|---|---|---|
| `GPT2Model` | brak (24 bloki + końcowy LayerNorm `ln_f`) | `(B, T, 1024)` |
| `GPT2LMHeadModel` | `lm_head` 1024 → 51 200, **te same wagi co embeddingi wejściowe** (sprawdzone w kodzie: `True`) | rozkład następnego tokenu dla każdej pozycji |
| `GPT2ForSequenceClassification` | `score` 1024 → 4 | `(B, 4)`, liczone na **ostatnim tokenie różnym od `pad_token_id`** |
| `GPT2ForTokenClassification` | `classifier` 1024 → 7 | `(B, T, 7)` |

**Pułapka, którą warto pokazać grupie.** `GPT2ForSequenceClassification` szuka ostatniego tokenu różnego od `pad`. Skoro `pad == eos`, dopisany na końcu `</s>` model traktuje jak padding i go pomija. Dlatego w klasyfikacji tekstu `</s>` **nie dopisujemy**, a w klasyfikacji tokenów dopisujemy (szczegóły w 3b).

---

## 5. 3a: klasyfikacja tekstu + PEFT

**Polecenie (pkt 3 → 1a):** strojenie `GPT2ForSequenceClassification` z adapterem PEFT na danych z Zadania 1, model `sdadas/polish-gpt2-medium`.

**Dane.** PolEmo (recenzje): 4 klasy `plus_m` (pozytywna), `minus_m` (negatywna), `zero` (neutralna), `amb` (ambiwalentna). Podział 6573 / 823 / 820, taki sam jak u osoby A.

**Co robimy.** Protokół jest **1:1 z BERT-em**, żeby różnice wynikały z modelu, a nie z ustawień:

| | BERT (osoba A) | GPT-2 (my) |
|---|---|---|
| rangi | `r = 0, 4, 8, 16` | to samo |
| `r = 0` (baseline) | zamrożony model, uczy się tylko głowica | to samo |
| epoki, lr, `MAX_LEN` | 5, `2e-4`, 400 | to samo |
| batch | 16 | 4 × gradient accumulation 4 = efektywnie 16 |
| wybór checkpointu | najlepsze macro-F1 na dev | to samo |
| LoRA na | `query`, `value` | `c_attn` (Q, K, V) |
| reprezentacja dla głowicy | `[CLS]` → pooler | ostatni token |

**Dlaczego tak.**
- **Baseline `r = 0`** pokazuje, ile informacji o sentymencie ma model *przed* strojeniem: uczy się tylko liniowa głowica na zamrożonych reprezentacjach. Technicznie to ten sam kod co LoRA, z zamrożonymi `A`, `B` (a `B = 0` na starcie, więc poprawka wynosi 0).
- **Batch 4 × 4:** batch 16 (a nawet 8) nie mieści się w 8 GB VRAM. Gradient accumulation sumuje gradienty z 4 kroków przed aktualizacją wag. Matematycznie to ten sam efektywny batch.
- **Macro-F1, a nie accuracy:** klasy są niezbalansowane (`minus_m` 38%, `zero` 15%), a macro-F1 waży każdą klasę tak samo.

**Wyniki pilota (test):**

| | BERT (pełny) F1 | GPT-2 (pilot) F1 |
|---|---|---|
| baseline `r=0` | 0,533 | **0,753** |
| LoRA `r=4` | 0,870 | 0,838 |
| LoRA `r=8` | 0,868 | 0,843 |
| LoRA `r=16` | 0,872 | 0,836 |

- LoRA daje ok. +0,09 F1 względem baseline'u. `r` prawie nie ma znaczenia (tak samo jak u BERT-a).
- Najsłabsza klasa to `amb` (F1 ok. 0,60). Najłatwiejsza to `zero` (0,95): teksty neutralne to często artykuły, a nie recenzje.
- **Hipoteza zapisana przed runem się nie potwierdziła:** spodziewaliśmy się, że baseline GPT-2 będzie *gorszy* od BERT-a, a jest wyraźnie lepszy (0,75 vs 0,53). Możliwe wyjaśnienia do sprawdzenia przy wnioskach:
  - głowica BERT-a stoi na `pooler_output` (`tanh` po warstwie wytrenowanej do innego zadania, NSP), a głowica GPT-2 bezpośrednio na stanie ukrytym,
  - GPT-2 medium jest prawie 3× większy i ma 1024 wymiary zamiast 768,
  - przewidywanie „co będzie dalej” po całej recenzji może kodować jej ton.

  Osoba A zauważyła też, że baseline BERT-a „wolno się uczył”, więc 5 epok mogło mu nie wystarczyć.

---

## 6. 3b: klasyfikacja tokenów + PEFT (padding i `<eos>` → `-100`)

**Polecenie (pkt 3 → 1b):** strojenie `GPT2ForTokenClassification` z PEFT. **„Proszę również zamaskować tokeny paddingu i znacznik `<eos>` za pomocą etykietki `-100`.”**

**Dane.** Anotacje z Zadania 1: tagi BIO dla `wzmocnienie`, `oslabienie`, `negacja` (7 tagów z `O`). Tylko **35 tekstów** (24 train / 4 dev / 7 test), te same co u BERT-a.

**Jak spełniamy wymóg `-100`.**
1. Tokenizer GPT-2 sam **nie dopisuje** `</s>`, więc dopisujemy go ręcznie na końcu każdej sekwencji, z etykietą `-100`.
2. Padding dokłada `DataCollatorForTokenClassification`, który też daje mu `-100`.
3. Każde słowo dostaje etykietę tylko na **pierwszym** subtokenie, a reszta ma `-100` (`utils/align.py`, wspólne z BERT-em).
4. **Asserty w notebooku** sprawdzają na prawdziwym batchu, że: padding ma `-100`, `</s>` ma `-100`, ostatni prawdziwy token to `</s>`, a liczba etykiet ≠ `-100` równa się liczbie słów. Przykład z wyjścia: `(' horm', 0), ('onów', -100), (' .', 0), ('</s>', -100), ('</s>', -100)`.

Tokenizer w tej sekcji ma `add_prefix_space=True`, bo GPT-2 zapisuje spację jako część tokenu. Bez tej opcji pierwsze słowo listy tokenizowałoby się inaczej niż reszta.

**Ustawienia** jak u BERT-a: 200 epok, `lr = 1e-3` (przy 24 tekstach model potrzebuje dużo przejść), efektywny batch 8. **Metryka:** macro-F1 po 6 tagach **bez `O`**, bo `O` to ok. 96% tokenów. Model „zawsze `O`” ma accuracy 0,97 i nic nie umie.

**Wyniki pilota (50 epok, nie 200!):** macro-F1 0,19–0,22 przy BERT-cie 0,24–0,39.

| Tag | GPT-2 LoRA (pilot) F1 | support w teście |
|---|---|---|
| `B-wzmocnienie` | **0,94** | 9 |
| `B-negacja` | 0,18–0,36 | 18 |
| reszta | 0 | 0–2 |

**Interpretacja (do potwierdzenia w pełnym runie).** Wzmocnienie („bardzo”, „naprawdę”) rozpoznaje się po samym słowie, więc GPT-2 radzi sobie dobrze. Negacja jest trudna: „nie” jest bardzo częste, a o tym, czy to *negacja sentymentu*, decyduje to, co stoi **po nim**. Tego GPT-2 przez maskę przyczynową nie widzi. BERT widzi.

**Zastrzeżenie:** 7 tekstów testowych to ok. 30 tagów innych niż `O`. Jedna zmieniona predykcja przesuwa F1 o kilka punktów, więc różnice między `r` to szum.

---

## 7. 3c: wizualizacja przestrzeni wektorowej

**Polecenie (pkt 3 → 2):** zwizualizować przestrzeń wektorową przykładów testowych (klasyfikacja tekstu) przed strojeniem i po nim, na interaktywnych wykresach, porównać i opisać. **„W przypadku wykorzystania adaptera PEFT proszę wykorzystać reprezentacje ostatniego tokenu sekwencji tekstowej jako reprezentację tekstu.”**

**Co robimy.**
- Reprezentacja = stan ukryty **ostatniego prawdziwego tokenu** (padding jest z prawej, więc indeks to `attention_mask.sum(1) − 1`) z ostatniej warstwy. To dokładnie wejście głowicy `score`.
- **Przed:** bazowy `GPT2Model`. **Po:** model z adapterem z 3a (`r = 4, 8, 16`) po `merge_and_unload()`, czyli z poprawką LoRA wpisaną w wagi.
- Wykresy: t-SNE do 2D (Plotly, interaktywne, tekst po najechaniu kursorem), kolor = prawdziwa etykieta.
- Miara liczbowa: **silhouette** (cosinus) liczony w pełnej przestrzeni 1024D.

**Dwa sprawdzenia poprawności w kodzie:**
1. Głowica `score` zastosowana do naszej reprezentacji daje **te same logity** co cały model, czyli patrzymy na ten sam token co klasyfikator.
2. Dla `r = 0` reprezentacja jest identyczna z „przed”, bo poprawka LoRA wynosi 0.

**Dlaczego silhouette w 1024D, a nie na wykresie?** t-SNE zachowuje sąsiedztwo punktów, ale zniekształca odległości: odstęp między skupiskami na wykresie nic nie znaczy. Silhouette mówi, czy punkt jest bliżej swojej klasy (→ 1), czy obcej (→ −1). Około 0 oznacza klasy wymieszane.

**Wyniki pilota:** silhouette **0,05 → 0,22–0,25** (BERT: −0,01 → 0,47–0,51). Kierunek się zgadza: strojenie porządkuje przestrzeń według etykiet. GPT-2 robi to słabiej niż BERT, ale pilot miał 2 epoki zamiast 5, więc z porównaniem czekamy na pełny run.

---

## 8. 5: augmentacja generacją GPT-2

**Polecenie (pkt 5):** „Wykorzystać generatywne zdolności GPT-2 i różne hiperparametry generacji, np. temperatura, top-p, top-k, [do] powiększenia zbiorów treningowych”. Uwaga z polecenia: „wystarczy wytrenować jeden model na powiększonym zbiorze danych”.

### 8.1 Jak generować, żeby zachować etykietę

GPT-2 bazowy nie wykonuje poleceń („napisz negatywną recenzję” nie zadziała). Używamy więc **kontynuacji prefiksu**:
- bierzemy tekst treningowy klasy *c* i zostawiamy pierwsze ok. 25% tokenów (min 10, max 40),
- model dopisuje resztę, mniej więcej do oryginalnej długości,
- nowy tekst (prefiks + kontynuacja) dostaje etykietę *c*.

Założenie: początek recenzji ustawia jej ton. Sprawdzamy to miarą „zgodności etykiety” (niżej).

### 8.2 Siatka hiperparametrów (5a)

Temperatura `T` skaluje rozkład następnego tokenu: `T < 1` wyostrza go (bezpieczniej, ale nudno), `T > 1` spłaszcza (różnorodnie, ale chaotycznie). **top-k** losuje z `k` najbardziej prawdopodobnych tokenów. **top-p** losuje z najmniejszego zbioru tokenów o łącznym prawdopodobieństwie ≥ `p`.

| | Parametry |
|---|---|
| A | `T=0.7, top-k=50` |
| B | `T=1.0, top-p=0.9` |
| C | `T=1.0, top-k=10` |
| D | `T=1.3, top-p=0.95` |
| E | `T=1.0`, bez obcinania (kontrola) |

`top_k=0` ustawiamy jawnie: model nie ma własnej konfiguracji generacji, a domyślnie biblioteka dokłada `top_k=50`, co po cichu zmieniłoby B, D i E.

**Miary** (te same prompty dla każdego ustawienia):
- **distinct-2:** różnorodność, czyli odsetek unikalnych par słów,
- **rep-4:** pętle, czyli odsetek powtórzonych 4-gramów w tekście,
- **perplexity:** płynność, liczona pod bazowym GPT-2 (im mniej, tym bardziej „typowy” tekst),
- **zgodność etykiety:** odsetek tekstów, którym klasyfikator z 3a (`r=8`) przypisuje etykietę promptu.

Odniesienia: te same miary dla prawdziwej dalszej części tekstu oraz zgodność na **samym prefiksie**.

**Wyniki pilota (64 prompty):**

| | distinct-2 | rep-4 | perplexity | zgodność |
|---|---|---|---|---|
| A: `T=0.7, k=50` | 0,79 | 0,046 | 6 | **0,77** |
| B: `T=1.0, p=0.9` | 0,94 | 0,001 | 19 | 0,63 |
| C: `T=1.0, k=10` | 0,84 | 0,014 | 8 | 0,69 |
| D: `T=1.3, p=0.95` | 1,00 | 0 | **456** | 0,70 |
| E: bez obcinania | 0,97 | 0 | 42 | 0,59 |
| *prawdziwy tekst* | 0,94 | 0,003 | 36 | 0,91 |
| *sam prefiks* | | | | 0,64 |

Widać oczekiwany kompromis. Niska temperatura (A, C) daje płynne teksty z powtórzeniami (przykład z C: lista dat i godzin „12:00 – 12:40… 12:00 – 13:00…”). Wysoka (D) daje bełkot („*zastała brak pieniedzy w czesne. zakupyami…*”). B ma różnorodność i powtórzenia na poziomie prawdziwego tekstu.

### 8.3 Reguła wyboru ustawienia i lekcja z pilota

Pełny run liczy się w nocy, więc ustawienie musi wybrać **reguła w kodzie**. Pierwsza wersja brzmiała: „najwyższa zgodność etykiety spośród ustawień z distinct-2 ≥ mediana”. **W pilocie wybrała D, czyli bełkot.** Dwa powody:
- bełkot jest „bardzo różnorodny”, więc przechodzi warunek distinct-2,
- klasyfikator ocenia głównie po prefiksie (sam prefiks daje już zgodność 0,64), a bełkotliwa kontynuacja niewiele zmienia.

**Poprawiona reguła (w pełnym runie):**
1. odrzuć ustawienia z perplexity > 2× perplexity prawdziwego tekstu,
2. z pozostałych zostaw te z distinct-2 ≥ mediana,
3. wybierz najwyższą zgodność etykiety.

Na danych z pilota wybiera **B**. Wniosek do sprawozdania: **każdą pojedynczą miarę jakości generacji da się „oszukać”**, więc potrzebne są co najmniej płynność, różnorodność i zgodność razem.

### 8.4 Augmentacja i trening (5b, 5c)

- Wybranym ustawieniem generujemy **2 kopie** każdego tekstu treningowego (w pilocie 1). To tyle samo co augmentacja `[MASK]` u BERT-a, więc porównanie jest uczciwe.
- Odrzucamy kontynuacje krótsze niż 20 tokenów i duplikaty. **Nie filtrujemy klasyfikatorem:** zbiór stałby się echem klasyfikatora, a model uczyłby się jego błędów. Zgodność tylko raportujemy (tabela krzyżowa „etykieta promptu × predykcja”).
- Test **nigdy** nie jest augmentowany. Wynik zapisujemy w `data/aug_gpt2.jsonl` (format wspólny z BERT-em: `text`, `label`).
- Trenujemy **jeden model** (LoRA `r=8`, zgodnie z uwagą w poleceniu) na `train + aug` i porównujemy z `r=8` bez augmentacji na tym samym teście.

**Wyniki pilota (z bełkotliwym ustawieniem D):** zgodność całej augmentacji 0,57. Najwięcej etykiet „gubi” `amb` (0,38), najmniej `zero` (0,70). Test F1: 0,843 → 0,844, czyli bez zmian. U BERT-a `[MASK]` dał +0,007. W pełnym runie, z ustawieniem B, zobaczymy, czy lepsza generacja coś zmienia.

---

## 9. Podsumowanie: GPT-2 vs HerBERT

Ostatnia sekcja notebooka zestawia w jednej tabeli i na wykresie wyniki BERT-a (przepisane z `bert.ipynb`) i GPT-2: F1 tekstu i tokenów per `r`, silhouette, liczbę trenowanych parametrów, efekt augmentacji oraz czasy eksperymentów.

Główne pytania do wniosków:
1. Tekst: czy dwukierunkowy enkoder (BERT) wygrywa z przyczynowym dekoderem (GPT-2)? Pilot: LoRA nieco gorsza, baseline wyraźnie lepszy.
2. Tokeny: czy GPT-2 przegrywa tam, gdzie potrzebny jest prawy kontekst (negacja, granice fraz)?
3. Przestrzeń: czy strojenie porządkuje ją w obu modelach tak samo (kształt skupisk, kolejność `minus_m` → `amb` → `plus_m`)?
4. Augmentacja: `[MASK]` (mała, kontrolowana zmiana) vs generacja (nowy tekst, ryzyko zmiany sentymentu): która daje więcej, a która jest bezpieczniejsza dla etykiet?

---

## 10. Zgodność z poleceniem: checklista

| Wymóg z polecenia | Jak spełniony |
|---|---|
| model polskojęzyczny, np. `sdadas/polish-gpt2-medium` | ten model |
| `GPT2ForSequenceClassification` i `GPT2ForTokenClassification` | 3a, 3b |
| adapter PEFT | LoRA (`r = 4, 8, 16`) + baseline bez adaptera |
| wydzielony zbiór testowy, ocena | wspólny test z BERT-em; accuracy i macro-F1, raport per klasa i tag, macierz pomyłek |
| reprezentacja = ostatni token sekwencji | 3c: ostatni prawdziwy token; assert, że to ten sam token co u głowicy |
| padding i `<eos>` → `-100` | 3b: `</s>` dopisany jawnie, asserty na batchu |
| interaktywne wykresy przed i po + opis | 3c: t-SNE (Plotly) dla bazowego i 3 dostrojonych modeli + silhouette; opis w „Wnioskach” |
| generacja z różnymi temperatura / top-p / top-k | 5a: 5 ustawień, 4 miary, przykłady tekstów |
| trening na powiększonym zbiorze | 5c: jeden model (`r=8`) na `train + aug`, porównanie z `r=8` bez aug |

**Decyzje, które warto umieć uzasadnić:**
- **Nie łączymy augmentacji BERT + GPT-2** (polecenie na to pozwala, ale nie wymaga). Każda jest oceniana osobno, co daje czystsze porównanie.
- **Klasyfikacja tokenów na 35 tekstach** to ograniczenie danych z Zadania 1, wspólne z osobą A. Wyniki są niestabilne i piszemy to wprost.
- W poleceniu Części II nazwa modelu ma literówkę (`polishgpt2-medium`). Poprawna to `polish-gpt2-medium`.

---

## 11. Pytania, które mogą paść

**Dlaczego nie bierzemy pierwszego tokenu, jak `[CLS]` u BERT-a?** W GPT-2 pierwszy token widzi tylko siebie, bo maska przyczynowa nie pozwala mu patrzeć w prawo. Cały tekst „zna” dopiero ostatni.

**Dlaczego w 3a nie dopisujemy `</s>`, skoro w 3b dopisujemy?** W 3a głowica i tak by go pominęła (`pad == eos`, a model szuka ostatniego tokenu różnego od `pad`). Wizualizacja patrzyłaby wtedy na inny token niż klasyfikator. W 3b polecenie każe zamaskować `<eos>`, więc musi on w sekwencji być.

**Dlaczego accuracy w tokenach jest bez sensu?** 96% tokenów to `O`. Model, który zawsze mówi `O`, ma 0,97 accuracy. Dlatego F1 liczymy bez `O`.

**Czy batch 4 zamiast 16 zmienia wyniki?** Nie: gradient accumulation 4 daje ten sam efektywny batch. Kosztuje tylko czas.

**Dlaczego adapter GPT-2 ma więcej parametrów przy tym samym `r`?** Ma 2× więcej warstw, a `c_attn` obejmuje Q, K i V (u BERT-a LoRA jest tylko na Q i V). Względem rozmiaru modelu udział jest podobny.

**Dlaczego nie odfiltrować wygenerowanych tekstów klasyfikatorem, skoro mamy zgodność?** Zostałyby tylko teksty, które klasyfikator już dobrze rozpoznaje, więc nowy model nie nauczyłby się niczego nowego i powielałby błędy starego.

**Dlaczego perplexity D jest taka wysoka, skoro generował ją ten sam model?** Przy `T=1.3` model często losuje tokeny, które sam uważa za mało prawdopodobne, a perplexity liczymy przy `T=1`. Z tego samego powodu A (`T=0.7`) ma perplexity *niższą* niż prawdziwy tekst. To miara typowości, a nie jakości.

**Czy wyniki pilota można porównać z BERT-em?** Tylko orientacyjnie: pilot ma 30% danych i 2 epoki. Test jest jednak ten sam, więc kierunki (np. baseline GPT-2 > BERT) są już widoczne.

---

## 12. Po pełnym runie: do zrobienia

1. Otworzyć `gpt2_full.ipynb` i sprawdzić, które ustawienie wybrała reguła w 5a (na pilocie: B).
2. Podmienić liczby z pilota w tym dokumencie.
3. Uzupełnić komórki **„Wnioski 3a / 3b / 3c / 5 / końcowe”**. Każda ma listę pytań prowadzących.
4. Zweryfikować trzy obserwacje z pilota: baseline GPT-2 > BERT, słaba `B-negacja`, brak zysku z augmentacji.
