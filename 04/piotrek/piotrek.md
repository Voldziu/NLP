# Piotrek: Sondowanie (Część II)

Zakres (4 pkt): Część II z [instrukcji](../Zadanie_4.md). **Nie zależysz od nikogo.**

## Materiały

1. Slajdy M. Iyyera o sondowaniu: [19-probes.pdf](https://people.cs.umass.edu/~miyyer/cs685_f20/slides/19-probes.pdf)
2. Artykuł o sondowaniu: [arXiv 2102.12452](https://arxiv.org/abs/2102.12452) (Belinkov, *Probing Classifiers: Promises, Shortcomings, and Advances*)
3. Notebook #3: [Probing_Gender.ipynb](https://github.com/stanfordnlp/pyvene/blob/main/tutorials/advanced_tutorials/Probing_Gender.ipynb). Szczególnie pętla zbierająca aktywacje (`pv.CollectIntervention` na `block_output`, każda warstwa × pozycja) i trening `LogisticRegression`.

Krótka notatka z 1–2 w notebooku: czym jest sonda, czemu prosta (liniowa), ograniczenia (wysoka dokładność ≠ model *używa* informacji, por. DAS/IIA w notebooku #3; zadania kontrolne / selectivity).

## Ustalenia

- **Model**: `EleutherAI/pythia-70m` (jak w notebooku #3; 6 warstw, szybki pełny przegląd warstw × pozycji). `load_causal_lm("EleutherAI/pythia-70m")` z `utils.infra`.
- **Zadania**: z `tasksource/babi_nli`, tylko przykłady z `label == 1` (entailed):
  1. **agents-motivations**: pomieszczenie z motywacji. Premisy są długie i mieszają kilka osób, więc **przykład budujesz z hipotezy**: `"Sumit went to the garden because she was bored."` → wejście `"Sumit is bored. Sumit will go to the"`, etykieta `garden`. Uwaga: `living room` → etykieta `living` (pierwsze słowo, jak w instrukcji).
  2. **compound-coreference**: premisa + początek hipotezy: `"John and Daniel went to the office. After that they went back to the kitchen. John is in the"`, etykieta `kitchen`.
- **Sonda**: `LogisticRegression` (wieloklasowa) na aktywacjach `block_output`, **dla każdej warstwy × pozycji**, dokładność (i macro-F1) na zbiorze testowym. DAS z notebooka #3 jest opcjonalny.
- **Pozycje**: wejścia mają różne długości, więc indeksuj pozycje **od końca** (np. ostatnie 4–6 tokenów: `... to the`, imię, ...) albo wyrównaj do wspólnego szablonu. W agents-motivations szablon jest stały (`<imię> is <motywacja>. <imię> will go to the`), więc pozycje mają stałe znaczenie.
- **Podział**: splity `train` / `test` z babi_nli (osobno), ziarno `42`.

## Do zrobienia

1. Moduł `piotrek/probing.py` (dane + zbieranie aktywacji + trening sondy), a w notebooku tylko wywołania:
   - `build_motivations_dataset(split)`, `build_coreference_dataset(split)` → listy `(text, label)`;
   - `collect_activations(model, tok, texts, layer, positions)` przez `pv.CollectIntervention` (jak w notebooku #3) albo `output_hidden_states=True` (prostsze, te same `block_output`; opisz, którego użyłeś);
   - `train_probe(X_train, y_train, X_test, y_test)` → dokładność, F1.
2. Statystyki danych: liczba przykładów, rozkład etykiet, **baseline klasy większościowej** (sonda musi go pobić).
3. Heatmapy dokładności warstwa × pozycja dla obu zadań: `plot_layer_position_heatmap(df, "acc", pos_labels=..., cmap="Reds")` z `utils.plot`. PNG do `piotrek/figures/`.
4. (Opcjonalnie) czy model sam przewiduje pomieszczenie jako następny token (top-k na wejściu), żeby zestawić z dokładnością sondy.
5. Analiza: w której warstwie / na którym tokenie informacja o pomieszczeniu staje się liniowo odczytywalna; różnica między zadaniami (motywacja → pomieszczenie to wiedza „zdroworozsądkowa”, koreferencja wymaga śledzenia osób w kontekście); ograniczenia sondowania.
