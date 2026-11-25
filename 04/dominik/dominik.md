# Dominik: Causal Tracing na BoolQ (Część I, plan pkt 3)

Zakres (3 pkt): plan realizacji pkt 3 z [instrukcji](../Zadanie_4.md). Modele: **gpt2-xl** i **utahnlp/boolq_gpt2-xl_seed-1**.

## Co jest gotowe, a co piszesz sam

**Możesz importować** [`../utils/tracing/`](../utils/tracing/) (gotowe od początku, nie czekasz na nikogo):
`trace_stream` / `trace_all_streams` (pętla warstwy × pozycje × strumienie, szum na osadzeniach, przywracanie stanów, zapis CSV), `find_token_positions`, `TraceResult`, `position_summary`, `streams_summary`, wykresy z `utils.tracing.plot`. API przyjmuje **dowolne pozycje do zaszumienia** (`noise_positions`), **dowolne pozycje do przywracania** (`restore_positions`) i **dowolną funkcję celu** (`score_fn`).

**Piszesz sam** (to jest „modyfikacja kodu” z pkt 3a–b i za to są Twoje punkty), w swoim folderze, najlepiej w module `dominik/boolq.py` zamiast w notebooku:

1. **Wczytanie BoolQ** (`google/boolq`, split `validation`) i wybór **8–10 przykładów**: krótkie pytania z wyraźną jednostką nazewniczą (np. „is **the Statue of Liberty** in New York”), mieszanka odpowiedzi True/False. Uwaga: pytania w BoolQ są **małymi literami i bez znaku zapytania** (`"is the statue of liberty in new york"`). `find_token_positions` rozróżnia wielkość liter, więc albo szukaj podmiotu małymi literami, albo popraw wielkość liter w pytaniu (i opisz to).
2. **Format wejścia** dla obu modeli (ustalone):
   - **gpt2-xl** (głowica LM): samo pytanie, bez fragmentu (passage), np. `"Question: {question}? Answer:"`. Miara: P(" Yes") i P(" No") z logitów następnego tokenu (uwaga na spację i wielką literę; sprawdź `tok.encode(" Yes")`). Dodatkowo `p_yes_norm = P(Yes) / (P(Yes) + P(No))`.
   - **utahnlp** (`GPT2ForSequenceClassification`, `id2label = {0: False, 1: True}`): nie ma tokenów Yes/No. „P(Yes)” to softmax po 2 klasach z `outputs.logits`. Model card nie podaje formatu treningu, więc **najpierw sprawdź 2–3 warianty** (np. `passage + "\n" + question`, `question + "\n" + passage`, `"question: ... passage: ..."`) na ok. 50 przykładach z walidacji i weź ten z najlepszą dokładnością. Zapisz w notatce, który wygrał i z jaką dokładnością.
3. **`score_fn`** dla każdego modelu, zwracające `{"p_yes": ..., "p_no": ...}` (kontrakt w [README](../README.md#kontrakt-score_fn)).
4. **Wybór tokenów**: zaszumiasz głównie tokeny **przedmiotu pytania** (jednostka nazewnicza, o którą pytamy) przez `find_token_positions(tok, text, "Statue of Liberty")`. Kod ma przyjmować **dowolny przykład i dowolne zaszumione tokeny** (funkcja typu `trace_boolq(example, model_kind, noise_substring, ...)`).
5. **Przywracane pozycje**: przy utahnlp wejście z passage ma ok. 150 tokenów, a pełny tracing to 48 warstw × 150 pozycji × 3 strumienie ≈ 20 tys. przebiegów gpt2-xl na przykład. Przywracaj tylko **tokeny pytania + ostatni token** (`restore_positions=[...]`). Do szybkich prób użyj `layers=range(0, 48, 2)`.

## Do zrobienia

1. Moduł `boolq.py`: dane, formaty, `score_fn`, `trace_boolq(...)`.
2. Sprawdzenie formatu utahnlp (dokładność na ok. 50 przykładach) i bazowej trafności gpt2-xl (P(Yes) vs P(No)).
3. Tracing 8–10 przykładów × 2 modele × 3 strumienie. Wyniki zapisuj przez `out_dir=artifacts_dir("dominik")` (cache, nie liczysz dwa razy).
4. Wykresy (`plot_trace_streams(results, key="p_yes")`, `plot_trace_grid(...)`), końcowe PNG do `dominik/figures/`.
5. **Porównanie modeli** (gpt2-xl vs utahnlp) i **porównanie z pkt 2** (fakty Mikołaja). Wyniki Mikołaja wczytasz przez `load_trace_streams(artifacts_dir("mikolaj"), "<nazwa_faktu>")`. Do tego czasu możesz sam puścić Space Needle (`next_token_prob(tok, " Seattle")`), żeby mieć punkt odniesienia.
6. Odpowiedzi na pytania pomocnicze:
   - Dlaczego wzorce dla `boolq_gpt2-xl_seed-1` różnią się od gpt2-xl? Na czym polegają różnice? (np. dostrojenie całego modelu pod klasyfikację, głowica `score` na ostatnim tokenie, inne wejście z passage)
   - Czym różnią się wzorce od tych z pkt 2? (pytanie tak/nie a przywołanie obiektu faktu; gdzie jest szczyt IE: ostatni token podmiotu czy ostatni token wejścia)

## Szkic

```python
from utils.infra import load_gpt2_xl, load_boolq_classifier, artifacts_dir
from utils.tracing import find_token_positions, trace_all_streams
from utils.tracing.plot import plot_trace_streams

clf, tok = load_boolq_classifier()
text = f"{passage}\n{question}?"                    # format ustalony w kroku 2
noise = find_token_positions(tok, text, "Statue of Liberty", occurrence=1)  # wystąpienie w pytaniu
q_start = len(tok(passage + "\n")["input_ids"])
restore = list(range(q_start, len(tok(text)["input_ids"])))

def clf_score(outputs):
    p = outputs.logits[0].float().softmax(-1)
    return {"p_yes": float(p[1]), "p_no": float(p[0])}

results = trace_all_streams(clf, tok, text, noise, clf_score, restore_positions=restore,
                            out_dir=artifacts_dir("dominik"), name="liberty__utahnlp")
plot_trace_streams(results, key="p_yes")
```

Uwaga: gpt2-xl i klasyfikator to razem ok. 13 GB. Na Macu ładuj je po kolei (`del model; gc.collect(); torch.mps.empty_cache()`).
