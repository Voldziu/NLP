# Zadanie 4: Lokalizowanie wiedzy i sondowanie

Instrukcja: [Zadanie_4.md](Zadanie_4.md) (przerobiona z `.docx`). Oddanie: lab 11.

## Podział zadań

| Osoba | Zakres | Folder | Pkt |
|---|---|---|---|
| **Mikołaj** (A) | Część I pkt 1–3 (survey KE, ROME / Causal Tracing, pyvene 101) + notatki. Plan pkt 1: odtworzenie notebooka #2 (`Causal_Tracing.ipynb`, Space Needle). Plan pkt 2: 5 własnych faktów na **gpt2-xl**, zaszumiony tylko podmiot, wykresy `block_output` / `mlp_activation` / `attention_output`, odpowiedzi na pytania pomocnicze. | `mikolaj/` | 3 |
| **Dominik** (B) | Część I plan pkt 3: **BoolQ** na **gpt2-xl** i **utahnlp/boolq_gpt2-xl_seed-1**. Kod przyjmujący dowolny przykład i dowolne zaszumione tokeny, P(Yes) / P(No), 8–10 pytań, wykresy, porównanie modeli i porównanie z pkt 2. | `dominik/` | 3 |
| **Piotrek** (C) | Część II: sondowanie (notebook #3 `Probing_Gender.ipynb`) na 2 zadaniach z `tasksource/babi_nli`: **agents-motivations** i **compound-coreference** (`label == 1`), model **pythia-70m**, sonda regresją logistyczną dla każdej warstwy × pozycji. DAS opcjonalnie. | `piotrek/` | 4 |

## Zależności

- **Nikt nie czeka na start.** Kod causal tracingu (`utils/tracing`) jest gotowy od początku.
- **Dominik importuje `utils/tracing`** (pętla warstwy × pozycje × strumienie, szum, wykresy). Sam pisze część BoolQ: format wejścia, `score_fn` dla P(Yes)/P(No), obsługę klasyfikatora utahnlp, wybór tokenów do zaszumienia i do przywracania, analizę.
- **Jedna miękka zależność na końcu:** Dominik porównuje swoje wzorce z wynikami Mikołaja (pkt 2). Mikołaj zapisuje wyniki do `artifacts/mikolaj/*.csv|json`, a końcowe wykresy do `mikolaj/figures/`. Do tego czasu Dominik może sam puścić 1–2 fakty (np. Space Needle) tym samym kodem.
- Piotrek jest całkowicie niezależny.

## Zasady współpracy

- Wspólny kod trafia do `utils/` (małe moduły). Kod własny zostaje w swoim folderze.
- Notebooki mają być cienkie: większy kod przenosimy do modułów `.py`.
- `artifacts/` (w `.gitignore`): CSV, JSON i wykresy robocze, w podfolderze z imieniem (`artifacts_dir("dominik")`).
- Wykresy do raportu: `<imię>/figures/*.png` (w repo).
- Ziarno `42` (`set_seed()`); szum w tracingu ma własne ziarno `seed=1` (jak w notebooku #2).

## Środowisko

```bash
uv sync   # z korzenia repo; dochodzą pyvene i plotnine
```

Przy wolnej sieci `uv` może przerwać pobieranie po 30 s. Pomaga `UV_HTTP_TIMEOUT=600 uv sync`.

gpt2-xl (i klasyfikator utahnlp) to ok. 6,5 GB każdy w float32. Mieszczą się na Macu z 16+ GB (MPS). Pierwsze pobranie z HF trwa kilka minut.

Notebooki leżą w podfolderach `04/`, więc na początku:

```python
import sys; sys.path.insert(0, "..")
from utils.infra import get_device, set_seed, load_gpt2_xl, load_boolq_classifier, artifacts_dir
from utils.tracing import find_token_positions, next_token_prob, trace_all_streams
from utils.tracing.plot import plot_trace_streams
```

## `utils/`

| Moduł | Co robi |
|---|---|
| `infra/` | `get_device()` (cuda → mps → cpu), `set_seed(42)`, `load_causal_lm(name)`, `load_gpt2_xl()`, `load_boolq_classifier()`, `artifacts_dir("imię")`, `LAB_DIR`. |
| `plot/` | `plot_layer_position_heatmap(df, value, pos_labels=..., cmap=...)`: ogólna heatmapa warstwa × pozycja (tracing i dokładność sondy), `save_figure(fig, path)`. |
| `tracing/` | Causal tracing na pyvene. `find_token_positions(tok, text, "podmiot")`, `next_token_prob(tok, " Seattle")` (`score_fn`), `trace_stream(...)` / `trace_all_streams(...)` → `TraceResult` (`df`, `clean`, `corrupted`, `save`/`load`), `load_trace_streams(dir, name)`, `position_summary`, `streams_summary`, `top_next_tokens`, `compute_noise_level`, `NOTEBOOK_NOISE_LEVEL_GPT2_XL`. |
| `tracing/plot.py` | `plot_trace(result)`, `plot_trace_streams(results)` (3 heatmapy obok siebie), `plot_trace_grid({nazwa: results})` (siatka przykładów × strumieni). |

### Kontrakt `score_fn`

`score_fn(outputs) -> dict[str, float]` dostaje wyjście modelu po interwencji. Każdy klucz staje się kolumną w `TraceResult.df`. Na wykresie wybierasz kolumnę przez `key=...`.

```python
# LM (gpt2-xl): P(" Seattle") na ostatniej pozycji
score_fn = next_token_prob(tok, " Seattle")            # -> {"prob": ...}

# klasyfikator BoolQ (Dominik pisze własny)
def boolq_score(outputs):
    p = outputs.logits[0].float().softmax(-1)         # [P(False), P(True)]
    return {"p_yes": float(p[1]), "p_no": float(p[0])}
```

pyvene obsługuje `GPT2LMHeadModel`, `GPT2ForSequenceClassification` i `GPTNeoXForCausalLM` (pythia), więc ta sama pętla działa dla obu modeli z BoolQ.
