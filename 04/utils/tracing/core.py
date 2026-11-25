"""Pętla causal tracingu: warstwy × pozycje × strumienie.

Typowe użycie (fakt z notebooka #2):

```python
model, tok = load_gpt2_xl()
text = "The Space Needle is in downtown"
subject = find_token_positions(tok, text, "The Space Needle")       # [0, 1, 2, 3]
results = trace_all_streams(model, tok, text, subject, next_token_prob(tok, " Seattle"),
                            out_dir=artifacts_dir("mikolaj"), name="space_needle")
plot_trace_streams(results)
```

Działa dla każdego modelu obsługiwanego przez pyvene (GPT-2 z głowicą LM,
`GPT2ForSequenceClassification`, GPT-NeoX...). Model wybiera `score_fn`.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyvene as pv
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .interventions import (
    DEFAULT_WINDOW,
    STREAMS,
    compute_noise_level,
    corrupted_config,
    make_noise_intervention,
    restore_config,
)
from .scores import ScoreFn
from .tokens import position_labels, tokenize


@dataclass
class TraceResult:
    """Wynik tracingu jednego strumienia dla jednego przykładu.

    `df`: kolumny `layer`, `pos`, `token` + po jednej kolumnie na klucz `score_fn`
    (np. `prob`), wartość po przywróceniu stanu z czystego przebiegu.
    `clean` / `corrupted`: te same miary bez interwencji i z samym szumem.
    """

    text: str
    tokens: list[str]
    noised: list[int]
    stream: str
    window: int
    noise_level: float
    clean: dict[str, float]
    corrupted: dict[str, float]
    df: pd.DataFrame = field(repr=False)

    @property
    def labels(self) -> list[str]:
        """Etykiety pozycji z `*` przy zaszumionych tokenach."""
        return position_labels(self.tokens, self.noised)

    @property
    def score_keys(self) -> list[str]:
        return list(self.clean)

    def indirect_effect(self, key: str | None = None) -> pd.DataFrame:
        """`df` z kolumną `ie` = wynik po przywróceniu − wynik zaszumiony (efekt pośredni)."""
        key = key or self.score_keys[0]
        out = self.df.copy()
        out["ie"] = out[key] - self.corrupted[key]
        return out

    def save(self, path: str | Path) -> Path:
        """Zapisuje `<path>.csv` (siatka) i `<path>.json` (metadane)."""
        path = Path(path).with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path.with_suffix(".csv"), index=False)
        meta = {k: v for k, v in asdict(self).items() if k != "df"}
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TraceResult":
        """Wczytuje wynik zapisany przez `save` (podaj ścieżkę bez rozszerzenia lub z nim)."""
        path = Path(path).with_suffix("")
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(**meta, df=pd.read_csv(path.with_suffix(".csv")))


def _num_layers(model: PreTrainedModel) -> int:
    cfg = model.config
    return int(getattr(cfg, "n_layer", None) or cfg.num_hidden_layers)


def _encode(
    tokenizer: PreTrainedTokenizerBase, text: str, model: PreTrainedModel
) -> Mapping[str, torch.Tensor]:
    return tokenizer(text, return_tensors="pt").to(model.device)


@torch.no_grad()
def run_clean(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str, score_fn: ScoreFn
) -> dict[str, float]:
    """Wynik bez żadnej interwencji."""
    return dict(score_fn(model(**_encode(tokenizer, text, model))))


@torch.no_grad()
def run_corrupted(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    noise_positions: Sequence[int],
    score_fn: ScoreFn,
    noise_level: float | None = None,
    seed: int = 1,
) -> dict[str, float]:
    """Wynik z zaszumionymi osadzeniami na `noise_positions`, bez przywracania."""
    level = compute_noise_level(model) if noise_level is None else noise_level
    noise = make_noise_intervention(level, seed)
    intervenable = pv.IntervenableModel(corrupted_config(noise), model)
    _, outputs = intervenable(
        _encode(tokenizer, text, model),
        unit_locations={"base": [[list(noise_positions)]]},
    )
    return dict(score_fn(outputs))


@torch.no_grad()
def trace_stream(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    noise_positions: Sequence[int],
    score_fn: ScoreFn,
    stream: str = "block_output",
    *,
    window: int | None = None,
    restore_positions: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    noise_level: float | None = None,
    seed: int = 1,
    progress: bool = True,
) -> TraceResult:
    """Causal tracing jednego strumienia.

    - `noise_positions`: tokeny, których osadzenia zaszumiamy (podmiot).
    - `stream`: `block_output`, `mlp_activation`, `attention_output` (lub inny komponent pyvene).
    - `window`: liczba przywracanych warstw; domyślnie 1 dla bloku, 10 dla MLP i atencji.
    - `restore_positions`: pozycje, które przywracamy; domyślnie wszystkie. Zawężenie skraca
      czas przy długich wejściach (np. BoolQ: tylko pytanie + ostatni token).
    - `layers`: środki okien; domyślnie wszystkie warstwy.

    Koszt: `len(layers) × len(restore_positions)` przebiegów modelu.
    """
    window = DEFAULT_WINDOW.get(stream, 1) if window is None else window
    level = compute_noise_level(model) if noise_level is None else noise_level
    noise = make_noise_intervention(level, seed)
    num_layers = _num_layers(model)

    base = _encode(tokenizer, text, model)
    seq_len = int(base["input_ids"].shape[1])
    tokens = tokenize(tokenizer, text)
    noised = [int(p) for p in noise_positions]
    positions = list(range(seq_len)) if restore_positions is None else list(restore_positions)
    layer_ids = list(range(num_layers)) if layers is None else list(layers)

    clean = run_clean(model, tokenizer, text, score_fn)
    corrupted = run_corrupted(model, tokenizer, text, noised, score_fn, level, seed)

    rows: list[dict[str, Any]] = []
    for layer in tqdm(layer_ids, desc=stream, disable=not progress):
        config, n_restores = restore_config(noise, layer, stream, window, num_layers)
        intervenable = pv.IntervenableModel(config, model)
        for pos in positions:
            _, outputs = intervenable(
                base,
                [None] + [base] * n_restores,
                {
                    "sources->base": (
                        [None] + [[[pos]]] * n_restores,
                        [[noised]] + [[[pos]]] * n_restores,
                    )
                },
            )
            rows.append({"layer": layer, "pos": pos, "token": tokens[pos], **score_fn(outputs)})

    return TraceResult(
        text=text,
        tokens=tokens,
        noised=noised,
        stream=stream,
        window=window,
        noise_level=level,
        clean=clean,
        corrupted=corrupted,
        df=pd.DataFrame(rows),
    )


def trace_all_streams(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    noise_positions: Sequence[int],
    score_fn: ScoreFn,
    streams: Sequence[str] = STREAMS,
    *,
    out_dir: str | Path | None = None,
    name: str | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> dict[str, TraceResult]:
    """`trace_stream` dla kilku strumieni (domyślnie trzech z ROME).

    Z `out_dir` i `name` każdy wynik trafia do `out_dir/name__<stream>.{csv,json}`.
    Jeśli plik już istnieje, wynik jest wczytywany zamiast liczony (`overwrite=True` liczy od nowa).
    Pozostałe argumenty idą do `trace_stream`.
    """
    results: dict[str, TraceResult] = {}
    for stream in streams:
        path = Path(out_dir) / f"{name}__{stream}" if out_dir and name else None
        if path is not None and path.with_suffix(".csv").exists() and not overwrite:
            results[stream] = TraceResult.load(path)
            continue
        result = trace_stream(model, tokenizer, text, noise_positions, score_fn, stream, **kwargs)
        if path is not None:
            result.save(path)
        results[stream] = result
    return results


def load_trace_streams(
    out_dir: str | Path, name: str, streams: Sequence[str] = STREAMS
) -> dict[str, TraceResult]:
    """Wczytuje wyniki zapisane przez `trace_all_streams` (np. Dominik czyta CSV Mikołaja)."""
    return {s: TraceResult.load(Path(out_dir) / f"{name}__{s}") for s in streams}
