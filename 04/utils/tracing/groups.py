"""Przywracanie **grupy tokenów naraz** (rozszerzenie causal tracingu z ROME).

`trace_stream` przywraca jedną pozycję na raz. Gdy informacja o podmiocie jest rozłożona
na kilka tokenów (np. „The E iff el Tower”), przywrócenie jednego z nich może nic nie dać,
bo reszta zostaje zaszumiona. Tu w każdej warstwie przywracamy wskazany zbiór pozycji
jednocześnie i porównujemy grupy (np. cały podmiot vs ostatni token podmiotu).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyvene as pv
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .core import _encode, _num_layers, run_clean, run_corrupted
from .interventions import (
    DEFAULT_WINDOW,
    STREAMS,
    compute_noise_level,
    make_noise_intervention,
    restore_config,
)
from .scores import ScoreFn


@torch.no_grad()
def trace_groups(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    noise_positions: Sequence[int],
    score_fn: ScoreFn,
    groups: Mapping[str, Sequence[int]],
    streams: Sequence[str] = STREAMS,
    *,
    noise_level: float | None = None,
    seed: int = 1,
    out_path: str | Path | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Dla każdej grupy pozycji, strumienia i warstwy: wynik po przywróceniu całej grupy naraz.

    Zwraca `DataFrame` z kolumnami `group`, `stream`, `layer` + klucze `score_fn`
    oraz `clean_<klucz>` i `corrupted_<klucz>` (stałe, do rysowania linii odniesienia).
    Okna warstw jak w `trace_stream` (blok: 1, MLP i atencja: 10).
    Z `out_path` wynik jest zapisywany do CSV i wczytywany przy kolejnym uruchomieniu.
    """
    if out_path is not None:
        out_path = Path(out_path).with_suffix(".csv")
        if out_path.exists() and not overwrite:
            return pd.read_csv(out_path)

    level = compute_noise_level(model) if noise_level is None else noise_level
    noise = make_noise_intervention(level, seed)
    num_layers = _num_layers(model)
    base = _encode(tokenizer, text, model)
    noised = [int(p) for p in noise_positions]
    clean = run_clean(model, tokenizer, text, score_fn)
    corrupted = run_corrupted(model, tokenizer, text, noised, score_fn, level, seed)

    rows: list[dict[str, Any]] = []
    for stream in streams:
        window = DEFAULT_WINDOW.get(stream, 1)
        for layer in tqdm(range(num_layers), desc=stream, disable=not progress):
            config, n_restores = restore_config(noise, layer, stream, window, num_layers)
            intervenable = pv.IntervenableModel(config, model)
            for group, positions in groups.items():
                pos = [int(p) for p in positions]
                _, outputs = intervenable(
                    base,
                    [None] + [base] * n_restores,
                    {
                        "sources->base": (
                            [None] + [[pos]] * n_restores,
                            [[noised]] + [[pos]] * n_restores,
                        )
                    },
                )
                rows.append({"group": group, "stream": stream, "layer": layer, **score_fn(outputs)})

    df = pd.DataFrame(rows)
    for key in clean:
        df[f"clean_{key}"] = clean[key]
        df[f"corrupted_{key}"] = corrupted[key]
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


def groups_summary(df: pd.DataFrame, key: str = "prob") -> pd.DataFrame:
    """Tabela: dla każdej grupy i strumienia maksimum wyniku i warstwa, w której wystąpiło."""
    idx = df.groupby(["group", "stream"])[key].idxmax()
    out = df.loc[idx, ["group", "stream", "layer", key]].rename(
        columns={"layer": "best_layer", key: f"max_{key}"}
    )
    return out.pivot(index="group", columns="stream", values=[f"max_{key}", "best_layer"])
