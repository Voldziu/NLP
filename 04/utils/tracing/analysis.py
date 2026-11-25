"""Podsumowania wyników tracingu: gdzie fakt „siedzi”.

Odpowiadają na pytania pomocnicze: które tokeny podmiotu mają największy wpływ
i w których warstwach, oraz czym różnią się strumienie.
"""

from collections.abc import Mapping

import pandas as pd

from .core import TraceResult


def position_summary(result: TraceResult, key: str | None = None) -> pd.DataFrame:
    """Dla każdej pozycji: maksymalny efekt pośredni (IE) i warstwa, w której wystąpił.

    Posortowane malejąco po `max_ie`; kolumna `noised` mówi, czy token był zaszumiony.
    """
    df = result.indirect_effect(key)
    idx = df.groupby("pos")["ie"].idxmax()
    out = df.loc[idx, ["pos", "token", "layer", "ie"]].rename(
        columns={"layer": "best_layer", "ie": "max_ie"}
    )
    out["noised"] = out["pos"].isin(result.noised)
    return out.sort_values("max_ie", ascending=False).reset_index(drop=True)


def peak(result: TraceResult, key: str | None = None) -> pd.Series:
    """Komórka (warstwa, pozycja) z największym efektem pośrednim."""
    df = result.indirect_effect(key)
    return df.loc[df["ie"].idxmax(), ["layer", "pos", "token", "ie"]]


def streams_summary(results: Mapping[str, TraceResult], key: str | None = None) -> pd.DataFrame:
    """Jeden wiersz na strumień: wynik czysty, zaszumiony, szczyt IE i jego położenie.

    Pozwala porównać `block_output`, `mlp_activation` i `attention_output` w tabeli.
    Rozróżnia dwa obszary z ROME: ostatni token podmiotu (`last_subject`) i ostatni token
    całego wejścia (`last_token`).
    """
    rows = []
    for stream, res in results.items():
        k = key or res.score_keys[0]
        ie = res.indirect_effect(k)
        last_subject = max(res.noised)
        last_token = len(res.tokens) - 1
        p = peak(res, k)
        rows.append(
            {
                "stream": stream,
                "clean": res.clean[k],
                "corrupted": res.corrupted[k],
                "peak_ie": p["ie"],
                "peak_layer": int(p["layer"]),
                "peak_token": p["token"],
                "max_ie_last_subject": ie.loc[ie["pos"] == last_subject, "ie"].max(),
                "max_ie_last_token": ie.loc[ie["pos"] == last_token, "ie"].max(),
            }
        )
    return pd.DataFrame(rows)
