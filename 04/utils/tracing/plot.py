"""Wykresy causal tracingu w stylu ROME (Rys. 1 artykułu / notebook #2).

Kolory jak w notebooku: blok fioletowy, MLP zielony, atencja czerwona.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..plot import plot_layer_position_heatmap, save_figure
from .core import TraceResult

STREAM_COLORS = {
    "block_output": "Purples",
    "mlp_activation": "Greens",
    "attention_output": "Reds",
}

STREAM_TITLES = {
    "block_output": "przywrócona pojedyncza warstwa (blok)",
    "mlp_activation": "środek okna {w} przywróconych warstw MLP",
    "attention_output": "środek okna {w} przywróconych warstw atencji",
}


def plot_trace(
    result: TraceResult,
    key: str | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Axes:
    """Heatmapa jednego strumienia: warstwa × token, kolor = miara po przywróceniu."""
    key = key or result.score_keys[0]
    xlabel = STREAM_TITLES.get(result.stream, result.stream).format(w=result.window)
    if title is None:
        title = (
            f"{result.stream}: clean={result.clean[key]:.3f}, corrupted={result.corrupted[key]:.3f}"
        )
    return plot_layer_position_heatmap(
        result.df,
        value=key,
        pos_labels=result.labels,
        cmap=STREAM_COLORS.get(result.stream, "viridis"),
        title=title,
        xlabel=xlabel,
        colorbar_label=key,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )


def plot_trace_streams(
    results: Mapping[str, TraceResult],
    key: str | None = None,
    *,
    suptitle: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Trzy heatmapy obok siebie (blok, MLP, atencja) dla jednego przykładu."""
    n = len(results)
    first = next(iter(results.values()))
    height = max(2.8, 0.32 * len(first.tokens) + 1.2)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, height), squeeze=False)
    for ax, res in zip(axes[0], results.values(), strict=True):
        plot_trace(res, key, ax=ax)
    fig.suptitle(suptitle or first.text, fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    return fig


def plot_trace_grid(
    examples: Mapping[str, Mapping[str, TraceResult]],
    key: str | None = None,
    *,
    streams: Sequence[str] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Siatka: wiersz = przykład, kolumna = strumień. Do zestawienia 5 faktów na jednym rysunku."""
    names = list(examples)
    streams = list(streams or next(iter(examples.values())).keys())
    max_tokens = max(len(next(iter(e.values())).tokens) for e in examples.values())
    row_h = max(2.6, 0.3 * max_tokens + 1.0)
    fig, axes = plt.subplots(
        len(names), len(streams), figsize=(5.2 * len(streams), row_h * len(names)), squeeze=False
    )
    for r, name in enumerate(names):
        for c, stream in enumerate(streams):
            res = examples[name][stream]
            plot_trace(res, key, ax=axes[r][c], title=f"{name} · {stream}")
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    return fig


def plot_groups(
    df: pd.DataFrame,
    key: str = "prob",
    *,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Wynik `trace_groups`: po jednym panelu na strumień, linia = grupa pozycji.

    Linie przerywane: wynik czysty (clean) i zaszumiony (corrupted).
    """
    streams = list(dict.fromkeys(df["stream"]))
    fig, axes = plt.subplots(1, len(streams), figsize=(5.2 * len(streams), 3.4), squeeze=False)
    for ax, stream in zip(axes[0], streams, strict=True):
        part = df[df["stream"] == stream]
        for group, g in part.groupby("group", sort=False):
            ax.plot(g["layer"], g[key], label=str(group))
        ax.axhline(part[f"clean_{key}"].iloc[0], ls="--", c="gray", lw=1, label="clean")
        ax.axhline(part[f"corrupted_{key}"].iloc[0], ls=":", c="gray", lw=1, label="corrupted")
        ax.set_title(stream, fontsize=10)
        window = 1 if stream == "block_output" else 10
        ax.set_xlabel(STREAM_TITLES.get(stream, stream).format(w=window))
        ax.set_ylabel(key)
        ax.set_ylim(0, 1)
    axes[0][0].legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    return fig
