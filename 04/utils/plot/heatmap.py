"""Ogólna heatmapa warstwa × pozycja (matplotlib).

Wspólna dla wszystkich: tracing (prawdopodobieństwo po przywróceniu stanu)
i sondowanie (dokładność sondy na danej warstwie i pozycji).
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_layer_position_heatmap(
    df: pd.DataFrame,
    value: str,
    *,
    layer_col: str = "layer",
    pos_col: str = "pos",
    pos_labels: Sequence[str] | None = None,
    cmap: str = "Purples",
    title: str | None = None,
    xlabel: str = "warstwa",
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Rysuje heatmapę: oś X to warstwy, oś Y to pozycje tokenów (od góry).

    `df` w formacie długim: jeden wiersz na parę (`layer_col`, `pos_col`) z kolumną `value`.
    `pos_labels[i]` to etykieta pozycji `i` (np. token, z `*` przy zaszumionych).
    """
    grid = df.pivot_table(index=pos_col, columns=layer_col, values=value, aggfunc="mean")
    grid = grid.sort_index().sort_index(axis=1)

    if ax is None:
        height = max(2.5, 0.35 * len(grid.index) + 1.0)
        _, ax = plt.subplots(figsize=(6, height))

    mesh = ax.imshow(
        grid.to_numpy(),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(grid.index)))
    if pos_labels is not None:
        ax.set_yticklabels([pos_labels[int(p)] for p in grid.index])
    else:
        ax.set_yticklabels([str(p) for p in grid.index])

    layers = list(grid.columns)
    step = max(1, len(layers) // 8)
    ax.set_xticks(range(0, len(layers), step))
    ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), step)])
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=10)

    cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(colorbar_label or value)
    return ax


def save_figure(fig: Figure, path: str | Path, dpi: int = 200) -> Path:
    """Zapisuje wykres (PNG/PDF po rozszerzeniu), tworząc katalog."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
