"""Interwencje pyvene do causal tracingu.

Schemat z notebooka #2 (`Causal_Tracing.ipynb`) i artykułu ROME:

1. **Przebieg zaszumiony**: do osadzeń tokenów podmiotu (`block_input` warstwy 0)
   dodajemy szum gaussowski. Model traci fakt (spada P(" Seattle")).
2. **Przebieg z przywróceniem**: jak wyżej, ale w jednej warstwie i na jednej pozycji
   wstawiamy stan ukryty z czystego przebiegu (`VanillaIntervention`, źródło = to samo wejście).
   Jeśli P wraca, ten stan niesie fakt.

Dla `mlp_activation` i `attention_output` przywracamy okno 10 kolejnych warstw
(środek okna = `layer`), dla `block_output` jedną warstwę, jak w ROME.
"""

from typing import Any

import pyvene as pv
import torch
from transformers import PreTrainedModel

STREAMS = ("block_output", "mlp_activation", "attention_output")
"""Strumienie analizowane w ROME: wyjście bloku, aktywacja MLP, wyjście atencji."""

DEFAULT_WINDOW = {"block_output": 1, "mlp_activation": 10, "attention_output": 10}

NOTEBOOK_NOISE_LEVEL_GPT2_XL = 0.13462981581687927
"""Stała z notebooka #2 (3 × std osadzeń gpt2-xl, policzona przez autorów ROME na korpusie)."""


def compute_noise_level(model: PreTrainedModel, multiplier: float = 3.0) -> float:
    """Poziom szumu = `multiplier` × odchylenie std macierzy osadzeń tokenów.

    ROME liczy std osadzeń podmiotów z korpusu; std całej macierzy `wte` daje wartość
    tego samego rzędu. Dla reprodukcji notebooka użyj `NOTEBOOK_NOISE_LEVEL_GPT2_XL`.
    """
    emb = model.get_input_embeddings().weight
    return float(multiplier * emb.detach().float().std().item())


def make_noise_intervention(noise_level: float, seed: int = 1) -> type[pv.Intervention]:
    """Tworzy klasę interwencji dodającej stały szum gaussowski.

    pyvene sam tworzy instancje interwencji (podaje `embed_dim` itd.), dlatego parametry
    przekazujemy przez klasę. Szum zależy tylko od `seed` i kształtu, więc każdy przebieg
    (zaszumiony i wszystkie przywrócenia) widzi **ten sam** szum, tak jak w notebooku.
    """

    class NoiseIntervention(pv.ConstantSourceIntervention, pv.LocalistRepresentationIntervention):  # type: ignore[misc]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.noise_level = noise_level
            self._cache: dict[tuple[int, ...], torch.Tensor] = {}

        def _noise(self, shape: tuple[int, ...]) -> torch.Tensor:
            if shape not in self._cache:
                gen = torch.Generator().manual_seed(seed)
                self._cache[shape] = torch.randn(shape, generator=gen, dtype=torch.float32)
            return self._cache[shape]

        def forward(
            self, base: torch.Tensor, source: Any = None, subspaces: Any = None
        ) -> torch.Tensor:
            # base: [batch, liczba_zaszumionych_pozycji, d_model]
            noise = self._noise((1, *base.shape[1:])).to(base.device, base.dtype)
            return base + noise * self.noise_level

        def __str__(self) -> str:
            return f"NoiseIntervention(noise_level={self.noise_level:.4f}, seed={seed})"

    return NoiseIntervention


def corrupted_config(noise: type[pv.Intervention]) -> pv.IntervenableConfig:
    """Konfiguracja: tylko szum na osadzeniach (warstwa 0, `block_input`)."""
    return pv.IntervenableConfig(
        representations=[pv.RepresentationConfig(0, "block_input")],
        intervention_types=noise,
    )


def restore_window(layer: int, window: int, num_layers: int) -> range:
    """Warstwy przywracane dla środka okna `layer` (obcięte do [0, num_layers))."""
    start = max(0, layer - window // 2)
    end = min(num_layers, layer - (-window // 2))
    return range(start, end)


def restore_config(
    noise: type[pv.Intervention],
    layer: int,
    stream: str,
    window: int,
    num_layers: int,
) -> tuple[pv.IntervenableConfig, int]:
    """Konfiguracja: szum na osadzeniach + przywrócenie `stream` w oknie wokół `layer`.

    Zwraca `(config, liczba_przywróceń)`.
    """
    layers = restore_window(layer, window, num_layers)
    config = pv.IntervenableConfig(
        representations=[pv.RepresentationConfig(0, "block_input")]
        + [pv.RepresentationConfig(i, stream) for i in layers],
        intervention_types=[noise] + [pv.VanillaIntervention] * len(layers),
    )
    return config, len(layers)
