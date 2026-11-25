"""Causal tracing (ROME) na pyvene: zaszumienie podmiotu i przywracanie stanów ukrytych.

Wykresy: `utils.tracing.plot`.
"""

from .analysis import peak, position_summary, streams_summary
from .core import (
    TraceResult,
    load_trace_streams,
    run_clean,
    run_corrupted,
    trace_all_streams,
    trace_stream,
)
from .interventions import (
    DEFAULT_WINDOW,
    NOTEBOOK_NOISE_LEVEL_GPT2_XL,
    STREAMS,
    compute_noise_level,
    corrupted_config,
    make_noise_intervention,
    restore_config,
)
from .scores import ScoreFn, next_token_prob, target_token_id, top_next_tokens
from .tokens import find_token_positions, position_labels, tokenize

__all__ = [
    "DEFAULT_WINDOW",
    "NOTEBOOK_NOISE_LEVEL_GPT2_XL",
    "STREAMS",
    "ScoreFn",
    "TraceResult",
    "compute_noise_level",
    "corrupted_config",
    "find_token_positions",
    "load_trace_streams",
    "make_noise_intervention",
    "next_token_prob",
    "peak",
    "position_labels",
    "position_summary",
    "restore_config",
    "run_clean",
    "run_corrupted",
    "streams_summary",
    "target_token_id",
    "tokenize",
    "top_next_tokens",
    "trace_all_streams",
    "trace_stream",
]
