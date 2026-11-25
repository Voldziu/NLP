"""Funkcje celu (`score_fn`): co mierzymy na wyjściu modelu.

`score_fn(outputs) -> dict[str, float]` dostaje wyjście modelu (np. `CausalLMOutput`)
i zwraca jedną lub kilka liczb. Każdy klucz staje się kolumną w `TraceResult.df`.
Przykład dla BoolQ (pisze Dominik): `{"p_yes": ..., "p_no": ...}`.
"""

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import torch
from transformers import PreTrainedTokenizerBase

ScoreFn = Callable[[Any], Mapping[str, float]]


def target_token_id(tokenizer: PreTrainedTokenizerBase, target: str) -> int:
    """Id pierwszego tokenu odpowiedzi. Pamiętaj o spacji: `" Seattle"`, nie `"Seattle"`."""
    ids = tokenizer.encode(target)
    if len(ids) > 1:
        first = tokenizer.decode(ids[:1])
        print(f"Uwaga: {target!r} to {len(ids)} tokeny, mierzę pierwszy: {first!r}")
    return int(ids[0])


def next_token_prob(tokenizer: PreTrainedTokenizerBase, target: str, key: str = "prob") -> ScoreFn:
    """`score_fn` dla modelu z głowicą LM: P(`target` | prompt) na ostatniej pozycji."""
    token_id = target_token_id(tokenizer, target)

    def score(outputs: Any) -> dict[str, float]:
        probs = torch.softmax(outputs.logits[0, -1].float(), dim=-1)
        return {key: float(probs[token_id].item())}

    return score


@torch.no_grad()
def top_next_tokens(
    model: Any, tokenizer: PreTrainedTokenizerBase, text: str, k: int = 10
) -> pd.DataFrame:
    """Top-`k` przewidywanych następnych tokenów (sprawdzenie, czy model zna fakt)."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    probs = torch.softmax(model(**inputs).logits[0, -1].float(), dim=-1)
    values, ids = probs.topk(k)
    return pd.DataFrame(
        {"token": [tokenizer.decode([int(i)]) for i in ids], "prob": values.cpu().numpy()}
    )
