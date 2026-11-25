"""Ładowanie modeli używanych w zadaniu 4.

Modele zwracane są w trybie `eval()`, na wskazanym urządzeniu, razem z tokenizerem.
gpt2-xl w float32 zajmuje ok. 6,5 GB; pierwsze pobranie z HF trwa kilka minut.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .device import get_device

BOOLQ_CLASSIFIER = "utahnlp/boolq_gpt2-xl_seed-1"
"""gpt2-xl dostrojony do BoolQ (`GPT2ForSequenceClassification`), id2label = {0: False, 1: True}."""


def _tokenizer(name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    name: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Model z głowicą LM + tokenizer, np. `gpt2`, `gpt2-xl`, `EleutherAI/pythia-70m`."""
    device = get_device() if device is None else torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()
    return model, _tokenizer(name)


def load_gpt2_xl(
    device: torch.device | str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """`gpt2-xl` (48 warstw, 1600 wymiarów) z głowicą LM."""
    return load_causal_lm("gpt2-xl", device)


def load_boolq_classifier(
    name: str = BOOLQ_CLASSIFIER,
    device: torch.device | str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Klasyfikator BoolQ (`GPT2ForSequenceClassification`), domyślnie `BOOLQ_CLASSIFIER`.

    Wyjście: `logits` o kształcie `[batch, 2]`; softmax daje P(False), P(True).
    Klasyfikator czyta ostatni token różny od `pad_token_id`.
    """
    device = get_device() if device is None else torch.device(device)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
    model.eval()
    return model, _tokenizer(name)
