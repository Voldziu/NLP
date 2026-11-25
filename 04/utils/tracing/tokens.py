"""Pozycje tokenów: który token to podmiot, jak go podpisać na wykresie."""

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase


def tokenize(tokenizer: PreTrainedTokenizerBase, text: str) -> list[str]:
    """Tokeny tekstu jako napisy (bez znaku `Ġ`; spacja na początku zostaje jako ` `)."""
    ids = tokenizer(text)["input_ids"]
    return [str(tokenizer.decode([i])) for i in ids]


def find_token_positions(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    substring: str,
    occurrence: int = 0,
) -> list[int]:
    """Pozycje tokenów, które pokrywają `substring` w `text`.

    Używa `offset_mapping` (szybki tokenizer), więc działa też, gdy podmiot dzieli się
    na kilka tokenów (`" Space Needle"` -> `[" Space", " Need", "le"]`).
    `occurrence` wybiera kolejne wystąpienie, gdy podmiot pojawia się kilka razy.

    >>> find_token_positions(tok, "The Space Needle is in downtown", "The Space Needle")
    [0, 1, 2, 3]
    """
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(substring, start + 1)
        if start == -1:
            raise ValueError(f"Brak wystąpienia nr {occurrence} napisu {substring!r} w {text!r}")
    end = start + len(substring)

    enc = tokenizer(text, return_offsets_mapping=True)
    positions = [
        i
        for i, (tok_start, tok_end) in enumerate(enc["offset_mapping"])
        if tok_start < end and tok_end > start
    ]
    if not positions:
        raise ValueError(f"Żaden token nie pokrywa {substring!r}")
    return positions


def position_labels(tokens: Sequence[str], noised: Sequence[int]) -> list[str]:
    """Etykiety osi Y: token z `*` przy pozycjach zaszumionych (jak w artykule ROME)."""
    noised_set = set(noised)
    return [f"{t.strip() or repr(t)}{'*' if i in noised_set else ''}" for i, t in enumerate(tokens)]
