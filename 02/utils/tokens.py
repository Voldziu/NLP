"""Klasyfikacja tokenów: anotacje z Label Studio (01/annotation) -> tagi BIO na słowach."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_text

ANNOT = Path(__file__).resolve().parents[2] / "01" / "annotation"
SPANS = ["wzmocnienie", "oslabienie", "negacja"]
TAGS = ["O"] + [f"{p}-{s}" for s in SPANS for p in "BI"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}


def _offsets(words: list[str], text: str) -> list[tuple[int, int]]:
    """Pozycje słów w `text` (detokenizacja usuwa tylko spacje, więc słowa występują dosłownie)."""
    pos, out = 0, []
    for w in words:
        pos = text.index(w, pos)
        out.append((pos, pos + len(w)))
        pos += len(w)
    return out


def _tags(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int, str]],
) -> list[str]:
    tags = ["O"] * len(offsets)
    for start, end, label in spans:
        inside = False
        for i, (a, b) in enumerate(offsets):
            if a < end and b > start:
                tags[i] = ("I-" if inside else "B-") + label
                inside = True
    return tags


def _vote(variants: list[list[str]]) -> list[str]:
    """Większość głosów per słowo; remis rozstrzyga etykieta inna niż O."""
    pick = lambda col: min(Counter(col).items(), key=lambda kv: (-kv[1], kv[0] == "O"))[
        0
    ]
    return [pick(col) for col in zip(*variants, strict=False)]


def load_tokens() -> pd.DataFrame:
    """Kolumny: id (jak w load_text('train')), tokens, tags. Tylko teksty zaanotowane w 01/."""
    train = load_text("train").set_index("id")
    per_doc: dict[int, list[list[str]]] = {}
    for path in sorted(ANNOT.glob("*/export_*.json")):
        for task in json.loads(path.read_text(encoding="utf-8")):
            ann = next((a for a in task["annotations"] if not a["was_cancelled"]), None)
            if ann is None:
                continue
            doc_id, text = task["data"]["id"], task["data"]["text"]
            assert train.at[doc_id, "text"] == text, doc_id
            spans = [
                (r["value"]["start"], r["value"]["end"], r["value"]["labels"][0])
                for r in ann["result"]
                if r["type"] == "labels"
            ]
            words = train.at[doc_id, "raw"].split()
            per_doc.setdefault(doc_id, []).append(_tags(_offsets(words, text), spans))
    rows = [
        {"id": i, "tokens": train.at[i, "raw"].split(), "tags": _vote(v)}
        for i, v in sorted(per_doc.items())
    ]
    return pd.DataFrame(rows)


def split_tokens(df: pd.DataFrame, test_size: float = 0.2, seed: int = 420):  # blaze it
    return train_test_split(df, test_size=test_size, random_state=seed)
