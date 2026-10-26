"""Klasyfikacja tekstu: wczytanie PolEmo (all.text.{train,dev,test}.txt)."""

import re
from pathlib import Path
from typing import Literal

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "dataset_conll"
LABELS = ["plus_m", "minus_m", "zero", "amb"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def detokenize(text: str) -> str:
    text = re.sub(r" ([.,!?;:)%])", r"\1", text)
    return re.sub(r"\( ", "(", text)


def parse(line: str) -> dict:
    raw, label = line.rsplit(" __label__", 1)
    label = label.strip().removeprefix("meta_")
    return {"raw": raw.strip(), "text": detokenize(raw.strip()), "label": label}


def load_text(split: Literal["train", "dev", "test"] = "train") -> pd.DataFrame:
    """Kolumny: id (numer linii, ten sam co w anotacjach z 01/), raw, text, label, label_id."""
    path = DATA / f"all.text.{split}.txt"
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame([{"id": i, **parse(l)} for i, l in enumerate(lines)])
    df["label_id"] = df["label"].map(LABEL2ID)
    return df


def load_aug(name: str) -> pd.DataFrame:
    """`data/aug_{name}.jsonl` (pola `text`, `label`) -> kolumny `text`, `label`, `label_id`, gotowe do `pd.concat` z `load_text("train")`."""
    df = pd.read_json(DATA.parent / f"aug_{name}.jsonl", lines=True)[["text", "label"]]
    df["label_id"] = df["label"].map(LABEL2ID)
    return df
