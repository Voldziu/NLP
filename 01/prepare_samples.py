"""Losuje dwie rozłączne próbki z PolEmo i zapisuje je w formacie importu Label Studio."""

import json
import random
import re
from pathlib import Path

DATA = Path(__file__).parent / "data" / "dataset_conll" / "all.text.train.txt"
OUT = Path(__file__).parent / "annotation"
N = 100
SEED = 42


def detokenize(text: str) -> str:
    text = re.sub(r" ([.,!?;:)%])", r"\1", text)
    return re.sub(r"\( ", "(", text)


def parse(line: str) -> tuple[str, str]:
    text, label = line.rsplit(" __label__", 1)
    return detokenize(text.strip()), label.strip()


lines = [l for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
docs = [
    {"id": i, **dict(zip(("text", "gold"), parse(l), strict=False))}
    for i, l in enumerate(lines)
]

random.Random(SEED).shuffle(docs)
OUT.mkdir(exist_ok=True)

for k, sample in enumerate((docs[:N], docs[N : 2 * N]), start=1):
    # tasks bez etykiety oryginalnej, żeby anotatorzy jej nie widzieli
    tasks = [{"data": {"id": d["id"], "text": d["text"]}} for d in sample]
    (OUT / f"sample{k}.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gold = {d["id"]: d["gold"] for d in sample}
    (OUT / f"gold_sample{k}.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8"
    )

print(f"Zapisano 2 próbki po {N} w {OUT}")
