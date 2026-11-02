# Zadanie 2: Modele językowe

Instrukcja: [Zadanie_2_zmodyfikowane.md](Zadanie_2_zmodyfikowane.md)

## Podział zadań

| Osoba | Zakres | Folder | Pkt |
|---|---|---|---|
| **Osoba A** (BERT) | I.A (klasy BERT). Pkt 1: HerBERT + PEFT, klasyfikacja tekstu i tokenów, mały zbiór testowy. Pkt 2: wizualizacja przestrzeni (`[CLS]`) przed i po strojeniu. Pkt 4: augmentacja przez `[MASK]`. | `bert/` | 3,25 |
| **Osoba B** (GPT-2) | I.B, I.C (PEFT + tutorial), I.D (klasy GPT-2). Pkt 3: `sdadas/polish-gpt2-medium` + PEFT, klasyfikacja tekstu i tokenów (ostatni token, padding i `<eos>` → `-100`), wizualizacja. Pkt 5: augmentacja przez generację (temperatura, top-p, top-k). | `gpt2/` | 2,75 |
| **Osoba C** (Część II) | II.A–C. Pkt 1: anizotropia i context-specificity (BERT i GPT-2, `wsd_polish_datasets`). Pkt 2: Logit Lens (≥20 promptów, heatmapa, ranga i prawdopodobieństwo, wnioski). | `part2/` | 4 |

Część II nie zależy od Części I, więc osoba C może zacząć od razu.

## Zasady współpracy

- Dane: `data/dataset_conll/` (skopiowane z `01/data/`). Folder `data` jest w `.gitignore`, więc każdy kopiuje go u siebie.
- Zbiory testowe są wspólne dla A i B: tekst → `all.text.test.txt`; tokeny → `split_tokens(df)` z `seed=42`.
- Augmentacja (pkt 4 i 5) dotyka tylko zbioru treningowego. Wyniki zapisujemy w `data/aug_bert.jsonl` i `data/aug_gpt2.jsonl` (pola `text`, `label`; `label` to nazwa etykiety).
- Końcowy model trenujemy na `train + augmentacja` i oceniamy na niezmienionym teście.
- Wspólny kod trafia do `utils/` (małe, niezależne moduły). Kod własny zostaje w swoim folderze.
- Artefakty (checkpointy, adaptery) idą do `artifacts/`.
- Modele bazowe: `allegro/herbert-base-cased` (BERT) i `sdadas/polish-gpt2-medium` (GPT-2).

## Środowisko

```bash
uv sync   # z korzenia repo
```

Notebooki i skrypty leżą w podfolderach `02/`, więc na początku:

```python
import sys; sys.path.insert(0, "..")
from utils.data import load_text, LABELS, LABEL2ID
```

## `utils/`

| Moduł | Co robi |
|---|---|
| `data.py` | `load_text("train" / "dev" / "test")` → `id, raw, text, label, label_id`. Etykiety: `plus_m, minus_m, zero, amb`. |
| `tokens.py` | `load_tokens()` → `id, tokens, tags` (BIO: `wzmocnienie`, `oslabienie`, `negacja`) z anotacji w `01/annotation`. `split_tokens(df)` → train/test. `TAGS`, `TAG2ID`. |
| `align.py` | `encode_tokens(tokenizer, tokens, tags, TAG2ID)` → tagi na subtokeny (pierwszy subtoken słowa, reszta `-100`). |
| `plots.py` | `plot_embeddings(emb, labels, texts, method="pca")` → interaktywny wykres 2D (Plotly); `method="tsne"` dla t-SNE. |
| `train.py` | `train(model, tok, train_ds, dev_ds, out_dir, ..., collator=None, compute_metrics=metrics)` (HF `Trainer`, ewaluacja co epokę, najlepszy checkpoint wg `f1` na dev), `plot_history({nazwa: log_history})` → krzywe uczenia. Tekst: `make_dataset(df, tok, max_length)`, `metrics` (accuracy, macro-F1). Tokeny: `make_token_dataset(df, tok, max_length)`, `token_metrics` (macro-F1 po tagach bez `O`, pomija `-100`) + `collator=DataCollatorForTokenClassification(tok)`. |

Uwaga: `load_tokens()` korzysta z anotacji z Zadania 1. Są w nich tylko 35 unikalne teksty, a tagi to głosowanie większościowe anotatorów.
