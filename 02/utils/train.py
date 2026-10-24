"""Klasyfikacja tekstu: trening dowolnego modelu HF (np. z adapterem PEFT) na DataFrame'ach z `load_text`."""

import pandas as pd
import plotly.express as px
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


def make_dataset(df: pd.DataFrame, tok, max_length: int = 400) -> Dataset:
    """Kolumny `text`, `label_id` -> `input_ids`, `attention_mask`, `labels` (bez paddingu, ten robi collator)."""
    ds = Dataset.from_dict({"text": list(df["text"]), "labels": list(df["label_id"])})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro")}


def train(model, tok, train_ds, dev_ds, out_dir, epochs=3, lr=2e-4, batch_size=16, seed=42) -> Trainer:
    """Ewaluacja na dev co epokę; na końcu ładowany jest checkpoint z najlepszym macro-F1 na dev."""
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        label_names=["labels"],
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
    )
    trainer.train()
    return trainer


def plot_history(histories: dict, title: str = ""):
    """`histories`: {nazwa przebiegu: trainer.state.log_history} -> krzywe train loss, dev loss, dev macro-F1."""
    keys = {"loss": "train loss", "eval_loss": "dev loss", "eval_f1": "dev macro-F1"}
    rows = [
        {"run": run, "epoch": log["epoch"], "metric": name, "value": log[key]}
        for run, history in histories.items()
        for log in history
        for key, name in keys.items()
        if key in log
    ]
    fig = px.line(pd.DataFrame(rows), x="epoch", y="value", color="run", facet_col="metric", markers=True, title=title)
    return fig.update_yaxes(matches=None, showticklabels=True)
