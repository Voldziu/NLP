"""Dopasowanie tagów słów do subtokenów (HF). Pierwszy subtoken słowa dostaje tag, reszta -100."""


def encode_tokens(tokenizer, tokens, tags, tag2id, max_length=256):
    """GPT-2: tokenizer wymaga `add_prefix_space=True` i ustawionego `pad_token`."""
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_length)
    labels, prev = [], None
    for w in enc.word_ids():
        labels.append(-100 if w is None or w == prev else tag2id[tags[w]])
        prev = w
    enc["labels"] = labels
    return enc
