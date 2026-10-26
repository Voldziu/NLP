"""Interaktywna wizualizacja przestrzeni wektorowej (zad. 2 część I, pkt 2)."""

import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_embeddings(emb, labels, texts=None, title="", method="tsne"):
    """`method`: "pca" (liniowa, deterministyczna) albo "tsne" (lokalne skupiska; odległości między nimi nic nie znaczą)."""
    reducer = (
        TSNE(n_components=2, random_state=42)
        if method == "tsne"
        else PCA(n_components=2)
    )
    xy = reducer.fit_transform(emb)
    hover = None if texts is None else [t[:200] for t in texts]
    return px.scatter(
        x=xy[:, 0], y=xy[:, 1], color=list(labels), hover_name=hover, title=title
    )
