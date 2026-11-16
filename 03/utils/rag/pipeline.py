import re
import string
from operator import itemgetter
from typing import TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from .base import Doc, Retriever


class RagOutput(TypedDict):
    """Result of the RAG chain."""

    question: str
    docs: list[Doc]
    answer: str


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using only the numbered context passages below. "
            "Answer briefly (one sentence at most) and cite the passage number in "
            "square brackets, e.g. [2]. If the context does not contain the answer, "
            "reply exactly: I don't know.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ],
)

PLAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question briefly (one sentence at most)."),
        ("human", "{question}"),
    ],
)

_CITATION_RE = re.compile(r"\[\d+\]")
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def format_docs(docs: list[Doc]) -> str:
    """Render documents as numbered passages for the prompt.

    Args:
        docs: Retrieved documents, best first.

    Returns:
        One ``[i] (title) text`` block per document, separated by blank lines.
    """
    return "\n\n".join(f"[{i}] ({d.title}) {d.text}" for i, d in enumerate(docs, 1))


def build_rag_chain(
    retriever: Retriever,
    llm: BaseChatModel,
    *,
    k: int = 5,
) -> Runnable[str, RagOutput]:
    """Build the retrieve -> prompt -> generate chain.

    Args:
        retriever: Source of context documents.
        llm: Chat model used to answer.
        k: Number of documents to retrieve.

    Returns:
        A chain that takes a question string and returns the question,
        the retrieved documents and the generated answer.
    """
    generate = (
        {
            "context": itemgetter("docs") | RunnableLambda(format_docs),
            "question": itemgetter("question"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    chain = RunnableParallel(
        docs=RunnableLambda(lambda q: retriever.retrieve(q, k)),
        question=RunnablePassthrough(),
    ) | RunnablePassthrough.assign(answer=generate)
    # LCEL types the merged output as dict[str, Any]; its keys match RagOutput.
    return cast("Runnable[str, RagOutput]", chain)


def build_plain_chain(llm: BaseChatModel) -> Runnable[dict[str, str], str]:
    """Build the no-retrieval baseline chain.

    Args:
        llm: Chat model used to answer.

    Returns:
        A chain that takes ``{"question": ...}`` and returns the answer text.
    """
    return PLAIN_PROMPT | llm | StrOutputParser()


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace (SQuAD style)."""
    text = text.lower().translate(_PUNCT_TABLE)
    return " ".join(_ARTICLE_RE.sub(" ", text).split())


def contains_gold(pred: str, golds: list[str]) -> bool:
    """Check whether any gold answer appears in the prediction as whole words.

    Args:
        pred: Model answer, possibly with ``[n]`` citations.
        golds: Acceptable gold answers.

    Returns:
        True if any non-empty normalized gold is contained in the normalized prediction.
    """
    # Citations like [2] would otherwise match numeric gold answers.
    p = f" {_normalize(_CITATION_RE.sub('', pred))} "
    normalized = (_normalize(g) for g in golds)
    return any(f" {g} " in p for g in normalized if g)
