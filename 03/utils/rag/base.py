from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class Doc(BaseModel):
    """A retrievable paragraph."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    title: str
    score: float = 0.0


class QAItem(BaseModel):
    """A question with its gold answers and the id of the paragraph that answers it."""

    model_config = ConfigDict(frozen=True)

    question: str
    answers: list[str]
    doc_id: str


class Retriever(ABC):
    """Common interface, so the hybrid retriever from Part I can be swapped in."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[Doc]:
        """Return the top-k documents for a query.

        Args:
            query: User question.
            k: Number of documents to return.

        Returns:
            Documents sorted from best to worst, with ``score`` filled in.
        """
