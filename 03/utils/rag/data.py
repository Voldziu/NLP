from datasets import load_dataset

from ..logger import get_logger
from .base import Doc, QAItem

logger = get_logger(__name__)


def load_squad(split: str = "validation") -> tuple[list[Doc], list[QAItem]]:
    """Load SQuAD and split it into unique paragraphs and questions.

    Args:
        split: SQuAD split name, e.g. ``"validation"`` or ``"train"``.

    Returns:
        Unique paragraphs in first-seen order (id is the paragraph index) and
        one ``QAItem`` per dataset row pointing at its paragraph.
    """
    ds = load_dataset("rajpurkar/squad", split=split)

    docs: list[Doc] = []
    qa: list[QAItem] = []
    doc_ids: dict[str, str] = {}
    for row in ds:
        context = row["context"]
        if context not in doc_ids:
            doc_ids[context] = str(len(docs))
            docs.append(
                Doc(id=doc_ids[context], text=context, title=row["title"].replace("_", " ")),
            )
        qa.append(
            QAItem(
                question=row["question"],
                answers=row["answers"]["text"],
                doc_id=doc_ids[context],
            ),
        )

    logger.info("SQuAD %s: %d unique paragraphs, %d questions", split, len(docs), len(qa))
    return docs, qa
