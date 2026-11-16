import openai
from langchain_openai import ChatOpenAI

from ..logger import get_logger
from .config import get_settings

logger = get_logger(__name__)


def list_models() -> list[str]:
    """List model ids available on the CLARIN LLM API.

    Returns:
        Sorted model ids.
    """
    s = get_settings()
    client = openai.OpenAI(base_url=s.base_url, api_key=s.api_key.get_secret_value())
    return sorted(m.id for m in client.models.list())


def get_llm(model: str | None = None, *, temperature: float = 0.0) -> ChatOpenAI:
    """Build a chat model for the CLARIN LLM API.

    Args:
        model: Model id. Falls back to ``CLARIN_MODEL`` from ``03/.env``.
        temperature: Sampling temperature.

    Returns:
        Configured ``ChatOpenAI`` client.

    Raises:
        ValueError: When no model is given and ``CLARIN_MODEL`` is empty.
    """
    s = get_settings()
    model = model or s.model
    if not model:
        raise ValueError("No model: pass `model` or set CLARIN_MODEL in 03/.env")
    logger.info("LLM: %s (temperature=%s)", model, temperature)
    return ChatOpenAI(
        model=model,
        base_url=s.base_url,
        api_key=s.api_key,
        temperature=temperature,
    )
