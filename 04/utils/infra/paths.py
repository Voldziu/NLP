"""Ścieżki niezależne od katalogu, z którego uruchomiono notebook."""

from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[2]
"""Katalog `04/`."""

ARTIFACTS_DIR = LAB_DIR / "artifacts"
"""`04/artifacts/` (w `.gitignore`): CSV, wykresy robocze, cache."""


def artifacts_dir(*parts: str) -> Path:
    """Zwraca (i tworzy) podkatalog `04/artifacts/...`, np. `artifacts_dir("mikolaj")`."""
    path = ARTIFACTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
