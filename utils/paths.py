from __future__ import annotations

from pathlib import Path

# Base directories
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATOS_DIR: Path = ROOT_DIR / "datos"
UTILS_DIR: Path = ROOT_DIR / "utils"
OUTPUT_DIR: Path = ROOT_DIR / "output"

# Ensure output directory exists at import time (safe no-op if already present)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def data_path(*parts: str | Path) -> Path:
    """Build a path under datos/.

    Example: data_path("hectarea.json") -> ROOT/datos/hectarea.json
    """
    return DATOS_DIR.joinpath(*map(str, parts))


def output_path(*parts: str | Path) -> Path:
    """Build a path under output/ and ensure its parent exists."""
    p = OUTPUT_DIR.joinpath(*map(str, parts))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def root_path(*parts: str | Path) -> Path:
    """Build an absolute path under the project root."""
    return ROOT_DIR.joinpath(*map(str, parts))


def ensure_dirs() -> None:
    """Ensure key directories exist."""
    DATOS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
