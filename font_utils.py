"""Font helpers for rendering translated text with Vietnamese diacritics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


WINDOWS_FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

try:
    from pipeline_config import FONT_FALLBACK_PATHS
except Exception:
    FONT_FALLBACK_PATHS = []


def _candidate_font_paths(preferred: str | os.PathLike[str] | None) -> Iterable[Path]:
    if preferred:
        preferred_path = Path(preferred)
        yield preferred_path
        if not preferred_path.is_absolute():
            yield Path.cwd() / preferred_path
            yield WINDOWS_FONT_DIR / preferred_path.name

    for path in FONT_FALLBACK_PATHS:
        yield Path(path)

    for name in (
        "arial.ttf",
        "calibri.ttf",
        "segoeui.ttf",
        "tahoma.ttf",
        "verdana.ttf",
        "NotoSans-Regular.ttf",
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        yield WINDOWS_FONT_DIR / name
        yield Path("/usr/share/fonts/truetype/dejavu") / name
        yield Path("/usr/share/fonts/truetype/noto") / name


def resolve_font_path(preferred: str | os.PathLike[str] | None = None) -> Optional[str]:
    """Return a real TTF/OTF path likely to support Vietnamese text."""
    seen: set[Path] = set()
    for path in _candidate_font_paths(preferred):
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            resolved = path.expanduser()

        if resolved in seen:
            continue
        seen.add(resolved)

        if resolved.is_file():
            return str(resolved)

    return None
