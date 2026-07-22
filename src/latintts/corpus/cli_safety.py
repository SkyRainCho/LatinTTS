from __future__ import annotations

import os
import re
from contextlib import suppress
from pathlib import Path

_DELIMITER = r"\s'\"\[\](){},;:"
_FILE_URI = re.compile(r"(?i)(?<!\w)file:/[^\r\n]*")
_QUOTED_ABSOLUTE_PATH = re.compile(
    r"(?P<quote>['\"])(?P<path>(?:[A-Za-z]:/|/)[^'\"\r\n]*)(?P=quote)"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![\w<>])(?:[A-Za-z]:/)[^\r\n]*")
_POSIX_ABSOLUTE_PATH = re.compile(r"(?<![\w<>:])/(?!/)[^\r\n]*")


def _normalized_display(value: str) -> str:
    normalized = value.replace("\\", "/")
    normalized = re.sub(r"/{2,}", "/", normalized)
    if normalized.casefold().startswith("/?/unc/"):
        return f"/{normalized[len('/?/unc/') :]}"
    if normalized.startswith("/?/"):
        return normalized[3:]
    return normalized


def _root_aliases(project_root: Path, *, resolve_project_root: bool) -> tuple[str, ...]:
    candidates: set[str] = set()
    raw_candidates = [str(project_root)]
    with suppress(OSError, RuntimeError):
        raw_candidates.append(os.path.abspath(project_root))
    if resolve_project_root:
        with suppress(OSError, RuntimeError):
            raw_candidates.append(str(project_root.resolve(strict=False)))
    for candidate in raw_candidates:
        normalized = _normalized_display(candidate).rstrip("/")
        if normalized:
            candidates.add(normalized)
    return tuple(sorted(candidates, key=len, reverse=True))


def _replace_project_root(
    message: str,
    project_root: Path,
    *,
    resolve_project_root: bool,
) -> str:
    safe = message
    for alias in _root_aliases(project_root, resolve_project_root=resolve_project_root):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_.-]){re.escape(alias)}(?=$|/|[{_DELIMITER}])",
            re.IGNORECASE,
        )
        safe = pattern.sub("<project-root>", safe)
    return safe


def _redact_quoted_absolute(match: re.Match[str]) -> str:
    quote = match.group("quote")
    return f"{quote}<outside-project-root>{quote}"


def safe_error_message(
    error: BaseException,
    project_root: Path,
    *,
    resolve_project_root: bool = True,
) -> str:
    """Render an exception without exposing project or external absolute paths."""
    try:
        message = _normalized_display(str(error))
        message = _replace_project_root(
            message,
            project_root,
            resolve_project_root=resolve_project_root,
        )
        message = _FILE_URI.sub("<outside-project-root>", message)
        message = _QUOTED_ABSOLUTE_PATH.sub(_redact_quoted_absolute, message)
        message = _WINDOWS_ABSOLUTE_PATH.sub("<outside-project-root>", message)
        return _POSIX_ABSOLUTE_PATH.sub("<outside-project-root>", message)
    except Exception:
        return "operation failed without safe detail"
