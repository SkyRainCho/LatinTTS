from __future__ import annotations

import os
import re
from contextlib import suppress
from pathlib import Path

_DELIMITER = r"\s'\"\[\](){},;:"
_FILE_URI = re.compile(
    r"(?i)(?<!\w)(?:f|%46|%66)(?:i|%49|%69)(?:l|%4c|%6c)"
    r"(?:e|%45|%65)(?::|%3a)(?:/|%2f|%5c)[^\r\n]*"
)
_QUOTED_ABSOLUTE_PATH = re.compile(
    r"(?P<quote>['\"])(?P<path>(?:[A-Za-z]:/|/)[^'\"\r\n]*)(?P=quote)"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:/)[^\r\n]*")
_POSIX_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?!/)[^\r\n]*")
_RELATIVE_PREFIX = re.compile(r"(?<![A-Za-z0-9._-])(?P<prefix>\.\.?/)")
_SAFE_PLACEHOLDERS = ("<project-root>", "<outside-project-root>")
_PROJECT_ROOT_PATH = re.compile(r"<project-root>(?P<suffix>/[^\r\n]*)")
_PARENT_SEGMENT = re.compile(r"(?:^|/)\.\.(?=/|$|[\s'\"\[\](){}<>,;:])")


def _normalized_display(value: str) -> str:
    normalized = value.replace("\\", "/")
    normalized = re.sub(r"/{2,}", "/", normalized)
    if normalized.casefold().startswith("/?/unc/"):
        return f"/{normalized[len('/?/unc/') :]}"
    if re.match(r"^/\?/[A-Za-z]:/", normalized):
        return normalized[3:]
    return normalized


def _is_windows_path_text(value: str) -> bool:
    display = value.replace("\\", "/")
    return bool(re.match(r"^[A-Za-z]:/", display)) or display.startswith("//")


def _root_aliases(
    project_root: Path,
    *,
    resolve_project_root: bool,
) -> tuple[tuple[str, bool], ...]:
    candidates: dict[str, bool] = {}
    raw_candidates = [str(project_root)]
    with suppress(OSError, RuntimeError):
        raw_candidates.append(os.path.abspath(project_root))
    if resolve_project_root:
        with suppress(OSError, RuntimeError):
            raw_candidates.append(str(project_root.resolve(strict=False)))
    for candidate in raw_candidates:
        normalized = _normalized_display(candidate).rstrip("/")
        if normalized:
            candidates[normalized] = candidates.get(normalized, False) or (
                _is_windows_path_text(candidate)
            )
    return tuple(sorted(candidates.items(), key=lambda item: len(item[0]), reverse=True))


def _replace_project_root(
    message: str,
    project_root: Path,
    *,
    resolve_project_root: bool,
) -> str:
    safe = message
    for alias, case_insensitive in _root_aliases(
        project_root,
        resolve_project_root=resolve_project_root,
    ):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_.-]){re.escape(alias)}(?=$|/|[{_DELIMITER}])",
            re.IGNORECASE if case_insensitive else 0,
        )
        safe = pattern.sub("<project-root>", safe)
    return safe


def _redact_quoted_absolute(match: re.Match[str]) -> str:
    quote = match.group("quote")
    return f"{quote}<outside-project-root>{quote}"


def _protect_safe_placeholders(message: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    protected = message
    replacements: list[tuple[str, str]] = []
    for index, placeholder in enumerate(_SAFE_PLACEHOLDERS):
        for suffix_index, suffix in enumerate(("/", "")):
            original = f"{placeholder}{suffix}"
            sentinel = f"\x00latintts-safe-placeholder-{index}-{suffix_index}\x00"
            while sentinel in protected:
                sentinel += "\x00"
            if original in protected:
                protected = protected.replace(original, sentinel)
                replacements.append((sentinel, original))
    return protected, tuple(replacements)


def _restore_safe_placeholders(
    message: str,
    replacements: tuple[tuple[str, str], ...],
) -> str:
    restored = message
    for sentinel, placeholder in reversed(replacements):
        restored = restored.replace(sentinel, placeholder)
    return restored


def _protect_relative_prefixes(message: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    replacements: list[tuple[str, str]] = []

    def protect(match: re.Match[str]) -> str:
        sentinel = f"\x00latintts-safe-relative-{len(replacements)}\x00"
        while sentinel in message:
            sentinel += "\x00"
        replacements.append((sentinel, match.group("prefix")))
        return sentinel

    return _RELATIVE_PREFIX.sub(protect, message), tuple(replacements)


def _redact_project_root_traversal(match: re.Match[str]) -> str:
    if _PARENT_SEGMENT.search(match.group("suffix")):
        return "<outside-project-root>"
    return match.group(0)


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
        message = _PROJECT_ROOT_PATH.sub(_redact_project_root_traversal, message)
        message, relative_prefixes = _protect_relative_prefixes(message)
        message, placeholders = _protect_safe_placeholders(message)
        message = _FILE_URI.sub("<outside-project-root>", message)
        message = _QUOTED_ABSOLUTE_PATH.sub(_redact_quoted_absolute, message)
        message = _WINDOWS_ABSOLUTE_PATH.sub("<outside-project-root>", message)
        message = _POSIX_ABSOLUTE_PATH.sub("<outside-project-root>", message)
        message = _restore_safe_placeholders(message, placeholders)
        return _restore_safe_placeholders(message, relative_prefixes)
    except Exception:
        return "operation failed without safe detail"
