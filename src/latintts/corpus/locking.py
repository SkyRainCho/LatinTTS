"""Cooperative cross-process locking for LatinTTS corpus mutations.

Every built-in write path that mutates mutable durable evidence consumed by the manifest
build participates in this lease. Content-addressed audio/cache writers retain their
independent key locks. Manual or hostile processes that ignore the shared lease, including
namespace ABA attacks, are outside its guarantee; descriptor and snapshot checks remain
defense in depth.
"""

from __future__ import annotations

import errno
import ntpath
import os
import stat
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from latintts.corpus.paths import require_canonical_descendant

_LOCK_NAME = ".corpus-mutation.lock"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_WINDOWS_COMPONENT_CASEFOLD = os.name == "nt"


class CorpusMutationLockBusy(BlockingIOError):
    """Another LatinTTS writer already owns the corpus mutation lease."""


class _Backend(Protocol):
    def close(self) -> None: ...

    def close_after_fork(self) -> None: ...


@dataclass(slots=True)
class _PosixBackend:
    directory_fd: int
    lock_fd: int

    def close(self) -> None:  # pragma: no cover - exercised on POSIX hosts
        fcntl: Any = __import__("fcntl")
        failure: OSError | None = None
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        except OSError as error:
            failure = error
        for descriptor in (self.lock_fd, self.directory_fd):
            try:
                os.close(descriptor)
            except OSError as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    def close_after_fork(self) -> None:  # pragma: no cover - exercised on POSIX hosts
        # Do not issue LOCK_UN on an inherited open-file description: doing so could
        # release the parent's flock. Closing only this child's duplicates is safe.
        for descriptor in (self.lock_fd, self.directory_fd):
            with suppress(OSError):
                os.close(descriptor)


@dataclass(slots=True)
class _WindowsBackend:
    directory_handle: object
    lock_handle: object

    def close(self) -> None:
        failure: OSError | None = None
        for handle in (self.lock_handle, self.directory_handle):
            try:
                _windows_close_handle(handle)
            except OSError as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    def close_after_fork(self) -> None:  # pragma: no cover - Windows has no fork
        self.close()


@dataclass(slots=True)
class _Entry:
    lock: Any = field(default_factory=threading.RLock)
    depth: int = 0
    backend: _Backend | None = None


_registry_guard = threading.Lock()
_registry_pid = os.getpid()
_entries: dict[str, _Entry] = {}


def _reset_registry_after_fork() -> None:  # pragma: no cover - exercised on POSIX hosts
    global _entries, _registry_guard, _registry_pid
    inherited = _entries
    _entries = {}
    _registry_guard = threading.Lock()
    _registry_pid = os.getpid()
    _close_inherited_entries(inherited)


def _close_inherited_entries(entries: dict[str, _Entry]) -> None:
    """Drop child copies without issuing an unlock against the parent's lease."""
    for entry in entries.values():
        if entry.backend is not None:
            entry.backend.close_after_fork()


if hasattr(os, "register_at_fork"):  # pragma: no branch - platform capability
    os.register_at_fork(  # pragma: no cover - exercised on POSIX hosts
        after_in_child=_reset_registry_after_fork
    )


def _component_key(component: str) -> str:
    if _WINDOWS_COMPONENT_CASEFOLD:
        return ntpath.normcase(component)
    return component


def _matches_corpus_layout(root: Path, target: Path) -> bool:
    relative = target.relative_to(root)
    parts = tuple(_component_key(part) for part in relative.parts)
    return (
        (len(parts) >= 2 and parts[0] == _component_key("manifests"))
        or (
            len(parts) >= 4
            and parts[:2] == (_component_key("derived"), _component_key("corpus-v1"))
        )
        or (
            len(parts) >= 3
            and parts[0] == _component_key("raw")
            and parts[1] in {_component_key("spoken"), _component_key("sung")}
        )
    )


def _named_corpus_candidates(target: Path) -> tuple[Path, ...]:
    return tuple(
        candidate
        for candidate in (target, *target.parents)
        if _component_key(candidate.name) == _component_key("local-data")
    )


def _layout_candidates(target: Path) -> tuple[Path, ...]:
    return tuple(
        candidate
        for candidate in _named_corpus_candidates(target)
        if _matches_corpus_layout(candidate, target)
    )


def _same_existing_path(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, NotADirectoryError):
        return False


def _matches_physical_corpus_layout(root: Path, target: Path) -> bool:
    parts = target.relative_to(root).parts
    return (
        (len(parts) >= 2 and _same_existing_path(root / parts[0], root / "manifests"))
        or (
            len(parts) >= 4
            and _same_existing_path(
                root.joinpath(*parts[:2]),
                root / "derived" / "corpus-v1",
            )
        )
        or (
            len(parts) >= 3
            and (
                _same_existing_path(
                    root.joinpath(*parts[:2]),
                    root / "raw" / "spoken",
                )
                or _same_existing_path(
                    root.joinpath(*parts[:2]),
                    root / "raw" / "sung",
                )
            )
        )
    )


def _physical_layout_candidates(target: Path) -> tuple[Path, ...]:
    return tuple(
        candidate
        for candidate in (target, *target.parents)
        if candidate.name
        and _same_existing_path(
            candidate,
            candidate.with_name("local-data"),
        )
        and _matches_physical_corpus_layout(candidate, target)
    )


def local_data_root_for(path: Path) -> Path | None:
    """Locate the unique ``local-data`` ancestor matching the fixed corpus layout."""
    absolute = Path(os.path.abspath(path))
    named_candidates = _named_corpus_candidates(absolute)
    if not named_candidates:
        canonical = absolute.resolve(strict=False)
        if _layout_candidates(canonical) or _physical_layout_candidates(canonical):
            raise ValueError("corpus mutation target uses an alias to a corpus root")
        return None
    layout_candidates = tuple(
        candidate for candidate in named_candidates if _matches_corpus_layout(candidate, absolute)
    )
    if len(layout_candidates) != 1:
        raise ValueError(
            "corpus mutation target has an unrecognized or ambiguous corpus root layout"
        )
    local_data = layout_candidates[0]
    require_canonical_descendant(
        local_data,
        absolute,
        kind="corpus mutation target",
    )
    return local_data


def _lock_path(local_data: Path) -> Path:
    local_data.mkdir(parents=True, exist_ok=True)
    require_canonical_descendant(
        local_data,
        local_data,
        kind="corpus mutation root",
    )
    manifests = local_data / "manifests"
    manifests.mkdir(exist_ok=True)
    require_canonical_descendant(
        local_data,
        manifests,
        kind="corpus mutation lock directory",
    )
    lock_path = manifests / _LOCK_NAME
    if lock_path.exists() or lock_path.is_symlink():
        metadata = os.lstat(lock_path)
        attributes = getattr(metadata, "st_file_attributes", 0)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or attributes & _REPARSE_POINT
        ):
            raise ValueError("corpus mutation lock must be a canonical regular file")
    return lock_path


def _windows_close_handle(handle: object) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_open_backend(lock_path: Path) -> _WindowsBackend:
    import ctypes
    from ctypes import wintypes

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        ]

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
    invalid = ctypes.c_void_p(-1).value

    def open_handle(
        path: Path,
        *,
        desired_access: int,
        share_mode: int,
        creation: int,
        flags: int,
    ) -> object:
        handle = kernel32.CreateFileW(
            str(path),
            desired_access,
            share_mode,
            None,
            creation,
            flags,
            None,
        )
        if handle == invalid:
            code = ctypes.get_last_error()
            if code in {32, 33}:
                raise CorpusMutationLockBusy(
                    errno.EWOULDBLOCK,
                    "corpus mutation lock is busy",
                    str(lock_path),
                )
            raise ctypes.WinError(code)
        return handle

    def attributes(handle: object) -> int:
        information = FileAttributeTagInfo()
        if not kernel32.GetFileInformationByHandleEx(
            handle,
            9,  # FileAttributeTagInfo
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(information.file_attributes)

    directory_handle = open_handle(
        lock_path.parent,
        desired_access=0x80,  # FILE_READ_ATTRIBUTES
        share_mode=0x1 | 0x2,  # deliberately deny FILE_SHARE_DELETE
        creation=3,  # OPEN_EXISTING
        flags=0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
    )
    try:
        directory_attributes = attributes(directory_handle)
        if not directory_attributes & 0x10 or directory_attributes & _REPARSE_POINT:
            raise ValueError("corpus mutation lock directory must be canonical")
        lock_handle = open_handle(
            lock_path,
            desired_access=0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
            share_mode=0,
            creation=4,  # OPEN_ALWAYS
            flags=0x00200000,  # OPEN_REPARSE_POINT
        )
        try:
            lock_attributes = attributes(lock_handle)
            if lock_attributes & (0x10 | _REPARSE_POINT):
                raise ValueError("corpus mutation lock must be a canonical regular file")
        except Exception:
            _windows_close_handle(lock_handle)
            raise
    except Exception:
        _windows_close_handle(directory_handle)
        raise
    return _WindowsBackend(directory_handle, lock_handle)


def _posix_open_backend(lock_path: Path) -> _PosixBackend:  # pragma: no cover
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(lock_path.parent, directory_flags)
    lock_fd: int | None = None
    try:
        pinned_directory = os.fstat(directory_fd)
        current_directory = os.stat(lock_path.parent, follow_symlinks=False)
        if not stat.S_ISDIR(pinned_directory.st_mode) or (
            pinned_directory.st_dev,
            pinned_directory.st_ino,
        ) != (current_directory.st_dev, current_directory.st_ino):
            raise ValueError("corpus mutation lock directory identity changed")
        lock_fd = os.open(
            lock_path.name,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        pinned_lock = os.fstat(lock_fd)
        current_lock = os.stat(lock_path.name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(pinned_lock.st_mode) or (pinned_lock.st_dev, pinned_lock.st_ino) != (
            current_lock.st_dev,
            current_lock.st_ino,
        ):
            raise ValueError("corpus mutation lock must be a canonical regular file")
        fcntl: Any = __import__("fcntl")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CorpusMutationLockBusy(
                errno.EWOULDBLOCK,
                "corpus mutation lock is busy",
                str(lock_path),
            ) from error
        return _PosixBackend(directory_fd, lock_fd)
    except Exception:
        if lock_fd is not None:
            with suppress(OSError):
                os.close(lock_fd)
        with suppress(OSError):
            os.close(directory_fd)
        raise


def _open_backend(local_data: Path) -> _Backend:
    lock_path = _lock_path(local_data)
    if os.name == "nt":
        return _windows_open_backend(lock_path)
    return _posix_open_backend(lock_path)  # pragma: no cover - exercised on POSIX hosts


def _entry_for(local_data: Path) -> _Entry:
    if os.getpid() != _registry_pid:
        # Check before touching the possibly inherited/locked registry mutex.
        _reset_registry_after_fork()
    key = os.path.normcase(os.path.abspath(local_data))
    with _registry_guard:
        return _entries.setdefault(key, _Entry())


@dataclass(slots=True)
class CorpusMutationLease:
    _entry: _Entry | None
    _owner_pid: int = field(default_factory=os.getpid)
    _owner_thread: int = field(default_factory=threading.get_ident)
    _closed: bool = False

    def __enter__(self) -> CorpusMutationLease:
        if self._closed:
            raise RuntimeError("corpus mutation lease is already closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        if os.getpid() != self._owner_pid:
            # The at-fork hook owns inherited descriptor cleanup. This stale lease
            # must not mutate the copied Entry or unlock the parent's open description.
            self._entry = None
            self._closed = True
            return
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("corpus mutation lease must be closed by its owning thread")
        entry = self._entry
        self._closed = True
        if entry is None:
            return
        try:
            if entry.depth < 1 or entry.backend is None:
                raise RuntimeError("corpus mutation lease registry is inconsistent")
            entry.depth -= 1
            if entry.depth == 0:
                backend = entry.backend
                entry.backend = None
                backend.close()
        finally:
            entry.lock.release()


def corpus_mutation_lease(*targets: Path) -> CorpusMutationLease:
    """Acquire the one cooperative mutation lease shared by a local corpus."""
    if not targets:
        raise ValueError("corpus_mutation_lease requires at least one target")
    roots = tuple(local_data_root_for(Path(target)) for target in targets)
    if all(root is None for root in roots):
        return CorpusMutationLease(None)
    if any(root is None for root in roots):
        raise ValueError("all mutation targets must belong to the same local-data root")
    local_data = roots[0]
    assert local_data is not None
    root_keys = {os.path.normcase(os.path.abspath(root)) for root in roots if root is not None}
    if len(root_keys) != 1:
        raise ValueError("all mutation targets must belong to the same local-data root")
    entry = _entry_for(local_data)
    if not entry.lock.acquire(blocking=False):
        raise CorpusMutationLockBusy(
            errno.EWOULDBLOCK,
            "corpus mutation lock is busy",
            str(local_data / "manifests" / _LOCK_NAME),
        )
    try:
        if entry.depth == 0:
            entry.backend = _open_backend(local_data)
        elif entry.backend is None:
            raise RuntimeError("corpus mutation lease registry is inconsistent")
        entry.depth += 1
    except Exception:
        entry.lock.release()
        raise
    return CorpusMutationLease(entry)
