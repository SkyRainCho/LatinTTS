from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from latintts.corpus import cli, locking, review, store
from latintts.corpus.inventory import write_intake_skeleton
from latintts.corpus.locking import (
    CorpusMutationLockBusy,
    corpus_mutation_lease,
)
from latintts.corpus.paths import CorpusPaths
from tests.corpus.test_store import _transition


def _paths(tmp_path: Path) -> CorpusPaths:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    return paths


def _run_in_thread(operation: object) -> list[BaseException]:
    failures: list[BaseException] = []

    def run() -> None:
        try:
            operation()  # type: ignore[operator]
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    return failures


def test_corpus_mutation_lease_is_reentrant_for_nested_store_writes(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = paths.manifests / "rights.jsonl"

    with corpus_mutation_lease(target):
        store.write_jsonl_atomic(target, ({"value": 1},))
        with corpus_mutation_lease(target):
            store.write_jsonl_atomic(target, ({"value": 2},))

    assert store.read_jsonl(target) == ({"value": 2},)
    assert (paths.manifests / ".corpus-mutation.lock").is_file()


@pytest.mark.parametrize(
    "relative_target",
    (
        "manifests/rights.jsonl",
        "manifests/transcripts.jsonl",
        "manifests/review.jsonl",
        "manifests/recordings.jsonl",
        "derived/corpus-v1/alignments/runs/test/rec-1/pairing.json",
    ),
)
def test_second_thread_cannot_bypass_process_local_reentrancy(
    tmp_path: Path,
    relative_target: str,
) -> None:
    paths = _paths(tmp_path)
    target = paths.local_data / Path(relative_target)
    store.write_jsonl_atomic(target, ({"value": "stable"},))
    before = target.read_bytes()
    processing_path = paths.alignments / "runs" / "test" / "processing-events.jsonl"

    with corpus_mutation_lease(paths.manifests / "segments.jsonl", processing_path):
        failures = _run_in_thread(lambda: store.write_jsonl_atomic(target, ({"value": "raced"},)))

    assert len(failures) == 1
    assert isinstance(failures[0], CorpusMutationLockBusy)
    assert target.read_bytes() == before
    store.write_jsonl_atomic(target, ({"value": "released"},))
    assert store.read_jsonl(target) == ({"value": "released"},)


def test_second_process_gets_busy_without_changing_corpus_bytes(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = paths.manifests / "rights.jsonl"
    store.write_jsonl_atomic(target, ({"value": "stable"},))
    before = target.read_bytes()
    program = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from latintts.corpus.locking import CorpusMutationLockBusy",
            "from latintts.corpus.store import write_jsonl_atomic",
            "try:",
            "    write_jsonl_atomic(Path(sys.argv[1]), ({'value': 'raced'},))",
            "except CorpusMutationLockBusy:",
            "    raise SystemExit(42)",
        )
    )

    with corpus_mutation_lease(target):
        completed = subprocess.run(
            [sys.executable, "-c", program, str(target)],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

    assert completed.returncode == 42, completed.stderr
    assert target.read_bytes() == before


def test_backend_acquire_and_close_failures_release_process_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    target = paths.manifests / "rights.jsonl"
    real_open = locking._open_backend

    def fail_open(_local_data: Path) -> object:
        raise OSError("injected acquire failure")

    monkeypatch.setattr(locking, "_open_backend", fail_open)
    with pytest.raises(OSError, match="acquire failure"):
        corpus_mutation_lease(target)

    class FailClose:
        def close(self) -> None:
            raise OSError("injected close failure")

        def close_after_fork(self) -> None:
            raise AssertionError("not forked")

    monkeypatch.setattr(locking, "_open_backend", lambda _root: FailClose())
    lease = corpus_mutation_lease(target)
    with pytest.raises(OSError, match="close failure"):
        lease.close()

    monkeypatch.setattr(locking, "_open_backend", real_open)
    with corpus_mutation_lease(target):
        store.write_jsonl_atomic(target, ({"value": "recovered"},))
    assert store.read_jsonl(target) == ({"value": "recovered"},)


def test_nested_refcount_keeps_other_threads_blocked_until_last_close(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = paths.manifests / "rights.jsonl"
    outer = corpus_mutation_lease(target)
    inner = corpus_mutation_lease(target)

    outer.close()
    failures = _run_in_thread(lambda: store.write_jsonl_atomic(target, ({"value": 1},)))
    assert len(failures) == 1
    assert isinstance(failures[0], CorpusMutationLockBusy)

    inner.close()
    assert not _run_in_thread(lambda: store.write_jsonl_atomic(target, ({"value": 2},)))
    assert store.read_jsonl(target) == ({"value": 2},)


def test_lease_rejects_cross_thread_close_without_changing_owner_state(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = paths.manifests / "rights.jsonl"
    lease = corpus_mutation_lease(target)

    close_failures = _run_in_thread(lease.close)
    assert len(close_failures) == 1
    assert isinstance(close_failures[0], RuntimeError)
    assert "owning thread" in str(close_failures[0])
    busy_failures = _run_in_thread(
        lambda: store.write_jsonl_atomic(target, ({"value": "blocked"},))
    )
    assert len(busy_failures) == 1
    assert isinstance(busy_failures[0], CorpusMutationLockBusy)

    lease.close()
    assert not _run_in_thread(lambda: store.write_jsonl_atomic(target, ({"value": "released"},)))
    assert store.read_jsonl(target) == ({"value": "released"},)


def test_lease_argument_and_lifecycle_contracts(tmp_path: Path) -> None:
    paths = _paths(tmp_path / "one")
    other_paths = _paths(tmp_path / "two")
    nonlocal_target = tmp_path / "ordinary" / "rows.jsonl"

    with pytest.raises(ValueError, match="at least one target"):
        corpus_mutation_lease()
    noop = corpus_mutation_lease(nonlocal_target)
    with noop:
        pass
    noop.close()
    with pytest.raises(RuntimeError, match="already closed"):
        noop.__enter__()
    with pytest.raises(ValueError, match="same local-data root"):
        corpus_mutation_lease(paths.manifests / "rows.jsonl", nonlocal_target)
    with pytest.raises(ValueError, match="same local-data root"):
        corpus_mutation_lease(
            paths.manifests / "rows.jsonl",
            other_paths.manifests / "rows.jsonl",
        )
    assert not (tmp_path / "ordinary" / "local-data").exists()


def test_registry_pid_reset_and_inconsistent_entries_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)

    class InheritedBackend:
        def __init__(self) -> None:
            self.closed_after_fork = False

        def close(self) -> None:
            raise AssertionError("normal close is not valid during child reset")

        def close_after_fork(self) -> None:
            self.closed_after_fork = True

    inherited_backend = InheritedBackend()
    stale = locking._Entry(depth=1, backend=inherited_backend)
    monkeypatch.setattr(locking, "_registry_pid", -1)
    monkeypatch.setattr(locking, "_entries", {"stale": stale})
    entry = locking._entry_for(paths.local_data)
    assert "stale" not in locking._entries
    assert inherited_backend.closed_after_fork

    assert entry.lock.acquire(blocking=False)
    entry.depth = 1
    entry.backend = None
    try:
        with pytest.raises(RuntimeError, match="registry is inconsistent"):
            corpus_mutation_lease(paths.manifests / "rows.jsonl")
    finally:
        entry.depth = 0
        entry.lock.release()

    broken = locking._Entry()
    assert broken.lock.acquire(blocking=False)
    broken.depth = 1
    lease = locking.CorpusMutationLease(broken)
    with pytest.raises(RuntimeError, match="registry is inconsistent"):
        lease.close()
    assert broken.lock.acquire(blocking=False)
    broken.lock.release()


def test_inherited_stale_lease_close_is_noop_for_parent_owned_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backend:
        def close(self) -> None:
            raise AssertionError("stale child lease must not close the parent backend")

        def close_after_fork(self) -> None:
            raise AssertionError("registry reset owns inherited backend cleanup")

    entry = locking._Entry(depth=1, backend=Backend())
    assert entry.lock.acquire(blocking=False)
    lease = locking.CorpusMutationLease(entry)
    owner_pid = lease._owner_pid

    with monkeypatch.context() as child:
        child.setattr(locking.os, "getpid", lambda: owner_pid + 1)
        lease.close()

    assert lease._closed
    assert entry.depth == 1
    assert entry.backend is not None
    entry.depth = 0
    entry.backend = None
    entry.lock.release()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork contract")
def test_forked_child_drops_inherited_lease_without_releasing_parent_flock(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    target = paths.manifests / "rights.jsonl"
    lease = corpus_mutation_lease(target)
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - exercised on POSIX hosts
        os.close(read_fd)
        payload = b"error"
        try:
            lease.close()
            try:
                child_lease = corpus_mutation_lease(target)
            except CorpusMutationLockBusy:
                payload = b"busy"
            else:
                child_lease.close()
                payload = b"acquired"
        finally:
            os.write(write_fd, payload)
            os.close(write_fd)
            os._exit(0)

    os.close(write_fd)
    try:
        payload = os.read(read_fd, 32)
        _, status = os.waitpid(child_pid, 0)
    finally:
        os.close(read_fd)
        lease.close()

    assert os.waitstatus_to_exitcode(status) == 0
    assert payload == b"busy"
    with corpus_mutation_lease(target):
        pass


@pytest.mark.skipif(sys.platform != "win32", reason="Windows handle cleanup contract")
def test_windows_backend_closes_every_handle_and_raises_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    first = object()
    second = object()

    def fail(handle: object) -> None:
        calls.append(handle)
        raise OSError(f"close-{len(calls)}")

    monkeypatch.setattr(locking, "_windows_close_handle", fail)
    with pytest.raises(OSError, match="close-1"):
        locking._WindowsBackend(second, first).close()
    assert calls == [first, second]


def test_review_atomic_helper_uses_the_shared_mutation_lease(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = paths.alignments / "runs" / "review" / "automatic.json"

    with corpus_mutation_lease(target):
        failures = _run_in_thread(lambda: review._write_bytes_atomic(target, b"raced"))

    assert len(failures) == 1
    assert isinstance(failures[0], CorpusMutationLockBusy)
    assert not target.exists()


def test_inventory_intake_writer_uses_the_shared_mutation_lease(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = paths.manifests / "intake.csv"

    with corpus_mutation_lease(target):
        failures = _run_in_thread(lambda: write_intake_skeleton(paths, target))

    assert len(failures) == 1
    assert isinstance(failures[0], CorpusMutationLockBusy)
    assert not target.exists()


def test_cli_atomic_text_helper_uses_the_shared_mutation_lease(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = paths.alignments / "runtime" / "environment.txt"

    with corpus_mutation_lease(target):
        failures = _run_in_thread(lambda: cli._write_text_atomic(target, "raced"))

    assert len(failures) == 1
    assert isinstance(failures[0], CorpusMutationLockBusy)
    assert not target.exists()


def test_persist_transitions_holds_one_lease_across_events_and_recordings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    recordings_path = paths.manifests / "recordings.jsonl"
    events_path = paths.alignments / "runs" / "test" / "processing-events.jsonl"
    rights_path = paths.manifests / "rights.jsonl"
    original, updated, event = _transition()
    store.write_jsonl_atomic(recordings_path, (original.to_dict(),))
    store.write_jsonl_atomic(rights_path, ({"value": "stable"},))
    rights_before = rights_path.read_bytes()
    failures: list[BaseException] = []
    real_write = store._write_jsonl_in_directory

    def write_and_compete(
        path: Path,
        rows: object,
        directory_fd: int | None,
    ) -> None:
        real_write(path, rows, directory_fd)  # type: ignore[arg-type]
        if path == events_path:
            failures.extend(
                _run_in_thread(lambda: store.write_jsonl_atomic(rights_path, ({"value": "raced"},)))
            )

    monkeypatch.setattr(store, "_write_jsonl_in_directory", write_and_compete)
    store.persist_recording_transitions(
        recordings_path=recordings_path,
        events_path=events_path,
        recordings=(updated,),
        events=(event,),
    )

    assert len(failures) == 1
    assert isinstance(failures[0], CorpusMutationLockBusy)
    assert rights_path.read_bytes() == rights_before
    assert store.read_jsonl(recordings_path) == (updated.to_dict(),)
    assert store.read_jsonl(events_path) == (event.to_dict(),)


def test_corpus_mutation_lease_rejects_nonregular_lock_file(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    lock_path = paths.manifests / ".corpus-mutation.lock"
    lock_path.mkdir()
    target = paths.manifests / "rights.jsonl"

    with pytest.raises((OSError, ValueError), match=r"regular|directory"):
        store.write_jsonl_atomic(target, ({"value": "blocked"},))

    assert not target.exists()


def test_corpus_mutation_lease_rejects_manifest_alias(tmp_path: Path) -> None:
    local_data = tmp_path / "local-data"
    external = tmp_path / "external-manifests"
    local_data.mkdir()
    external.mkdir()
    manifests = local_data / "manifests"
    try:
        manifests.symlink_to(external, target_is_directory=True)
    except OSError as error:
        if sys.platform != "win32":
            pytest.skip(f"directory symlinks unavailable: {error}")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(manifests), str(external)],
            capture_output=True,
            check=False,
            encoding="utf-8",
            text=True,
        )
        if completed.returncode:
            pytest.skip(f"directory aliases unavailable: {error}; {completed.stderr}")

    with pytest.raises(ValueError, match=r"alias|canonical|reparse"):
        store.write_jsonl_atomic(manifests / "rights.jsonl", ({"value": "blocked"},))

    assert not (external / "rights.jsonl").exists()
