from __future__ import annotations

import errno
import hashlib
import json
import math
import shutil
import stat
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from latintts.corpus import cli
from latintts.corpus import manifest as manifest_module
from latintts.corpus import report as report_module
from latintts.corpus import review as review_module
from latintts.corpus.cli import align_corpus, pair_corpus
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure, CorpusState, IssueCode
from latintts.corpus.pairing import (
    _pairing_cache_key,
    pairing_from_dict,
    pairing_to_dict,
)
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import (
    AudioMetadata,
    ProcessingEvent,
    RecordingRecord,
    advance_recording,
)
from latintts.corpus.report import (
    PilotMetrics,
    ReportTelemetry,
    build_report,
    classify_scale_readiness,
)
from latintts.corpus.review import read_textgrid, write_textgrid
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from tests.corpus.test_manifest_cli import (
    _build_with_fake_audio,
    _reviewed_project,
    _set_decisions,
    _two_recording_aligned_project,
    _write_rights,
    export_review_bundle,
    import_review_bundle,
)
from tests.corpus.test_pair_cli import _segment, _set_up
from tests.corpus.test_pairing import _audio_command, _FakeAligner
from tests.corpus.test_review_cli import _confirm_pairing_correction


def test_operator_guide_defines_complete_telemetry_and_error_recovery_contract() -> None:
    guide = (Path(__file__).parents[2] / "docs/corpus/alignment-pilot-operator-guide.md").read_text(
        encoding="utf-8"
    )

    assert "正式 `pair` 与 `align` 的全部 forced-alignment 工作负载" in guide
    assert "不能用仅命中 cache 的 `align`" in guide
    assert "停顿单元" in guide
    assert "pairing correction" in guide
    assert "模型权重与 Hugging Face cache" in guide
    assert "lock、临时文件、`report.json` 和 `report.md`" in guide
    assert "`.recovery.`" in guide
    assert "`.report-output-transaction.rolled-back.json`" in guide
    assert "`.report-output-transaction.completed.json`" in guide
    assert "`intent.completed`" in guide
    assert "只允许 roll-forward" in guide
    assert "rolled-back 状态只继续清理" in guide
    assert "协作锁" in guide
    assert "进程中断/重启" in guide
    assert "不承诺 OS crash 或断电" in guide
    assert "绝对路径" in guide
    for code in IssueCode:
        assert f"`{code.value}`" in guide


def test_design_spec_registers_report_output_recovery_error() -> None:
    design = (
        Path(__file__).parents[2]
        / "docs/superpowers/specs/2026-07-19-latintts-corpus-alignment-pilot-design.md"
    ).read_text(encoding="utf-8")

    assert "`REPORT_OUTPUT_RECOVERY_REQUIRED`" in design
    assert "进程中断/重启" in design
    assert "不承诺 OS crash 或断电" in design


def _valid_report_transaction_intent_raw() -> dict[str, Any]:
    directory_mode = stat.S_IFDIR | 0o700
    file_mode = stat.S_IFREG | 0o600

    def snapshot(inode: int, size: int, sha256: str) -> dict[str, object]:
        return {
            "device": 1,
            "inode": inode,
            "mode": file_mode,
            "size": size,
            "sha256": sha256,
        }

    outputs: list[dict[str, object]] = []
    for index, target_name in enumerate(("report.json", "report.md"), start=1):
        outputs.append(
            {
                "target_name": target_name,
                "prepared_name": f".{target_name}.prepared-{index}",
                "backup_name": f".recovery.{target_name}.backup-{index}",
                "restore_name": f".restore.{target_name}.restore-{index}",
                "old_snapshot": snapshot(index * 10, 3, "a" * 64),
                "prepared_snapshot": snapshot(index * 10 + 1, 4, "b" * 64),
                "backup_snapshot": snapshot(index * 10 + 2, 3, "a" * 64),
                "restore_snapshot": snapshot(index * 10 + 3, 3, "a" * 64),
            }
        )
    return {
        "schema_version": "1",
        "transaction_directory": ".report-output-transaction.fixture",
        "directory_identity": {"device": 1, "inode": 2, "mode": directory_mode},
        "outputs": outputs,
    }


def test_report_intent_decoder_accepts_complete_owned_identity_contract() -> None:
    decoded = report_module._decode_report_intent(_valid_report_transaction_intent_raw())

    assert decoded.transaction_directory == ".report-output-transaction.fixture"
    assert tuple(output.target_name for output in decoded.outputs) == (
        "report.json",
        "report.md",
    )


@pytest.mark.parametrize(
    "case",
    (
        "top_fields",
        "schema_version",
        "transaction_directory",
        "directory_fields",
        "directory_negative",
        "directory_mode",
        "outputs_shape",
        "output_fields",
        "target_name",
        "prepared_name",
        "backup_name",
        "restore_name",
        "old_coherence",
        "snapshot_fields",
        "snapshot_negative",
        "snapshot_mode",
        "snapshot_sha256",
        "backup_content",
        "restore_content",
        "output_order",
    ),
)
def test_report_intent_decoder_rejects_noncanonical_ownership_contract(case: str) -> None:
    raw = deepcopy(_valid_report_transaction_intent_raw())
    first = raw["outputs"][0]
    if case == "top_fields":
        raw["extra"] = None
    elif case == "schema_version":
        raw["schema_version"] = "2"
    elif case == "transaction_directory":
        raw["transaction_directory"] = 1
    elif case == "directory_fields":
        raw["directory_identity"]["extra"] = 1
    elif case == "directory_negative":
        raw["directory_identity"]["inode"] = -1
    elif case == "directory_mode":
        raw["directory_identity"]["mode"] = stat.S_IFREG | 0o600
    elif case == "outputs_shape":
        raw["outputs"] = []
    elif case == "output_fields":
        first["extra"] = None
    elif case == "target_name":
        first["target_name"] = "foreign.json"
    elif case == "prepared_name":
        first["prepared_name"] = "prepared.json"
    elif case == "backup_name":
        first["backup_name"] = "backup.json"
    elif case == "restore_name":
        first["restore_name"] = "restore.json"
    elif case == "old_coherence":
        first["restore_name"] = None
    elif case == "snapshot_fields":
        first["prepared_snapshot"]["extra"] = 1
    elif case == "snapshot_negative":
        first["prepared_snapshot"]["size"] = -1
    elif case == "snapshot_mode":
        first["prepared_snapshot"]["mode"] = stat.S_IFDIR | 0o700
    elif case == "snapshot_sha256":
        first["prepared_snapshot"]["sha256"] = "invalid"
    elif case == "backup_content":
        first["backup_snapshot"]["sha256"] = "c" * 64
    elif case == "restore_content":
        first["restore_snapshot"]["size"] = 4
    else:
        raw["outputs"].reverse()

    with pytest.raises(ValueError):
        report_module._decode_report_intent(raw)


@pytest.mark.parametrize("kind", ("invalid-json", "noncanonical-json"))
def test_report_intent_marker_rejects_noncanonical_bytes(tmp_path: Path, kind: str) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    transaction_directory = manifests / ".report-output-transaction.fixture"
    transaction_directory.mkdir()
    metadata = transaction_directory.stat()
    raw = _valid_report_transaction_intent_raw()
    raw["directory_identity"] = {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
    }
    marker = manifests / ".report-output-transaction.json"
    if kind == "invalid-json":
        marker.write_bytes(b"{\n")
    else:
        marker.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        report_module._read_report_intent_marker(manifests, marker)


def _write_canonical_report_intent_marker(
    manifests: Path,
    marker: Path,
    transaction_directory: Path,
) -> report_module._ReportTransactionIntent:
    raw = _valid_report_transaction_intent_raw()
    metadata = transaction_directory.stat()
    raw["transaction_directory"] = transaction_directory.name
    raw["directory_identity"] = {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
    }
    intent = report_module._decode_report_intent(raw)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(report_module._canonical_intent_bytes(intent))
    return intent


def _dummy_loaded_report_transaction(
    manifests: Path,
    intent: report_module._ReportTransactionIntent,
    *,
    state: report_module._ReportTransactionState = report_module._ReportTransactionState.ACTIVE,
) -> report_module._LoadedReportTransaction:
    marker_content = b"fixture marker\n"
    marker_snapshot = report_module._ReportOutputSnapshot(
        device=0,
        inode=0,
        mode=stat.S_IFREG | 0o600,
        size=len(marker_content),
        sha256=hashlib.sha256(marker_content).hexdigest(),
        content=marker_content,
    )
    return report_module._LoadedReportTransaction(
        state=state,
        intent=intent,
        marker_paths=(),
        marker_snapshot=marker_snapshot,
        transaction_directory=manifests / intent.transaction_directory,
        directory_identity=intent.directory_identity,
        directory_present=True,
    )


def test_report_transaction_directory_guards_name_and_file_type(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    regular_file = manifests / ".report-output-transaction.file"
    regular_file.write_bytes(b"not a directory\n")

    with pytest.raises(ValueError, match="directory name is invalid"):
        report_module._transaction_directory_path(manifests, "transaction.fixture")
    with pytest.raises(ValueError, match="must be a directory"):
        report_module._directory_identity(regular_file)


def test_active_report_marker_requires_its_private_directory(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    marker = manifests / ".report-output-transaction.json"
    intent = report_module._decode_report_intent(_valid_report_transaction_intent_raw())
    marker.write_bytes(report_module._canonical_intent_bytes(intent))

    with pytest.raises(ValueError, match="directory is missing"):
        report_module._read_report_intent_marker(manifests, marker)


def test_report_loader_rejects_active_marker_with_private_final_marker(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    transaction_directory = manifests / ".report-output-transaction.fixture"
    transaction_directory.mkdir(parents=True)
    (manifests / ".report-output-transaction.json").write_bytes(b"active\n")
    (transaction_directory / "intent.completed").write_bytes(b"final\n")

    with pytest.raises(ValueError, match="rollback and committed"):
        report_module._load_report_transaction(manifests)


def test_report_loader_rejects_committed_marker_with_unrelated_final_marker(
    tmp_path: Path,
) -> None:
    manifests = tmp_path / "manifests"
    transaction_directory = manifests / ".report-output-transaction.fixture"
    transaction_directory.mkdir(parents=True)
    _write_canonical_report_intent_marker(
        manifests,
        manifests / ".report-output-transaction.completed.json",
        transaction_directory,
    )
    unrelated_directory = manifests / ".report-output-transaction.unrelated"
    unrelated_directory.mkdir()
    (unrelated_directory / "intent.completed").write_bytes(b"unrelated final\n")

    with pytest.raises(ValueError, match="multiple committed"):
        report_module._load_report_transaction(manifests)


def test_report_loader_rejects_multiple_private_final_markers(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    for suffix in ("first", "second"):
        directory = manifests / f".report-output-transaction.{suffix}"
        directory.mkdir(parents=True)
        (directory / "intent.completed").write_bytes(b"final\n")

    with pytest.raises(ValueError, match="multiple final"):
        report_module._load_report_transaction(manifests)


def test_report_loader_rejects_private_final_marker_in_wrong_directory(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    transaction_directory = manifests / ".report-output-transaction.fixture"
    marker_directory = manifests / ".report-output-transaction.marker"
    transaction_directory.mkdir(parents=True)
    marker_directory.mkdir()
    _write_canonical_report_intent_marker(
        manifests,
        marker_directory / "intent.completed",
        transaction_directory,
    )

    with pytest.raises(ValueError, match="wrong directory"):
        report_module._load_report_transaction(manifests)


def _publication_guard_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    report_module._ReportOutputSnapshot,
    report_module._LoadedReportTransaction,
]:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    temporary = manifests / ".report.json.fixture"
    temporary.write_bytes(b"prepared report\n")
    prepared = report_module._prepared_report_snapshot(temporary)
    intent = report_module._decode_report_intent(_valid_report_transaction_intent_raw())
    output = replace(
        intent.outputs[0],
        prepared_name=temporary.name,
        prepared_snapshot=report_module._ReportSnapshotIdentity.from_snapshot(prepared),
    )
    intent = replace(intent, outputs=(output, intent.outputs[1]))
    return (
        temporary,
        manifests / "report.json",
        prepared,
        _dummy_loaded_report_transaction(manifests, intent),
    )


def test_report_publication_requires_active_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary, target, prepared, _loaded = _publication_guard_fixture(tmp_path)
    monkeypatch.setattr(report_module, "_load_report_transaction", lambda _manifests: None)

    with pytest.raises(OSError, match="intent is missing"):
        report_module._publish_report_output(temporary, target, old=None, prepared=prepared)


def test_report_publication_requires_registered_prepared_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary, target, prepared, loaded = _publication_guard_fixture(tmp_path)
    output = replace(loaded.intent.outputs[0], prepared_name=".report.json.other")
    loaded = replace(
        loaded, intent=replace(loaded.intent, outputs=(output, loaded.intent.outputs[1]))
    )
    monkeypatch.setattr(report_module, "_load_report_transaction", lambda _manifests: loaded)

    with pytest.raises(OSError, match="does not match transaction intent"):
        report_module._publish_report_output(temporary, target, old=None, prepared=prepared)


def test_report_publication_rejects_changed_prepared_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary, target, prepared, loaded = _publication_guard_fixture(tmp_path)
    monkeypatch.setattr(report_module, "_load_report_transaction", lambda _manifests: loaded)
    temporary.write_bytes(b"changed prepared report\n")

    with pytest.raises(OSError, match="changed before publication"):
        report_module._publish_report_output(temporary, target, old=None, prepared=prepared)


def test_report_publication_rejects_missing_old_snapshot_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary, target, prepared, loaded = _publication_guard_fixture(tmp_path)
    monkeypatch.setattr(report_module, "_load_report_transaction", lambda _manifests: loaded)

    with pytest.raises(OSError, match="old output identity is inconsistent"):
        report_module._publish_report_output(temporary, target, old=None, prepared=prepared)


def test_report_publication_rejects_mismatched_old_snapshot_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary, target, prepared, loaded = _publication_guard_fixture(tmp_path)
    old_path = temporary.parent / "old-report.json"
    old_path.write_bytes(b"old report\n")
    old = report_module._prepared_report_snapshot(old_path)
    monkeypatch.setattr(report_module, "_load_report_transaction", lambda _manifests: loaded)

    with pytest.raises(OSError, match="old output identity is inconsistent"):
        report_module._publish_report_output(temporary, target, old=old, prepared=prepared)


@pytest.mark.parametrize(
    "claim_name",
    ("_claim_completed_intent", "_claim_rolled_back_intent"),
)
def test_report_marker_handoff_requires_shared_parent(
    tmp_path: Path,
    claim_name: str,
) -> None:
    intent_path = tmp_path / "first" / ".report-output-transaction.json"
    transaction_directory = tmp_path / "second" / ".report-output-transaction.fixture"
    marker_snapshot = _dummy_loaded_report_transaction(
        tmp_path,
        report_module._decode_report_intent(_valid_report_transaction_intent_raw()),
    ).marker_snapshot

    with pytest.raises(OSError, match="private directory disagree"):
        getattr(report_module, claim_name)(intent_path, transaction_directory, marker_snapshot)


def test_private_final_anchor_requires_root_committed_marker(tmp_path: Path) -> None:
    intent_path = tmp_path / ".report-output-transaction.json"
    transaction_directory = tmp_path / ".report-output-transaction.fixture"
    marker_snapshot = _dummy_loaded_report_transaction(
        tmp_path,
        report_module._decode_report_intent(_valid_report_transaction_intent_raw()),
    ).marker_snapshot

    with pytest.raises(OSError, match="only the committed"):
        report_module._claim_final_intent(intent_path, transaction_directory, marker_snapshot)


def test_private_final_anchor_rejects_changed_root_marker(tmp_path: Path) -> None:
    intent_path = tmp_path / ".report-output-transaction.completed.json"
    transaction_directory = tmp_path / ".report-output-transaction.fixture"
    transaction_directory.mkdir()
    intent_path.write_bytes(b"original marker\n")
    marker_snapshot = report_module._prepared_report_snapshot(intent_path)
    intent_path.write_bytes(b"changed marker\n")

    with pytest.raises(OSError, match="changed before final anchoring"):
        report_module._claim_final_intent(intent_path, transaction_directory, marker_snapshot)


def test_root_anchor_rebuild_rejects_changed_private_marker(tmp_path: Path) -> None:
    final_path = tmp_path / ".report-output-transaction.fixture" / "intent.completed"
    final_path.parent.mkdir()
    final_path.write_bytes(b"original marker\n")
    marker_snapshot = report_module._prepared_report_snapshot(final_path)
    final_path.write_bytes(b"changed marker\n")

    with pytest.raises(OSError, match="changed before rebuilding"):
        report_module._claim_root_committed_intent(final_path, tmp_path, marker_snapshot)


def test_terminal_cleanup_rejects_registered_root_payload(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    intent = report_module._decode_report_intent(_valid_report_transaction_intent_raw())
    (manifests / intent.outputs[0].prepared_name).write_bytes(b"remaining payload\n")

    with pytest.raises(OSError, match="root payload remains"):
        report_module._require_registered_root_payloads_absent(manifests, intent)


@pytest.mark.parametrize("case", ("missing", "changed"))
def test_owned_report_cleanup_rejects_missing_or_changed_file(tmp_path: Path, case: str) -> None:
    path = tmp_path / "owned-report-artifact"
    if case == "changed":
        path.write_bytes(b"current artifact\n")
        current = report_module._prepared_report_snapshot(path)
        expected = replace(
            report_module._ReportSnapshotIdentity.from_snapshot(current),
            sha256="0" * 64,
        )
    else:
        expected = report_module._ReportSnapshotIdentity(
            device=0,
            inode=0,
            mode=stat.S_IFREG | 0o600,
            size=0,
            sha256=hashlib.sha256(b"").hexdigest(),
        )

    with pytest.raises(OSError, match="owned report transaction file"):
        report_module._delete_owned_report_file(path, expected)


def test_owned_report_cleanup_rejects_replaced_transaction_directory(tmp_path: Path) -> None:
    directory = tmp_path / ".report-output-transaction.fixture"
    directory.mkdir()
    expected = replace(
        report_module._directory_identity(directory), inode=directory.stat().st_ino + 1
    )

    with pytest.raises(OSError, match="directory identity changed"):
        report_module._remove_owned_transaction_directory(directory, expected)


def test_report_claim_syncs_both_directories_after_cross_directory_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_directory = tmp_path / "source"
    claimed_directory = tmp_path / "claimed"
    source_directory.mkdir()
    claimed_directory.mkdir()
    target = source_directory / "report.json"
    claimed = claimed_directory / "report.json.displaced"
    target.write_bytes(b"old report\n")
    expected = report_module._ReportSnapshotIdentity.from_snapshot(
        report_module._prepared_report_snapshot(target)
    )
    synced: list[Path] = []
    monkeypatch.setattr(report_module, "_sync_report_directory", synced.append)

    report_module._claim_report_output(
        target,
        claimed,
        expected=expected,
        operation="test claim",
    )

    assert synced == [source_directory, claimed_directory]


def test_report_claim_syncs_shared_directory_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "report.json"
    claimed = tmp_path / "report.json.displaced"
    target.write_bytes(b"old report\n")
    expected = report_module._ReportSnapshotIdentity.from_snapshot(
        report_module._prepared_report_snapshot(target)
    )
    synced: list[Path] = []
    monkeypatch.setattr(report_module, "_sync_report_directory", synced.append)

    report_module._claim_report_output(
        target,
        claimed,
        expected=expected,
        operation="test claim",
    )

    assert synced == [tmp_path]


def test_report_claim_compensation_syncs_both_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_directory = tmp_path / "source"
    claimed_directory = tmp_path / "claimed"
    source_directory.mkdir()
    claimed_directory.mkdir()
    target = source_directory / "report.json"
    claimed = claimed_directory / "report.json.displaced"
    old_content = b"old report\n"
    target.write_bytes(old_content)
    current = report_module._prepared_report_snapshot(target)
    mismatched = replace(
        report_module._ReportSnapshotIdentity.from_snapshot(current),
        sha256="0" * 64,
    )
    synced: list[Path] = []
    monkeypatch.setattr(report_module, "_sync_report_directory", synced.append)

    with pytest.raises(report_module._ReportPublicationVerificationError):
        report_module._claim_report_output(
            target,
            claimed,
            expected=mismatched,
            operation="test compensation",
        )

    assert target.read_bytes() == old_content
    assert not claimed.exists()
    assert synced == [
        source_directory,
        claimed_directory,
        claimed_directory,
        source_directory,
    ]


class _FailingRenamePrimitive:
    def __init__(self) -> None:
        self.argtypes: object = None
        self.restype: object = None
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        return -1


def _patch_failing_linux_renameat2(
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> _FailingRenamePrimitive:
    primitive = _FailingRenamePrimitive()
    library = type("FakeCLibrary", (), {"renameat2": primitive})()
    monkeypatch.setattr(report_module.sys, "platform", "linux")
    monkeypatch.setattr(report_module.ctypes, "CDLL", lambda *_args, **_kwargs: library)
    monkeypatch.setattr(report_module.ctypes, "get_errno", lambda: error_number)
    return primitive


def test_posix_noreplace_rename_propagates_eexist_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"source\n")
    destination.write_bytes(b"destination\n")
    primitive = _patch_failing_linux_renameat2(monkeypatch, errno.EEXIST)
    monkeypatch.setattr(
        report_module.os,
        "link",
        lambda *_args, **_kwargs: pytest.fail("rename must not fall back to hard links"),
    )
    monkeypatch.setattr(
        report_module.os,
        "rename",
        lambda *_args, **_kwargs: pytest.fail("rename must not fall back to os.rename"),
    )

    with pytest.raises(OSError) as failure:
        report_module._rename_noreplace_posix(source, destination)

    assert failure.value.errno == errno.EEXIST
    assert len(primitive.calls) == 1
    assert source.read_bytes() == b"source\n"
    assert destination.read_bytes() == b"destination\n"


def test_posix_noreplace_rename_fails_closed_without_renameat2_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"source\n")
    monkeypatch.setattr(report_module.sys, "platform", "linux")
    monkeypatch.setattr(report_module.ctypes, "CDLL", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        report_module.os,
        "link",
        lambda *_args, **_kwargs: pytest.fail("rename must not fall back to hard links"),
    )
    monkeypatch.setattr(
        report_module.os,
        "rename",
        lambda *_args, **_kwargs: pytest.fail("rename must not fall back to os.rename"),
    )

    with pytest.raises(OSError) as failure:
        report_module._rename_noreplace_posix(source, destination)

    assert failure.value.errno == errno.ENOTSUP
    assert source.read_bytes() == b"source\n"
    assert not destination.exists()


@pytest.mark.parametrize("error_name", ("ENOSYS", "ENOTSUP"))
def test_posix_noreplace_rename_fails_closed_when_primitive_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_name: str,
) -> None:
    if not hasattr(errno, error_name):
        pytest.skip(f"{error_name} is unavailable on this platform")
    error_number = getattr(errno, error_name)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"source\n")
    primitive = _patch_failing_linux_renameat2(monkeypatch, error_number)
    monkeypatch.setattr(
        report_module.os,
        "link",
        lambda *_args, **_kwargs: pytest.fail("rename must not fall back to hard links"),
    )
    monkeypatch.setattr(
        report_module.os,
        "rename",
        lambda *_args, **_kwargs: pytest.fail("rename must not fall back to os.rename"),
    )

    with pytest.raises(OSError) as failure:
        report_module._rename_noreplace_posix(source, destination)

    assert failure.value.errno == error_number
    assert len(primitive.calls) == 1
    assert source.read_bytes() == b"source\n"
    assert not destination.exists()


def _copy_config(project: Path) -> None:
    destination = project / "config" / "corpus" / "pilot-v1.json"
    destination.parent.mkdir(parents=True)
    source = Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json"
    destination.write_bytes(source.read_bytes())


def _recording_for_report_unit_tests() -> RecordingRecord:
    return RecordingRecord(
        schema_version="1",
        recording_id="rec-unit",
        relative_path="raw/spoken/unit.wav",
        sha256="a" * 64,
        content_type="spoken",
        title_or_citation="Unit fixture",
        speaker_id="speaker-unit",
        rights_id="rights-unit",
        notes="",
        metadata=AudioMetadata(
            duration_seconds=1.0,
            sample_rate=16_000,
            channels=1,
            codec="pcm_s16le",
            bit_rate=256_000,
        ),
        state=CorpusState.APPROVED,
    )


def test_report_cli_loads_config_dispatches_and_prints_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _copy_config(tmp_path)
    calls: list[tuple[Path, str]] = []

    def report(paths: object, config: object) -> object:
        calls.append((paths.project_root, config.raw["corpus_version"]))  # type: ignore[attr-defined]
        decision = type("Decision", (), {"value": "optimize"})()
        return type("Report", (), {"scale_decision": decision})()

    monkeypatch.setattr(cli, "build_report", report, raising=False)

    assert cli.main(["--project-root", str(tmp_path), "report"]) == 0
    assert calls == [(tmp_path, "corpus-v1")]
    assert capsys.readouterr().out == "corpus-report: optimize\n"


@pytest.mark.parametrize(
    ("failure", "exit_code", "message"),
    (
        (CorpusFailure("REVIEW_REQUIRED", "review first"), 1, "REVIEW_REQUIRED: review first\n"),
        (ValueError("bad telemetry"), 2, "MANIFEST_SCHEMA_MISMATCH: bad telemetry\n"),
    ),
)
def test_report_cli_maps_stable_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
    exit_code: int,
    message: str,
) -> None:
    _copy_config(tmp_path)
    monkeypatch.setattr(
        cli,
        "build_report",
        lambda *_args: (_ for _ in ()).throw(failure),
        raising=False,
    )

    assert cli.main(["--project-root", str(tmp_path), "report"]) == exit_code
    assert capsys.readouterr().err == message


def test_scale_readiness_is_scalable_at_all_green_boundaries() -> None:
    metrics = PilotMetrics(
        input_duration_seconds=600.0,
        speech_duration_seconds=500.0,
        auto_pairing_correct_ratio=0.95,
        boundary_unchanged_ratio=0.85,
        review_minutes_per_audio_minute=3.0,
        approved_speech_ratio=0.80,
        gpu_realtime_factor=0.2,
        peak_gpu_memory_bytes=4_000_000_000,
    )

    assert classify_scale_readiness(metrics).value == "scalable"


def test_any_red_metric_makes_batch_processing_not_ready() -> None:
    metrics = PilotMetrics(
        input_duration_seconds=600.0,
        speech_duration_seconds=500.0,
        auto_pairing_correct_ratio=0.84,
        boundary_unchanged_ratio=0.90,
        review_minutes_per_audio_minute=2.0,
        approved_speech_ratio=0.90,
        gpu_realtime_factor=0.2,
        peak_gpu_memory_bytes=4_000_000_000,
    )

    assert classify_scale_readiness(metrics).value == "not_ready"


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"auto_pairing_correct_ratio": 0.85}, "optimize"),
        ({"auto_pairing_correct_ratio": 0.949999}, "optimize"),
        ({"boundary_unchanged_ratio": 0.70}, "optimize"),
        ({"boundary_unchanged_ratio": 0.849999}, "optimize"),
        ({"review_minutes_per_audio_minute": 3.000001}, "optimize"),
        ({"review_minutes_per_audio_minute": 6.0}, "optimize"),
        ({"approved_speech_ratio": 0.60}, "optimize"),
        ({"approved_speech_ratio": 0.799999}, "optimize"),
    ),
)
def test_scale_readiness_uses_exact_yellow_boundaries(
    changes: dict[str, float], expected: str
) -> None:
    values: dict[str, float | int] = {
        "input_duration_seconds": 600.0,
        "speech_duration_seconds": 500.0,
        "auto_pairing_correct_ratio": 0.95,
        "boundary_unchanged_ratio": 0.85,
        "review_minutes_per_audio_minute": 3.0,
        "approved_speech_ratio": 0.80,
        "gpu_realtime_factor": 0.2,
        "peak_gpu_memory_bytes": 4_000_000_000,
    }
    values.update(changes)

    metrics = PilotMetrics(**values)  # type: ignore[arg-type]

    assert classify_scale_readiness(metrics).value == expected


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("input_duration_seconds", 0.0),
        ("input_duration_seconds", True),
        ("speech_duration_seconds", 0.0),
        ("speech_duration_seconds", 601.0),
        ("auto_pairing_correct_ratio", -0.1),
        ("boundary_unchanged_ratio", 1.1),
        ("review_minutes_per_audio_minute", -0.1),
        ("approved_speech_ratio", math.nan),
        ("gpu_realtime_factor", math.inf),
        ("peak_gpu_memory_bytes", 0),
        ("peak_gpu_memory_bytes", True),
    ),
)
def test_pilot_metrics_reject_invalid_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "input_duration_seconds": 600.0,
        "speech_duration_seconds": 500.0,
        "auto_pairing_correct_ratio": 0.95,
        "boundary_unchanged_ratio": 0.85,
        "review_minutes_per_audio_minute": 3.0,
        "approved_speech_ratio": 0.80,
        "gpu_realtime_factor": 0.2,
        "peak_gpu_memory_bytes": 4_000_000_000,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError)):
        PilotMetrics(**values)  # type: ignore[arg-type]


def test_scale_readiness_rejects_non_metrics() -> None:
    with pytest.raises(TypeError, match="PilotMetrics"):
        classify_scale_readiness(object())  # type: ignore[arg-type]


def test_report_public_entrypoint_rejects_invalid_types_and_config(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)

    with pytest.raises(TypeError, match="CorpusPaths and CorpusConfig"):
        build_report(object(), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="schema 1 corpus-v1"):
        build_report(
            paths,
            CorpusConfig(
                raw={"schema_version": "2", "corpus_version": "corpus-v1"}, digest="a" * 64
            ),
        )


def _telemetry_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "1",
        "text_preparation_seconds": 10.0,
        "review_seconds": 20.0,
        "alignment_wall_seconds": 1.48,
        "gpu_retry_count": 0,
        "peak_gpu_memory_bytes": 4_000_000_000,
        "pilot_storage_bytes": 1_000,
    }
    row.update(changes)
    return row


def _completed_two_recording_project(
    tmp_path: Path, *, decision: str = "approved"
) -> tuple[object, object]:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    for group in export_review_bundle(paths, config):
        _set_decisions(group, decision=decision)
    assert import_review_bundle(paths, config)
    assert _build_with_fake_audio(paths, config)
    return paths, config


def _completed_corrected_pairing_project(tmp_path: Path) -> tuple[object, object]:
    paths, _ = _set_up(tmp_path)
    config_path = tmp_path / "config" / "corpus" / "pilot-v1.json"
    config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    config_raw["pairing"]["minimum_duration_ratio"] = 1.0
    config_raw["pairing"]["maximum_duration_ratio"] = 1.0
    config_path.write_text(json.dumps(config_raw), encoding="utf-8")
    config = CorpusConfig.load(config_path)
    source_one = paths.raw_spoken / "rec-1.wav"
    source_two = paths.raw_spoken / "rec-2.wav"
    shutil.copyfile(source_one, source_two)
    recordings = list(read_jsonl(paths.manifests / "recordings.jsonl"))
    second_recording = deepcopy(recordings[0])
    second_recording.update(
        recording_id="rec-2",
        relative_path="raw/spoken/rec-2.wav",
        title_or_citation="rec-2",
        sha256=hashlib.sha256(source_two.read_bytes()).hexdigest(),
    )
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (*recordings, second_recording))
    transcripts = list(read_jsonl(paths.manifests / "transcripts.jsonl"))
    second_transcript = deepcopy(transcripts[0])
    second_transcript["recording_id"] = "rec-2"
    for unit in second_transcript["spoken_units"]:
        unit["unit_id"] = unit["unit_id"].replace("rec-1-", "rec-2-")
    write_jsonl_atomic(paths.manifests / "transcripts.jsonl", (*transcripts, second_transcript))
    write_jsonl_atomic(
        paths.manifests / "pilot-selection.json",
        (
            {
                "schema_version": "1",
                "strategy": "explicit-v1",
                "recording_ids": ["rec-1", "rec-2"],
                "inventory_hashes": [recordings[0]["sha256"], second_recording["sha256"]],
            },
        ),
    )
    _segment(paths, config)
    backend = _FakeAligner()
    assert not pair_corpus(
        paths,
        config,
        backend,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    for correction in export_review_bundle(paths, config):
        _confirm_pairing_correction(correction)
    assert import_review_bundle(paths, config)
    assert align_corpus(paths, config, backend)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    assert import_review_bundle(paths, config)
    _write_rights(paths)
    assert _build_with_fake_audio(paths, config)
    return paths, config


def _first_selected_run_directory(paths: object, config: object) -> Path:
    selection = read_jsonl(paths.manifests / "pilot-selection.json")[0]  # type: ignore[attr-defined]
    recording_id = selection["recording_ids"][0]
    return paths.alignments / "runs" / config.digest / recording_id  # type: ignore[attr-defined]


def _first_review_automatic_path(paths: object, config: object) -> Path:
    review_root = _first_selected_run_directory(paths, config) / "review"
    return next(review_root.glob("*/automatic.json"))


def _first_analysis_audio_path(paths: object, config: object) -> Path:
    segmentation = json.loads(
        (_first_selected_run_directory(paths, config) / "segmentation.json").read_text(
            encoding="utf-8"
        )
    )
    return paths.resolve_local(segmentation["analysis_audio"]["relative_path"])  # type: ignore[attr-defined]


def _first_lossless_audio_path(paths: object) -> Path:
    segment = read_jsonl(paths.manifests / "segments.jsonl")[0]  # type: ignore[attr-defined]
    return paths.resolve_local(segment["derived_audio_relative_path"])  # type: ignore[attr-defined]


def _completed_project_with_unselected_spoken(tmp_path: Path) -> tuple[object, object]:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    for group in export_review_bundle(paths, config):
        _set_decisions(group)
    assert import_review_bundle(paths, config)
    recordings = list(read_jsonl(paths.manifests / "recordings.jsonl"))
    unselected_path = paths.raw_spoken / "rec-unselected.wav"
    unselected_path.write_bytes(b"unselected immutable fixture")
    unselected = deepcopy(recordings[0])
    unselected.update(
        recording_id="rec-unselected",
        relative_path="raw/spoken/rec-unselected.wav",
        sha256=hashlib.sha256(unselected_path.read_bytes()).hexdigest(),
        title_or_citation="unselected full-corpus item",
        state="INVENTORIED",
    )
    write_jsonl_atomic(paths.manifests / "recordings.jsonl", (*recordings, unselected))

    assert _build_with_fake_audio(paths, config)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    return paths, config


def test_manifest_build_preserves_unselected_spoken_inventory_for_projection(
    tmp_path: Path,
) -> None:
    paths, config = _completed_project_with_unselected_spoken(tmp_path)

    after = read_jsonl(paths.manifests / "recordings.jsonl")
    assert [row["recording_id"] for row in after] == [
        "rec-1",
        "rec-2",
        "rec-unselected",
    ]
    assert [row["state"] for row in after] == ["APPROVED", "APPROVED", "INVENTORIED"]
    assert len(read_jsonl(paths.manifests / "segments.jsonl")) == 12

    report = build_report(paths, config)
    assert report.metrics.input_duration_seconds == 20.0
    assert report.full_corpus_spoken_duration_seconds == 30.0
    assert report.text_unit_count == 6
    assert report.reading_count == 12
    assert report.projected_full_corpus_person_hours == pytest.approx(45.0 / 3600.0)
    assert report.projected_gpu_hours == pytest.approx(1.48 * 1.5 / 3600.0)
    assert report.projected_storage_bytes == 1_500


@pytest.mark.parametrize("tamper", ("changed", "missing"))
def test_build_report_rejects_selected_raw_drift(tmp_path: Path, tamper: str) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    selected = read_jsonl(paths.manifests / "recordings.jsonl")[0]
    source = paths.resolve_local(selected["relative_path"])
    if tamper == "changed":
        source.write_bytes(source.read_bytes() + b"tampered")
    else:
        source.unlink()

    with pytest.raises(CorpusFailure) as failure:
        build_report(paths, config)

    assert failure.value.code == "INVENTORY_HASH_MISMATCH"


@pytest.mark.parametrize("tamper", ("changed", "missing"))
def test_build_report_rejects_unselected_spoken_raw_drift(tmp_path: Path, tamper: str) -> None:
    paths, config = _completed_project_with_unselected_spoken(tmp_path)
    unselected = read_jsonl(paths.manifests / "recordings.jsonl")[-1]
    source = paths.resolve_local(unselected["relative_path"])
    if tamper == "changed":
        source.write_bytes(source.read_bytes() + b"tampered")
    else:
        source.unlink()

    with pytest.raises(CorpusFailure) as failure:
        build_report(paths, config)

    assert failure.value.code == "INVENTORY_HASH_MISMATCH"


def test_build_report_rejects_unselected_spoken_metadata_drift(tmp_path: Path) -> None:
    paths, config = _completed_project_with_unselected_spoken(tmp_path)
    recordings_path = paths.manifests / "recordings.jsonl"
    recordings = list(read_jsonl(recordings_path))
    recordings[-1]["metadata"]["duration_seconds"] = 1_000.0
    write_jsonl_atomic(recordings_path, recordings)

    with pytest.raises(ValueError, match=r"inventory.*evidence|terminal.*evidence"):
        build_report(paths, config)


@pytest.mark.parametrize("artifact", ("analysis", "lossless"))
@pytest.mark.parametrize("tamper", ("changed", "missing"))
def test_build_report_rejects_missing_or_tampered_derived_audio_without_publishing(
    tmp_path: Path, artifact: str, tamper: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    output_paths = (paths.manifests / "report.json", paths.manifests / "report.md")
    before = tuple(path.read_bytes() for path in output_paths)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    audio_path = (
        _first_analysis_audio_path(paths, config)
        if artifact == "analysis"
        else _first_lossless_audio_path(paths)
    )
    if tamper == "changed":
        audio_path.write_bytes(audio_path.read_bytes() + b"tampered")
    else:
        audio_path.unlink()

    with pytest.raises(CorpusFailure) as failure:
        build_report(paths, config)

    assert failure.value.code == "CACHE_ARTIFACT_INVALID"
    assert tuple(path.read_bytes() for path in output_paths) == before


@pytest.mark.parametrize("artifact", ("analysis", "lossless"))
def test_build_report_rejects_derived_audio_alias_without_publishing(
    tmp_path: Path, artifact: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    build_report(paths, config)
    output_paths = (paths.manifests / "report.json", paths.manifests / "report.md")
    before = tuple(path.read_bytes() for path in output_paths)
    audio_path = (
        _first_analysis_audio_path(paths, config)
        if artifact == "analysis"
        else _first_lossless_audio_path(paths)
    )
    real_path = audio_path.with_name(f"real-{audio_path.name}")
    audio_path.replace(real_path)
    try:
        audio_path.symlink_to(real_path)
    except OSError as error:
        real_path.replace(audio_path)
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(CorpusFailure) as failure:
        build_report(paths, config)

    assert failure.value.code == "CACHE_ARTIFACT_INVALID"
    assert tuple(path.read_bytes() for path in output_paths) == before


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"unexpected": 1}, "exact fields"),
        ({"schema_version": 1}, "schema_version"),
        ({"text_preparation_seconds": "invalid"}, "text_preparation_seconds"),
        ({"text_preparation_seconds": -1.0}, "text_preparation_seconds"),
        ({"alignment_wall_seconds": math.nan}, "alignment_wall_seconds"),
        ({"gpu_retry_count": True}, "gpu_retry_count"),
        ({"peak_gpu_memory_bytes": 0}, "peak_gpu_memory_bytes"),
        ({"pilot_storage_bytes": 0}, "pilot_storage_bytes"),
    ),
)
def test_report_telemetry_strictly_validates_input(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ReportTelemetry.from_dict(_telemetry_row(**changes))


def test_report_telemetry_allows_zero_review_seconds() -> None:
    telemetry = ReportTelemetry.from_dict(_telemetry_row(review_seconds=0.0))

    assert telemetry.review_seconds == 0.0
    assert telemetry.to_dict() == _telemetry_row(review_seconds=0.0)


def test_report_single_row_and_recording_loaders_reject_empty_files(tmp_path: Path) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    recordings_path = paths.manifests / "recordings.jsonl"
    write_jsonl_atomic(telemetry_path, ())
    write_jsonl_atomic(recordings_path, ())

    with pytest.raises(ValueError, match="exactly one row"):
        report_module._one_row(telemetry_path, "pilot telemetry")
    with pytest.raises(ValueError, match="unique recordings"):
        report_module._load_recordings(paths)
    recording = _recording_for_report_unit_tests().to_dict()
    write_jsonl_atomic(recordings_path, (recording, recording))
    with pytest.raises(ValueError, match="unique recordings"):
        report_module._load_recordings(paths)


@pytest.mark.parametrize(
    ("invalid_part", "message"),
    (
        ("identity", "identity"),
        ("vad", "16 kHz"),
        ("interval-array", "must be an array"),
        ("interval-fields", "exact fields"),
        ("interval-range", "half-open ranges"),
        ("empty", "zero VAD"),
    ),
)
def test_report_vad_interval_validation_rejects_each_invalid_layer(
    invalid_part: str, message: str
) -> None:
    recording = _recording_for_report_unit_tests()
    raw: dict[str, object] = {
        "schema_version": "1",
        "recording_id": recording.recording_id,
        "vad": {
            "sample_rate": 16_000,
            "speech_intervals": [{"start_sample": 0, "end_sample": 8_000}],
        },
    }
    vad = raw["vad"]
    assert type(vad) is dict
    if invalid_part == "identity":
        raw["recording_id"] = "rec-other"
    elif invalid_part == "vad":
        vad["sample_rate"] = 8_000
    elif invalid_part == "interval-array":
        vad["speech_intervals"] = "invalid"
    elif invalid_part == "interval-fields":
        vad["speech_intervals"] = [{"start_sample": 0}]
    elif invalid_part == "interval-range":
        vad["speech_intervals"] = [{"start_sample": 8_000, "end_sample": 8_000}]
    else:
        vad["speech_intervals"] = []

    with pytest.raises(ValueError, match=message):
        report_module._vad_intervals(raw, recording, analysis_sample_count=16_000)


@pytest.mark.parametrize("issue", (1, "NOT_A_STABLE_ISSUE"))
def test_report_issue_counter_rejects_invalid_issue_values(issue: object) -> None:
    with pytest.raises(ValueError):
        report_module._count_issue(Counter(), issue)


def test_terminal_evidence_requires_processing_history_for_every_recording() -> None:
    with pytest.raises(ValueError, match="processing history"):
        report_module._validate_terminal_processing_evidence(
            recordings=[_recording_for_report_unit_tests()],
            processing_rows=(),
            config_sha256="a" * 64,
            inventory_sha256="f" * 64,
            rights_sha256="b" * 64,
            transcripts_sha256="c" * 64,
            review_sha256="d" * 64,
            segments_sha256="e" * 64,
            pairing_sha256s={},
            alignment_sha256s={},
            attestations=(),
        )


def test_report_atomic_preparation_removes_temporary_file_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(report_module.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        report_module._prepare_atomic(tmp_path / "report.json", b"report")

    assert not tuple(tmp_path.glob(".report.json.*"))


def test_report_atomic_preparation_preserves_primary_failure_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_unlink = type(tmp_path).unlink

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("primary fsync failure")

    def fail_temporary_cleanup(self: Path, missing_ok: bool = False) -> None:
        if self.parent == tmp_path and self.name.startswith(".report.json."):
            raise OSError("secondary cleanup failure")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(report_module.os, "fsync", fail_fsync)
    monkeypatch.setattr(type(tmp_path), "unlink", fail_temporary_cleanup)

    with pytest.raises(OSError, match="primary fsync failure"):
        report_module._prepare_atomic(tmp_path / "report.json", b"report")


def test_report_snapshot_rejects_metadata_drift_without_changing_output_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_fstat = report_module.os.fstat
    calls = 0

    def drift_during_second_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            json_path.write_bytes(before[0] + b"external drift")
            changed = real_fstat(descriptor)
            json_path.write_bytes(before[0])
            return changed
        return real_fstat(descriptor)

    monkeypatch.setattr(report_module.os, "fstat", drift_during_second_fstat)

    with pytest.raises(OSError, match="changed while snapshotting"):
        build_report(paths, config)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == before
    assert not tuple(paths.manifests.glob(".recovery.*"))


def test_missing_prepared_markdown_fails_without_changing_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_prepare = report_module._prepare_atomic

    def remove_prepared_markdown(path: Path, content: bytes) -> Path:
        temporary = real_prepare(path, content)
        if path == markdown_path:
            temporary.unlink()
        return temporary

    monkeypatch.setattr(report_module, "_prepare_atomic", remove_prepared_markdown)

    with pytest.raises(OSError, match="prepared report output is missing"):
        build_report(paths, config)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == before
    assert not tuple(paths.manifests.glob(".recovery.*"))
    assert not tuple(paths.manifests.glob(".report.*"))


def test_build_report_derives_every_metric_and_writes_deterministic_outputs(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))

    report = build_report(paths, config)

    assert report.scale_decision.value == "scalable"
    assert report.text_unit_count == 6
    assert report.reading_count == 12
    assert report.repetition_group_count == 6
    assert report.approved_segment_count == 12
    assert report.metrics.input_duration_seconds == 20.0
    assert report.metrics.speech_duration_seconds == 10.4
    assert report.metrics.auto_pairing_correct_ratio == 1.0
    assert report.metrics.boundary_unchanged_ratio == 1.0
    assert report.metrics.review_minutes_per_audio_minute == 1.0
    assert report.metrics.approved_speech_ratio == 1.0
    assert report.metrics.gpu_realtime_factor == pytest.approx(0.074)
    assert report.approved_duration_seconds == pytest.approx(14.8)
    assert report.rejected_duration_seconds == 0.0
    assert report.unreviewed_duration_seconds == 0.0
    assert report.issue_code_counts == ()
    assert report.rejection_reason_counts == ()
    assert report.projected_full_corpus_person_hours == pytest.approx(30.0 / 3600.0)
    assert report.projected_gpu_hours == pytest.approx(1.48 / 3600.0)
    assert report.projected_storage_bytes == 1_000

    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    first = (json_path.read_bytes(), markdown_path.read_bytes())
    assert json.loads(json_path.read_text(encoding="utf-8")) == report.to_dict()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# LatinTTS Corpus Alignment Pilot Report" in markdown
    assert "Automatic pairing accepted without correction ratio" in markdown
    assert "Automatic pairing correct ratio" not in markdown
    assert "Approved effective take duration ratio" in markdown
    assert "Approved VAD speech ratio" not in markdown
    assert "Text preparation (seconds) | 10.000000" in markdown
    assert "Human review (seconds) | 20.000000" in markdown
    assert "GPU retries | 0" in markdown
    assert "Pilot storage (bytes) | 1000" in markdown

    assert build_report(paths, config) == report
    assert (json_path.read_bytes(), markdown_path.read_bytes()) == first


def test_build_report_rejects_zero_denominators_instead_of_emitting_nan(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(
        paths.manifests / "pilot-telemetry.json",
        (_telemetry_row(alignment_wall_seconds=0.0),),
    )

    with pytest.raises(ValueError, match="alignment_wall_seconds"):
        build_report(paths, config)

    assert not (paths.manifests / "report.json").exists()
    assert not (paths.manifests / "report.md").exists()


def test_build_report_counts_rejection_reasons_and_marks_not_ready(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path, decision="rejected")
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))

    report = build_report(paths, config)

    assert report.scale_decision.value == "not_ready"
    assert report.approved_segment_count == 0
    assert report.approved_duration_seconds == 0.0
    assert report.rejected_duration_seconds == pytest.approx(14.8)
    assert report.rejection_reason_counts == (("listened in full", 12),)
    assert report.metrics.approved_speech_ratio == 0.0
    assert read_jsonl(paths.manifests / "segments.jsonl") == ()


def test_build_report_rejects_rejected_recording_with_approved_takes(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    recordings_path = paths.manifests / "recordings.jsonl"
    recordings = list(read_jsonl(recordings_path))
    recordings[0]["state"] = "REJECTED"
    write_jsonl_atomic(recordings_path, recordings)

    with pytest.raises(ValueError, match="terminal state"):
        build_report(paths, config)


def test_build_report_rejects_approved_recording_with_only_rejected_takes(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path, decision="rejected")
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    recordings_path = paths.manifests / "recordings.jsonl"
    recordings = list(read_jsonl(recordings_path))
    recordings[0]["state"] = "APPROVED"
    write_jsonl_atomic(recordings_path, recordings)

    with pytest.raises(ValueError, match="terminal state"):
        build_report(paths, config)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("terminal-event-missing", "processing"),
        ("rights-file-missing", "rights.jsonl"),
        ("recording-rights-id", "rights"),
        ("segment-rights-id", r"segment.*evidence"),
        ("transcript-selected-candidate", "selected source"),
        ("transcript-source-hash", "source candidate hash"),
        ("segment-source-text-id", r"segment.*evidence"),
    ),
)
def test_build_report_rejects_broken_terminal_evidence_chain(
    tmp_path: Path, tamper: str, message: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    recordings_path = paths.manifests / "recordings.jsonl"
    rights_path = paths.manifests / "rights.jsonl"
    transcripts_path = paths.manifests / "transcripts.jsonl"
    segments_path = paths.manifests / "segments.jsonl"
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"

    if tamper == "terminal-event-missing":
        events = tuple(
            row for row in read_jsonl(processing_path) if row["previous_state"] != "REVIEWED"
        )
        write_jsonl_atomic(processing_path, events)
    elif tamper == "rights-file-missing":
        rights_path.unlink()
    elif tamper == "recording-rights-id":
        recordings = list(read_jsonl(recordings_path))
        recordings[0]["rights_id"] = "rights-unknown"
        write_jsonl_atomic(recordings_path, recordings)
    elif tamper == "segment-rights-id":
        segments = list(read_jsonl(segments_path))
        segments[0]["rights_id"] = "rights-unknown"
        write_jsonl_atomic(segments_path, segments)
    elif tamper == "transcript-selected-candidate":
        transcripts = list(read_jsonl(transcripts_path))
        transcripts[0]["selected_candidate_id"] = "source-unknown"
        write_jsonl_atomic(transcripts_path, transcripts)
    elif tamper == "transcript-source-hash":
        transcripts = list(read_jsonl(transcripts_path))
        transcripts[0]["source_candidates"][0]["source_sha256"] = "f" * 64
        write_jsonl_atomic(transcripts_path, transcripts)
    else:
        segments = list(read_jsonl(segments_path))
        segments[0]["source_text_id"] = "source-unknown"
        write_jsonl_atomic(segments_path, segments)

    with pytest.raises((OSError, ValueError, CorpusFailure), match=message):
        build_report(paths, config)


def test_build_report_rejects_rights_without_internal_evaluation_permission(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    rights_path = paths.manifests / "rights.jsonl"
    rights = list(read_jsonl(rights_path))
    rights[0]["allow_internal_evaluation"] = False
    write_jsonl_atomic(rights_path, rights)

    with pytest.raises(CorpusFailure) as failure:
        build_report(paths, config)
    assert failure.value.code == "RIGHTS_SCOPE_UNCONFIRMED"


def _replace_terminal_events_with_old_rows(
    paths: object, config: CorpusConfig, version: str
) -> None:
    recordings = {
        raw["recording_id"]: review_module._decode_recording(raw)
        for raw in read_jsonl(paths.manifests / "recordings.jsonl")  # type: ignore[attr-defined]
    }
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    rewritten = []
    for raw in read_jsonl(processing_path):
        event = ProcessingEvent.from_dict(raw)
        if event.previous_state is not CorpusState.REVIEWED:
            rewritten.append(raw)
            continue
        recording = recordings[event.recording_id]
        if version == "v1":
            inputs = (
                event.input_sha256s[0],
                event.input_sha256s[2],
                event.input_sha256s[3],
                event.input_sha256s[4],
                event.input_sha256s[5],
                event.input_sha256s[6],
                event.input_sha256s[7],
            )
        else:
            inputs = (
                event.input_sha256s[0],
                event.input_sha256s[4],
                event.input_sha256s[5],
                event.input_sha256s[6],
                event.input_sha256s[7],
            )
        _, old_event = advance_recording(
            replace(recording, state=CorpusState.REVIEWED),
            recording.state,
            input_sha256s=inputs,
            config_sha256=event.config_sha256,
            tool_versions=("approved-manifest-v1", event.tool_versions[1]),
            started_at=event.started_at,
            finished_at=event.finished_at,
            result=event.result,
        )
        rewritten.append(old_event.to_dict())
    write_jsonl_atomic(processing_path, rewritten)


def _terminal_processing_by_recording(
    paths: object, config: CorpusConfig
) -> dict[str, ProcessingEvent]:
    processing_path = (
        paths.alignments / "runs" / config.digest / "processing-events.jsonl"  # type: ignore[attr-defined]
    )
    return {
        event.recording_id: event
        for event in (ProcessingEvent.from_dict(raw) for raw in read_jsonl(processing_path))
        if event.previous_state is CorpusState.REVIEWED
    }


def _attestation_path(paths: object, config: CorpusConfig) -> Path:
    return (
        paths.alignments  # type: ignore[attr-defined]
        / "runs"
        / config.digest
        / "manifest-attestations.jsonl"
    )


def _attestation_rows(
    old_events: dict[str, ProcessingEvent],
    v2_events: dict[str, ProcessingEvent],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "schema_version": "1",
            "terminal_event_id": old_events[recording_id].event_id,
            "evidence": v2_events[recording_id].to_dict(),
        }
        for recording_id in v2_events
    )


@pytest.mark.parametrize(
    ("version", "tamper_rights_metadata"),
    (("v1", False), ("v1", True), ("legacy", False), ("legacy", True)),
)
def test_build_report_requires_current_terminal_evidence_not_legacy(
    tmp_path: Path, version: str, tamper_rights_metadata: bool
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    _replace_terminal_events_with_old_rows(paths, config, version)
    if tamper_rights_metadata:
        rights_path = paths.manifests / "rights.jsonl"
        rights = list(read_jsonl(rights_path))
        rights[0]["basis"] = "changed after legacy manifest build"
        write_jsonl_atomic(rights_path, rights)

    expected = (
        r"processing manifest terminal event"
        if version == "v1" and tamper_rights_metadata
        else r"legacy terminal.*rebuild"
    )
    with pytest.raises(ValueError, match=expected):
        build_report(paths, config)


@pytest.mark.parametrize("version", ("v1", "legacy"))
def test_build_manifest_recovery_appends_v2_attestations_for_old_terminal_evidence(
    tmp_path: Path, version: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    _replace_terminal_events_with_old_rows(paths, config, version)
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    processing_before = processing_path.read_bytes()
    terminal_by_recording = {
        event.recording_id: event
        for event in (ProcessingEvent.from_dict(raw) for raw in read_jsonl(processing_path))
        if event.previous_state is CorpusState.REVIEWED
    }

    assert _build_with_fake_audio(paths, config)
    assert processing_path.read_bytes() == processing_before
    migrated = build_report(paths, config)

    attestation_path = paths.alignments / "runs" / config.digest / "manifest-attestations.jsonl"
    attestation_bytes = attestation_path.read_bytes()
    attestations = read_jsonl(attestation_path)
    assert len(attestations) == len(terminal_by_recording) == 2
    for raw in attestations:
        assert set(raw) == {"schema_version", "terminal_event_id", "evidence"}
        assert raw["schema_version"] == "1"
        evidence_raw = raw["evidence"]
        assert type(evidence_raw) is dict
        evidence = ProcessingEvent.from_dict(evidence_raw)
        terminal = terminal_by_recording[evidence.recording_id]
        assert raw["terminal_event_id"] == terminal.event_id
        assert evidence.previous_state is CorpusState.REVIEWED
        assert evidence.target_state is terminal.target_state
        assert len(evidence.input_sha256s) == 8
        assert evidence.tool_versions[0] == "approved-manifest-v2"
        identity = {
            "recording_id": evidence.recording_id,
            "previous_state": evidence.previous_state.value,
            "target_state": evidence.target_state.value,
            "inputs": evidence.input_sha256s,
            "config": evidence.config_sha256,
            "tools": evidence.tool_versions,
            "result": evidence.result,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        assert evidence.event_id == (
            "state-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        )

    assert _build_with_fake_audio(paths, config)
    assert attestation_path.read_bytes() == attestation_bytes
    assert build_report(paths, config).to_dict() == migrated.to_dict()


def test_fresh_v2_report_uses_manifest_builder_not_pairing_ffmpeg_version(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    recordings = {
        raw["recording_id"]: review_module._decode_recording(raw)
        for raw in read_jsonl(paths.manifests / "recordings.jsonl")
    }
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    rewritten = []
    for raw in read_jsonl(processing_path):
        event = ProcessingEvent.from_dict(raw)
        if event.previous_state is not CorpusState.REVIEWED:
            rewritten.append(raw)
            continue
        recording = recordings[event.recording_id]
        _, rebuilt = advance_recording(
            replace(recording, state=CorpusState.REVIEWED),
            recording.state,
            input_sha256s=event.input_sha256s,
            config_sha256=event.config_sha256,
            tool_versions=("approved-manifest-v2", "ffmpeg-manifest-2"),
            started_at=event.started_at,
            finished_at=event.finished_at,
            result=event.result,
        )
        rewritten.append(rebuilt.to_dict())
    write_jsonl_atomic(processing_path, rewritten)

    report = build_report(paths, config)

    assert report.approved_segment_count == 12


def test_manifest_attestation_append_preserves_prefix_and_is_byte_idempotent(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    v2_events = _terminal_processing_by_recording(paths, config)
    _replace_terminal_events_with_old_rows(paths, config, "legacy")
    old_events = _terminal_processing_by_recording(paths, config)
    rows = _attestation_rows(old_events, v2_events)
    attestation_path = _attestation_path(paths, config)
    write_jsonl_atomic(attestation_path, rows[:1])
    existing_prefix = attestation_path.read_bytes()

    assert _build_with_fake_audio(paths, config)

    appended = attestation_path.read_bytes()
    assert appended.startswith(existing_prefix)
    assert len(read_jsonl(attestation_path)) == 2
    assert _build_with_fake_audio(paths, config)
    assert attestation_path.read_bytes() == appended


def test_manifest_attestation_first_publish_does_not_clobber_concurrent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    _replace_terminal_events_with_old_rows(paths, config, "legacy")
    attestation_path = _attestation_path(paths, config)
    assert not attestation_path.exists()
    foreign_bytes = b"concurrent manifest attestation bytes\n"
    real_persist = manifest_module.persist_recording_transitions

    def persist_then_publish_foreign_file(**kwargs: object) -> None:
        real_persist(**kwargs)  # type: ignore[arg-type]
        attestation_path.write_bytes(foreign_bytes)

    monkeypatch.setattr(
        manifest_module,
        "persist_recording_transitions",
        persist_then_publish_foreign_file,
    )

    with pytest.raises(FileExistsError):
        _build_with_fake_audio(paths, config)

    assert attestation_path.read_bytes() == foreign_bytes


def test_manifest_attestation_publish_preserves_primary_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    _replace_terminal_events_with_old_rows(paths, config, "legacy")
    attestation_path = _attestation_path(paths, config)
    foreign_bytes = b"concurrent manifest attestation bytes\n"
    real_persist = manifest_module.persist_recording_transitions
    real_unlink = type(attestation_path).unlink

    def persist_then_publish_foreign_file(**kwargs: object) -> None:
        real_persist(**kwargs)  # type: ignore[arg-type]
        attestation_path.write_bytes(foreign_bytes)

    def fail_attestation_temporary_cleanup(self: Path, missing_ok: bool = False) -> None:
        if self.parent == attestation_path.parent and self.name.startswith(
            ".manifest-attestations.jsonl."
        ):
            raise OSError("secondary attestation cleanup failure")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(
        manifest_module,
        "persist_recording_transitions",
        persist_then_publish_foreign_file,
    )
    monkeypatch.setattr(type(attestation_path), "unlink", fail_attestation_temporary_cleanup)

    with pytest.raises(FileExistsError):
        _build_with_fake_audio(paths, config)

    assert attestation_path.read_bytes() == foreign_bytes


def test_noncanonical_manifest_attestation_is_rejected_before_state_write(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    v2_events = _terminal_processing_by_recording(paths, config)
    _replace_terminal_events_with_old_rows(paths, config, "legacy")
    old_events = _terminal_processing_by_recording(paths, config)
    recordings_path = paths.manifests / "recordings.jsonl"
    recordings = list(read_jsonl(recordings_path))
    for recording in recordings:
        recording["state"] = CorpusState.REVIEWED.value
    write_jsonl_atomic(recordings_path, recordings)
    attestation_path = _attestation_path(paths, config)
    noncanonical = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in _attestation_rows(old_events, v2_events)
    ).encode("utf-8")
    attestation_path.write_bytes(noncanonical)
    protected_paths = (
        recordings_path,
        paths.alignments / "runs" / config.digest / "processing-events.jsonl",
        paths.manifests / "review.jsonl",
        paths.manifests / "segments.jsonl",
        attestation_path,
    )
    before = {path: path.read_bytes() for path in protected_paths}

    with pytest.raises(ValueError, match="canonical"):
        _build_with_fake_audio(paths, config)

    assert {path: path.read_bytes() for path in protected_paths} == before


def test_unlinked_manifest_attestation_is_rejected_before_manifest_publication(
    tmp_path: Path,
) -> None:
    paths, config = _reviewed_project(tmp_path)
    recordings_path = paths.manifests / "recordings.jsonl"
    recording = review_module._decode_recording(read_jsonl(recordings_path)[0])
    timestamp = "1970-01-01T00:00:00+00:00"
    _, evidence = advance_recording(
        recording,
        CorpusState.APPROVED,
        input_sha256s=("a" * 64,) * 8,
        config_sha256=config.digest,
        tool_versions=("approved-manifest-v2", "ffmpeg-test-1"),
        started_at=timestamp,
        finished_at=timestamp,
        result="success",
    )
    attestation_path = _attestation_path(paths, config)
    write_jsonl_atomic(
        attestation_path,
        (
            {
                "schema_version": "1",
                "terminal_event_id": "state-ffffffffffffffff",
                "evidence": evidence.to_dict(),
            },
        ),
    )
    processing_path = paths.alignments / "runs" / config.digest / "processing-events.jsonl"
    protected_paths = (
        recordings_path,
        processing_path,
        paths.manifests / "review.jsonl",
        attestation_path,
    )
    protected_before = {path: path.read_bytes() for path in protected_paths}
    manifest_path = paths.manifests / "segments.jsonl"
    segment_files_before = {
        path.relative_to(paths.segments): path.read_bytes()
        for path in paths.segments.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ValueError, match="extraneous manifest attestation"):
        _build_with_fake_audio(paths, config)

    assert {path: path.read_bytes() for path in protected_paths} == protected_before
    assert not manifest_path.exists()
    assert {
        path.relative_to(paths.segments): path.read_bytes()
        for path in paths.segments.rglob("*")
        if path.is_file()
    } == segment_files_before


def test_fresh_v2_report_rejects_extraneous_manifest_attestation(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    v2_events = _terminal_processing_by_recording(paths, config)
    first = next(iter(v2_events.values()))
    write_jsonl_atomic(
        _attestation_path(paths, config),
        (
            {
                "schema_version": "1",
                "terminal_event_id": first.event_id,
                "evidence": first.to_dict(),
            },
        ),
    )

    with pytest.raises(ValueError, match="extraneous manifest attestation"):
        build_report(paths, config)


@pytest.mark.parametrize(
    "tamper",
    (
        "outer-key",
        "nested-key",
        "event-id",
        "marker",
        "input-length",
        "terminal-event-id",
        "duplicate",
    ),
)
def test_old_terminal_report_rejects_invalid_manifest_attestation(
    tmp_path: Path, tamper: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    v2_events = _terminal_processing_by_recording(paths, config)
    _replace_terminal_events_with_old_rows(paths, config, "legacy")
    old_events = _terminal_processing_by_recording(paths, config)
    rows = [deepcopy(row) for row in _attestation_rows(old_events, v2_events)]
    if tamper == "outer-key":
        rows[0]["unknown"] = True
    elif tamper == "nested-key":
        rows[0]["evidence"]["unknown"] = True  # type: ignore[index]
    elif tamper == "event-id":
        rows[0]["evidence"]["event_id"] = "state-0000000000000000"  # type: ignore[index]
    elif tamper == "marker":
        rows[0]["evidence"]["tool_versions"][0] = "unknown-manifest"  # type: ignore[index]
    elif tamper == "input-length":
        rows[0]["evidence"]["input_sha256s"].pop()  # type: ignore[index,union-attr]
    elif tamper == "terminal-event-id":
        rows[0]["terminal_event_id"] = "state-ffffffffffffffff"
    else:
        rows.append(deepcopy(rows[0]))
    write_jsonl_atomic(_attestation_path(paths, config), rows)

    with pytest.raises(ValueError, match="manifest attestation"):
        build_report(paths, config)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("selection-schema", "selection identity"),
        ("selection-hash", "spoken inventory"),
        ("recording-state", "terminal review states"),
        ("rights-speaker", "rights speaker"),
        ("analysis-audio", "analysis_audio"),
    ),
)
def test_build_report_rejects_invalid_pilot_identity_layers(
    tmp_path: Path, tamper: str, message: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    if tamper.startswith("selection-"):
        selection_path = paths.manifests / "pilot-selection.json"
        selection = dict(read_jsonl(selection_path)[0])
        if tamper == "selection-schema":
            selection["schema_version"] = "2"
        else:
            selection["inventory_hashes"][0] = "f" * 64
        write_jsonl_atomic(selection_path, (selection,))
    elif tamper == "recording-state":
        recordings_path = paths.manifests / "recordings.jsonl"
        recordings = list(read_jsonl(recordings_path))
        recordings[0]["state"] = "REVIEWED"
        write_jsonl_atomic(recordings_path, recordings)
    elif tamper == "rights-speaker":
        rights_path = paths.manifests / "rights.jsonl"
        rights = list(read_jsonl(rights_path))
        rights[0]["speaker_id"] = "speaker-other"
        write_jsonl_atomic(rights_path, rights)
    else:
        segmentation_path = _first_selected_run_directory(paths, config) / "segmentation.json"
        segmentation = dict(read_jsonl(segmentation_path)[0])
        segmentation["analysis_audio"] = []
        write_jsonl_atomic(segmentation_path, (segmentation,))

    with pytest.raises((CorpusFailure, ValueError), match=message):
        build_report(paths, config)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("identity", "identity"),
        ("takes", "exactly two takes"),
        ("take-index", "take identities"),
        ("take-index-type", "take identities"),
        ("entity-id", "entity identity"),
        ("provenance-type", "take_provenance"),
        ("provenance-bounds", "take provenance"),
    ),
)
def test_build_report_rejects_invalid_review_automatic_layers(
    tmp_path: Path, tamper: str, message: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    automatic_path = _first_review_automatic_path(paths, config)
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    if tamper == "identity":
        automatic["unit_id"] = "unit-other"
    elif tamper == "takes":
        automatic["takes"] = []
    elif tamper == "take-index":
        automatic["takes"][1]["take_index"] = 1
    elif tamper == "take-index-type":
        automatic["takes"][0]["take_index"] = []
    elif tamper == "entity-id":
        automatic["takes"][0]["entity_id"] = "review:other"
    elif tamper == "provenance-type":
        automatic["takes"][0]["take_provenance"] = []
    else:
        automatic["takes"][0]["take_provenance"]["source_start_sample"] += 1
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_report(paths, config)


def test_build_report_rejects_segmentation_provenance_tampering(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    segmentation_path = _first_selected_run_directory(paths, config) / "segmentation.json"
    segmentation = deepcopy(read_jsonl(segmentation_path)[0])
    segmentation["config_sha256"] = "f" * 64
    write_jsonl_atomic(segmentation_path, (segmentation,))

    with pytest.raises(ValueError, match="segmentation"):
        build_report(paths, config)

    assert not (paths.manifests / "report.json").exists()
    assert not (paths.manifests / "report.md").exists()


def test_build_report_rejects_review_automatic_binding_tampering(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    automatic_path = _first_review_automatic_path(paths, config)
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    automatic["artifact_binding"]["config_sha256"] = "f" * 64
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact binding"):
        build_report(paths, config)


def test_build_report_rejects_original_pairing_with_different_run_identity(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    run_directory = _first_selected_run_directory(paths, config)
    original = pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0])
    changed_config = "f" * 64
    changed_cache = _pairing_cache_key(
        original.recording_id,
        original.windows,
        original.analysis_audio_relative_path,
        original.analysis_audio_sha256,
        original.segmentation_artifact_sha256,
        changed_config,
        original.pairing_parameters,
        original.ffmpeg_version,
    )
    stale_original = replace(
        original,
        config_sha256=changed_config,
        cache_key=changed_cache,
        integrity_sha256="",
    )
    write_jsonl_atomic(
        run_directory / "pairing-automatic.json",
        (pairing_to_dict(stale_original),),
    )

    with pytest.raises(ValueError, match=r"automatic pairing|pairing correction"):
        build_report(paths, config)


def test_build_report_converts_pairing_shape_type_error(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    pairing_path = _first_selected_run_directory(paths, config) / "pairing.json"
    pairing = dict(read_jsonl(pairing_path)[0])
    pairing["windows"] = "invalid"
    write_jsonl_atomic(pairing_path, (pairing,))

    with pytest.raises(ValueError, match="pairing"):
        build_report(paths, config)


def test_build_report_converts_segment_shape_type_error(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    segments_path = paths.manifests / "segments.jsonl"
    segments = list(read_jsonl(segments_path))
    segments[0]["quality_metrics"] = []
    write_jsonl_atomic(segments_path, segments)

    with pytest.raises(ValueError, match="segment"):
        build_report(paths, config)


def test_corrected_pairing_metrics_use_original_automatic_evidence(tmp_path: Path) -> None:
    paths, config = _completed_corrected_pairing_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))

    report = build_report(paths, config)

    assert report.metrics.auto_pairing_correct_ratio == pytest.approx(2 / 3)
    assert report.issue_code_counts == (("TAKE_DURATION_MISMATCH", 2),)
    assert report.scale_decision.value == "not_ready"
    assert report.approved_segment_count == 12


def test_build_report_does_not_fall_back_to_final_pairing_when_correction_event_exists(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    run_directory = _first_selected_run_directory(paths, config)
    pairing = pairing_from_dict(read_jsonl(run_directory / "pairing.json")[0])
    group = pairing.groups[0].group
    assert group is not None
    event = review_module._new_review_event(
        review_module._pairing_entity_id(pairing.recording_id, group.unit_id),
        "pairing_selected_split",
        None,
        group.selected_evidence.split_sample,
        {
            "reason": "human pairing correction",
            "reviewer": "owner",
            "reviewed_at": "2026-07-19T13:00:00+08:00",
        },
    )
    review_path = paths.manifests / "review.jsonl"
    write_jsonl_atomic(review_path, (*read_jsonl(review_path), event.to_dict()))

    with pytest.raises(ValueError, match=r"pairing correction|pairing history"):
        build_report(paths, config)


@pytest.mark.parametrize(
    "tamper",
    ("source_audio", "text_layers", "take_provenance"),
)
def test_build_report_rejects_review_automatic_evidence_tampering(
    tmp_path: Path, tamper: str
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    automatic_path = _first_review_automatic_path(paths, config)
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    if tamper == "source_audio":
        automatic["source_audio"]["sha256"] = "f" * 64
    elif tamper == "text_layers":
        automatic["text_layers"]["unit_spoken_text"] = "tampered text"
    else:
        automatic["takes"][0]["take_provenance"]["candidate_audio_sha256"] = "f" * 64
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")

    with pytest.raises(ValueError, match=r"review (source|text|take)|review automatic"):
        build_report(paths, config)


def test_build_report_rejects_unknown_review_entity(tmp_path: Path) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    event = review_module._new_review_event(
        "review:unknown-entity",
        "review_decision",
        "unreviewed",
        "approved",
        {
            "reason": "orphan event",
            "reviewer": "owner",
            "reviewed_at": "2026-07-19T13:00:00+08:00",
        },
    )
    review_path = paths.manifests / "review.jsonl"
    write_jsonl_atomic(review_path, (*read_jsonl(review_path), event.to_dict()))

    with pytest.raises(ValueError, match=r"unknown.*entity"):
        build_report(paths, config)


@pytest.mark.parametrize(
    ("field", "tampered"),
    (
        ("config_sha256", "f" * 64),
        ("source_audio_sha256", "f" * 64),
        ("source_start_sample", 1),
        ("text_unit_id", "tampered-unit"),
    ),
)
def test_build_report_rejects_same_key_segment_provenance_tampering(
    tmp_path: Path, field: str, tampered: object
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    segments_path = paths.manifests / "segments.jsonl"
    segments = list(read_jsonl(segments_path))
    segments[0][field] = tampered
    write_jsonl_atomic(segments_path, segments)

    with pytest.raises(ValueError, match=r"segment.*evidence"):
        build_report(paths, config)


def test_boundary_moved_then_restored_is_still_counted_as_changed(tmp_path: Path) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    grid_path = groups[0] / "take-1.TextGrid"
    original = read_textgrid(grid_path)
    assert original.words[-1].end_seconds < original.take_end
    write_textgrid(
        grid_path,
        duration_seconds=original.duration_seconds,
        take_start=original.take_start,
        take_end=original.words[-1].end_seconds,
        words=original.words,
    )
    assert import_review_bundle(paths, config)
    write_textgrid(
        grid_path,
        duration_seconds=original.duration_seconds,
        take_start=original.take_start,
        take_end=original.take_end,
        words=original.words,
    )
    assert import_review_bundle(paths, config)
    assert _build_with_fake_audio(paths, config)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))

    report = build_report(paths, config)

    assert report.metrics.boundary_unchanged_ratio == pytest.approx(11 / 12)


def test_rejection_reason_comes_from_final_effective_rejection(tmp_path: Path) -> None:
    paths, config = _two_recording_aligned_project(tmp_path)
    _write_rights(paths)
    groups = export_review_bundle(paths, config)
    for group in groups:
        _set_decisions(group)
    _set_decisions(
        groups[0],
        decision="rejected",
        reason="initial rejection",
        reviewed_at="2026-07-19T12:01:00+08:00",
        take_indexes=(1,),
    )
    assert import_review_bundle(paths, config)
    _set_decisions(
        groups[0],
        decision="approved",
        reason="temporary approval",
        reviewed_at="2026-07-19T12:02:00+08:00",
        take_indexes=(1,),
    )
    assert import_review_bundle(paths, config)
    _set_decisions(
        groups[0],
        decision="rejected",
        reason="final rejection",
        reviewed_at="2026-07-19T12:03:00+08:00",
        take_indexes=(1,),
    )
    assert import_review_bundle(paths, config)
    assert _build_with_fake_audio(paths, config)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))

    report = build_report(paths, config)

    assert report.rejection_reason_counts == (("final rejection", 1),)
    assert report.approved_segment_count == 11


def test_report_output_pair_rolls_back_when_second_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    failed = False

    def fail_markdown_once(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal failed
        if target == markdown_path and not failed:
            failed = True
            raise OSError("injected second publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_markdown_once)

    with pytest.raises(OSError, match="injected second publication failure"):
        build_report(paths, config)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == before
    assert not tuple(paths.manifests.glob(".report.*"))
    assert not tuple(paths.manifests.glob(".recovery.*"))
    assert not tuple(paths.manifests.glob(".report-output-transaction.*"))


def _inject_markdown_report_publication_failure(
    paths: CorpusPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    real_publish = report_module._publish_report_output
    markdown_path = paths.manifests / "report.md"

    def fail_markdown_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected Markdown publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_markdown_publication)
    return real_publish


def test_rolled_back_handoff_survives_hard_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    old_pair = tuple((paths.manifests / name).read_bytes() for name in ("report.json", "report.md"))
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = _inject_markdown_report_publication_failure(paths, monkeypatch)
    real_claim = getattr(
        report_module,
        "_claim_rolled_back_intent",
        report_module._claim_completed_intent,
    )

    def crash_after_rolled_back_handoff(
        intent_path: Path,
        transaction_directory: Path,
        marker_snapshot: report_module._ReportOutputSnapshot,
    ) -> object:
        real_claim(intent_path, transaction_directory, marker_snapshot)
        raise SystemExit("simulated rolled-back handoff crash")

    monkeypatch.setattr(
        report_module,
        "_claim_rolled_back_intent",
        crash_after_rolled_back_handoff,
        raising=False,
    )

    with pytest.raises(SystemExit, match="rolled-back handoff crash"):
        build_report(paths, config)

    rolled_back = paths.manifests / ".report-output-transaction.rolled-back.json"
    assert rolled_back.is_file()
    assert (
        tuple((paths.manifests / name).read_bytes() for name in ("report.json", "report.md"))
        == old_pair
    )
    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)
    monkeypatch.setattr(report_module, "_claim_rolled_back_intent", real_claim)
    report = build_report(paths, config)

    assert json.loads((paths.manifests / "report.json").read_text(encoding="utf-8")) == (
        report.to_dict()
    )
    assert not rolled_back.exists()


def test_rolled_back_cleanup_resumes_after_first_payload_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = _inject_markdown_report_publication_failure(paths, monkeypatch)
    real_delete = report_module._delete_owned_report_file
    crashed = False

    def crash_after_first_rolled_back_payload_delete(
        path: Path,
        expected: report_module._ReportSnapshotIdentity,
        *,
        missing_ok: bool = False,
    ) -> None:
        nonlocal crashed
        is_rolled_back_payload = (
            path.parent == paths.manifests
            and path.name.startswith(".report.json.")
            and (paths.manifests / ".report-output-transaction.rolled-back.json").is_file()
        )
        real_delete(path, expected, missing_ok=missing_ok)
        if is_rolled_back_payload and not crashed:
            crashed = True
            raise SystemExit("simulated rolled-back payload cleanup crash")

    monkeypatch.setattr(
        report_module,
        "_delete_owned_report_file",
        crash_after_first_rolled_back_payload_delete,
    )

    with pytest.raises(SystemExit, match="rolled-back payload cleanup crash"):
        build_report(paths, config)

    assert crashed
    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)
    monkeypatch.setattr(report_module, "_delete_owned_report_file", real_delete)
    report = build_report(paths, config)

    assert json.loads((paths.manifests / "report.json").read_text(encoding="utf-8")) == (
        report.to_dict()
    )
    assert not tuple(paths.manifests.glob(".report-output-transaction.*"))


@pytest.mark.parametrize("target_name", ("report.json", "report.md"))
def test_first_report_publication_does_not_clobber_concurrent_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    target = paths.manifests / target_name
    other = (
        paths.manifests / ({"report.json": "report.md", "report.md": "report.json"}[target_name])
    )
    foreign_bytes = f"foreign {target_name}\n".encode()
    real_link = report_module.os.link
    injected = False

    def inject_before_link(
        source: Path, destination: Path, *args: object, **kwargs: object
    ) -> None:
        nonlocal injected
        if Path(destination) == target and not injected:
            injected = True
            target.write_bytes(foreign_bytes)
        real_link(source, destination, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(report_module.os, "link", inject_before_link)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert injected
    assert target.read_bytes() == foreign_bytes
    assert not other.exists()
    assert paths.manifests / ".report-output-transaction.json" in failure.value.recovery_paths


@pytest.mark.parametrize("target_name", ("report.json", "report.md"))
@pytest.mark.parametrize("drift", ("digest", "identity"))
def test_existing_report_drift_after_backup_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    drift: str,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    target = paths.manifests / target_name
    other = markdown_path if target == json_path else json_path
    other_before = other.read_bytes()
    foreign_bytes = f"foreign {drift} {target_name}\n".encode()
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_prepare = report_module._prepare_atomic
    injected = False

    def drift_after_backup(path: Path, content: bytes) -> Path:
        nonlocal injected
        temporary = real_prepare(path, content)
        if path.name == f"recovery.{target_name}" and not injected:
            injected = True
            if drift == "digest":
                target.write_bytes(foreign_bytes)
            else:
                replacement = target.with_name(f"foreign.{target.name}")
                replacement.write_bytes(foreign_bytes)
                replacement.replace(target)
        return temporary

    monkeypatch.setattr(report_module, "_prepare_atomic", drift_after_backup)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert injected
    assert target.read_bytes() == foreign_bytes
    assert other.read_bytes() == other_before
    assert any(
        path.is_file() and path.read_bytes() != foreign_bytes
        for path in failure.value.recovery_paths
    )


def test_report_rollback_refuses_to_overwrite_concurrent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    foreign_bytes = b"foreign replacement after json publication\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_replace = report_module.os.replace
    injected = False

    def replace_then_take_over_before_second_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal injected
        if target == markdown_path and not injected:
            injected = True
            replacement = json_path.with_name("foreign.report.json")
            replacement.write_bytes(foreign_bytes)
            real_replace(replacement, json_path)
            raise OSError("injected second publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    monkeypatch.setattr(
        report_module,
        "_publish_report_output",
        replace_then_take_over_before_second_publication,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert injected
    assert json_path.read_bytes() == foreign_bytes
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_pair_commit_revalidates_json_after_markdown_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    foreign_bytes = b"foreign replacement after json publish returned\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_replace = report_module.os.replace
    injected = False

    def publish_then_take_over_json(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal injected
        published = real_publish(temporary, target, old=old, prepared=prepared)
        if target == json_path and not injected:
            injected = True
            replacement = json_path.with_name("foreign.report.json")
            replacement.write_bytes(foreign_bytes)
            real_replace(replacement, json_path)
        return published

    monkeypatch.setattr(report_module, "_publish_report_output", publish_then_take_over_json)

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert injected
    assert json_path.read_bytes() == foreign_bytes
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_publication_snapshot_failure_preserves_old_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_snapshot = report_module._prepared_report_snapshot

    def fail_published_json_snapshot(path: Path) -> report_module._ReportOutputSnapshot:
        if path == json_path:
            raise OSError("injected published JSON snapshot failure")
        return real_snapshot(path)

    monkeypatch.setattr(
        report_module,
        "_prepared_report_snapshot",
        fail_published_json_snapshot,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert "injected published JSON snapshot failure" in str(failure.value.__cause__)
    assert json_path.is_file()
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_publication_rejects_target_that_differs_from_prepared_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    foreign_bytes = b"foreign JSON before publication verification\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_link = report_module.os.link
    real_replace = report_module.os.replace
    injected = False

    def link_then_take_over_json(source: Path, destination: Path) -> None:
        nonlocal injected
        real_link(source, destination)
        if Path(destination) == json_path and not injected:
            replacement = json_path.with_name("foreign.report.json")
            replacement.write_bytes(foreign_bytes)
            real_replace(replacement, json_path)
            injected = True

    monkeypatch.setattr(report_module.os, "link", link_then_take_over_json)

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert injected
    assert "foreign report output blocks recovery" in str(failure.value.__cause__)
    assert json_path.read_bytes() == foreign_bytes
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_pair_commit_preserves_concurrent_markdown_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    foreign_bytes = b"foreign Markdown after publish returned\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_replace = report_module.os.replace
    injected = False

    def publish_then_take_over_markdown(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal injected
        published = real_publish(temporary, target, old=old, prepared=prepared)
        if target == markdown_path and not injected:
            injected = True
            replacement = markdown_path.with_name("foreign.report.md")
            replacement.write_bytes(foreign_bytes)
            real_replace(replacement, markdown_path)
        return published

    monkeypatch.setattr(
        report_module,
        "_publish_report_output",
        publish_then_take_over_markdown,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert injected
    assert json_path.read_bytes() == old_json
    assert markdown_path.read_bytes() == foreign_bytes
    assert "foreign report output blocks recovery" in str(failure.value.__cause__)
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_markdown for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_rollback_requires_a_snapshot_for_existing_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_snapshot = report_module._prepared_report_snapshot
    real_publish = report_module._publish_report_output

    def omit_json_backup_snapshot(path: Path) -> report_module._ReportOutputSnapshot:
        if path.name.startswith(".recovery.report.json."):
            return None  # type: ignore[return-value]
        return real_snapshot(path)

    def fail_markdown_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected Markdown publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    monkeypatch.setattr(
        report_module,
        "_prepared_report_snapshot",
        omit_json_backup_snapshot,
    )
    monkeypatch.setattr(
        report_module,
        "_publish_report_output",
        fail_markdown_publication,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert "backup is invalid" in str(failure.value.__cause__)
    assert json_path.read_bytes() == old_json
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_rollback_detects_corrupted_restored_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_markdown = markdown_path.read_bytes()
    corrupt_bytes = b"corrupt restored JSON\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    old_json = json_path.read_bytes()
    real_publish = report_module._publish_report_output
    real_link = report_module.os.link

    def fail_markdown_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected Markdown publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def corrupt_json_restore(source: Path, destination: Path) -> None:
        real_link(source, destination)
        if Path(destination) == json_path and Path(source).name.startswith(".restore.report.json."):
            json_path.write_bytes(corrupt_bytes)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_markdown_publication)
    monkeypatch.setattr(report_module.os, "link", corrupt_json_restore)

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery") as failure:
        build_report(paths, config)

    assert "differs from old content" in str(failure.value.__cause__)
    assert json_path.read_bytes() == corrupt_bytes
    assert markdown_path.read_bytes() == old_markdown
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() == old_json for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_output_pair_surfaces_rollback_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_link = report_module.os.link

    def fail_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def fail_json_restore(source: Path, destination: Path) -> None:
        if Path(destination) == json_path and Path(source).name.startswith(".restore.report.json."):
            raise OSError("injected rollback failure")
        real_link(source, destination)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_publication)
    monkeypatch.setattr(report_module.os, "link", fail_json_restore)

    with pytest.raises(report_module.ReportOutputRecoveryError, match="recovery failed") as failure:
        build_report(paths, config)

    assert isinstance(failure.value.__cause__, OSError)
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    assert any(path.read_bytes() in before for path in recovery_backups)
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)
    recovery_restores = tuple(paths.manifests.glob(".restore.*"))
    assert recovery_restores
    for path in recovery_restores:
        assert path in failure.value.recovery_paths
        assert f"manifests/{path.name}" in str(failure.value)


def test_report_cli_maps_output_recovery_failure_and_lists_backups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    real_publish = report_module._publish_report_output
    real_link = report_module.os.link

    def fail_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def fail_json_restore(source: Path, destination: Path) -> None:
        if Path(destination) == json_path and Path(source).name.startswith(".restore.report.json."):
            raise OSError("injected rollback failure")
        real_link(source, destination)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_publication)
    monkeypatch.setattr(report_module.os, "link", fail_json_restore)

    assert cli.main(["--project-root", str(tmp_path), "report"]) == 2
    stderr = capsys.readouterr().err
    assert stderr.startswith("REPORT_OUTPUT_RECOVERY_REQUIRED:")
    assert "recovery" in stderr
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    for path in recovery_backups:
        assert f"manifests/{path.name}" in stderr


def test_report_cli_uses_dedicated_recovery_code_without_absolute_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _copy_config(tmp_path)
    manifests = tmp_path / "local-data" / "manifests"
    backup = manifests / ".recovery.report.json.fixture"
    outside = tmp_path / "outside-recovery-evidence"
    error = report_module.ReportOutputRecoveryError((backup, outside), manifests_root=manifests)
    monkeypatch.setattr(
        cli,
        "build_report",
        lambda *_args: (_ for _ in ()).throw(error),
    )

    assert cli.main(["--project-root", str(tmp_path), "report"]) == 2
    stderr = capsys.readouterr().err
    assert stderr.startswith("REPORT_OUTPUT_RECOVERY_REQUIRED:")
    assert "manifests/.recovery.report.json.fixture" in stderr
    assert "manifests/<invalid-recovery-path>" in stderr
    assert str(tmp_path.resolve()) not in stderr


def test_existing_report_claim_race_preserves_foreign_and_old_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_json = json_path.read_bytes()
    old_markdown = markdown_path.read_bytes()
    foreign = b"foreign writer won the claim race\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_rename = report_module._rename_noreplace
    real_replace = report_module.os.replace
    injected = False

    def replace_before_claim(source: Path, destination: Path) -> None:
        nonlocal injected
        if Path(source) == json_path and Path(destination).name == "report.json.displaced":
            injected = True
            foreign_source = paths.manifests / "foreign.report.json"
            foreign_source.write_bytes(foreign)
            real_replace(foreign_source, json_path)
        real_rename(source, destination)

    monkeypatch.setattr(report_module, "_rename_noreplace", replace_before_claim)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert injected
    preserved = [
        path.read_bytes() for path in (json_path, *failure.value.recovery_paths) if path.is_file()
    ]
    assert foreign in preserved
    assert old_json in preserved
    assert markdown_path.read_bytes() == old_markdown


def test_failed_rollback_verification_keeps_immutable_old_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    old_json = json_path.read_bytes()
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_link = report_module.os.link

    def fail_markdown_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target.name == "report.md":
            raise OSError("injected Markdown publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def corrupt_restored_json(source: Path, destination: Path) -> None:
        real_link(source, destination)
        if Path(destination) == json_path and Path(source).name.startswith(".restore.report.json."):
            json_path.write_bytes(b"corrupt rollback target\n")

    monkeypatch.setattr(report_module, "_publish_report_output", fail_markdown_publication)
    monkeypatch.setattr(report_module.os, "link", corrupt_restored_json)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert any(
        path.is_file() and path.read_bytes() == old_json for path in failure.value.recovery_paths
    )


def test_report_recovers_durable_intent_left_after_json_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_pair = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    crashed = False

    def crash_after_json_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal crashed
        published = real_publish(temporary, target, old=old, prepared=prepared)
        if target == json_path and not crashed:
            crashed = True
            raise SystemExit("simulated hard exit")
        return published

    monkeypatch.setattr(report_module, "_publish_report_output", crash_after_json_publication)
    with pytest.raises(SystemExit, match="simulated hard exit"):
        build_report(paths, config)

    intent_path = paths.manifests / ".report-output-transaction.json"
    assert crashed
    assert intent_path.is_file()
    assert json_path.read_bytes() != old_pair[0]
    assert markdown_path.read_bytes() == old_pair[1]

    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)
    report = build_report(paths, config)

    assert json.loads(json_path.read_text(encoding="utf-8")) == report.to_dict()
    assert "Human review minutes / audio minute | 1.500000" in markdown_path.read_text(
        encoding="utf-8"
    )
    assert not intent_path.exists()


def test_active_recovery_resumes_registered_restore_after_hard_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    old_pair = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    real_restore = report_module._restore_old_report_output
    crashed = False

    def fail_markdown_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected Markdown publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def crash_after_json_restore(
        manifests: Path,
        transaction_directory: Path,
        output: report_module._ReportTransactionOutput,
    ) -> report_module._ReportOutputSnapshot | None:
        nonlocal crashed
        restored = real_restore(manifests, transaction_directory, output)
        if output.target_name == "report.json" and not crashed:
            crashed = True
            raise SystemExit("simulated rollback hard exit")
        return restored

    monkeypatch.setattr(report_module, "_publish_report_output", fail_markdown_publication)
    monkeypatch.setattr(report_module, "_restore_old_report_output", crash_after_json_restore)
    with pytest.raises(SystemExit, match="rollback hard exit"):
        build_report(paths, config)

    assert crashed
    assert (json_path.read_bytes(), markdown_path.read_bytes()) == old_pair
    assert (paths.manifests / ".report-output-transaction.json").is_file()
    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)
    monkeypatch.setattr(report_module, "_restore_old_report_output", real_restore)
    report = build_report(paths, config)

    assert json.loads(json_path.read_text(encoding="utf-8")) == report.to_dict()
    assert not tuple(paths.manifests.glob(".report-output-transaction.*"))


def test_report_rejects_malformed_recovery_intent_before_evidence_validation(
    tmp_path: Path,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    outputs = (paths.manifests / "report.json", paths.manifests / "report.md")
    before = tuple(path.read_bytes() for path in outputs)
    telemetry_path.unlink()
    intent_path = paths.manifests / ".report-output-transaction.json"
    intent_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert tuple(path.read_bytes() for path in outputs) == before
    assert intent_path in failure.value.recovery_paths
    assert intent_path.is_file()


def test_intent_link_followed_by_directory_sync_failure_recovers_without_dangling_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    outputs = (paths.manifests / "report.json", paths.manifests / "report.md")
    before = tuple(path.read_bytes() for path in outputs)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    intent_path = paths.manifests / ".report-output-transaction.json"
    real_sync = report_module._sync_report_directory
    injected = False

    def fail_first_sync_after_intent_link(directory: Path) -> None:
        nonlocal injected
        if intent_path.is_file() and not injected:
            injected = True
            raise OSError("injected intent directory sync failure")
        real_sync(directory)

    monkeypatch.setattr(report_module, "_sync_report_directory", fail_first_sync_after_intent_link)

    with pytest.raises(OSError, match="intent directory sync failure"):
        build_report(paths, config)

    assert injected
    assert tuple(path.read_bytes() for path in outputs) == before
    assert not intent_path.exists()
    assert not tuple(paths.manifests.glob(".report-output-transaction.*"))


def test_first_report_recovers_durable_intent_left_after_json_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    real_publish = report_module._publish_report_output
    crashed = False

    def crash_after_json_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        nonlocal crashed
        published = real_publish(temporary, target, old=old, prepared=prepared)
        if target == json_path and not crashed:
            crashed = True
            raise SystemExit("simulated first-report hard exit")
        return published

    monkeypatch.setattr(report_module, "_publish_report_output", crash_after_json_publication)
    with pytest.raises(SystemExit, match="first-report hard exit"):
        build_report(paths, config)

    intent_path = paths.manifests / ".report-output-transaction.json"
    assert intent_path.is_file()
    assert json_path.is_file()
    assert not markdown_path.exists()

    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)
    report = build_report(paths, config)

    assert json.loads(json_path.read_text(encoding="utf-8")) == report.to_dict()
    assert markdown_path.is_file()
    assert not intent_path.exists()


def test_report_recovery_rejects_unregistered_private_transaction_path_before_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    json_path = paths.manifests / "report.json"
    real_publish = report_module._publish_report_output

    def crash_after_json_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        published = real_publish(temporary, target, old=old, prepared=prepared)
        if target == json_path:
            raise SystemExit("simulated private-path crash")
        return published

    monkeypatch.setattr(report_module, "_publish_report_output", crash_after_json_publication)
    with pytest.raises(SystemExit, match="private-path crash"):
        build_report(paths, config)

    intent_path = paths.manifests / ".report-output-transaction.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    private_directory = paths.manifests / intent["transaction_directory"]
    unexpected = private_directory / "unregistered.evidence"
    unexpected.write_bytes(b"must not be silently removed")
    telemetry_path.unlink()
    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert "private" in str(failure.value.__cause__)
    assert unexpected.is_file()
    assert intent_path.is_file()


def test_report_handoff_revalidates_pair_before_destroying_backups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    old_json = json_path.read_bytes()
    foreign = b"foreign JSON after intent handoff\n"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_claim = report_module._claim_completed_intent
    real_replace = report_module.os.replace
    injected = False

    def claim_then_replace_json(
        intent_path: Path,
        transaction_directory: Path,
        marker_snapshot: report_module._ReportOutputSnapshot,
    ) -> object:
        nonlocal injected
        result = real_claim(intent_path, transaction_directory, marker_snapshot)
        replacement = json_path.with_name("foreign.after-handoff.json")
        replacement.write_bytes(foreign)
        real_replace(replacement, json_path)
        injected = True
        return result

    monkeypatch.setattr(report_module, "_claim_completed_intent", claim_then_replace_json)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert injected
    assert json_path.read_bytes() == foreign
    assert any(
        path.is_file() and path.read_bytes() == old_json for path in failure.value.recovery_paths
    )
    assert (paths.manifests / ".report-output-transaction.completed.json").is_file()


def test_committed_handoff_before_private_anchor_recovers_after_hard_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_claim = report_module._claim_completed_intent

    def crash_after_committed_handoff(
        intent_path: Path,
        transaction_directory: Path,
        marker_snapshot: report_module._ReportOutputSnapshot,
    ) -> object:
        real_claim(intent_path, transaction_directory, marker_snapshot)
        raise SystemExit("simulated committed handoff crash")

    monkeypatch.setattr(report_module, "_claim_completed_intent", crash_after_committed_handoff)
    with pytest.raises(SystemExit, match="committed handoff crash"):
        build_report(paths, config)

    committed = paths.manifests / ".report-output-transaction.completed.json"
    assert committed.is_file()
    assert not tuple(paths.manifests.glob(".report-output-transaction.*/intent.completed"))
    monkeypatch.setattr(report_module, "_claim_completed_intent", real_claim)
    report = build_report(paths, config)

    assert json.loads((paths.manifests / "report.json").read_text(encoding="utf-8")) == (
        report.to_dict()
    )
    assert not committed.exists()


def test_final_report_verification_failure_is_not_masked_by_unpublished_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    json_path = paths.manifests / "report.json"
    foreign = b"foreign JSON at final verification\n"
    real_require = report_module._require_report_output_snapshot
    real_replace = report_module.os.replace
    injected = False

    def replace_before_final_verification(
        path: Path,
        expected: report_module._ReportOutputSnapshot | None,
        *,
        operation: str,
    ) -> None:
        nonlocal injected
        if path == json_path and operation == "final report commit" and not injected:
            replacement = json_path.with_name("foreign.final-verification.json")
            replacement.write_bytes(foreign)
            real_replace(replacement, json_path)
            injected = True
        real_require(path, expected, operation=operation)

    monkeypatch.setattr(
        report_module,
        "_require_report_output_snapshot",
        replace_before_final_verification,
    )

    with pytest.raises(OSError, match="changed before final report commit"):
        build_report(paths, config)

    assert injected
    assert json_path.read_bytes() == foreign


def test_existing_report_same_bytes_new_inode_is_not_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    old_json = json_path.read_bytes()
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_rename = report_module.os.rename
    real_replace = report_module.os.replace
    injected = False

    def replace_source_before_claim(source: Path, destination: Path) -> None:
        nonlocal injected
        if Path(source) == json_path and Path(destination).name == "report.json.displaced":
            replacement = paths.manifests / "same-bytes-new-inode.json"
            replacement.write_bytes(old_json)
            real_replace(replacement, json_path)
            injected = True
        real_rename(source, destination)

    monkeypatch.setattr(
        report_module,
        "_rename_noreplace",
        replace_source_before_claim,
        raising=False,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert injected
    assert any(
        path.is_file() and path.read_bytes() == old_json
        for path in (json_path, *failure.value.recovery_paths)
    )


def test_report_claim_destination_race_is_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_noreplace = getattr(report_module, "_rename_noreplace", report_module.os.rename)
    foreign = b"foreign private claim destination\n"
    injected_path: Path | None = None

    def occupy_destination_before_claim(source: Path, destination: Path) -> None:
        nonlocal injected_path
        if Path(destination).name == "report.json.displaced" and injected_path is None:
            injected_path = Path(destination)
            injected_path.write_bytes(foreign)
        real_noreplace(source, destination)

    monkeypatch.setattr(
        report_module,
        "_rename_noreplace",
        occupy_destination_before_claim,
        raising=False,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError):
        build_report(paths, config)

    assert injected_path is not None
    assert injected_path.read_bytes() == foreign


@pytest.mark.parametrize(
    ("name", "content"),
    (
        ("unregistered.after-completion", b"unregistered cleanup evidence\n"),
        ("report.json.rollback-current", b"foreign allowed-name evidence\n"),
    ),
)
def test_committed_cleanup_rejects_foreign_private_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    content: bytes,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_claim = report_module._claim_completed_intent
    injected_path: Path | None = None

    def claim_then_inject(
        intent_path: Path,
        transaction_directory: Path,
        marker_snapshot: report_module._ReportOutputSnapshot,
    ) -> object:
        nonlocal injected_path
        result = real_claim(intent_path, transaction_directory, marker_snapshot)
        injected_path = transaction_directory / name
        injected_path.write_bytes(content)
        return result

    monkeypatch.setattr(report_module, "_claim_completed_intent", claim_then_inject)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert injected_path is not None
    assert injected_path.read_bytes() == content
    assert tuple(paths.manifests.glob(".recovery.report.*"))
    assert (
        paths.manifests / ".report-output-transaction.completed.json"
    ) in failure.value.recovery_paths


@pytest.mark.parametrize("remove_root_marker", (False, True))
def test_report_recovers_completed_marker_orphan_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remove_root_marker: bool,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_claim = report_module._claim_final_intent

    def crash_after_completed_claim(
        intent_path: Path,
        transaction_directory: Path,
        marker_snapshot: report_module._ReportOutputSnapshot,
    ) -> object:
        real_claim(intent_path, transaction_directory, marker_snapshot)
        raise SystemExit("simulated completed-marker crash")

    monkeypatch.setattr(report_module, "_claim_final_intent", crash_after_completed_claim)
    with pytest.raises(SystemExit, match="completed-marker crash"):
        build_report(paths, config)

    assert tuple(paths.manifests.glob(".report-output-transaction.*/intent.completed"))
    if remove_root_marker:
        (paths.manifests / ".report-output-transaction.completed.json").unlink()
    monkeypatch.setattr(report_module, "_claim_final_intent", real_claim)
    report = build_report(paths, config)

    assert json.loads((paths.manifests / "report.json").read_text(encoding="utf-8")) == (
        report.to_dict()
    )
    assert not tuple(paths.manifests.glob(".report-output-transaction.*"))


def test_report_resumes_committed_cleanup_after_one_payload_was_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_delete = report_module._delete_owned_report_file
    crashed = False

    def crash_after_first_committed_payload_delete(
        path: Path,
        expected: report_module._ReportSnapshotIdentity,
        *,
        missing_ok: bool = False,
    ) -> None:
        nonlocal crashed
        is_committed_payload = (
            path.parent == paths.manifests
            and path.name.startswith(".report.json.")
            and (paths.manifests / ".report-output-transaction.completed.json").is_file()
            and bool(tuple(paths.manifests.glob(".report-output-transaction.*/intent.completed")))
        )
        real_delete(path, expected, missing_ok=missing_ok)
        if is_committed_payload and not crashed:
            crashed = True
            raise SystemExit("simulated partial committed cleanup")

    monkeypatch.setattr(
        report_module,
        "_delete_owned_report_file",
        crash_after_first_committed_payload_delete,
    )
    with pytest.raises(SystemExit, match="partial committed cleanup"):
        build_report(paths, config)

    assert crashed
    assert (paths.manifests / ".report-output-transaction.completed.json").is_file()
    monkeypatch.setattr(report_module, "_delete_owned_report_file", real_delete)
    report = build_report(paths, config)

    assert json.loads((paths.manifests / "report.json").read_text(encoding="utf-8")) == (
        report.to_dict()
    )
    assert not tuple(paths.manifests.glob(".report-output-transaction.*"))


def test_redundant_committed_markers_must_share_full_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_claim = report_module._claim_final_intent

    def crash_after_final_anchor(
        intent_path: Path,
        transaction_directory: Path,
        marker_snapshot: report_module._ReportOutputSnapshot,
    ) -> object:
        real_claim(intent_path, transaction_directory, marker_snapshot)
        raise SystemExit("simulated redundant-marker crash")

    monkeypatch.setattr(report_module, "_claim_final_intent", crash_after_final_anchor)
    with pytest.raises(SystemExit, match="redundant-marker crash"):
        build_report(paths, config)

    final_marker = next(paths.manifests.glob(".report-output-transaction.*/intent.completed"))
    replacement = final_marker.with_name("replacement.intent.completed")
    replacement.write_bytes(final_marker.read_bytes())
    report_module.os.replace(replacement, final_marker)
    telemetry_path.unlink()
    monkeypatch.setattr(report_module, "_claim_final_intent", real_claim)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert final_marker.is_file()
    assert final_marker in failure.value.recovery_paths
    assert (paths.manifests / ".report-output-transaction.completed.json").is_file()


def test_root_committed_marker_survives_private_directory_cleanup_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_delete = report_module._delete_owned_report_file
    foreign_path: Path | None = None

    def inject_after_private_marker_delete(
        path: Path,
        expected: report_module._ReportSnapshotIdentity,
        *,
        missing_ok: bool = False,
    ) -> None:
        nonlocal foreign_path
        real_delete(path, expected, missing_ok=missing_ok)
        if path.name == "intent.completed" and foreign_path is None:
            foreign_path = path.parent / "foreign-after-private-marker"
            foreign_path.write_bytes(b"foreign private cleanup race\n")

    monkeypatch.setattr(
        report_module,
        "_delete_owned_report_file",
        inject_after_private_marker_delete,
    )

    with pytest.raises(report_module.ReportOutputRecoveryError):
        build_report(paths, config)

    assert foreign_path is not None
    assert foreign_path.read_bytes() == b"foreign private cleanup race\n"
    assert (paths.manifests / ".report-output-transaction.completed.json").is_file()


def _leave_outcome_marker_after_private_directory_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> tuple[CorpusPaths, CorpusConfig, Path]:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output
    if outcome == "rolled-back":
        real_publish = _inject_markdown_report_publication_failure(paths, monkeypatch)
    marker_name = {
        "rolled-back": ".report-output-transaction.rolled-back.json",
        "committed": ".report-output-transaction.completed.json",
    }[outcome]
    marker = paths.manifests / marker_name
    real_remove = report_module._remove_owned_transaction_directory
    crashed = False

    def crash_after_outcome_directory_removal(
        directory: Path,
        expected_identity: report_module._ReportDirectoryIdentity,
    ) -> None:
        nonlocal crashed
        real_remove(directory, expected_identity)
        if marker.is_file() and not crashed:
            crashed = True
            raise SystemExit(f"simulated {outcome} terminal cleanup crash")

    monkeypatch.setattr(
        report_module,
        "_remove_owned_transaction_directory",
        crash_after_outcome_directory_removal,
    )
    with pytest.raises(SystemExit, match=f"{outcome} terminal cleanup crash"):
        build_report(paths, config)

    assert crashed
    assert marker.is_file()
    monkeypatch.setattr(report_module, "_remove_owned_transaction_directory", real_remove)
    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)
    return paths, config, marker


@pytest.mark.parametrize("outcome", ("rolled-back", "committed"))
def test_terminal_outcome_marker_recovers_after_private_directory_hard_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    paths, config, marker = _leave_outcome_marker_after_private_directory_removal(
        tmp_path,
        monkeypatch,
        outcome,
    )

    report = build_report(paths, config)

    assert json.loads((paths.manifests / "report.json").read_text(encoding="utf-8")) == (
        report.to_dict()
    )
    assert not marker.exists()


@pytest.mark.parametrize("outcome", ("rolled-back", "committed"))
def test_terminal_outcome_rejects_corrupted_authoritative_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    paths, config, marker = _leave_outcome_marker_after_private_directory_removal(
        tmp_path,
        monkeypatch,
        outcome,
    )
    json_path = paths.manifests / "report.json"
    replacement = json_path.with_name("foreign.terminal-report.json")
    replacement.write_bytes(b"foreign terminal report\n")
    report_module.os.replace(replacement, json_path)
    (paths.manifests / "pilot-telemetry.json").unlink()

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert marker.is_file()
    assert marker in failure.value.recovery_paths


def test_report_rejects_corrupted_completed_marker_orphan_before_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    json_path = paths.manifests / "report.json"
    real_claim = report_module._claim_final_intent

    def crash_after_completed_claim(
        intent_path: Path,
        transaction_directory: Path,
        marker_snapshot: report_module._ReportOutputSnapshot,
    ) -> object:
        real_claim(intent_path, transaction_directory, marker_snapshot)
        raise SystemExit("simulated corrupted-completed crash")

    monkeypatch.setattr(report_module, "_claim_final_intent", crash_after_completed_claim)
    with pytest.raises(SystemExit, match="corrupted-completed crash"):
        build_report(paths, config)

    json_path.write_bytes(b"corrupted committed report\n")
    telemetry_path.unlink()
    monkeypatch.setattr(report_module, "_claim_final_intent", real_claim)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert tuple(paths.manifests.glob(".recovery.report.*"))
    assert any(path.name == "intent.completed" for path in failure.value.recovery_paths)


def test_active_intent_requires_every_prepared_output_before_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output

    def crash_before_first_publication(*_args: object, **_kwargs: object) -> object:
        raise SystemExit("simulated active-intent crash")

    monkeypatch.setattr(report_module, "_publish_report_output", crash_before_first_publication)
    with pytest.raises(SystemExit, match="active-intent crash"):
        build_report(paths, config)

    intent_path = paths.manifests / ".report-output-transaction.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    prepared = paths.manifests / intent["outputs"][0]["prepared_name"]
    prepared.unlink()
    telemetry_path.unlink()
    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert intent_path.is_file()
    assert intent_path in failure.value.recovery_paths


def test_active_intent_rejects_same_name_transaction_directory_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output

    def crash_before_first_publication(*_args: object, **_kwargs: object) -> object:
        raise SystemExit("simulated directory-replacement crash")

    monkeypatch.setattr(report_module, "_publish_report_output", crash_before_first_publication)
    with pytest.raises(SystemExit, match="directory-replacement crash"):
        build_report(paths, config)

    intent_path = paths.manifests / ".report-output-transaction.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    transaction_directory = paths.manifests / intent["transaction_directory"]
    preserved_owned_directory = transaction_directory.with_name(
        f"preserved-{transaction_directory.name}"
    )
    transaction_directory.rename(preserved_owned_directory)
    transaction_directory.mkdir()
    telemetry_path.unlink()
    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert transaction_directory.is_dir()
    assert intent_path.is_file()
    assert transaction_directory in failure.value.recovery_paths


def test_active_and_committed_markers_are_rejected_as_ambiguous_before_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_publish = report_module._publish_report_output

    def crash_before_first_publication(*_args: object, **_kwargs: object) -> object:
        raise SystemExit("simulated ambiguous-marker crash")

    monkeypatch.setattr(report_module, "_publish_report_output", crash_before_first_publication)
    with pytest.raises(SystemExit, match="ambiguous-marker crash"):
        build_report(paths, config)

    active = paths.manifests / ".report-output-transaction.json"
    committed = paths.manifests / ".report-output-transaction.completed.json"
    report_module.os.link(active, committed)
    telemetry_path.unlink()
    monkeypatch.setattr(report_module, "_publish_report_output", real_publish)

    with pytest.raises(report_module.ReportOutputRecoveryError) as failure:
        build_report(paths, config)

    assert active.is_file()
    assert committed.is_file()
    assert active in failure.value.recovery_paths
    assert committed in failure.value.recovery_paths


def test_report_output_cleanup_failure_does_not_mask_recovery_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    real_publish = report_module._publish_report_output
    real_link = report_module.os.link
    real_unlink = type(telemetry_path).unlink

    def fail_publication(
        temporary: Path,
        target: Path,
        *,
        old: report_module._ReportOutputSnapshot | None,
        prepared: report_module._ReportOutputSnapshot,
    ) -> report_module._ReportOutputSnapshot:
        if target == markdown_path:
            raise OSError("injected publication failure")
        return real_publish(temporary, target, old=old, prepared=prepared)

    def fail_json_restore(source: Path, destination: Path) -> None:
        if Path(destination) == json_path and Path(source).name.startswith(".restore.report.json."):
            raise OSError("injected rollback failure")
        real_link(source, destination)

    def fail_report_temporary_cleanup(self: Path, missing_ok: bool = False) -> None:
        if self.parent == paths.manifests and self.name.startswith(".report."):
            raise OSError("injected cleanup failure")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(report_module, "_publish_report_output", fail_publication)
    monkeypatch.setattr(report_module.os, "link", fail_json_restore)
    monkeypatch.setattr(type(telemetry_path), "unlink", fail_report_temporary_cleanup)

    with pytest.raises(OSError, match="recovery failed") as failure:
        build_report(paths, config)

    assert isinstance(failure.value.__cause__, OSError)
    assert "injected rollback failure" in str(failure.value.__cause__)
    recovery_backups = tuple(paths.manifests.glob(".recovery.*"))
    assert recovery_backups
    for path in recovery_backups:
        assert f"manifests/{path.name}" in str(failure.value)


@pytest.mark.parametrize("failed_prepare_call", (2, 3, 4, 5, 6))
def test_report_output_pair_preserves_existing_files_when_preparation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_prepare_call: int,
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    telemetry_path = paths.manifests / "pilot-telemetry.json"
    write_jsonl_atomic(telemetry_path, (_telemetry_row(),))
    build_report(paths, config)
    json_path = paths.manifests / "report.json"
    markdown_path = paths.manifests / "report.md"
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    write_jsonl_atomic(telemetry_path, (_telemetry_row(review_seconds=30.0),))
    real_prepare = report_module._prepare_atomic
    calls = 0

    def fail_prepare(path: Path, content: bytes) -> Path:
        nonlocal calls
        calls += 1
        if calls == failed_prepare_call:
            raise OSError("injected preparation failure")
        return real_prepare(path, content)

    monkeypatch.setattr(report_module, "_prepare_atomic", fail_prepare)

    with pytest.raises(OSError, match="injected preparation failure"):
        build_report(paths, config)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == before
    assert not tuple(paths.manifests.glob(".report.*"))


def test_first_report_failure_leaves_neither_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, config = _completed_two_recording_project(tmp_path)
    write_jsonl_atomic(paths.manifests / "pilot-telemetry.json", (_telemetry_row(),))
    markdown_path = paths.manifests / "report.md"
    real_link = report_module.os.link

    def fail_markdown(source: Path, destination: Path) -> None:
        if Path(destination) == markdown_path:
            raise OSError("injected first publication failure")
        real_link(source, destination)

    monkeypatch.setattr(report_module.os, "link", fail_markdown)

    with pytest.raises(OSError, match="injected first publication failure"):
        build_report(paths, config)

    assert not (paths.manifests / "report.json").exists()
    assert not markdown_path.exists()
    assert not tuple(paths.manifests.glob(".report.*"))
