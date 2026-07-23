import re
from pathlib import Path

import pytest

from latintts.corpus import cli
from latintts.corpus.cli import main
from latintts.corpus.cli_safety import safe_error_message


def test_doctor_returns_failure_when_ffmpeg_is_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("latintts.corpus.cli.shutil.which", lambda name: None)
    exit_code = main(["--project-root", str(tmp_path), "doctor"])
    assert exit_code == 1
    assert "ALIGNER_UNAVAILABLE: ffmpeg not found in PATH" in capsys.readouterr().out


_CONFIG_COMMANDS = (
    "segment",
    "pair",
    "align",
    "export-review",
    "import-review",
    "build-manifest",
    "report",
)


@pytest.mark.parametrize("command", _CONFIG_COMMANDS)
def test_missing_config_errors_keep_relative_clue_without_project_root(
    command: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    relative_config = "config/corpus/__missing__.json"

    assert (
        main(
            [
                "--project-root",
                str(tmp_path),
                command,
                "--config",
                relative_config,
            ]
        )
        == 2
    )

    stderr = capsys.readouterr().err.replace("\\", "/")
    assert stderr.startswith("MANIFEST_SCHEMA_MISMATCH:")
    assert relative_config in stderr
    assert str(tmp_path.resolve()).replace("\\", "/").casefold() not in stderr.casefold()


@pytest.mark.parametrize("operation", ("snapshotting", "final report validation"))
def test_report_os_error_uses_safe_project_relative_path(
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    absolute = tmp_path / "local-data" / "manifests" / "report.json"
    differently_cased = str(absolute.resolve()).replace("\\", "/").swapcase()

    monkeypatch.setattr(cli.CorpusConfig, "load", lambda _path: object())
    monkeypatch.setattr(cli.CorpusPaths, "ensure_layout", lambda _paths: None)

    def fail_report(*_args: object) -> object:
        raise OSError(f"report output changed while {operation}: {differently_cased}")

    monkeypatch.setattr(cli, "build_report", fail_report)

    assert main(["--project-root", str(tmp_path), "report"]) == 2

    stderr = capsys.readouterr().err.replace("\\", "/")
    assert stderr.startswith("MANIFEST_SCHEMA_MISMATCH:")
    assert "<project-root>/local-data/manifests/report.json" in stderr.casefold()
    assert str(tmp_path.resolve()).replace("\\", "/").casefold() not in stderr.casefold()


def test_report_os_error_redacts_absolute_path_outside_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    external = (tmp_path.parent / "private-user-home" / "evidence.json").resolve()
    monkeypatch.setattr(cli.CorpusConfig, "load", lambda _path: object())
    monkeypatch.setattr(cli.CorpusPaths, "ensure_layout", lambda _paths: None)

    def fail_report(*_args: object) -> object:
        raise OSError(f"unable to read external evidence: {external}")

    monkeypatch.setattr(cli, "build_report", fail_report)

    assert main(["--project-root", str(tmp_path), "report"]) == 2

    stderr = capsys.readouterr().err.replace("\\", "/")
    assert "<outside-project-root>" in stderr
    assert str(external).replace("\\", "/").casefold() not in stderr.casefold()


@pytest.mark.parametrize(
    ("arguments", "target"),
    (
        (("doctor",), "_doctor"),
        (("inventory", "--init-intake"), "write_intake_skeleton"),
        (("inventory",), "inventory_from_manifests"),
        (("select-pilot",), "_select_pilot"),
        (("prepare-text",), "_prepare_text"),
    ),
)
def test_non_config_command_os_errors_are_safe_and_have_stable_code(
    arguments: tuple[str, ...],
    target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    absolute = tmp_path / "local-data" / "manifests" / "evidence.json"
    monkeypatch.setattr(cli.CorpusPaths, "ensure_layout", lambda _paths: None)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise OSError(f"injected failure at {absolute}")

    monkeypatch.setattr(cli, target, fail)

    assert main(["--project-root", str(tmp_path), *arguments]) == 2

    stderr = capsys.readouterr().err.replace("\\", "/")
    assert stderr.startswith("MANIFEST_SCHEMA_MISMATCH:")
    assert "<project-root>/local-data/manifests/evidence.json" in stderr
    assert str(tmp_path.resolve()).replace("\\", "/").casefold() not in stderr.casefold()


def test_argument_parser_error_redacts_absolute_argument(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    external = (tmp_path.parent / "private-user-home" / "argument.json").resolve()

    assert (
        main(
            [
                "--project-root",
                str(tmp_path),
                "report",
                "--unsupported-path",
                str(external),
            ]
        )
        == 2
    )

    stderr = capsys.readouterr().err.replace("\\", "/")
    assert stderr == "MANIFEST_SCHEMA_MISMATCH: invalid command-line arguments\n"
    assert str(external).replace("\\", "/").casefold() not in stderr.casefold()


@pytest.mark.parametrize(
    ("argument", "secret"),
    (
        (r"C:\Users\Doe, John\secret.txt", "John"),
        (r"\\?\UNC\server\share,private\Users\alice\secret.txt", "alice"),
        ("file://server/share/Users/alice/secret.txt", "Users/alice"),
        ("/home/alice, private/secret.txt", "private"),
    ),
)
def test_argument_parser_error_redacts_absolute_path_variants(
    argument: str,
    secret: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--project-root", str(tmp_path), "report", "--unsupported", argument]) == 2

    stderr = capsys.readouterr().err.replace("\\", "/")
    assert stderr == "MANIFEST_SCHEMA_MISMATCH: invalid command-line arguments\n"
    assert secret.casefold() not in stderr.casefold()


@pytest.mark.parametrize(
    ("message", "secret"),
    (
        (r"failed at C:\Users\Doe, John\secret.txt", "John"),
        (r"failed at \\?\UNC\server\share,private\Users\alice\secret.txt", "alice"),
        ("failed at file://server/share/Users/alice/secret.txt", "Users/alice"),
        ("failed at /home/alice, private/secret.txt", "private"),
    ),
)
def test_safe_error_message_redacts_absolute_path_variants(
    message: str,
    secret: str,
    tmp_path: Path,
) -> None:
    rendered = safe_error_message(OSError(message), tmp_path)
    assert "<outside-project-root>" in rendered
    assert secret.casefold() not in rendered.casefold()


@pytest.mark.parametrize(
    "message",
    (
        "failed at file:%2F%2F%2Fhome%2Falice%2Fsecret.txt",
        "failed at FILE:%5C%5Cserver%5Cshare%5CUsers%5Calice%5Csecret.txt",
        "failed at file%3A%2F%2F%2Fhome%2Falice%2Fsecret.txt",
        "failed at FILE%3a%5C%5Cserver%5Cshare%5CUsers%5Calice%5Csecret.txt",
        "failed at %66ile%3A%2F%2F%2Fhome%2Falice%2Fsecret.txt",
        "failed at f%69le:%5C%5Cserver%5Cshare%5CUsers%5Calice%5Csecret.txt",
        "failed at %46%49%4C%45%3a%2f%2f%2fhome%2falice%2fsecret.txt",
    ),
)
def test_safe_error_message_redacts_percent_encoded_file_uris(
    message: str,
    tmp_path: Path,
) -> None:
    rendered = safe_error_message(OSError(message), tmp_path)
    assert "<outside-project-root>" in rendered
    assert "alice" not in rendered.casefold()


@pytest.mark.parametrize(
    "message",
    (
        r"\\?\GLOBALROOT\Device\HarddiskVolumeShadowCopy1\Users\alice\secret.txt",
        r"\\?\Volume{abc}\Users\alice\secret.txt",
    ),
)
def test_safe_error_message_redacts_exact_windows_device_namespace(
    message: str,
    tmp_path: Path,
) -> None:
    assert safe_error_message(OSError(message), tmp_path) == "<outside-project-root>"


@pytest.mark.parametrize(
    ("message", "secret"),
    (
        (r"failed at <C:\Users\alice\secret.txt>", "alice"),
        (r"failed at <\\server\share\Users\alice\secret.txt>", "alice"),
        ("failed at </home/alice/secret.txt>", "alice"),
        ("failed at path:/home/alice/secret.txt", "alice"),
    ),
)
def test_safe_error_message_redacts_delimiter_adjacent_absolute_paths(
    message: str,
    secret: str,
    tmp_path: Path,
) -> None:
    rendered = safe_error_message(OSError(message), tmp_path)
    assert "<outside-project-root>" in rendered
    assert secret.casefold() not in rendered.casefold()


@pytest.mark.parametrize(
    "message",
    (
        "failed at marker./home/alice/secret.txt",
        "failed at marker-/home/alice/secret.txt",
        "failed at marker_/home/alice/secret.txt",
        r"failed at marker.\server\share\alice\secret.txt",
    ),
)
def test_safe_error_message_redacts_punctuation_adjacent_absolute_paths(
    message: str,
    tmp_path: Path,
) -> None:
    rendered = safe_error_message(OSError(message), tmp_path)
    assert "<outside-project-root>" in rendered
    assert "alice" not in rendered.casefold()


@pytest.mark.parametrize(
    "message",
    (
        "failed at relative/path.txt",
        "failed at ./relative/path.txt",
        "failed at ../relative/path.txt",
    ),
)
def test_safe_error_message_preserves_unambiguous_relative_paths(
    message: str,
    tmp_path: Path,
) -> None:
    assert safe_error_message(OSError(message), tmp_path) == message


def test_safe_error_message_preserves_project_root_placeholder(tmp_path: Path) -> None:
    rendered = safe_error_message(
        OSError(f"failed at {tmp_path / 'local-data' / 'evidence.json'}"),
        tmp_path,
    )
    assert rendered == "failed at <project-root>/local-data/evidence.json"


def test_safe_error_message_redacts_project_root_parent_traversal(tmp_path: Path) -> None:
    outside = tmp_path / ".." / "private" / "alice" / "evidence.json"

    rendered = safe_error_message(OSError(f"failed at {outside}"), tmp_path)

    assert "<outside-project-root>" in rendered
    assert "alice" not in rendered.casefold()
    assert "<project-root>/.." not in rendered


def test_safe_error_message_treats_posix_root_aliases_as_case_sensitive() -> None:
    rendered = safe_error_message(
        OSError(
            "inside /srv/LatinTTS/local-data/evidence.json; "
            "outside /srv/latintts/private/alice/evidence.json"
        ),
        Path("/srv/LatinTTS"),
        resolve_project_root=False,
    )

    assert "<project-root>/local-data/evidence.json" in rendered
    assert "<outside-project-root>" in rendered
    assert "alice" not in rendered.casefold()


@pytest.mark.parametrize(
    ("command", "target"),
    (
        ("build-manifest", "build_manifest_corpus"),
        ("report", "build_report"),
        ("export-review", "export_review_bundle"),
        ("import-review", "import_review_bundle"),
        ("pair", "pair_corpus"),
        ("segment", "segment_corpus"),
    ),
)
def test_command_core_programmer_type_error_propagates(
    command: str,
    target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.CorpusConfig, "load", lambda _path: object())
    monkeypatch.setattr(cli.CorpusPaths, "ensure_layout", lambda _paths: None)
    monkeypatch.setattr(cli, "_ffmpeg_version", lambda: "test-ffmpeg")
    monkeypatch.setattr(cli, "create_alignment_backend", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli,
        "_validate_segmentation_config",
        lambda _config: (
            {
                "threshold": 0.5,
                "neg_threshold": 0.35,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 100,
                "max_speech_duration_s": 30.0,
                "speech_pad_ms": 30,
                "min_silence_at_max_speech": 98,
                "use_max_poss_sil_at_max_speech": True,
                "sample_rate": 16_000,
                "window_samples": 512,
            },
            object(),
        ),
    )
    monkeypatch.setattr(cli, "SileroVadBackend", lambda **_kwargs: object())

    def fail(*_args: object, **_kwargs: object) -> object:
        raise TypeError("programmer bug")

    monkeypatch.setattr(cli, target, fail)

    with pytest.raises(TypeError, match="programmer bug"):
        main(["--project-root", str(tmp_path), command])


@pytest.mark.parametrize("target", ("SileroVadBackend", "classify_pauses"))
def test_segment_config_validator_programmer_type_error_propagates(
    target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json"
    monkeypatch.setattr(cli.CorpusPaths, "ensure_layout", lambda _paths: None)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise TypeError("programmer bug")

    monkeypatch.setattr(cli, target, fail)

    with pytest.raises(TypeError, match="programmer bug"):
        main(
            [
                "--project-root",
                str(tmp_path),
                "segment",
                "--config",
                str(config_path),
            ]
        )


@pytest.mark.parametrize(
    "argument",
    (
        r"--unsupportedC:\Users\alice\secret.txt",
        "--unsupported/home/alice/secret.txt",
    ),
)
def test_argument_parser_error_redacts_absolute_path_embedded_in_option(
    argument: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--project-root", str(tmp_path), "report", argument]) == 2

    stderr = capsys.readouterr().err.replace("\\", "/")
    assert stderr == "MANIFEST_SCHEMA_MISMATCH: invalid command-line arguments\n"


@pytest.mark.parametrize(
    "command",
    (
        r"prefixC:\Users\alice\secret.txt",
        "prefix/home/alice/secret.txt",
        "prefixfile://server/share/Users/alice/secret.txt",
    ),
)
def test_invalid_subcommand_never_echoes_user_token(
    command: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--project-root", str(tmp_path), command]) == 2
    assert capsys.readouterr().err == ("MANIFEST_SCHEMA_MISMATCH: invalid command-line arguments\n")


def test_dry_run_does_not_resolve_or_inspect_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_resolve(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run must not resolve filesystem paths")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "resolve", forbidden_resolve)
        assert main(["--project-root", str(tmp_path / "project"), "doctor", "--dry-run"]) == 0

    assert capsys.readouterr().err == ""


def test_failed_dry_run_does_not_resolve_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    resolve_calls = 0

    def observed_resolve(*_args: object, **_kwargs: object) -> object:
        nonlocal resolve_calls
        resolve_calls += 1
        raise OSError("unexpected resolution")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "resolve", observed_resolve)
        assert (
            main(
                [
                    "--project-root",
                    str(tmp_path / "project"),
                    "report",
                    "--dry-run",
                    "--config",
                    str(tmp_path / "outside.json"),
                ]
            )
            == 2
        )

    assert resolve_calls == 0
    assert capsys.readouterr().err == (
        "MANIFEST_SCHEMA_MISMATCH: --dry-run config must be within the project root\n"
    )


def test_safe_error_formatter_survives_project_root_resolution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable_resolve(*_args: object, **_kwargs: object) -> object:
        raise OSError("root resolution unavailable")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "resolve", unavailable_resolve)
        assert main(["--project-root", str(tmp_path), "report", "--unsupported"]) == 2

    stderr = capsys.readouterr().err
    assert stderr == "MANIFEST_SCHEMA_MISMATCH: invalid command-line arguments\n"


_DRY_RUN_CASES = (
    (
        ("doctor", "--dry-run"),
        (
            "WRITE local-data/raw/spoken/",
            "WRITE local-data/manifests/",
        ),
    ),
    (
        ("inventory", "--dry-run", "--init-intake"),
        (
            "READ local-data/raw/spoken/**/*",
            "READ local-data/raw/sung/**/*",
            "WRITE local-data/manifests/intake.csv",
        ),
    ),
    (
        ("inventory", "--dry-run"),
        (
            "READ local-data/manifests/intake.csv",
            "READ local-data/manifests/rights.jsonl",
            "WRITE local-data/manifests/recordings.jsonl",
            "WRITE local-data/derived/corpus-v1/alignments/runs/inventory/processing-events.jsonl",
        ),
    ),
    (
        (
            "select-pilot",
            "--dry-run",
            "--recording-id",
            "rec-one",
            "--recording-id",
            "rec-two",
        ),
        (
            "READ local-data/manifests/recordings.jsonl",
            "WRITE local-data/manifests/pilot-selection.json",
        ),
    ),
    (
        ("prepare-text", "--dry-run", "--init", "--replace"),
        (
            "READ local-data/manifests/pilot-selection.json",
            "WRITE local-data/manifests/transcript-intake.jsonl",
        ),
    ),
    (
        ("prepare-text", "--dry-run"),
        (
            "READ local-data/manifests/transcript-intake.jsonl",
            "READ local-data/**/*",
            "WRITE local-data/manifests/transcripts.jsonl",
            "WRITE local-data/manifests/pronunciation-review-*.json",
            (
                "WRITE local-data/derived/corpus-v1/alignments/runs/prepare-text/"
                "processing-events.jsonl"
            ),
        ),
    ),
    (
        ("segment", "--dry-run"),
        (
            "READ config/corpus/pilot-v1.json",
            "READ local-data/manifests/transcripts.jsonl",
            "WRITE local-data/derived/corpus-v1/normalized/analysis-*.wav",
            "WRITE local-data/derived/corpus-v1/normalized/analysis-*.wav.lock",
            "WRITE local-data/derived/corpus-v1/alignments/runs/*/*/segmentation.json",
        ),
    ),
    (
        ("pair", "--dry-run"),
        (
            "READ config/corpus/pilot-v1.json",
            "READ local-data/derived/corpus-v1/alignments/runs/*/*/segmentation.json",
            "WRITE local-data/derived/corpus-v1/segments/candidates/candidate-*.wav",
            "WRITE local-data/derived/corpus-v1/segments/candidates/candidate-*.wav.lock",
            "WRITE local-data/derived/corpus-v1/alignments/runs/*/*/.pairing.lock",
            "WRITE local-data/derived/corpus-v1/alignments/runs/*/*/pairing.json",
        ),
    ),
    (
        ("align", "--dry-run"),
        (
            "READ config/corpus/pilot-v1.json",
            "READ local-data/derived/corpus-v1/alignments/runs/*/*/pairing.json",
            "WRITE local-data/derived/corpus-v1/alignments/runs/*/*/alignment.json",
        ),
    ),
    (
        ("align", "--dry-run", "--smoke-test", "--allow-download"),
        (
            "READ config/corpus/pilot-v1.json",
            "READ .git",
            "READ .git/**/*",
            "WRITE local-data/derived/corpus-v1/alignments/runtime/environment.txt",
            "WRITE local-data/derived/corpus-v1/alignments/runtime/smoke.json",
        ),
    ),
    (
        ("export-review", "--dry-run"),
        (
            "READ config/corpus/pilot-v1.json",
            "READ local-data/derived/corpus-v1/alignments/runs/*/*/alignment.json",
            "WRITE local-data/derived/corpus-v1/alignments/runs/*/*/review/*/decision.json",
            "WRITE local-data/derived/corpus-v1/segments/review/review-*.wav",
            "WRITE local-data/derived/corpus-v1/segments/review/review-*.wav.lock",
        ),
    ),
    (
        ("import-review", "--dry-run"),
        (
            "READ config/corpus/pilot-v1.json",
            "READ local-data/derived/corpus-v1/alignments/runs/*/*/review/**/*",
            "WRITE local-data/manifests/review.jsonl",
            "WRITE local-data/manifests/recordings.jsonl",
        ),
    ),
    (
        ("build-manifest", "--dry-run"),
        (
            "READ config/corpus/pilot-v1.json",
            "READ local-data/manifests/review.jsonl",
            "READ local-data/derived/corpus-v1/segments/review/review-*.wav",
            "READ local-data/derived/corpus-v1/segments/.manifest-staging/**/*",
            "WRITE local-data/manifests/segments.jsonl",
            "WRITE local-data/derived/corpus-v1/segments/lossless/lossless-*.flac",
            "WRITE local-data/derived/corpus-v1/alignments/runs/*/*/pairing.json",
            "WRITE local-data/derived/corpus-v1/alignments/runs/*/*/pairing-automatic.json",
            "RECOVER local-data/derived/corpus-v1/segments/.manifest-staging/**/*",
            "RECOVER local-data/derived/corpus-v1/alignments/runs/*/*/pairing.json",
            "RECOVER local-data/derived/corpus-v1/alignments/runs/*/*/pairing-automatic.json",
        ),
    ),
    (
        ("report", "--dry-run"),
        (
            "READ config/corpus/pilot-v1.json",
            "READ local-data/manifests/pilot-telemetry.json",
            "READ local-data/manifests/recordings.jsonl",
            "READ local-data/manifests/pilot-selection.json",
            "READ local-data/manifests/rights.jsonl",
            "READ local-data/manifests/transcripts.jsonl",
            "READ local-data/manifests/review.jsonl",
            "READ local-data/manifests/segments.jsonl",
            "READ local-data/manifests/.report-output-transaction.json",
            "READ local-data/manifests/.report-output-transaction.*/**/*",
            "READ local-data/manifests/.report.json.*",
            "READ local-data/manifests/.report.md.*",
            "WRITE local-data/manifests/report.json",
            "WRITE local-data/manifests/report.md",
            "WRITE local-data/manifests/.report-output-transaction.json",
            "WRITE local-data/manifests/.report-output-transaction.*/**/*",
            "WRITE local-data/manifests/..report-output-transaction.json.*",
            "WRITE local-data/manifests/.report.json.*",
            "WRITE local-data/manifests/.report.md.*",
            "RECOVER local-data/manifests/.report-output-transaction.json",
            "RECOVER local-data/manifests/.report-output-transaction.rolled-back.json",
            "RECOVER local-data/manifests/.report-output-transaction.completed.json",
            "RECOVER local-data/manifests/.report-output-transaction.*/**/*",
            "RECOVER local-data/manifests/.recovery.report.*",
            "RECOVER local-data/manifests/.restore.report.*",
        ),
    ),
)


@pytest.mark.parametrize(("arguments", "expected_lines"), _DRY_RUN_CASES)
def test_dry_run_is_declarative_relative_and_has_no_side_effects(
    arguments: tuple[str, ...],
    expected_lines: tuple[str, ...],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_root = tmp_path / "uncreated-project"

    assert main(["--project-root", str(project_root), *arguments]) == 0

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert captured.err == ""
    assert lines
    assert not project_root.exists()
    assert lines == sorted(
        lines, key=lambda line: (("READ", "WRITE", "RECOVER").index(line.split()[0]), line)
    )
    assert set(expected_lines) <= set(lines)
    for line in lines:
        action, relative_path = line.split(" ", 1)
        assert action in {"READ", "WRITE", "RECOVER"}
        assert relative_path
        assert "\\" not in relative_path
        assert not relative_path.startswith("/")
        assert re.match(r"^[A-Za-z]:", relative_path) is None
        assert str(project_root.resolve()).replace("\\", "/").casefold() not in line.casefold()


def test_pair_dry_run_does_not_claim_cached_analysis_lock(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--project-root", str(tmp_path), "pair", "--dry-run"]) == 0
    assert (
        "WRITE local-data/derived/corpus-v1/normalized/analysis-*.wav.lock"
        not in capsys.readouterr().out.splitlines()
    )
