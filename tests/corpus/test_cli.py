from pathlib import Path

from latintts.corpus.cli import main


def test_doctor_returns_failure_when_ffmpeg_is_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("latintts.corpus.cli.shutil.which", lambda name: None)
    exit_code = main(["--project-root", str(tmp_path), "doctor"])
    assert exit_code == 1
    assert "ALIGNER_UNAVAILABLE: ffmpeg not found in PATH" in capsys.readouterr().out
