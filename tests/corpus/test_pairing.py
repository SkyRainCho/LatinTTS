import hashlib
import json
import shutil
import wave
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from latintts.corpus.alignment import (
    AlignmentRequest,
    AlignmentResult,
    AlignmentToken,
    WordSpan,
)
from latintts.corpus.audio import DerivedAudio, derive_analysis_audio
from latintts.corpus.config import CorpusConfig
from latintts.corpus.domain import CorpusFailure
from latintts.corpus.pairing import (
    PairingParameters,
    SplitEvidence,
    TakeCandidate,
    TextUnitWindow,
    choose_split,
    map_text_units,
    pair_recording,
    pairing_from_dict,
    pairing_run_directory,
    pairing_to_dict,
)
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.pauses import PauseAnalysis, PauseInterval
from latintts.corpus.store import read_jsonl, write_jsonl_atomic
from latintts.corpus.transcripts import SpokenUnit
from latintts.corpus.vad import SpeechInterval
from tests.corpus.factories import recording


def test_choose_split_selects_unique_two_take_candidate() -> None:
    unit = TextUnitWindow("unit-1", "Pater noster", 0, 10000, ((4500, 5000),))
    evidence = (SplitEvidence(4750, 0.92, 0.90, True, 0.95),)
    group = choose_split(
        "rec-1",
        unit,
        evidence,
        sample_rate=1000,
        minimum_duration_ratio=0.65,
        maximum_duration_ratio=1.35,
        minimum_score_margin=0.02,
    )
    assert group.repetition_group_id == "rec-1-unit-1"
    assert [take.take_index for take in group.takes] == [1, 2]
    assert group.takes[0].end_sample == 4750
    assert group.takes[1].start_sample == 4750


def test_choose_split_rejects_text_mismatch() -> None:
    unit = TextUnitWindow("unit-1", "Pater noster", 0, 10000, ((4500, 5000),))
    with pytest.raises(CorpusFailure) as error:
        choose_split(
            "rec-1",
            unit,
            (SplitEvidence(4750, 0.9, 0.9, False, 0.95),),
            sample_rate=1000,
            minimum_duration_ratio=0.65,
            maximum_duration_ratio=1.35,
            minimum_score_margin=0.02,
        )
    assert error.value.code == "TAKE_TEXT_MISMATCH"


def test_choose_split_classifies_missing_candidate_as_take_count_mismatch() -> None:
    unit = TextUnitWindow("unit-1", "Pater noster", 0, 10000, ())
    with pytest.raises(CorpusFailure) as error:
        choose_split(
            "rec-1",
            unit,
            (),
            sample_rate=1000,
            minimum_duration_ratio=0.65,
            maximum_duration_ratio=1.35,
            minimum_score_margin=0.02,
        )
    assert error.value.code == "TAKE_COUNT_MISMATCH"


def _units() -> tuple[SpokenUnit, ...]:
    return (
        SpokenUnit("unit-1", 1, "Pater noster", 0, 2),
        SpokenUnit("unit-2", 2, "qui es", 2, 4),
    )


def _speech() -> tuple[SpeechInterval, ...]:
    return (
        SpeechInterval(100, 1_000),
        SpeechInterval(1_200, 2_000),
        SpeechInterval(2_800, 3_700),
        SpeechInterval(3_900, 5_000),
    )


def _pauses() -> PauseAnalysis:
    return PauseAnalysis(
        (
            PauseInterval(1_000, 1_200, 0.2, "short"),
            PauseInterval(2_000, 2_800, 0.8, "long"),
            PauseInterval(3_700, 3_900, 0.2, "short"),
        ),
        0.4,
        0.2,
        0.8,
    )


def test_map_text_units_uses_long_pause_midpoints_and_preserves_short_pauses() -> None:
    windows = map_text_units(_units(), _speech(), _pauses())
    assert windows == (
        TextUnitWindow("unit-1", "Pater noster", 100, 2_400, ((1_000, 1_200),), 0, 2),
        TextUnitWindow("unit-2", "qui es", 2_400, 5_000, ((3_700, 3_900),), 2, 4),
    )


def test_map_text_units_rejects_long_pause_count_mismatch() -> None:
    pauses = PauseAnalysis(_pauses().pauses[:1], 0.4, 0.2, 0.8)
    with pytest.raises(CorpusFailure) as error:
        map_text_units(_units(), _speech(), pauses)
    assert error.value.code == "TRANSCRIPT_SPOKEN_MISMATCH"


def test_map_text_units_rejects_missing_take_pause() -> None:
    pauses = PauseAnalysis(
        (
            PauseInterval(1_000, 1_200, 0.2, "short"),
            PauseInterval(2_000, 2_800, 0.8, "long"),
        ),
        0.4,
        0.2,
        0.8,
    )
    with pytest.raises(CorpusFailure) as error:
        map_text_units(_units(), _speech(), pauses)
    assert error.value.code == "TAKE_COUNT_MISMATCH"


@pytest.mark.parametrize(
    "units",
    [
        (
            SpokenUnit("unit-1", 1, "Pater noster", 0, 2),
            SpokenUnit("unit-2", 2, "qui es", 3, 5),
        ),
        (
            SpokenUnit("unit-1", 1, "Pater noster", 0, 2),
            SpokenUnit("unit-2", 2, "qui es", 1, 3),
        ),
    ],
)
def test_map_text_units_rejects_incomplete_or_overlapping_token_ranges(
    units: tuple[SpokenUnit, ...],
) -> None:
    with pytest.raises(CorpusFailure) as error:
        map_text_units(units, _speech(), _pauses())
    assert error.value.code == "TRANSCRIPT_SPOKEN_MISMATCH"


def test_choose_split_rejects_duration_mismatch() -> None:
    unit = TextUnitWindow("unit-1", "Pater noster", 0, 10_000, ((1_500, 2_000),))
    with pytest.raises(CorpusFailure) as error:
        choose_split(
            "rec-1",
            unit,
            (SplitEvidence(1_750, 0.95, 0.95, True, 1.0),),
            sample_rate=1_000,
            minimum_duration_ratio=0.65,
            maximum_duration_ratio=1.35,
            minimum_score_margin=0.02,
        )
    assert error.value.code == "TAKE_DURATION_MISMATCH"


def test_choose_split_rejects_ambiguous_multiple_candidates() -> None:
    unit = TextUnitWindow("unit-1", "Pater noster", 0, 10_000, ((4_400, 4_600), (5_400, 5_600)))
    with pytest.raises(CorpusFailure) as error:
        choose_split(
            "rec-1",
            unit,
            (
                SplitEvidence(4_500, 0.91, 0.90, True, 1.0),
                SplitEvidence(5_500, 0.90, 0.90, True, 1.0),
            ),
            sample_rate=1_000,
            minimum_duration_ratio=0.65,
            maximum_duration_ratio=1.35,
            minimum_score_margin=0.02,
        )
    assert error.value.code == "TAKE_COUNT_MISMATCH"


def test_choose_split_uses_coverage_then_rejects_exact_tie_deterministically() -> None:
    unit = TextUnitWindow("unit-1", "Pater noster", 0, 10_000, ((4_400, 4_600), (5_400, 5_600)))
    tied = (
        SplitEvidence(5_500, 0.9, 0.9, True, 0.98),
        SplitEvidence(4_500, 0.9, 0.9, True, 0.98),
    )
    for evidence in (tied, tuple(reversed(tied))):
        with pytest.raises(CorpusFailure) as error:
            choose_split(
                "rec-1",
                unit,
                evidence,
                sample_rate=1_000,
                minimum_duration_ratio=0.65,
                maximum_duration_ratio=1.35,
                minimum_score_margin=0.0,
            )
        assert error.value.code == "TAKE_COUNT_MISMATCH"


def test_choose_split_rejects_evidence_outside_short_pause() -> None:
    unit = TextUnitWindow("unit-1", "Pater noster", 0, 10_000, ((4_500, 5_000),))
    with pytest.raises(ValueError, match="short-pause midpoint"):
        choose_split(
            "rec-1",
            unit,
            (SplitEvidence(4_999, 0.95, 0.95, True, 1.0),),
            sample_rate=1_000,
            minimum_duration_ratio=0.65,
            maximum_duration_ratio=1.35,
            minimum_score_margin=0.02,
        )


def test_choose_split_rejects_impostor_unit_and_mutable_evidence() -> None:
    with pytest.raises(TypeError, match="unit"):
        choose_split(
            "rec-1",
            object(),  # type: ignore[arg-type]
            (),
            sample_rate=1_000,
            minimum_duration_ratio=0.65,
            maximum_duration_ratio=1.35,
            minimum_score_margin=0.02,
        )
    with pytest.raises(TypeError, match="evidence"):
        choose_split(
            "rec-1",
            TextUnitWindow("unit-1", "Pater", 0, 10_000, ((4_000, 6_000),)),
            [],  # type: ignore[arg-type]
            sample_rate=1_000,
            minimum_duration_ratio=0.65,
            maximum_duration_ratio=1.35,
            minimum_score_margin=0.02,
        )


class _FakeAligner:
    def __init__(self, *, mismatched: bool = False) -> None:
        self.calls = 0
        self.mismatched = mismatched

    def build_request(
        self,
        *,
        audio_path: Path,
        audio_sha256: str,
        spoken_text: str,
        segmentation_artifact_sha256: str,
        config_sha256: str,
    ) -> AlignmentRequest:
        return AlignmentRequest(
            audio_path=audio_path,
            audio_sha256=audio_sha256,
            spoken_text=spoken_text,
            segmentation_artifact_sha256=segmentation_artifact_sha256,
            backend="fake-aligner",
            backend_version="1.0",
            model_id="fake/model",
            model_revision="a" * 40,
            model_license="test-only",
            config_sha256=config_sha256,
            effective_parameters={"fixture": "pairing-v1"},
        )

    def align(self, request: AlignmentRequest) -> AlignmentResult:
        self.calls += 1
        forms = request.alignment_text.split()
        surfaces = request.spoken_text.split()
        indexes = (
            tuple(reversed(range(len(forms)))) if self.mismatched else tuple(range(len(forms)))
        )
        words = tuple(
            WordSpan(form, index * 0.1, (index + 1) * 0.1, 0.9, spoken_index)
            for index, (form, spoken_index) in enumerate(zip(forms, indexes, strict=True))
        )
        return AlignmentResult(
            backend=request.backend,
            backend_version=request.backend_version,
            model_id=request.model_id,
            model_revision=request.model_revision,
            model_license=request.model_license,
            alignment_text=request.alignment_text,
            tokens=tuple(
                AlignmentToken(index, form, index, surfaces[index])
                for index, form in enumerate(forms)
            ),
            words=words,
            coverage=1.0,
            mean_score=0.9,
            alignment_level="word",
            phoneme_timing_status="not_estimated",
            warnings=(),
            raw_output={"fixture": True},
        )


def _audio_command(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
    source = Path(command[command.index("-i") + 1])
    destination = Path(command[-1])
    if "-ss" not in command:
        shutil.copyfile(source, destination)
        return CompletedProcess(command, 0, "", "")
    start = float(command[command.index("-ss") + 1])
    end = float(command[command.index("-to") + 1])
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        reader.setpos(round(start * rate))
        frames = reader.readframes(round((end - start) * rate))
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(frames)
    return CompletedProcess(command, 0, "", "")


def _analysis_fixture(tmp_path: Path) -> tuple[CorpusPaths, CorpusConfig, object]:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    config_path = Path(__file__).parents[2] / "config" / "corpus" / "pilot-v1.json"
    config = CorpusConfig.load(config_path)
    source = paths.raw_spoken / "rec-1.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x01\x00" * 80_000 + b"\x02\x00" * 80_000)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = replace(
        recording("rec-1", 10.0),
        relative_path="raw/spoken/rec-1.wav",
        sha256=digest,
    )
    analysis = derive_analysis_audio(
        record,
        paths,
        config,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    return paths, config, analysis


def test_pair_recording_extracts_both_takes_caches_alignment_and_reuses_result(
    tmp_path: Path,
) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    aligner = _FakeAligner()
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    result = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        aligner,
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    assert result.groups[0].status == "selected"
    assert result.groups[0].group is not None
    assert [take.take_index for take in result.groups[0].group.takes] == [1, 2]
    assert all(take.audio_sha256 for take in result.groups[0].group.takes)
    assert aligner.calls == 2
    cache_dir = paths.alignments / "runs" / config.digest / "rec-1" / "alignment-cache"
    assert len(tuple(cache_dir.glob("*.json"))) == 2

    cached = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        aligner,
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("valid pairing cache must not re-extract")
        ),
    )
    assert cached == result
    assert aligner.calls == 2


def test_pair_recording_preserves_text_mismatch_evidence_for_review(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    result = pair_recording(
        "rec-1",
        (TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2),),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(mismatched=True),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    outcome = result.groups[0]
    assert outcome.status == "review"
    assert outcome.issue_code == "TAKE_TEXT_MISMATCH"
    assert outcome.group is None
    assert len(outcome.candidates) == 1
    assert outcome.candidates[0].word_order_same is False


def test_pair_recording_preserves_word_text_drift_as_text_mismatch_review(tmp_path: Path) -> None:
    class WordDriftAligner(_FakeAligner):
        def align(self, request: AlignmentRequest) -> AlignmentResult:
            result = super().align(request)
            words = list(result.words)
            first = words[0]
            words[0] = WordSpan(
                "wrong",
                first.start_seconds,
                first.end_seconds,
                first.score,
                first.spoken_token_index,
            )
            return AlignmentResult(
                backend=result.backend,
                backend_version=result.backend_version,
                model_id=result.model_id,
                model_revision=result.model_revision,
                model_license=result.model_license,
                alignment_text=result.alignment_text,
                tokens=result.tokens,
                words=tuple(words),
                coverage=result.coverage,
                mean_score=result.mean_score,
                alignment_level=result.alignment_level,
                phoneme_timing_status=result.phoneme_timing_status,
                warnings=result.warnings,
                raw_output={"fixture": "word-drift"},
            )

    paths, config, analysis = _analysis_fixture(tmp_path)
    result = pair_recording(
        "rec-1",
        (TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2),),
        analysis,  # type: ignore[arg-type]
        paths,
        WordDriftAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    outcome = result.groups[0]
    assert outcome.status == "review"
    assert outcome.issue_code == "TAKE_TEXT_MISMATCH"
    assert len(outcome.candidates) == 1
    assert outcome.candidates[0].word_order_same is False
    cached = pair_recording(
        "rec-1",
        (TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2),),
        analysis,  # type: ignore[arg-type]
        paths,
        WordDriftAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    assert cached == result


def test_pair_recording_preserves_alignment_text_drift_as_text_mismatch_review(
    tmp_path: Path,
) -> None:
    class AlignmentTextDriftAligner(_FakeAligner):
        def align(self, request: AlignmentRequest) -> AlignmentResult:
            result = super().align(request)
            return AlignmentResult(
                backend=result.backend,
                backend_version=result.backend_version,
                model_id=result.model_id,
                model_revision=result.model_revision,
                model_license=result.model_license,
                alignment_text=f"wrong {result.alignment_text.split(maxsplit=1)[1]}",
                tokens=result.tokens,
                words=result.words,
                coverage=result.coverage,
                mean_score=result.mean_score,
                alignment_level=result.alignment_level,
                phoneme_timing_status=result.phoneme_timing_status,
                warnings=result.warnings,
                raw_output={"fixture": "alignment-text-drift"},
            )

    paths, config, analysis = _analysis_fixture(tmp_path)
    result = pair_recording(
        "rec-1",
        (TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2),),
        analysis,  # type: ignore[arg-type]
        paths,
        AlignmentTextDriftAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    assert result.groups[0].status == "review"
    assert result.groups[0].issue_code == "TAKE_TEXT_MISMATCH"


def test_pair_recording_preserves_missing_candidate_for_review(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    result = pair_recording(
        "rec-1",
        (TextUnitWindow("unit-1", "Pater noster", 0, 160_000, (), 0, 2),),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    outcome = result.groups[0]
    assert outcome.status == "review"
    assert outcome.issue_code == "TAKE_COUNT_MISMATCH"
    assert outcome.candidates == ()


@pytest.mark.parametrize(
    "factory",
    (
        lambda: TextUnitWindow("", "text", 0, 10, ()),
        lambda: TextUnitWindow("unit", "", 0, 10, ()),
        lambda: TextUnitWindow("unit", "text", -1, 10, ()),
        lambda: TextUnitWindow("unit", "text", 10, 10, ()),
        lambda: TextUnitWindow("unit", "text", 0, 10, (), 2, 1),
        lambda: TextUnitWindow("unit", "text", 0, 10, []),  # type: ignore[arg-type]
        lambda: TextUnitWindow("unit", "text", 0, 10, (("1", 2),)),  # type: ignore[arg-type]
        lambda: TextUnitWindow("unit", "text", 0, 10, ((8, 11),)),
        lambda: SplitEvidence(-1, 0.9, 0.9, True, 1.0),
        lambda: SplitEvidence(1, float("nan"), 0.9, True, 1.0),
        lambda: SplitEvidence(1, 1.1, 0.9, True, 1.0),
        lambda: SplitEvidence(1, 0.9, 0.9, 1, 1.0),  # type: ignore[arg-type]
        lambda: TakeCandidate(3, 0, 1),
        lambda: TakeCandidate(1, 1, 1),
        lambda: TakeCandidate(1, 0, 1, "path"),
        lambda: TakeCandidate(1, 0, 1, alignment_score=1.1),
        lambda: PairingParameters(-1, 1, 0.1),
        lambda: PairingParameters(2, 1, 0.1),
    ),
)
def test_pairing_domain_rejects_noncanonical_values(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("units", "speech", "pauses", "error_type"),
    (
        ((), _speech(), _pauses(), CorpusFailure),
        ((object(),), _speech(), _pauses(), TypeError),
        (_units(), (), _pauses(), CorpusFailure),
        (_units(), (object(),), _pauses(), TypeError),
        (
            _units(),
            (SpeechInterval(100, 1_000), SpeechInterval(900, 2_000)),
            _pauses(),
            ValueError,
        ),
        (
            _units(),
            _speech(),
            PauseAnalysis((object(),), 0.4, 0.2, 0.8),  # type: ignore[arg-type]
            TypeError,
        ),
        (
            _units(),
            _speech(),
            PauseAnalysis((PauseInterval(101, 102, 0.1, "short"),), 0.4, 0.2, 0.8),
            ValueError,
        ),
    ),
)
def test_map_text_units_rejects_invalid_evidence_contracts(
    units: object, speech: object, pauses: object, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        map_text_units(units, speech, pauses)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    (
        {"recording_id": ""},
        {"sample_rate": 0},
        {"minimum_duration_ratio": -1.0},
        {"minimum_duration_ratio": 2.0, "maximum_duration_ratio": 1.0},
    ),
)
def test_choose_split_rejects_invalid_parameters(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "recording_id": "rec-1",
        "sample_rate": 1_000,
        "minimum_duration_ratio": 0.65,
        "maximum_duration_ratio": 1.35,
        "minimum_score_margin": 0.02,
    }
    values.update(changes)
    with pytest.raises(ValueError):
        choose_split(
            values["recording_id"],  # type: ignore[arg-type]
            TextUnitWindow("unit-1", "Pater", 0, 10_000, ((4_000, 6_000),)),
            (SplitEvidence(5_000, 0.9, 0.9, True, 1.0),),
            sample_rate=values["sample_rate"],  # type: ignore[arg-type]
            minimum_duration_ratio=values["minimum_duration_ratio"],  # type: ignore[arg-type]
            maximum_duration_ratio=values["maximum_duration_ratio"],  # type: ignore[arg-type]
            minimum_score_margin=values["minimum_score_margin"],  # type: ignore[arg-type]
        )


def test_pairing_json_round_trip_is_strict_and_cache_key_is_bound(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    result = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    raw = pairing_to_dict(result)
    assert len(raw["integrity_sha256"]) == 64
    assert pairing_from_dict(raw) == result
    malformed = deepcopy(raw)
    malformed["integrity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="integrity"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["status"] = "approved"
    with pytest.raises(ValueError, match="status"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["group"]["takes"] = []
    with pytest.raises(ValueError, match="exactly two"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["windows"][0]["short_pause_ranges"] = "invalid"
    with pytest.raises(TypeError, match="short_pause_ranges"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["candidates"][0]["first_audio"] = None
    with pytest.raises(TypeError, match="audio"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["windows"] = "invalid"
    with pytest.raises(TypeError, match="arrays"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["candidates"] = "invalid"
    with pytest.raises(TypeError, match="candidates"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["status"] = "review"
    malformed["groups"][0]["issue_code"] = "TAKE_COUNT_MISMATCH"
    with pytest.raises(ValueError, match="disagree"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["schema_version"] = "2"
    with pytest.raises(ValueError, match="schema_version"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["unit_id"] = "wrong-unit"
    with pytest.raises(ValueError, match="cover windows"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["config_sha256"] = "invalid"
    with pytest.raises(ValueError, match="SHA-256"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["cache_key"] = "0" * 64
    with pytest.raises(ValueError, match="cache identity"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["group"]["takes"][0]["audio_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="take evidence"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["group"]["selected_evidence"]["split_sample"] += 1
    with pytest.raises(ValueError, match="selected evidence"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["candidates"] = []
    with pytest.raises(ValueError, match="short-pause"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["group"]["repetition_group_id"] = "wrong-group"
    with pytest.raises(ValueError, match="group identity"):
        pairing_from_dict(malformed)
    malformed = deepcopy(raw)
    malformed["groups"][0]["status"] = "review"
    malformed["groups"][0]["issue_code"] = "REVIEW_REQUIRED"
    malformed["groups"][0]["group"] = None
    with pytest.raises(ValueError, match="issue code"):
        pairing_from_dict(malformed)

    path = paths.alignments / "runs" / config.digest / "rec-1" / "pairing.json"
    malformed = deepcopy(raw)
    malformed["cache_key"] = "0" * 64
    write_jsonl_atomic(path, (malformed,))
    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_pair_recording_rejects_locked_run_and_tampered_candidate(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    run_directory = paths.alignments / "runs" / config.digest / "rec-1"
    run_directory.mkdir(parents=True)
    lock = run_directory / ".pairing.lock"
    lock.write_text("held", encoding="utf-8")
    with pytest.raises(CorpusFailure, match="locked"):
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    lock.unlink()
    result = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    audio = result.groups[0].candidates[0].first_audio
    assert audio is not None
    audio_path = paths.resolve_local(audio.relative_path)
    original_bytes = audio_path.read_bytes()
    audio_path.unlink()
    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"
    audio_path.write_bytes(original_bytes)
    audio_path.write_bytes(b"tampered")
    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_pair_recording_rejects_backend_contract_and_provenance(tmp_path: Path) -> None:
    class NonResultAligner(_FakeAligner):
        def align(self, request: AlignmentRequest) -> AlignmentResult:
            self.calls += 1
            return object()  # type: ignore[return-value]

    class WrongProvenanceAligner(_FakeAligner):
        def align(self, request: AlignmentRequest) -> AlignmentResult:
            result = super().align(request)
            return AlignmentResult(
                backend="wrong-backend",
                backend_version=result.backend_version,
                model_id=result.model_id,
                model_revision=result.model_revision,
                model_license=result.model_license,
                alignment_text=result.alignment_text,
                tokens=result.tokens,
                words=result.words,
                coverage=result.coverage,
                mean_score=result.mean_score,
                alignment_level=result.alignment_level,
                phoneme_timing_status=result.phoneme_timing_status,
                warnings=result.warnings,
                raw_output={"fixture": True},
            )

    for name, backend, error_type in (
        ("non-result", NonResultAligner(), TypeError),
        ("wrong-provenance", WrongProvenanceAligner(), ValueError),
    ):
        paths, config, analysis = _analysis_fixture(tmp_path / name)
        with pytest.raises(error_type):
            pair_recording(
                "rec-1",
                (
                    TextUnitWindow(
                        "unit-1",
                        "Pater noster",
                        0,
                        160_000,
                        ((78_000, 82_000),),
                        0,
                        2,
                    ),
                ),
                analysis,  # type: ignore[arg-type]
                paths,
                backend,
                segmentation_artifact_sha256="b" * 64,
                config_sha256=config.digest,
                pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
                ffmpeg_version="ffmpeg-test-1",
                run_command=_audio_command,
            )


@pytest.mark.parametrize(
    "change",
    (
        {"recording_id": ""},
        {"windows": ()},
        {"analysis_audio": object()},
        {"paths": object()},
        {"pairing_parameters": object()},
        {"segmentation_artifact_sha256": "bad"},
        {"config_sha256": "0" * 64},
    ),
)
def test_pair_recording_rejects_invalid_entry_contract(
    tmp_path: Path, change: dict[str, object]
) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    values: dict[str, object] = {
        "recording_id": "rec-1",
        "windows": (
            TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2),
        ),
        "analysis_audio": analysis,
        "paths": paths,
        "pairing_parameters": PairingParameters(0.65, 1.35, 0.02),
        "segmentation_artifact_sha256": "b" * 64,
        "config_sha256": config.digest,
    }
    values.update(change)
    with pytest.raises((TypeError, ValueError)):
        pair_recording(
            values["recording_id"],  # type: ignore[arg-type]
            values["windows"],  # type: ignore[arg-type]
            values["analysis_audio"],  # type: ignore[arg-type]
            values["paths"],  # type: ignore[arg-type]
            _FakeAligner(),
            segmentation_artifact_sha256=values[  # type: ignore[arg-type]
                "segmentation_artifact_sha256"
            ],
            config_sha256=values["config_sha256"],  # type: ignore[arg-type]
            pairing_parameters=values["pairing_parameters"],  # type: ignore[arg-type]
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )


@pytest.mark.parametrize("unsafe_id", ("../escape", "unit/child", ".", "..", "C:\\root"))
def test_pair_recording_rejects_unsafe_recording_and_unit_ids(
    tmp_path: Path, unsafe_id: str
) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window_id = "unit-1" if unsafe_id == "../escape" else unsafe_id
    recording_id = unsafe_id if unsafe_id == "../escape" else "rec-1"
    with pytest.raises(ValueError, match="safe path component"):
        pair_recording(
            recording_id,
            (TextUnitWindow(window_id, "Pater", 0, 160_000, ((78_000, 82_000),), 0, 1),),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )


def test_pair_recording_binds_ffmpeg_version_to_artifact_and_candidate_cache(
    tmp_path: Path,
) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    first = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    assert first.ffmpeg_version == "ffmpeg-test-1"
    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-2",
            run_command=_audio_command,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


@pytest.mark.parametrize("tamper", ("status", "take", "word-order", "evidence"))
def test_pair_recording_recomputes_cached_decision_despite_valid_artifact_integrity(
    tmp_path: Path, tamper: str
) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    result = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    raw = pairing_to_dict(result)
    group = raw["groups"][0]
    if tamper == "status":
        group["status"] = "review"
        group["issue_code"] = "TAKE_DURATION_MISMATCH"
        group["group"] = None
    elif tamper == "take":
        group["group"]["takes"][0]["end_sample"] -= 1
    elif tamper == "word-order":
        group["candidates"][0]["word_order_same"] = False
        group["status"] = "review"
        group["issue_code"] = "TAKE_TEXT_MISMATCH"
        group["group"] = None
    else:
        group["candidates"][0]["first_alignment_score"] = 0.1
        group["group"]["selected_evidence"]["first_alignment_score"] = 0.1
        group["group"]["takes"][0]["alignment_score"] = 0.1
    payload = {key: value for key, value in raw.items() if key != "integrity_sha256"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raw["integrity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    pairing_path = paths.alignments / "runs" / config.digest / "rec-1" / "pairing.json"
    write_jsonl_atomic(pairing_path, (raw,))

    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def _resign_candidate(raw: dict[str, object], **changes: object) -> dict[str, object]:
    resigned = deepcopy(raw)
    resigned.update(changes)
    identity = {
        "schema_version": "1",
        "mode": resigned["mode"],
        "source_relative_path": resigned["source_relative_path"],
        "source_sha256": resigned["source_sha256"],
        "config_sha256": resigned["config_sha256"],
        "output_config_sha256": resigned["output_config_sha256"],
        "ffmpeg_version": resigned["ffmpeg_version"],
        "start_seconds": float(resigned["start_seconds"]).hex(),
        "end_seconds": float(resigned["end_seconds"]).hex(),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    resigned["cache_key"] = cache_key
    resigned["relative_path"] = f"derived/corpus-v1/segments/candidates/candidate-{cache_key}.wav"
    return resigned


def test_pair_recording_rejects_candidate_swapped_from_another_window(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    source_window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    source = pair_recording(
        "rec-1",
        (source_window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    evidence = source.groups[0].candidates[0]
    extracted = iter((evidence.first_audio, evidence.second_audio))

    with pytest.raises(ValueError, match=r"boundary|sample"):
        pair_recording(
            "rec-2",
            (TextUnitWindow("unit-2", "Pater noster", 8_000, 160_000, ((78_000, 82_000),), 0, 2),),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
            extract_candidate=lambda *_args, **_kwargs: next(extracted),  # type: ignore[arg-type]
        )


def test_pair_recording_rejects_candidate_pcm_count_drift(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    source = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    evidence = source.groups[0].candidates[0]
    audio_items: list[DerivedAudio] = []
    for audio in (evidence.first_audio, evidence.second_audio):
        assert audio is not None and audio.metrics is not None
        audio_items.append(
            replace(
                audio, metrics=replace(audio.metrics, sample_count=audio.metrics.sample_count + 2)
            )
        )
    extracted = iter(audio_items)
    with pytest.raises(ValueError, match="sample_count"):
        pair_recording(
            "rec-2",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
            extract_candidate=lambda *_args, **_kwargs: next(extracted),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("mismatch", ("path", "hash"))
def test_pair_recording_rejects_alignment_request_candidate_mismatch(
    tmp_path: Path, mismatch: str
) -> None:
    class MismatchedRequestAligner(_FakeAligner):
        def build_request(self, **kwargs: object) -> AlignmentRequest:
            request = super().build_request(**kwargs)  # type: ignore[arg-type]
            return AlignmentRequest(
                audio_path=(
                    request.audio_path.with_name("wrong.wav")
                    if mismatch == "path"
                    else request.audio_path
                ),
                audio_sha256="f" * 64 if mismatch == "hash" else request.audio_sha256,
                spoken_text=request.spoken_text,
                segmentation_artifact_sha256=request.segmentation_artifact_sha256,
                backend=request.backend,
                backend_version=request.backend_version,
                model_id=request.model_id,
                model_revision=request.model_revision,
                model_license=request.model_license,
                config_sha256=request.config_sha256,
                effective_parameters=dict(request.effective_parameters),
                alignment_transform_version=request.alignment_transform_version,
            )

    paths, config, analysis = _analysis_fixture(tmp_path)
    with pytest.raises(ValueError, match=r"request.*audio"):
        pair_recording(
            "rec-1",
            (TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2),),
            analysis,  # type: ignore[arg-type]
            paths,
            MismatchedRequestAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )


def test_pair_recording_rejects_resigned_cached_candidate_source(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    result = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    raw = pairing_to_dict(result)
    original_audio = raw["groups"][0]["candidates"][0]["first_audio"]
    resigned = _resign_candidate(original_audio, source_sha256="f" * 64)
    original_path = paths.resolve_local(original_audio["relative_path"])
    resigned_path = paths.resolve_local(resigned["relative_path"])
    resigned_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original_path, resigned_path)
    raw["groups"][0]["candidates"][0]["first_audio"] = deepcopy(resigned)
    raw["groups"][0]["group"]["selected_evidence"]["first_audio"] = deepcopy(resigned)
    raw["groups"][0]["group"]["takes"][0]["audio_relative_path"] = resigned["relative_path"]
    payload = {key: value for key, value in raw.items() if key != "integrity_sha256"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raw["integrity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    write_jsonl_atomic(paths.alignments / "runs" / config.digest / "rec-1" / "pairing.json", (raw,))
    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_pair_recording_rejects_resigned_cached_candidate_boundary(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    result = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    raw = pairing_to_dict(result)
    original_audio = raw["groups"][0]["candidates"][0]["first_audio"]
    resigned = _resign_candidate(original_audio, start_seconds=0.5, end_seconds=5.5)
    original_path = paths.resolve_local(original_audio["relative_path"])
    resigned_path = paths.resolve_local(resigned["relative_path"])
    resigned_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original_path, resigned_path)
    raw["groups"][0]["candidates"][0]["first_audio"] = deepcopy(resigned)
    raw["groups"][0]["group"]["selected_evidence"]["first_audio"] = deepcopy(resigned)
    raw["groups"][0]["group"]["takes"][0]["audio_relative_path"] = resigned["relative_path"]
    payload = {key: value for key, value in raw.items() if key != "integrity_sha256"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raw["integrity_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    write_jsonl_atomic(paths.alignments / "runs" / config.digest / "rec-1" / "pairing.json", (raw,))
    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_pair_recording_rejects_resigned_candidate_output_format(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    source = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    evidence = source.groups[0].candidates[0]
    resigned_items: list[DerivedAudio] = []
    for audio in (evidence.first_audio, evidence.second_audio):
        assert audio is not None
        resigned_raw = _resign_candidate(audio.to_dict(), output_config_sha256="f" * 64)
        resigned = DerivedAudio.from_dict(resigned_raw)
        original_path = paths.resolve_local(audio.relative_path)
        resigned_path = paths.resolve_local(resigned.relative_path)
        resigned_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_path, resigned_path)
        resigned_items.append(resigned)
    extracted = iter(resigned_items)
    with pytest.raises(ValueError, match=r"format|PCM16"):
        pair_recording(
            "rec-2",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
            extract_candidate=lambda *_args, **_kwargs: next(extracted),  # type: ignore[arg-type]
        )


def test_pairing_run_directory_rejects_constructed_layout_field_drift(tmp_path: Path) -> None:
    paths, config, _ = _analysis_fixture(tmp_path)
    drifted = replace(paths, local_data=tmp_path / "other-local-data")
    with pytest.raises(ValueError, match="fixed corpus root"):
        pairing_run_directory(drifted, config.digest, "rec-1")


def test_pairing_run_directory_rejects_ancestor_alias(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside-local-data"
    outside.mkdir()
    try:
        (project / "local-data").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this Windows host")
    paths = CorpusPaths.from_project_root(project)
    with pytest.raises(ValueError, match="alias"):
        pairing_run_directory(paths, "a" * 64, "rec-1")


def test_pair_recording_rejects_aliased_config_run_root(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    runs = paths.alignments / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-runs"
    outside.mkdir()
    alias = runs / config.digest
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this Windows host")

    with pytest.raises(ValueError, match="alias"):
        pair_recording(
            "rec-1",
            (TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2),),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert not (outside / "rec-1" / "pairing.json").exists()


def test_pairing_artifact_and_run_directory_defensive_contracts(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    result = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    runtime = result.alignment_runtime_sha256s[0]
    with pytest.raises(ValueError, match="ffmpeg_version"):
        replace(result, ffmpeg_version="")
    with pytest.raises(ValueError, match="runtime"):
        replace(result, alignment_runtime_sha256s=("bad",))
    with pytest.raises(ValueError, match="unique"):
        replace(result, alignment_runtime_sha256s=(runtime, runtime))
    with pytest.raises(TypeError, match="PairingRecording"):
        pairing_to_dict(object())  # type: ignore[arg-type]
    object.__setattr__(result, "integrity_sha256", "0" * 64)
    with pytest.raises(ValueError, match="integrity"):
        pairing_to_dict(result)
    with pytest.raises(TypeError, match="CorpusPaths"):
        pairing_run_directory(object(), config.digest, "rec-1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="config_sha256"):
        pairing_run_directory(paths, "bad", "rec-1")
    with pytest.raises(ValueError, match="ffmpeg_version"):
        pair_recording(
            "rec-2",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="",
            run_command=_audio_command,
        )
    with pytest.raises(ValueError, match="unique"):
        pair_recording(
            "rec-2",
            (window, window),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )


def test_map_text_units_rejects_duplicate_ids_and_invalid_pause_analysis() -> None:
    duplicate_units = (
        SpokenUnit("unit-1", 1, "Pater", 0, 1),
        SpokenUnit("unit-1", 2, "noster", 1, 2),
    )
    speech = (SpeechInterval(0, 100), SpeechInterval(200, 300))
    pauses = PauseAnalysis((PauseInterval(100, 200, 0.00625, "long"),), 0.1, 0.05, 0.2)
    with pytest.raises(CorpusFailure, match="unique"):
        map_text_units(duplicate_units, speech, pauses)
    with pytest.raises(TypeError, match="PauseAnalysis"):
        map_text_units((SpokenUnit("unit-1", 1, "Pater", 0, 1),), speech, object())  # type: ignore[arg-type]


def test_pair_recording_rejects_duplicate_cache_rows(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    result = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    path = paths.alignments / "runs" / config.digest / "rec-1" / "pairing.json"
    row = pairing_to_dict(result)
    write_jsonl_atomic(path, (row, row))
    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_pair_recording_rejects_duplicate_alignment_cache_rows(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    result = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        _FakeAligner(),
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    key = result.groups[0].group.takes[0].alignment_cache_key  # type: ignore[union-attr]
    path = paths.alignments / "runs" / config.digest / "rec-1" / "alignment-cache" / f"{key}.json"
    row = read_jsonl(path)[0]
    write_jsonl_atomic(path, (row, row))
    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"


def test_pair_recording_rejects_invalid_alignment_cache_schema_before_backend_reuse(
    tmp_path: Path,
) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    window = TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 2)
    backend = _FakeAligner()
    result = pair_recording(
        "rec-1",
        (window,),
        analysis,  # type: ignore[arg-type]
        paths,
        backend,
        segmentation_artifact_sha256="b" * 64,
        config_sha256=config.digest,
        pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
        ffmpeg_version="ffmpeg-test-1",
        run_command=_audio_command,
    )
    outcome = result.groups[0]
    assert outcome.group is not None
    evidence = outcome.group.selected_evidence
    assert evidence.first_audio is not None
    assert evidence.second_audio is not None
    cache_path = (
        paths.alignments
        / "runs"
        / config.digest
        / "rec-1"
        / "alignment-cache"
        / f"{outcome.group.takes[0].alignment_cache_key}.json"
    )
    cached = read_jsonl(cache_path)[0]
    cached["unexpected"] = True
    write_jsonl_atomic(cache_path, (cached,))
    (paths.alignments / "runs" / config.digest / "rec-1" / "pairing.json").unlink()
    extracted = iter((evidence.first_audio, evidence.second_audio))

    def reuse_audio(*_args: object, **_kwargs: object) -> object:
        return next(extracted)

    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (window,),
            analysis,  # type: ignore[arg-type]
            paths,
            backend,
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
            extract_candidate=reuse_audio,  # type: ignore[arg-type]
        )
    assert error.value.code == "CACHE_ARTIFACT_INVALID"
    assert backend.calls == 2


def test_pair_recording_rejects_unit_token_count_mismatch(tmp_path: Path) -> None:
    paths, config, analysis = _analysis_fixture(tmp_path)
    with pytest.raises(CorpusFailure) as error:
        pair_recording(
            "rec-1",
            (TextUnitWindow("unit-1", "Pater noster", 0, 160_000, ((78_000, 82_000),), 0, 3),),
            analysis,  # type: ignore[arg-type]
            paths,
            _FakeAligner(),
            segmentation_artifact_sha256="b" * 64,
            config_sha256=config.digest,
            pairing_parameters=PairingParameters(0.65, 1.35, 0.02),
            ffmpeg_version="ffmpeg-test-1",
            run_command=_audio_command,
        )
    assert error.value.code == "TRANSCRIPT_SPOKEN_MISMATCH"
