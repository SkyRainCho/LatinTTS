from __future__ import annotations

import pytest

from latintts.corpus.domain import CorpusFailure
from latintts.corpus.pauses import classify_pauses
from latintts.corpus.vad import SpeechInterval


def _classify(speech: tuple[SpeechInterval, ...]):
    return classify_pauses(
        speech,
        sample_rate=1000,
        minimum_gap_ms=100,
        minimum_gap_count=4,
        separation_ratio=1.8,
        maximum_iterations=50,
    )


def test_classify_pauses_separates_short_and_long_per_recording() -> None:
    speech = (
        SpeechInterval(0, 1000),
        SpeechInterval(1200, 2000),
        SpeechInterval(4000, 5000),
        SpeechInterval(5250, 6000),
        SpeechInterval(7800, 8500),
    )

    result = _classify(speech)

    assert [pause.kind for pause in result.pauses] == ["short", "long", "short", "long"]
    assert result.long_median_seconds > result.short_median_seconds
    assert [(pause.start_sample, pause.end_sample) for pause in result.pauses] == [
        (1000, 1200),
        (2000, 4000),
        (5000, 5250),
        (6000, 7800),
    ]


def test_classify_pauses_reports_ambiguous_single_peak_distribution() -> None:
    speech = tuple(SpeechInterval(index * 1200, index * 1200 + 1000) for index in range(5))

    with pytest.raises(CorpusFailure) as error:
        _classify(speech)

    assert error.value.code == "PAUSE_CLASSES_AMBIGUOUS"


@pytest.mark.parametrize("speech", ((), (SpeechInterval(0, 1000),)))
def test_classify_pauses_reports_ambiguous_when_no_internal_pauses(
    speech: tuple[SpeechInterval, ...],
) -> None:
    with pytest.raises(CorpusFailure) as error:
        _classify(speech)

    assert error.value.code == "PAUSE_CLASSES_AMBIGUOUS"


def test_classify_pauses_handles_outlier_and_is_deterministic() -> None:
    gap_sizes = (180, 220, 1900, 2100, 20000)
    speech = [SpeechInterval(0, 1000)]
    for gap in gap_sizes:
        start = speech[-1].end_sample + gap
        speech.append(SpeechInterval(start, start + 1000))

    first = _classify(tuple(speech))
    second = _classify(tuple(speech))

    assert first == second
    assert [pause.kind for pause in first.pauses] == ["short", "short", "long", "long", "long"]


def test_classify_pauses_rejects_overlapping_or_unsorted_speech() -> None:
    speech = (
        SpeechInterval(0, 1000),
        SpeechInterval(900, 1100),
        SpeechInterval(1300, 1400),
        SpeechInterval(1600, 1700),
        SpeechInterval(1900, 2000),
        SpeechInterval(2200, 2300),
    )

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        _classify(speech)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"sample_rate": 0}, "sample_rate"),
        ({"minimum_gap_ms": -1}, "minimum_gap_ms"),
        ({"minimum_gap_count": 0}, "minimum_gap_count"),
        ({"separation_ratio": 1.0}, "separation_ratio"),
        ({"maximum_iterations": 0}, "maximum_iterations"),
    ),
)
def test_classify_pauses_validates_parameters(kwargs: dict[str, int | float], message: str) -> None:
    parameters: dict[str, int | float] = {
        "sample_rate": 1000,
        "minimum_gap_ms": 100,
        "minimum_gap_count": 4,
        "separation_ratio": 1.8,
        "maximum_iterations": 50,
    }
    parameters.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message):
        classify_pauses((SpeechInterval(0, 100),), **parameters)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    (
        {"sample_rate": 1.5},
        {"separation_ratio": "wide"},
    ),
)
def test_classify_pauses_rejects_wrong_parameter_types(overrides: dict[str, object]) -> None:
    parameters: dict[str, object] = {
        "sample_rate": 1000,
        "minimum_gap_ms": 100,
        "minimum_gap_count": 4,
        "separation_ratio": 1.8,
        "maximum_iterations": 50,
    }
    parameters.update(overrides)

    with pytest.raises(TypeError):
        classify_pauses((), **parameters)  # type: ignore[arg-type]


def test_classify_pauses_requires_tuple_of_intervals() -> None:
    with pytest.raises(TypeError, match="tuple"):
        classify_pauses(  # type: ignore[arg-type]
            [SpeechInterval(0, 100)],
            sample_rate=1000,
            minimum_gap_ms=100,
            minimum_gap_count=4,
            separation_ratio=1.8,
            maximum_iterations=50,
        )


def test_classify_pauses_rejects_two_clusters_below_separation_ratio() -> None:
    gap_sizes = (200, 210, 300, 310)
    speech = [SpeechInterval(0, 1000)]
    for gap in gap_sizes:
        start = speech[-1].end_sample + gap
        speech.append(SpeechInterval(start, start + 1000))

    with pytest.raises(CorpusFailure, match="not separated"):
        _classify(tuple(speech))
