import hashlib
import wave
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from latintts.corpus import review as review_module
from latintts.corpus.alignment import WordSpan
from latintts.corpus.paths import CorpusPaths
from latintts.corpus.records import ReviewEvent
from latintts.corpus.review import (
    ReviewedWordSpan,
    read_textgrid,
    replay_review_events,
    write_textgrid,
)
from tests.corpus.factories import recording


def _opaque_source_transcoder(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
    start = float(command[command.index("-ss") + 1])
    end = float(command[command.index("-to") + 1])
    with wave.open(command[-1], "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(3)
        writer.setframerate(48_000)
        writer.writeframes(b"\x01\x00\x00" * round((end - start) * 48_000))
    return CompletedProcess(command, 0, "", "")


@pytest.mark.parametrize(("extension", "codec"), [("flac", "flac"), ("mp3", "mp3")])
def test_export_review_audio_transcodes_lossless_and_lossy_raw_sources(
    tmp_path: Path, extension: str, codec: str
) -> None:
    paths = CorpusPaths.from_project_root(tmp_path)
    paths.ensure_layout()
    source = paths.raw_spoken / f"rec-1.{extension}"
    source.write_bytes(f"opaque-{extension}-fixture".encode())
    base = recording("rec-1", 2.0)
    record = replace(
        base,
        relative_path=f"raw/spoken/rec-1.{extension}",
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        metadata=replace(base.metadata, codec=codec),
    )
    destination = tmp_path / f"review-{extension}.wav"

    provenance = review_module._export_review_audio(
        record,
        paths,
        destination,
        start_seconds=0.25,
        end_seconds=1.25,
        ffmpeg_version="ffmpeg-test-1",
        run_command=_opaque_source_transcoder,
    )

    assert provenance.source_relative_path.endswith(f".{extension}")
    assert provenance.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    with wave.open(str(destination), "rb") as reader:
        assert (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) == (
            48_000,
            1,
            3,
        )


def test_textgrid_round_trip_preserves_take_and_word_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "take.TextGrid"
    words = (
        WordSpan("Pater", 0.1, 0.7, 0.9, 0),
        WordSpan("noster", 0.7, 1.4, 0.8, 1),
    )
    write_textgrid(path, duration_seconds=1.5, take_start=0.0, take_end=1.5, words=words)

    result = read_textgrid(path)

    assert result.take_start == 0.0
    assert result.take_end == 1.5
    assert [(word.text, word.start_seconds, word.end_seconds) for word in result.words] == [
        ("Pater", 0.1, 0.7),
        ("noster", 0.7, 1.4),
    ]


def test_textgrid_escapes_quotes_and_omits_gap_intervals_from_reviewed_words(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quoted.TextGrid"
    words = (ReviewedWordSpan('Pater"noster', 0.2, 0.8),)

    write_textgrid(path, duration_seconds=1.0, take_start=0.1, take_end=0.9, words=words)

    assert 'text = "Pater""noster"' in path.read_text(encoding="utf-8")
    result = read_textgrid(path)
    assert result.words == (result.words[0],)
    assert result.words[0].text == 'Pater"noster'


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('name = "words"', 'name = "phones"', "tiers"),
        ("xmax = 1.5\ntiers?", "xmax = 0.5\ntiers?", "duration"),
        ("xmin = 0.1\n            xmax = 0.7", "xmin = -0.1\n            xmax = 0.7", "interval"),
        ("xmin = 0.7\n            xmax = 1.4", "xmin = 0.6\n            xmax = 1.4", "overlap"),
        ("intervals [2]:", "intervals [1]:", "order"),
    ],
)
def test_textgrid_reader_rejects_invalid_generated_structure(
    tmp_path: Path, old: str, new: str, match: str
) -> None:
    path = tmp_path / "invalid.TextGrid"
    write_textgrid(
        path,
        duration_seconds=1.5,
        take_start=0.0,
        take_end=1.5,
        words=(
            WordSpan("Pater", 0.1, 0.7, 0.9, 0),
            WordSpan("noster", 0.7, 1.4, 0.8, 1),
        ),
    )
    content = path.read_text(encoding="utf-8")
    assert old in content
    path.write_text(content.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        read_textgrid(path)


@pytest.mark.parametrize(
    ("duration", "take_start", "take_end", "words"),
    [
        (0.0, 0.0, 0.0, ()),
        (1.0, -0.1, 0.9, ()),
        (1.0, 0.1, 1.1, ()),
        (
            1.0,
            0.0,
            1.0,
            (WordSpan("Pater", 0.5, 0.8, 0.9, 0), WordSpan("noster", 0.4, 0.9, 0.8, 1)),
        ),
    ],
)
def test_textgrid_writer_rejects_invalid_boundaries(
    tmp_path: Path,
    duration: float,
    take_start: float,
    take_end: float,
    words: tuple[WordSpan, ...],
) -> None:
    with pytest.raises(ValueError):
        write_textgrid(
            tmp_path / "invalid.TextGrid",
            duration_seconds=duration,
            take_start=take_start,
            take_end=take_end,
            words=words,
        )


def test_review_replay_preserves_automatic_values_and_applies_events() -> None:
    automatic = {
        "segment_start": 0.0,
        "segment_end": 1.4,
        "review_decision": "unreviewed",
    }
    events = (
        ReviewEvent(
            "1",
            "event-boundary",
            "take-1",
            "segment_end",
            1.4,
            1.5,
            "final consonant retained",
            "owner",
            "2026-07-19T12:00:00+08:00",
        ),
        ReviewEvent(
            "1",
            "event-decision",
            "take-1",
            "review_decision",
            "unreviewed",
            "approved",
            "listened in full",
            "owner",
            "2026-07-19T12:00:00+08:00",
        ),
    )

    effective = replay_review_events(automatic, events)

    assert automatic["segment_end"] == 1.4
    assert effective["segment_end"] == 1.5
    assert effective["review_decision"] == "approved"


@pytest.mark.parametrize("failure", ("before", "entity", "decision", "duplicate-id"))
def test_review_replay_strictly_validates_existing_history(failure: str) -> None:
    first = ReviewEvent(
        "1",
        "event-one",
        "take-1",
        "segment_end",
        1.4,
        1.5,
        "boundary checked",
        "owner",
        "2026-07-19T12:00:00+08:00",
    )
    second = ReviewEvent(
        "1",
        "event-two",
        "take-1",
        "review_decision",
        "unreviewed",
        "approved",
        "listened in full",
        "owner",
        "2026-07-19T12:01:00+08:00",
    )
    if failure == "before":
        second = ReviewEvent(
            "1",
            "event-two",
            "take-1",
            "segment_end",
            1.4,
            1.6,
            "boundary checked",
            "owner",
            "2026-07-19T12:01:00+08:00",
        )
    elif failure == "entity":
        second = ReviewEvent(
            "1",
            "event-two",
            "take-2",
            "review_decision",
            "unreviewed",
            "approved",
            "listened in full",
            "owner",
            "2026-07-19T12:01:00+08:00",
        )
    elif failure == "decision":
        second = ReviewEvent(
            "1",
            "event-two",
            "take-1",
            "review_decision",
            "unreviewed",
            "maybe",
            "listened in full",
            "owner",
            "2026-07-19T12:01:00+08:00",
        )
    else:
        second = ReviewEvent(
            "1",
            "event-one",
            "take-1",
            "review_decision",
            "unreviewed",
            "approved",
            "listened in full",
            "owner",
            "2026-07-19T12:01:00+08:00",
        )

    with pytest.raises(ValueError):
        replay_review_events({"segment_end": 1.4, "review_decision": "unreviewed"}, (first, second))


def test_review_event_from_dict_requires_exact_fields() -> None:
    raw = ReviewEvent(
        "1",
        "event-one",
        "take-1",
        "review_decision",
        "unreviewed",
        "rejected",
        "background noise",
        "owner",
        "2026-07-19T12:00:00+08:00",
    ).to_dict()
    assert ReviewEvent.from_dict(raw).to_dict() == raw
    raw["unexpected"] = True
    with pytest.raises(ValueError, match="exact fields"):
        ReviewEvent.from_dict(raw)


def test_textgrid_writer_rejects_empty_word_and_nonfinite_duration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="text"):
        write_textgrid(
            tmp_path / "empty.TextGrid",
            duration_seconds=1.0,
            take_start=0.0,
            take_end=1.0,
            words=(ReviewedWordSpan("", 0.1, 0.2),),
        )
    with pytest.raises(ValueError, match="finite"):
        write_textgrid(
            tmp_path / "nan.TextGrid",
            duration_seconds=float("nan"),
            take_start=0.0,
            take_end=1.0,
            words=(),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "missing"),
        ("assignment", "xmin"),
        ("literal", "structure"),
        ("number", "numeric"),
        ("header", "header"),
        ("count", "interval count"),
        ("trailing", "unexpected"),
        ("global", "duration"),
        ("tier-bounds", "tier bounds"),
        ("gap", "cover"),
        ("outside", "exceeds"),
        ("empty-intervals", "cover"),
    ],
)
def test_textgrid_reader_rejects_each_strict_structure_boundary(
    tmp_path: Path, mutation: str, match: str
) -> None:
    path = tmp_path / f"{mutation}.TextGrid"
    write_textgrid(
        path,
        duration_seconds=1.0,
        take_start=0.0,
        take_end=1.0,
        words=(ReviewedWordSpan("Pater", 0.2, 0.8),),
    )
    content = path.read_text(encoding="utf-8")
    if mutation == "missing":
        content = content[: content.index('        name = "take"')]
    elif mutation == "assignment":
        content = content.replace("xmin = 0.0", "xmin: 0.0", 1)
    elif mutation == "literal":
        content = content.replace("tiers? <exists>", "tiers? <missing>")
    elif mutation == "number":
        content = content.replace("xmax = 1.0\n", "xmax = nope\n", 1)
    elif mutation == "header":
        content = content.replace("ooTextFile", "bad", 1)
    elif mutation == "count":
        content = content.replace("intervals: size = 3", "intervals: size = nope")
    elif mutation == "trailing":
        content += "unexpected\n"
    elif mutation == "global":
        content = content.replace("xmin = 0\nxmax = 1.0\ntiers?", "xmin = -1\nxmax = 1.0\ntiers?")
    elif mutation == "tier-bounds":
        content = content.replace(
            'name = "words"\n        xmin = 0.0',
            'name = "words"\n        xmin = 0.1',
        )
    elif mutation == "gap":
        content = content.replace(
            "intervals [2]:\n            xmin = 0.2",
            "intervals [2]:\n            xmin = 0.3",
        )
    elif mutation == "outside":
        content = content.replace(
            "intervals [3]:\n            xmin = 0.8\n            xmax = 1.0",
            "intervals [3]:\n            xmin = 0.8\n            xmax = 1.1",
        )
    else:
        content = content.replace("intervals: size = 3", "intervals: size = 0")
        content = content[
            : content.index("        intervals [1]:", content.index('name = "words"'))
        ]
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        read_textgrid(path)


@pytest.mark.parametrize("cut_marker", ('name = "words"', "intervals: size = 3"))
def test_textgrid_reader_converts_critical_truncation_to_value_error(
    tmp_path: Path, cut_marker: str
) -> None:
    path = tmp_path / "truncated.TextGrid"
    write_textgrid(
        path,
        duration_seconds=1.0,
        take_start=0.0,
        take_end=1.0,
        words=(ReviewedWordSpan("Pater", 0.2, 0.8),),
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    cut = next(index for index, line in enumerate(lines) if cut_marker in line)
    if cut_marker.startswith("intervals"):
        cut += 1
    path.write_text("\n".join(lines[:cut]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="TextGrid"):
        read_textgrid(path)


def test_canonical_descendant_accepts_only_direct_unaliased_path(tmp_path: Path) -> None:
    root = tmp_path / "review"
    group = root / "group"
    group.mkdir(parents=True)
    target = group / "automatic.json"
    target.write_text("{}", encoding="utf-8")

    assert (
        review_module._require_canonical_descendant(
            root, target, kind="review file", require_file=True
        )
        == target.absolute()
    )


@pytest.mark.parametrize("alias_level", ("root", "group", "file"))
def test_canonical_descendant_rejects_alias_at_every_review_level(
    tmp_path: Path, alias_level: str
) -> None:
    canonical = tmp_path / "canonical"
    canonical_group = canonical / "group"
    canonical_group.mkdir(parents=True)
    canonical_file = canonical_group / "automatic.json"
    canonical_file.write_text("{}", encoding="utf-8")
    review_root = tmp_path / "review"
    group = review_root / "group"
    target = group / "automatic.json"
    try:
        if alias_level == "root":
            review_root.symlink_to(canonical, target_is_directory=True)
        elif alias_level == "group":
            review_root.mkdir()
            group.symlink_to(canonical_group, target_is_directory=True)
        else:
            group.mkdir(parents=True)
            target.symlink_to(canonical_file)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match=r"alias|canonical"):
        review_module._require_canonical_descendant(
            review_root, target, kind="review file", require_file=True
        )


def test_replay_rejects_invalid_container_and_missing_nested_fields() -> None:
    with pytest.raises(TypeError, match="object"):
        replay_review_events([], ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        replay_review_events({}, [])  # type: ignore[arg-type]
    missing = ReviewEvent(
        "1",
        "event-missing",
        "take-1",
        "missing",
        1,
        2,
        "reason",
        "owner",
        "2026-07-19T12:00:00+08:00",
    )
    with pytest.raises(ValueError, match="absent"):
        replay_review_events({}, (missing,))
    word = ReviewEvent(
        "1",
        "event-word",
        "take-1",
        "word:0:text",
        "Pater",
        "pater",
        "case corrected",
        "owner",
        "2026-07-19T12:00:00+08:00",
    )
    with pytest.raises(ValueError, match="word field"):
        replay_review_events({"words": []}, (word,))
    with pytest.raises(ValueError, match="word field"):
        replay_review_events({"words": [{}]}, (word,))


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"segment_start": 0.0, "segment_end": 1.0, "words": [], "review_decision": "approved"},
        {"segment_start": 1.0, "segment_end": 1.0, "words": [], "review_decision": "unreviewed"},
        {"segment_start": 0.0, "segment_end": 1.0, "words": {}, "review_decision": "unreviewed"},
        {
            "segment_start": 0.0,
            "segment_end": 1.0,
            "words": [None],
            "review_decision": "unreviewed",
        },
        {
            "segment_start": 0.0,
            "segment_end": 1.0,
            "words": [{"text": "", "start_seconds": 0.1, "end_seconds": 0.2}],
            "review_decision": "unreviewed",
        },
        {
            "segment_start": 0.0,
            "segment_end": 1.0,
            "words": [{"text": "Pater", "start_seconds": 0.8, "end_seconds": 1.1}],
            "review_decision": "unreviewed",
        },
    ],
)
def test_automatic_value_schema_rejects_invalid_nested_values(raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        review_module._validated_automatic_values(raw)


def test_decision_and_change_helpers_reject_invalid_values() -> None:
    with pytest.raises(TypeError, match="object"):
        review_module._validate_decision_item(None)
    base = {
        "entity_id": "take-1",
        "decision": "maybe",
        "reason": "reason",
        "reviewer": "owner",
        "reviewed_at": "2026-07-19T12:00:00+08:00",
    }
    with pytest.raises(ValueError, match="enum"):
        review_module._validate_decision_item(base)
    with pytest.raises(ValueError, match="metadata"):
        review_module._validate_decision_item(base | {"decision": "unreviewed"})
    with pytest.raises(ValueError, match="word values"):
        review_module._event_fields(
            {
                "segment_start": 0.0,
                "segment_end": 1.0,
                "words": {},
                "review_decision": "unreviewed",
            },
            {"segment_start": 0.1, "segment_end": 1.0, "words": [], "review_decision": "approved"},
        )
    with pytest.raises(ValueError, match="word count"):
        review_module._event_fields(
            {
                "segment_start": 0.0,
                "segment_end": 1.0,
                "words": [],
                "review_decision": "unreviewed",
            },
            {
                "segment_start": 0.0,
                "segment_end": 1.0,
                "words": [{"text": "Pater", "start_seconds": 0.1, "end_seconds": 0.9}],
                "review_decision": "approved",
            },
        )
    with pytest.raises(ValueError, match="word values"):
        review_module._event_fields(
            {
                "segment_start": 0.0,
                "segment_end": 1.0,
                "words": [None],
                "review_decision": "unreviewed",
            },
            {
                "segment_start": 0.0,
                "segment_end": 1.0,
                "words": [None],
                "review_decision": "approved",
            },
        )
    with pytest.raises(ValueError, match="word count"):
        review_module._target_from_textgrid(
            {"words": []},
            review_module.ReviewedBoundaries(1.0, 0.0, 1.0, (ReviewedWordSpan("Pater", 0.1, 0.9),)),
            "approved",
        )


def test_review_json_reader_rejects_duplicate_constants_and_nonobjects(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    path.write_text('{"id":1,"id":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid review JSON"):
        review_module._read_json(path)
    path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid review JSON"):
        review_module._read_json(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON object"):
        review_module._read_json(path)


def test_existing_review_history_rejects_duplicate_and_noop_events(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    decision = {
        "reason": "listened",
        "reviewer": "owner",
        "reviewed_at": "2026-07-19T12:00:00+08:00",
    }
    event = review_module._new_review_event(
        "take-1", "review_decision", "unreviewed", "approved", decision
    )
    from latintts.corpus.store import write_jsonl_atomic

    write_jsonl_atomic(path, (event.to_dict(), event.to_dict()))
    with pytest.raises(ValueError, match="duplicate"):
        review_module._load_existing_review_events(path)
    noop = review_module._new_review_event("take-1", "segment_start", 0.0, 0.0, decision)
    write_jsonl_atomic(path, (noop.to_dict(),))
    with pytest.raises(ValueError, match="no-op"):
        review_module._load_existing_review_events(path)


def test_review_bundle_public_functions_reject_wrong_argument_types() -> None:
    with pytest.raises(TypeError):
        review_module.export_review_bundle(object(), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        review_module.import_review_bundle(object(), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reference"):
        review_module._bundle_file(Path.cwd(), "wrong.wav", "take-1.wav", "audio")
