import pytest

from latintts.corpus.domain import CorpusState, require_transition


def test_state_machine_accepts_next_state_only() -> None:
    assert require_transition(CorpusState.DISCOVERED, CorpusState.INVENTORIED) is None


def test_state_machine_rejects_skipped_state() -> None:
    with pytest.raises(ValueError, match="illegal corpus state transition"):
        require_transition(CorpusState.DISCOVERED, CorpusState.SEGMENTED)
