from latintts.sources import load_source_registry


def test_registry_contains_normative_and_evaluation_sources() -> None:
    sources = load_source_registry()
    assert "liber-usualis-1962" in sources
    assert sources["liber-usualis-1962"].authority_rank == 1
    assert "wikimedia-ecclesiastical-pronunciation" in sources
    assert all(record.accessed_on == "2026-07-17" for record in sources.values())


def test_registry_has_unique_ids_and_nonempty_usage_terms() -> None:
    sources = load_source_registry()
    assert len(sources) == 6
    assert all(record.license_or_terms.strip() for record in sources.values())
    assert all(record.usage_note.strip() for record in sources.values())
