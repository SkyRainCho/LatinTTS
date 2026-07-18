from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest

import latintts
from latintts import pipeline as pipeline_module
from latintts.domain import Diagnostic, PronunciationOverride, ResolutionMethod, Severity
from latintts.g2p import G2PResult
from latintts.pipeline import Pronouncer


def test_analyze_preserves_source_spans_and_builds_plan() -> None:
    plan = Pronouncer.default().analyze("Ave, cælum!")

    assert plan.original_text == "Ave, cælum!"
    assert plan.normalized_text == "Ave, cælum!"
    assert [token.normalized for token in plan.tokens] == ["ave", "caelum"]
    assert [token.source_span for token in plan.tokens] == [(0, 3), (5, 10)]
    assert "expand-ae-ligature" in plan.tokens[1].normalization_transforms
    assert plan.tokens[1].ipa == "ˈt͡ʃe.lum"


def test_request_override_is_used_and_recorded() -> None:
    override = PronunciationOverride(stress_index=1)

    plan = Pronouncer.default().analyze("Dominus", overrides={0: override})

    assert plan.tokens[0].stress_index == 1
    assert plan.tokens[0].ipa == "doˈmi.nus"
    assert plan.tokens[0].resolution_method is ResolutionMethod.OVERRIDE
    assert plan.tokens[0].override == override


def test_request_override_wins_over_acute_mark_and_lexicon() -> None:
    override = PronunciationOverride(stress_index=2)

    token = Pronouncer.default().analyze("Domínus", overrides={0: override}).tokens[0]

    assert token.normalized == "dominus"
    assert token.stress_index == 2
    assert token.applied_rule_ids[0] == "override-stress"


def test_acute_mark_position_maps_to_its_real_syllable() -> None:
    token = Pronouncer.default().analyze("Domi\u0301nus").tokens[0]

    assert token.syllables == ("do", "mi", "nus")
    assert token.stress_index == 1
    assert token.ipa == "doˈmi.nus"
    assert "remove-acute-stress-mark" in token.normalization_transforms
    assert "explicit-stress" in token.applied_rule_ids


def test_full_override_uses_structured_non_initial_stress_rendering() -> None:
    override = PronunciationOverride(
        stress_index=1,
        ipa="doˈmi.nus",
        model_phonemes=("d", "o", ".", "ˈ", "m", "i", ".", "n", "u", "s"),
    )

    token = Pronouncer.default().analyze("Dominus", overrides={0: override}).tokens[0]

    assert token.ipa == override.ipa
    assert token.model_phonemes == override.model_phonemes
    assert token.stress_index == 1


def test_full_override_stress_must_match_resolved_stress() -> None:
    override = PronunciationOverride(
        ipa="doˈmi.nus",
        model_phonemes=("d", "o", ".", "ˈ", "m", "i", ".", "n", "u", "s"),
    )

    with pytest.raises(ValueError, match=r"token 0 'Dominus'.*stress token"):
        Pronouncer.default().analyze("Dominus", overrides={0: override})


def test_full_override_omits_discarded_g2p_rule_provenance() -> None:
    override = PronunciationOverride(
        ipa="ˈo.i",
        model_phonemes=("ˈ", "o", ".", "i"),
    )

    token = Pronouncer.default().analyze("Ave", overrides={0: override}).tokens[0]

    assert token.applied_rule_ids == ("disyllable-stress",)
    assert token.source_ids == ("allen-greenough-accents",)
    assert not {"simple-a", "simple-v", "simple-e"} & set(token.applied_rule_ids)
    assert "liber-usualis-1962" not in token.source_ids


def test_full_override_preserves_lexicon_stress_provenance() -> None:
    override = PronunciationOverride(
        ipa="ˈo.i.u",
        model_phonemes=("ˈ", "o", ".", "i", ".", "u"),
    )

    token = Pronouncer.default().analyze("Dominus", overrides={0: override}).tokens[0]

    assert token.applied_rule_ids == ("stress-lexicon",)
    assert token.source_ids == (
        "perseus-lewis-short",
        "allen-greenough-accents",
    )


def test_full_override_keeps_stress_warnings_and_omits_g2p_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_g2p = pipeline_module.ecclesiastical_g2p

    def g2p_with_warning(*args: Any, **kwargs: Any) -> G2PResult:
        result = real_g2p(*args, **kwargs)
        warning = Diagnostic(
            code="DISCARDED_AUTOMATIC_G2P_WARNING",
            message="warning from discarded automatic phonemes",
            severity=Severity.WARNING,
        )
        return replace(result, warnings=(warning,))

    monkeypatch.setattr(pipeline_module, "ecclesiastical_g2p", g2p_with_warning)
    override = PronunciationOverride(
        ipa="ˈo.i.u",
        model_phonemes=("ˈ", "o", ".", "i", ".", "u"),
    )

    plan = Pronouncer.default().analyze("Fabula", overrides={0: override})

    assert [warning.code for warning in plan.tokens[0].warnings] == ["PRONUNCIATION_NEEDS_REVIEW"]
    assert plan.warnings == plan.tokens[0].warnings


def test_unknown_stress_warning_reaches_token_and_plan_with_context() -> None:
    plan = Pronouncer.default().analyze("Fabula")

    warning = plan.tokens[0].warnings[0]
    assert warning.code == "PRONUNCIATION_NEEDS_REVIEW"
    assert warning.source_span == (0, 6)
    assert warning.token_index == 0
    assert plan.warnings == (warning,)


def test_g2p_exception_marks_token_resolution_method() -> None:
    token = Pronouncer.default().analyze("mihi").tokens[0]

    assert token.ipa == "ˈmi.ki"
    assert token.resolution_method is ResolutionMethod.EXCEPTION
    assert "h-mihi-nihil" in token.applied_rule_ids


def test_hei_word_level_exception_does_not_generalize_ei() -> None:
    hei, mei, diaeresis = (
        Pronouncer.default().analyze(word).tokens[0] for word in ("hei", "mei", "heï")
    )

    assert hei.syllables == ("hei",)
    assert hei.stress_index == 0
    assert hei.ipa == "ˈei̯"
    assert hei.model_phonemes == ("ˈ", "e", "i̯")
    assert hei.applied_rule_ids == ("monosyllable-stress", "hei-ei-diphthong")
    assert hei.source_ids == ("allen-greenough-accents", "liber-usualis-1962")
    assert hei.resolution_method is ResolutionMethod.EXCEPTION
    assert hei.warnings == ()

    assert mei.syllables == ("me", "i")
    assert mei.ipa == "ˈme.i"
    assert "hei-ei-diphthong" not in mei.applied_rule_ids
    assert diaeresis.syllables == ("he", "ï")
    assert diaeresis.ipa == "ˈe.i"
    assert "hei-ei-diphthong" not in diaeresis.applied_rule_ids


def test_invalid_override_error_identifies_token() -> None:
    override = PronunciationOverride(
        ipa="ˈaunknown",
        model_phonemes=("ˈ", "a", "unknown"),
    )

    with pytest.raises(ValueError, match=r"token 0 'Ave'.*unknown model phonemes"):
        Pronouncer.default().analyze("Ave", overrides={0: override})


def test_out_of_range_override_stress_error_identifies_token() -> None:
    override = PronunciationOverride(stress_index=2)

    with pytest.raises(ValueError, match=r"token 0 'Ave'.*override stress_index"):
        Pronouncer.default().analyze("Ave", overrides={0: override})


@pytest.mark.parametrize("stress_index", [False, 1.0, "1"])
def test_non_integer_override_stress_error_identifies_token(stress_index: object) -> None:
    override = PronunciationOverride(stress_index=cast(int, stress_index))

    with pytest.raises(
        ValueError,
        match=r"token 0 'Dominus'.*override stress_index must be an integer",
    ):
        Pronouncer.default().analyze("Dominus", overrides={0: override})


@pytest.mark.parametrize(
    "overrides",
    [
        {-1: PronunciationOverride(stress_index=0)},
        {1: PronunciationOverride(stress_index=0)},
        cast(
            Mapping[int, PronunciationOverride],
            {"first": PronunciationOverride(stress_index=0)},
        ),
        cast(
            Mapping[int, PronunciationOverride],
            {0.0: PronunciationOverride(stress_index=0)},
        ),
        cast(
            Mapping[int, PronunciationOverride],
            {False: PronunciationOverride(stress_index=0)},
        ),
    ],
)
def test_invalid_override_key_is_rejected(
    overrides: Mapping[int, PronunciationOverride],
) -> None:
    with pytest.raises(ValueError, match="override token indexes do not exist"):
        Pronouncer.default().analyze("Ave", overrides=overrides)


def test_phrase_phonemes_put_one_boundary_between_words() -> None:
    plan = Pronouncer.default().analyze("Ave Maria")

    assert plan.phrase_phonemes == (
        *plan.tokens[0].model_phonemes,
        "|",
        *plan.tokens[1].model_phonemes,
    )
    assert plan.phrase_phonemes.count("|") == 1


def test_empty_text_builds_an_empty_plan() -> None:
    plan = Pronouncer.default().analyze("")

    assert plan.original_text == ""
    assert plan.normalized_text == ""
    assert plan.tokens == ()
    assert plan.phrase_phonemes == ()
    assert plan.warnings == ()


def test_overrides_do_not_modify_pronouncer_dictionaries() -> None:
    pronouncer = Pronouncer.default()
    stress_before = dict(pronouncer.stress_lexicon)
    exceptions_before = dict(pronouncer.g2p_exceptions)

    pronouncer.analyze("Dominus", overrides={0: PronunciationOverride(stress_index=1)})

    assert dict(pronouncer.stress_lexicon) == stress_before
    assert dict(pronouncer.g2p_exceptions) == exceptions_before


def test_package_exports_only_the_stable_pronunciation_api() -> None:
    assert latintts.__all__ == [
        "Pronouncer",
        "PronunciationOverride",
        "PronunciationPlan",
        "PronunciationToken",
    ]
    assert latintts.Pronouncer is Pronouncer
    assert latintts.PronunciationOverride is PronunciationOverride
    assert latintts.__version__ == "0.1.0"
