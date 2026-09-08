"""Scoring engine: ATS, structure, content, aggregation and recommendations."""

from __future__ import annotations

import pytest

from app.core.extraction.base import PageStats, SourceFormat
from app.core.parsing.parser import parse_resume
from app.core.scoring.aggregate import analyze
from app.core.scoring.ats import score_ats, score_structure
from app.core.scoring.base import (
    WEIGHTS_WITH_JOB,
    WEIGHTS_WITHOUT_JOB,
    DimensionId,
    Severity,
    band,
    make_signal,
)
from app.core.scoring.content import score_content
from tests.conftest import make_document


class TestSignalPrimitives:
    def test_score_is_clamped(self):
        assert make_signal("x", "X", 1.7).score == 1.0
        assert make_signal("x", "X", -3.0).score == 0.0

    def test_severity_is_derived_from_score(self):
        assert make_signal("x", "X", 1.0).severity is Severity.OK
        assert make_signal("x", "X", 0.8).severity is Severity.INFO
        assert make_signal("x", "X", 0.5).severity is Severity.WARNING
        assert make_signal("x", "X", 0.1).severity is Severity.CRITICAL

    def test_passing_signals_carry_no_recommendation(self):
        assert make_signal("x", "X", 1.0, recommendation="do a thing").recommendation is None

    def test_points_lost_is_weighted(self):
        assert make_signal("x", "X", 0.5, weight=4.0).points_lost == 2.0

    def test_dimension_severity_surfaces_the_worst_signal(self):
        from app.core.scoring.base import DimensionScore

        dimension = DimensionScore(
            id=DimensionId.ATS,
            label="ATS",
            signals=[make_signal("a", "A", 1.0), make_signal("b", "B", 0.1)],
        )
        assert dimension.severity is Severity.CRITICAL

    @pytest.mark.parametrize(
        ("score", "expected"),
        [(95, "excellent"), (75, "strong"), (60, "fair"), (45, "weak"), (10, "poor")],
    )
    def test_bands(self, score, expected):
        assert band(score) == expected

    def test_weights_sum_to_one(self):
        assert sum(WEIGHTS_WITH_JOB.values()) == pytest.approx(1.0)
        assert sum(WEIGHTS_WITHOUT_JOB.values()) == pytest.approx(1.0)


class TestAtsSignals:
    @staticmethod
    def _signal(dimension, signal_id):
        return next(s for s in dimension.signals if s.id == signal_id)

    def test_missing_contact_is_critical(self):
        text = "SOMEONE\n\nEXPERIENCE\nCompany\n- Did a thing that was useful\n"
        document = make_document(text)
        signal = self._signal(score_ats(document, parse_resume(text)), "ats.contact")
        assert signal.severity is Severity.CRITICAL
        assert signal.score == 0.0

    def test_multi_column_layout_is_penalised(self, strong_resume_text):
        document = make_document(strong_resume_text, has_multi_column_hint=True)
        signal = self._signal(score_ats(document, parse_resume(strong_resume_text)), "ats.layout")
        assert signal.score < 0.5
        assert "single-column" in (signal.recommendation or "").lower()

    def test_tables_are_penalised_by_count(self, strong_resume_text):
        resume = parse_resume(strong_resume_text)
        few = make_document(strong_resume_text, has_tables=True, table_count=1)
        many = make_document(strong_resume_text, has_tables=True, table_count=6)
        assert (
            self._signal(score_ats(many, resume), "ats.tables").score
            < self._signal(score_ats(few, resume), "ats.tables").score
        )

    def test_header_footer_content_is_penalised(self, strong_resume_text):
        document = make_document(strong_resume_text, has_text_in_headers_footers=True)
        signal = self._signal(
            score_ats(document, parse_resume(strong_resume_text)), "ats.header_footer"
        )
        assert signal.score < 0.5

    def test_pdf_without_a_text_layer_scores_zero(self, strong_resume_text):
        document = make_document(
            strong_resume_text,
            SourceFormat.PDF,
            page_count=2,
            pages=[
                PageStats(index=0, char_count=10, line_count=1, has_extractable_text=False),
                PageStats(index=1, char_count=10, line_count=1, has_extractable_text=False),
            ],
        )
        signal = self._signal(
            score_ats(document, parse_resume(strong_resume_text)), "ats.text_layer"
        )
        assert signal.score == 0.0

    def test_encrypted_pdf_is_flagged(self, strong_resume_text):
        document = make_document(strong_resume_text, SourceFormat.PDF, is_encrypted=True)
        signal = self._signal(
            score_ats(document, parse_resume(strong_resume_text)), "ats.file_format"
        )
        assert signal.score < 1.0

    def test_unknown_headings_are_reported(self):
        text = "Jane Doe\njane@example.com\n\nMy Journey\n- Built a thing that worked well\n"
        signal = self._signal(
            score_ats(make_document(text), parse_resume(text)), "ats.headings"
        )
        assert signal.score < 1.0
        assert signal.evidence

    def test_clean_resume_scores_highly(self, strong_resume_text):
        dimension = score_ats(make_document(strong_resume_text), parse_resume(strong_resume_text))
        assert dimension.score > 90


class TestStructureSignals:
    @staticmethod
    def _signal(dimension, signal_id):
        return next(s for s in dimension.signals if s.id == signal_id)

    def test_missing_core_sections_are_penalised(self, weak_resume_text):
        dimension = score_structure(
            make_document(weak_resume_text), parse_resume(weak_resume_text)
        )
        signal = self._signal(dimension, "structure.core_sections")
        assert signal.score < 1.0
        assert signal.evidence

    def test_short_resume_is_flagged(self):
        text = "Jane Doe\njane@example.com\n\nEXPERIENCE\nAcme\n- Did one thing here\n"
        signal = self._signal(
            score_structure(make_document(text), parse_resume(text)), "structure.length"
        )
        assert signal.score < 0.5

    def test_overlong_resume_is_flagged(self, strong_resume_text):
        padded = strong_resume_text + ("\n- Delivered another measurable outcome here. " * 260)
        signal = self._signal(
            score_structure(make_document(padded), parse_resume(padded)), "structure.length"
        )
        assert signal.score < 0.7

    def test_length_in_the_ideal_band_scores_full(self, strong_resume_text):
        """400-1000 words is the target band; pad the fixture into it."""
        padded = strong_resume_text + "\n- Delivered a measurable outcome for the team here.\n" * 20
        signal = self._signal(
            score_structure(make_document(padded), parse_resume(padded)), "structure.length"
        )
        assert 400 <= len(padded.split()) <= 1000
        assert signal.score == 1.0

    def test_slightly_thin_resume_scores_between_the_extremes(self, strong_resume_text):
        """The fixture sits just under the ideal floor and should be nudged, not punished."""
        signal = self._signal(
            score_structure(make_document(strong_resume_text), parse_resume(strong_resume_text)),
            "structure.length",
        )
        assert 0.5 < signal.score < 1.0


class TestContentSignals:
    @staticmethod
    def _signal(dimension, signal_id):
        return next(s for s in dimension.signals if s.id == signal_id)

    def test_filler_language_is_caught(self, weak_resume_text):
        signal = self._signal(score_content(parse_resume(weak_resume_text)), "content.filler")
        assert signal.score < 0.6
        assert "responsible for" in (signal.detail or "").lower()

    def test_buzzwords_are_caught(self, weak_resume_text):
        signal = self._signal(score_content(parse_resume(weak_resume_text)), "content.buzzwords")
        assert signal.score < 1.0

    def test_quantified_bullets_score_well(self, strong_resume_text):
        signal = self._signal(
            score_content(parse_resume(strong_resume_text)), "content.quantification"
        )
        assert signal.score > 0.8

    def test_unquantified_bullets_score_poorly(self):
        text = (
            "Jane Doe\njane@example.com\n\nEXPERIENCE\n"
            "Engineer | Acme | Jan 2020 - Present\n"
            "- Built some internal services for the platform team\n"
            "- Improved the deployment process for the team\n"
            "- Maintained the legacy reporting stack for the business\n"
            "- Supported the on-call rotation for the group\n"
        )
        signal = self._signal(score_content(parse_resume(text)), "content.quantification")
        assert signal.score < 0.4
        assert signal.evidence

    def test_first_person_is_flagged(self, weak_resume_text):
        signal = self._signal(score_content(parse_resume(weak_resume_text)), "content.first_person")
        assert signal.score < 1.0

    def test_repeated_opening_verbs_are_flagged(self):
        bullets = "\n".join(
            f"- Managed the {n} workstream and its reporting line" for n in range(6)
        )
        text = f"Jane Doe\njane@example.com\n\nEXPERIENCE\nAcme | Jan 2020 - Present\n{bullets}\n"
        signal = self._signal(score_content(parse_resume(text)), "content.verb_variety")
        assert signal.score < 0.5
        assert signal.evidence

    def test_too_few_bullets_scores_neutral_not_zero(self):
        """A thin resume should not be punished twice for the same defect."""
        text = "Jane Doe\njane@example.com\n\nEXPERIENCE\nAcme\n- One bullet only\n"
        signal = self._signal(score_content(parse_resume(text)), "content.action_verbs")
        assert signal.score == 0.5


class TestAggregation:
    def test_strong_resume_beats_weak_resume(self, strong_resume_text, weak_resume_text):
        strong = analyze(make_document(strong_resume_text))
        weak = analyze(make_document(weak_resume_text))
        assert strong.overall_score > weak.overall_score + 25

    def test_analysis_is_deterministic(self, strong_resume_text):
        first = analyze(make_document(strong_resume_text))
        second = analyze(make_document(strong_resume_text))
        assert first.overall_score == second.overall_score
        assert [r.id for r in first.recommendations] == [r.id for r in second.recommendations]

    def test_dimensions_depend_on_whether_a_job_was_supplied(self, strong_resume_text, job_text):
        without = analyze(make_document(strong_resume_text))
        with_job = analyze(make_document(strong_resume_text), job_text=job_text)
        assert DimensionId.MATCH not in without.dimensions
        assert DimensionId.MATCH in with_job.dimensions
        assert without.weights == WEIGHTS_WITHOUT_JOB
        assert with_job.weights == WEIGHTS_WITH_JOB

    def test_overall_score_matches_the_weighted_mean(self, matched_analysis):
        expected = sum(
            matched_analysis.dimensions[d].score * w for d, w in matched_analysis.weights.items()
        )
        assert matched_analysis.overall_score == pytest.approx(round(expected, 1), abs=0.05)

    def test_recommendations_are_ranked_critical_first_then_by_impact(self, weak_resume_text):
        result = analyze(make_document(weak_resume_text))
        severities = [r.severity for r in result.recommendations]
        critical_positions = [i for i, s in enumerate(severities) if s is Severity.CRITICAL]
        other_positions = [i for i, s in enumerate(severities) if s is not Severity.CRITICAL]
        assert not critical_positions or not other_positions or max(critical_positions) < min(
            other_positions
        )

        criticals = [r for r in result.recommendations if r.severity is Severity.CRITICAL]
        impacts = [r.impact for r in criticals]
        assert impacts == sorted(impacts, reverse=True)

    def test_every_recommendation_is_actionable(self, weak_resume_text):
        for recommendation in analyze(make_document(weak_resume_text)).recommendations:
            assert recommendation.action.strip()
            assert recommendation.impact >= 0

    def test_passing_checks_are_reported_as_strengths(self, strong_resume_text):
        assert analyze(make_document(strong_resume_text)).strengths

    def test_scores_stay_in_range_for_degenerate_input(self):
        for text in ("x", "a b c", "Name\n\n\n", "•\n•\n•"):
            result = analyze(make_document(text))
            assert 0.0 <= result.overall_score <= 100.0
            for dimension in result.dimensions.values():
                assert 0.0 <= dimension.score <= 100.0
