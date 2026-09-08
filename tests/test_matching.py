"""Job matching, gap analysis and the similarity primitives behind them."""

from __future__ import annotations

import pytest

from app.core.embeddings.vectorizer import (
    build_idf,
    cosine,
    missing_terms,
    similarity,
    top_overlapping_terms,
    vectorize,
)
from app.core.parsing.jobdesc import parse_job
from app.core.parsing.parser import parse_resume
from app.core.scoring.aggregate import skill_contexts
from app.core.scoring.matching import score_match
from app.core.skills.matcher import extract_skills


class TestVectorizer:
    def test_identical_text_is_maximally_similar(self):
        text = "Built distributed payment services in Python on Kubernetes."
        assert similarity(text, text) == pytest.approx(1.0, abs=1e-6)

    def test_unrelated_text_is_dissimilar(self):
        assert similarity(
            "Built distributed payment services in Python on Kubernetes.",
            "Prepared seasonal menus and managed kitchen inventory for a restaurant.",
        ) < 0.15

    def test_related_text_scores_between(self):
        score = similarity(
            "Built distributed payment services in Python on Kubernetes.",
            "Seeking an engineer to build payment services in Python.",
        )
        assert 0.15 < score < 1.0

    def test_similarity_is_symmetric(self):
        a, b = "Python and Kafka pipelines", "Kafka pipelines built with Python"
        assert similarity(a, b) == pytest.approx(similarity(b, a))

    def test_empty_input_is_zero_not_an_error(self):
        assert similarity("", "anything at all") == 0.0
        assert cosine({}, {"a": 1.0}) == 0.0

    def test_repetition_does_not_dominate(self):
        """Sublinear TF: saying Python nine times is not nine times more relevant."""
        job = "We need a Python engineer."
        honest = "Built services in Python."
        stuffed = "Python " * 9
        assert similarity(stuffed, job) < similarity(honest, job) * 3

    def test_vectors_are_unit_length(self):
        idf = build_idf(["alpha beta", "beta gamma", "gamma delta"])
        vector = vectorize("alpha beta gamma", idf)
        assert sum(v * v for v in vector.values()) == pytest.approx(1.0, abs=1e-9)

    def test_overlapping_terms_explain_the_score(self):
        overlap = top_overlapping_terms(
            "Built Kafka pipelines and PostgreSQL schemas.",
            "You will build Kafka pipelines against PostgreSQL.",
        )
        terms = {term for term, _ in overlap}
        assert "kafka" in terms

    def test_missing_terms_are_from_the_target_only(self):
        absent = missing_terms("I know Python.", "We need Python and Kubernetes and Terraform.")
        assert "kubernetes" in absent
        assert "python" not in absent


class TestMatching:
    @staticmethod
    def _match(resume_text: str, job_text: str):
        # Build skill contexts exactly as the real pipeline does, so that
        # "demonstrated in a bullet" vs "listed in a keyword dump" is preserved.
        resume = parse_resume(resume_text)
        job = parse_job(job_text)
        skills = extract_skills(skill_contexts(resume))
        return score_match(resume, job, skills)

    def test_strong_candidate_covers_most_requirements(self, strong_resume_text, job_text):
        match = self._match(strong_resume_text, job_text)
        assert match.required_coverage > 0.7
        assert match.dimension.score > 65
        assert {"Python", "Kubernetes", "PostgreSQL", "Kafka"} <= set(match.matched_required)

    def test_weak_candidate_covers_almost_nothing(self, weak_resume_text, job_text):
        match = self._match(weak_resume_text, job_text)
        assert match.required_coverage < 0.2
        assert match.dimension.score < 40
        assert match.critical_gaps

    def test_gaps_are_split_by_priority(self, weak_resume_text, job_text):
        match = self._match(weak_resume_text, job_text)
        assert all(gap.required for gap in match.critical_gaps)
        assert {gap.priority for gap in match.gaps} <= {"high", "medium", "low"}

    def test_adjacent_skills_soften_a_gap(self):
        """Owning Docker makes a Kubernetes gap medium priority, not high."""
        resume = "Jane\njane@x.com\n\nEXPERIENCE\nAcme | Jan 2020 - Present\n- Shipped Docker images daily\n"
        job = "Requirements\n- Must have Kubernetes in production\n"
        match = self._match(resume, job)
        gap = next(g for g in match.gaps if g.name == "Kubernetes")
        assert "Docker" in gap.adjacent_owned
        assert gap.priority == "medium"

    def test_demonstrated_skills_beat_keyword_lists(self, job_text):
        """Two resumes with identical keywords score differently on evidence."""
        listed = (
            "Jane\njane@x.com\n\nSKILLS\nPython, PostgreSQL, Kubernetes, Docker, Kafka, AWS\n"
        )
        shown = (
            "Jane\njane@x.com\n\nEXPERIENCE\nEngineer | Acme | Jan 2019 - Present\n"
            "- Built Python services on Kubernetes with Docker, backed by PostgreSQL\n"
            "- Streamed events through Kafka on AWS for 2 million users\n"
        )
        listed_score = self._match(listed, job_text).dimension.score
        shown_score = self._match(shown, job_text).dimension.score
        assert shown_score > listed_score

    def test_seniority_mismatch_is_penalised(self):
        job = "Principal Engineer\nRequirements\n- Deep Python expertise\n"
        junior = (
            "Jane\njane@x.com\n\nEXPERIENCE\n"
            "Junior Developer | Acme | Jan 2023 - Present\n- Wrote Python scripts daily\n"
        )
        match = self._match(junior, job)
        signal = next(s for s in match.dimension.signals if s.id == "match.seniority")
        assert signal.score < 1.0

    def test_experience_shortfall_is_penalised(self):
        job = "Requirements\n- 10+ years of experience with Python\n"
        resume = (
            "Jane\njane@x.com\n\nEXPERIENCE\n"
            "Engineer | Acme | Jan 2023 - Present\n- Built Python services for the team\n"
        )
        signal = next(
            s for s in self._match(resume, job).dimension.signals if s.id == "match.experience"
        )
        assert signal.score < 1.0
        assert "short by" in (signal.detail or "")

    def test_missing_keywords_are_reported(self, weak_resume_text, job_text):
        assert self._match(weak_resume_text, job_text).missing_keywords

    def test_surplus_skills_are_reported(self, strong_resume_text, job_text):
        """Skills the posting never asks for are still worth surfacing."""
        assert self._match(strong_resume_text, job_text).surplus_skills

    def test_posting_with_no_requirements_scores_neutrally(self, strong_resume_text):
        match = self._match(strong_resume_text, "Come work with us. We are a friendly team.")
        signal = next(s for s in match.dimension.signals if s.id == "match.required_skills")
        assert 0.5 <= signal.score <= 0.85
