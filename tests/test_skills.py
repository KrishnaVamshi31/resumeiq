"""Skill taxonomy, extraction and job-description parsing."""

from __future__ import annotations

from app.core.parsing.jobdesc import Seniority, detect_seniority, parse_job
from app.core.skills.matcher import extract_skills, top_skills
from app.core.skills.taxonomy import load_taxonomy, normalise_surface


class TestTaxonomy:
    def test_loads_a_substantial_ontology(self):
        taxonomy = load_taxonomy()
        assert len(taxonomy) > 100
        assert taxonomy.get("Python") is not None

    def test_aliases_resolve_to_canonical_names(self):
        taxonomy = load_taxonomy()
        assert taxonomy.canonical("golang") == "Go"
        assert taxonomy.canonical("k8s") == "Kubernetes"
        assert taxonomy.canonical("postgres") == "PostgreSQL"

    def test_normalisation_keeps_meaningful_punctuation(self):
        assert normalise_surface("C++") == "c++"
        assert normalise_surface("Node.JS") == "node.js"
        assert normalise_surface("  CI/CD  ") == "ci/cd"

    def test_related_skills_are_available_for_gap_framing(self):
        assert "Kubernetes" in load_taxonomy().related_to("Docker")


class TestSkillExtraction:
    def test_finds_skills_across_contexts(self, strong_resume_text):
        found = extract_skills({"other": strong_resume_text})
        for expected in ("Python", "Go", "Kubernetes", "PostgreSQL", "Kafka", "Terraform"):
            assert expected in found, f"expected to find {expected}"

    def test_longer_skill_names_win_over_substrings(self):
        """"React Native" must not also register a bare "React"."""
        found = extract_skills({"skills": "React Native"})
        assert "React Native" in found
        assert "React" not in found

    def test_cpp_is_not_matched_inside_c(self):
        found = extract_skills({"skills": "C++"})
        assert "C++" in found
        assert "C" not in found

    def test_context_drives_confidence(self):
        listed = extract_skills({"skills": "Python, Django"})
        demonstrated = extract_skills(
            {"experience": "Built a Python service handling 2M requests using Django."}
        )
        assert demonstrated["Python"].confidence > listed["Python"].confidence
        assert demonstrated["Python"].is_demonstrated
        assert not listed["Python"].is_demonstrated

    def test_repetition_raises_confidence(self):
        once = extract_skills({"experience": "Used Kafka."})
        thrice = extract_skills(
            {"experience": "Used Kafka. Tuned Kafka consumers. Migrated Kafka clusters."}
        )
        assert thrice["Kafka"].count == 3
        assert thrice["Kafka"].confidence >= once["Kafka"].confidence

    def test_fuzzy_recovery_catches_a_misspelling(self):
        found = extract_skills({"skills": "Kubernets, Postgres"}, fuzzy=True)
        assert "Kubernetes" in found
        assert found["Kubernetes"].fuzzy is True
        # A fuzzy hit is explicitly less trusted than an exact one.
        assert found["Kubernetes"].confidence < found["PostgreSQL"].confidence

    def test_fuzzy_can_be_disabled(self):
        assert "Kubernetes" not in extract_skills({"skills": "Kubernets"}, fuzzy=False)

    def test_evidence_is_captured_for_the_ui(self):
        found = extract_skills({"experience": "Rebuilt the settlement pipeline in Go."})
        assert found["Go"].evidence
        assert "settlement" in found["Go"].evidence[0]

    def test_top_skills_orders_by_confidence(self, strong_resume_text):
        ranked = top_skills(extract_skills({"experience": strong_resume_text}), limit=5)
        assert len(ranked) == 5
        confidences = [s.confidence for s in ranked]
        assert confidences == sorted(confidences, reverse=True)


class TestJobParsing:
    def test_splits_required_from_preferred(self, job_text):
        job = parse_job(job_text)
        required = {s.name for s in job.required_skills}
        preferred = {s.name for s in job.preferred_skills}

        assert {"Python", "SQL", "PostgreSQL", "Kubernetes", "Docker", "Kafka"} <= required
        assert {"Go", "Terraform"} <= preferred
        assert not required & preferred, "a skill cannot be both required and preferred"

    def test_extracts_role_metadata(self, job_text):
        job = parse_job(job_text)
        assert job.seniority is Seniority.SENIOR
        assert job.min_years_experience == 6
        assert job.degree_required is not None
        assert job.title is not None and "Backend" in job.title

    def test_benefits_section_does_not_contribute_requirements(self, job_text):
        job = parse_job(job_text)
        # "Competitive salary, equity..." must not become a requirement.
        assert not any("salary" in s.name.lower() for s in job.all_skills)

    def test_keywords_capture_domain_terms_outside_the_taxonomy(self, job_text):
        job = parse_job(job_text)
        assert job.keywords
        assert all(isinstance(k, str) for k in job.keywords)

    def test_posting_without_headings_still_yields_requirements(self):
        job = parse_job("We need someone strong in Python and PostgreSQL to build APIs.")
        assert {s.name for s in job.required_skills} >= {"Python", "PostgreSQL"}

    def test_responsibilities_are_captured(self, job_text):
        assert parse_job(job_text).responsibilities

    def test_seniority_detection(self):
        assert detect_seniority("Staff Engineer") is Seniority.STAFF
        assert detect_seniority("Junior Data Analyst") is Seniority.JUNIOR
        assert detect_seniority("Director of Engineering") is Seniority.DIRECTOR
        assert detect_seniority("Software Engineer") is Seniority.UNSPECIFIED
