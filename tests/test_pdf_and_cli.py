"""End-to-end PDF extraction and the command-line interface."""

from __future__ import annotations

import json

import pytest

from app.cli import main
from app.core.extraction.base import SourceFormat
from app.core.extraction.detector import extract
from app.core.scoring.aggregate import analyze
from tests.pdf_builder import build_pdf

RESUME_PAGE = [
    "Priya Raghavan",
    "Bengaluru, India | priya.raghavan@example.com | +91 98450 12345",
    "",
    "PROFESSIONAL SUMMARY",
    "Senior backend engineer with 8 years building payment platforms.",
    "",
    "PROFESSIONAL EXPERIENCE",
    "Senior Software Engineer | Razorline Technologies | Mar 2021 - Present",
    "- Rebuilt the settlement pipeline in Go, cutting processing to 90 seconds",
    "- Migrated 38 microservices to Kubernetes, reducing spend by 34%",
    "- Designed a Kafka reconciliation service saving 1,200 hours per quarter",
    "",
    "EDUCATION",
    "B.Tech in Computer Science, National Institute of Technology, 2016",
    "",
    "TECHNICAL SKILLS",
    "Python, Go, SQL, PostgreSQL, Kafka, Docker, Kubernetes, AWS, Terraform",
]


class TestPdfPipeline:
    def test_extracts_text_from_a_real_pdf(self):
        document = extract(build_pdf([RESUME_PAGE]), "resume.pdf")
        assert document.source_format is SourceFormat.PDF
        assert document.page_count == 1
        assert "Razorline" in document.text
        assert document.pages[0].has_extractable_text

    def test_multi_page_pdf_is_read_in_order(self):
        document = extract(build_pdf([["Page one content here."], ["Page two content here."]]), "r.pdf")
        assert document.page_count == 2
        assert document.text.index("Page one") < document.text.index("Page two")

    def test_full_analysis_runs_on_a_pdf(self, job_text):
        document = extract(build_pdf([RESUME_PAGE]), "resume.pdf")
        result = analyze(document, job_text=job_text)

        assert result.overall_score > 50
        assert result.resume.contact.email == "priya.raghavan@example.com"
        assert result.resume.experience
        assert "Kubernetes" in result.skills
        assert result.match is not None

    def test_sparse_pdf_is_flagged_as_a_thin_text_layer(self):
        """One short line across two pages should not read as a healthy document."""
        document = extract(build_pdf([["Jane Doe jane@example.com"], ["Engineer at Acme"]]), "r.pdf")
        assert document.chars_per_page < 200
        assert any(not page.has_extractable_text for page in document.pages)

    def test_api_accepts_a_pdf_upload(self, client):
        response = client.post(
            "/api/v1/analyses",
            files={"file": ("resume.pdf", build_pdf([RESUME_PAGE]), "application/pdf")},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["document"]["source_format"] == "pdf"
        assert body["contact"]["email"] == "priya.raghavan@example.com"


class TestCli:
    @pytest.fixture
    def resume_file(self, tmp_path, strong_resume_text):
        path = tmp_path / "resume.txt"
        path.write_text(strong_resume_text, encoding="utf-8")
        return path

    def test_prints_a_readable_report(self, resume_file, capsys):
        assert main(["analyze", str(resume_file), "-q"]) == 0
        out = capsys.readouterr().out
        assert "resume.txt" in out
        assert "/100" in out
        assert "Top recommendations" in out

    def test_json_output_is_machine_readable(self, resume_file, capsys):
        assert main(["analyze", str(resume_file), "--json", "-q"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 1
        assert 0 <= payload[0]["overall_score"] <= 100
        assert payload[0]["dimensions"]
        assert payload[0]["ai_feedback"] is None

    def test_job_description_adds_match_output(self, resume_file, tmp_path, job_text, capsys):
        job_file = tmp_path / "job.txt"
        job_file.write_text(job_text, encoding="utf-8")

        main(["analyze", str(resume_file), "--job", str(job_file), "--json", "-q"])
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["match"]["required_coverage"] > 0.5

    def test_scans_a_directory(self, tmp_path, strong_resume_text, weak_resume_text, capsys):
        (tmp_path / "a.txt").write_text(strong_resume_text, encoding="utf-8")
        (tmp_path / "b.txt").write_text(weak_resume_text, encoding="utf-8")
        (tmp_path / "ignore.md").write_text("# notes", encoding="utf-8")

        assert main(["analyze", str(tmp_path), "--json", "-q"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 3  # .md is a supported suffix

    def test_missing_path_exits_non_zero(self, tmp_path, capsys):
        assert main(["analyze", str(tmp_path / "nope.txt"), "-q"]) == 1
        assert "No supported resume files" in capsys.readouterr().err

    def test_unreadable_file_is_reported_and_skipped(self, tmp_path, strong_resume_text, capsys):
        (tmp_path / "good.txt").write_text(strong_resume_text, encoding="utf-8")
        (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4\nnot really a pdf")

        # One file fails, the other still produces output, and the exit code says so.
        assert main(["analyze", str(tmp_path), "-q"]) == 1
        captured = capsys.readouterr()
        assert "bad.pdf" in captured.err
        assert "good.txt" in captured.out
