"""HTTP contract: endpoints, error envelope, persistence and ops routes."""

from __future__ import annotations

import io
import zipfile

import pytest

from app.api.deps import get_limiter


def _upload(text: str, name: str = "resume.txt") -> dict:
    return {"file": (name, io.BytesIO(text.encode("utf-8")), "text/plain")}


class TestOps:
    def test_health_is_cheap_and_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]

    def test_readiness_reports_each_dependency(self, client):
        body = client.get("/health/ready").json()
        assert body["checks"]["database"] == "ok"
        assert body["checks"]["taxonomy"] == "ok"
        # The LLM is unconfigured in tests and must NOT fail readiness.
        assert body["checks"]["llm"] in {"unconfigured", "disabled"}
        assert body["status"] == "ok"

    def test_metrics_render_in_prometheus_format(self, client):
        client.get("/health")
        text = client.get("/metrics").text
        assert "resumeiq_http_requests_total" in text
        assert "# TYPE" in text

    def test_metrics_json_snapshot(self, client):
        client.get("/health")
        assert "counters" in client.get("/metrics/json").json()

    def test_every_response_carries_a_request_id(self, client):
        assert client.get("/health").headers["x-request-id"]

    def test_supplied_request_id_is_echoed(self, client):
        response = client.get("/health", headers={"x-request-id": "trace-abc-123"})
        assert response.headers["x-request-id"] == "trace-abc-123"

    def test_openapi_is_published(self, client):
        schema = client.get("/openapi.json").json()
        assert "/api/v1/analyses" in schema["paths"]


class TestAnalysisEndpoint:
    def test_analyses_an_uploaded_resume(self, client, strong_resume_text):
        response = client.post("/api/v1/analyses", files=_upload(strong_resume_text))
        assert response.status_code == 201

        body = response.json()
        assert body["id"]
        assert 0 <= body["overall_score"] <= 100
        assert body["band"] in {"excellent", "strong", "fair", "weak", "poor"}
        assert body["contact"]["email"] == "priya.raghavan@example.com"
        assert body["years_of_experience"] >= 8
        assert len(body["experience"]) == 3
        assert body["skills"]
        assert body["dimensions"]
        assert body["summary"]

    def test_dimensions_carry_their_signals(self, client, strong_resume_text):
        body = client.post("/api/v1/analyses", files=_upload(strong_resume_text)).json()
        ats = next(d for d in body["dimensions"] if d["id"] == "ats")
        assert ats["signals"]
        signal = ats["signals"][0]
        assert {"id", "label", "score", "weight", "severity"} <= set(signal)

    def test_match_is_absent_without_a_job_description(self, client, strong_resume_text):
        body = client.post("/api/v1/analyses", files=_upload(strong_resume_text)).json()
        assert body["match"] is None
        assert body["job"] is None
        assert not any(d["id"] == "match" for d in body["dimensions"])

    def test_match_is_present_with_a_job_description(self, client, strong_resume_text, job_text):
        response = client.post(
            "/api/v1/analyses",
            files=_upload(strong_resume_text),
            data={"job_description": job_text, "company": "Northwind"},
        )
        body = response.json()
        assert body["match"] is not None
        assert body["match"]["required_coverage"] > 0.5
        assert body["job"]["seniority"] == "senior"
        assert any(d["id"] == "match" for d in body["dimensions"])

    def test_recommendations_are_ranked_and_actionable(self, client, weak_resume_text):
        body = client.post("/api/v1/analyses", files=_upload(weak_resume_text)).json()
        recommendations = body["recommendations"]
        assert recommendations
        assert all(r["action"] for r in recommendations)
        assert recommendations[0]["severity"] in {"critical", "warning"}

    def test_ai_feedback_is_absent_but_explained_when_unconfigured(
        self, client, strong_resume_text
    ):
        body = client.post("/api/v1/analyses", files=_upload(strong_resume_text)).json()
        assert body["ai_feedback"]["available"] is False
        assert body["ai_feedback"]["content"] is None
        assert body["ai_feedback"]["status"]
        # The deterministic analysis must be complete regardless.
        assert body["overall_score"] > 0

    def test_ai_can_be_declined_per_request(self, client, strong_resume_text):
        body = client.post(
            "/api/v1/analyses", files=_upload(strong_resume_text), data={"use_ai": "false"}
        ).json()
        assert "not requested" in body["ai_feedback"]["status"]

    def test_text_endpoint_matches_the_upload_endpoint(self, client, strong_resume_text):
        uploaded = client.post("/api/v1/analyses", files=_upload(strong_resume_text)).json()
        posted = client.post(
            "/api/v1/analyses/text", data={"resume_text": strong_resume_text}
        ).json()
        assert uploaded["overall_score"] == posted["overall_score"]

    def test_docx_upload_is_accepted(self, client):
        docx = pytest.importorskip("docx")
        document = docx.Document()
        document.add_paragraph("Jane Doe")
        document.add_paragraph("jane.doe@example.com | +1 555 0100")
        document.add_paragraph("EXPERIENCE")
        document.add_paragraph("Engineer | Acme | Jan 2020 - Present")
        document.add_paragraph("Built Python services", style="List Bullet")
        buffer = io.BytesIO()
        document.save(buffer)

        response = client.post(
            "/api/v1/analyses",
            files={
                "file": (
                    "resume.docx",
                    buffer.getvalue(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert response.status_code == 201
        assert response.json()["document"]["source_format"] == "docx"


class TestErrorContract:
    @staticmethod
    def _assert_envelope(response, code: str, status: int):
        assert response.status_code == status
        error = response.json()["error"]
        assert error["code"] == code
        assert error["message"]
        assert error["request_id"]

    def test_empty_upload(self, client):
        self._assert_envelope(
            client.post("/api/v1/analyses", files=_upload("")), "empty_document", 422
        )

    def test_unsupported_file_type(self, client):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("image.png", b"binary")
        response = client.post(
            "/api/v1/analyses", files={"file": ("photo.zip", buffer.getvalue(), "application/zip")}
        )
        self._assert_envelope(response, "unsupported_file_type", 415)

    def test_legacy_doc_is_rejected_with_guidance(self, client):
        response = client.post(
            "/api/v1/analyses",
            files={"file": ("old.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1data", "application/msword")},
        )
        self._assert_envelope(response, "unsupported_file_type", 415)
        assert ".docx" in response.json()["error"]["message"]

    def test_missing_file_is_a_validation_error(self, client):
        self._assert_envelope(client.post("/api/v1/analyses"), "validation_error", 422)

    def test_unknown_analysis_id(self, client):
        self._assert_envelope(client.get("/api/v1/analyses/doesnotexist"), "not_found", 404)

    def test_empty_job_text_is_rejected(self, client):
        response = client.post("/api/v1/jobs/parse", json={"text": "   "})
        assert response.status_code in (400, 422)


class TestPersistence:
    def test_analysis_is_retrievable_by_id(self, client, strong_resume_text):
        created = client.post("/api/v1/analyses", files=_upload(strong_resume_text)).json()
        fetched = client.get(f"/api/v1/analyses/{created['id']}").json()
        assert fetched["id"] == created["id"]
        assert fetched["overall_score"] == created["overall_score"]
        assert fetched["recommendations"] == created["recommendations"]

    def test_history_lists_most_recent_first(self, client, strong_resume_text, weak_resume_text):
        client.post("/api/v1/analyses", files=_upload(weak_resume_text, "weak.txt"))
        client.post("/api/v1/analyses", files=_upload(strong_resume_text, "strong.txt"))

        body = client.get("/api/v1/analyses", params={"limit": 5}).json()
        assert body["total"] >= 2
        assert body["items"][0]["filename"] == "strong.txt"

    def test_pagination_parameters_are_validated(self, client):
        assert client.get("/api/v1/analyses", params={"limit": 0}).status_code == 422
        assert client.get("/api/v1/analyses", params={"limit": 500}).status_code == 422

    def test_delete_removes_the_record(self, client, strong_resume_text):
        created = client.post("/api/v1/analyses", files=_upload(strong_resume_text)).json()
        assert client.delete(f"/api/v1/analyses/{created['id']}").status_code == 204
        assert client.get(f"/api/v1/analyses/{created['id']}").status_code == 404

    def test_delete_is_not_silently_idempotent(self, client):
        assert client.delete("/api/v1/analyses/nope").status_code == 404


class TestJobsEndpoint:
    def test_parses_a_posting(self, client, job_text):
        body = client.post("/api/v1/jobs/parse", json={"text": job_text}).json()
        job = body["job"]
        assert "Python" in job["required_skills"]
        assert "Go" in job["preferred_skills"]
        assert job["min_years_experience"] == 6
        assert body["responsibilities"]

    def test_enforces_a_length_cap(self, client):
        response = client.post("/api/v1/jobs/parse", json={"text": "x" * 40_000})
        assert response.status_code == 422


class TestRateLimiting:
    def test_returns_429_with_retry_hint_when_exceeded(self, client, monkeypatch):
        limiter = get_limiter()
        monkeypatch.setattr(limiter, "limit", 2)
        limiter.reset()
        try:
            payload = {"text": "We need a Python engineer with Kubernetes experience."}
            statuses = [
                client.post("/api/v1/jobs/parse", json=payload).status_code for _ in range(4)
            ]
            assert 429 in statuses
            last = client.post("/api/v1/jobs/parse", json=payload)
            assert last.json()["error"]["details"]["retry_after_seconds"] >= 1
        finally:
            limiter.reset()

    def test_read_endpoints_are_not_rate_limited(self, client, monkeypatch):
        limiter = get_limiter()
        monkeypatch.setattr(limiter, "limit", 1)
        limiter.reset()
        try:
            assert all(client.get("/health").status_code == 200 for _ in range(5))
        finally:
            limiter.reset()
