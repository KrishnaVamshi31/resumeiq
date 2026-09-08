"""End-to-end test of the Streamlit dashboard against a live API.

`AppTest` runs the app script in-process, so this exercises the real render
path - every widget, tab and dataframe - without a browser. A real uvicorn
server is started in a thread because the UI is deliberately a thin HTTP client
with no in-process access to the scoring engine.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn

from app.main import create_app

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_FILE = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"
STARTUP_TIMEOUT = 30.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_api() -> Iterator[str]:
    """Run the API on a random port for the lifetime of this module."""
    port = _free_port()
    config = uvicorn.Config(
        create_app(), host="127.0.0.1", port=port, log_level="error", log_config=None
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if server.started:
            try:
                if httpx.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
        time.sleep(0.1)
    else:
        server.should_exit = True
        pytest.fail("The API server did not start in time.")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _run_app(api_url: str, resume: str = "", job: str = "") -> AppTest:
    at = AppTest.from_file(str(APP_FILE), default_timeout=180)
    at.run()
    # Sidebar "API URL" is the third text input on the page.
    at.text_input[2].set_value(api_url)
    if resume:
        at.text_area[0].set_value(resume)
    if job:
        at.text_area[1].set_value(job)
    at.button[0].click().run()
    return at


class TestDashboard:
    def test_initial_render_has_no_errors(self, live_api):
        at = AppTest.from_file(str(APP_FILE), default_timeout=60)
        at.run()
        assert not at.exception
        # Nothing analysed yet, so the app should prompt and stop.
        assert any("Upload a resume" in info.value for info in at.info)

    def test_full_analysis_renders_every_section(self, live_api, strong_resume_text, job_text):
        at = _run_app(live_api, strong_resume_text, job_text)

        assert not at.exception
        assert not at.error
        assert at.session_state["result"] is not None

        labels = {metric.label for metric in at.metric}
        assert {"ATS compatibility", "Content quality", "Structure & completeness"} <= labels
        assert {"Match score", "Required coverage", "Gaps"} <= labels

        assert len(at.tabs) == 6
        assert at.dataframe, "skills and gap tables should render"
        assert any("/ 100" in block.value for block in at.markdown)

    def test_analysis_without_a_job_hides_match_section(self, live_api, strong_resume_text):
        at = _run_app(live_api, strong_resume_text)

        assert not at.exception
        assert at.session_state["result"]["match"] is None
        assert "Match score" not in {metric.label for metric in at.metric}
        assert any("job description" in info.value.lower() for info in at.info)

    def test_submitting_nothing_is_rejected(self, live_api):
        at = _run_app(live_api)
        assert not at.exception
        assert any("Upload a resume file or paste" in error.value for error in at.error)

    def test_unreachable_api_is_reported_not_crashed(self, strong_resume_text):
        at = _run_app(f"http://127.0.0.1:{_free_port()}", strong_resume_text)
        assert not at.exception
        assert any("Could not reach the API" in error.value for error in at.error)

    def test_ai_tab_explains_its_absence(self, live_api, strong_resume_text):
        """With no API key configured, the UI must say so rather than look broken."""
        at = _run_app(live_api, strong_resume_text)
        feedback = at.session_state["result"]["ai_feedback"]
        assert feedback["available"] is False
        assert feedback["status"]
