"""ResumeIQ dashboard.

A thin client over the ResumeIQ API: it renders what the service returns and
holds no scoring logic of its own, so the UI can never disagree with the API.

    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

# In Docker the API is another service; locally it is on localhost.
DEFAULT_API_URL = os.environ.get("RESUMEIQ_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180.0

SEVERITY_COLOR = {"critical": "red", "warning": "orange", "info": "blue", "ok": "green"}
SEVERITY_ICON = {
    "critical": ":material/error:",
    "warning": ":material/warning:",
    "info": ":material/info:",
    "ok": ":material/check_circle:",
}
BAND_COLOR = {
    "excellent": "green",
    "strong": "green",
    "fair": "orange",
    "weak": "red",
    "poor": "red",
}

st.set_page_config(
    page_title="ResumeIQ",
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_client(base_url: str) -> httpx.Client:
    """One pooled HTTP client per API URL, reused across reruns."""
    return httpx.Client(base_url=base_url, timeout=REQUEST_TIMEOUT)


def api_error_message(response: httpx.Response) -> str:
    try:
        error = response.json()["error"]
        return f"{error['message']} (code: {error['code']})"
    except Exception:
        return f"Request failed with status {response.status_code}."


if "result" not in st.session_state:
    st.session_state.result = None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("ResumeIQ", width="stretch")
    st.caption("AI resume intelligence and job matching")

    api_url = st.text_input("API URL", value=DEFAULT_API_URL, help="Where the ResumeIQ API is running.")
    use_ai = st.toggle(
        "AI coaching",
        value=True,
        help="Adds narrative feedback and rewrites. Requires an ANTHROPIC_API_KEY on the server.",
    )

    st.divider()
    with st.container(border=True):
        st.markdown("**Service status**")
        try:
            health = get_client(api_url).get("/health/ready", timeout=5.0).json()
            for name, state in health.get("checks", {}).items():
                color = "green" if state == "ok" else "orange" if state != "error" else "red"
                st.badge(f"{name}: {state}", color=color)
        except Exception:
            st.badge("unreachable", icon=":material/cloud_off:", color="red")
            st.caption("Start the API with `uvicorn app.main:app --reload`.")


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

st.title("Resume analysis", width="stretch")
st.caption(
    "Scores are computed deterministically from the document itself. "
    "AI coaching, when enabled, explains them - it never changes them."
)

with st.form("analysis_form", border=True):
    left, right = st.columns([1, 1])

    with left:
        uploaded = st.file_uploader(
            "Resume", type=["pdf", "docx", "txt", "md"], help="PDF, DOCX or plain text."
        )
        pasted = st.text_area(
            "...or paste resume text",
            height=140,
            placeholder="Paste the resume here if you do not have a file to hand.",
        )

    with right:
        job_description = st.text_area(
            "Job description (optional)",
            height=220,
            placeholder="Paste the full posting to unlock job matching and gap analysis.",
        )
        with st.container(horizontal=True):
            job_title = st.text_input("Job title", placeholder="Senior Backend Engineer")
            company = st.text_input("Company", placeholder="Northwind Financial")

    submitted = st.form_submit_button(
        "Analyse resume", icon=":material/play_arrow:", type="primary", width="stretch"
    )

if submitted:
    if not uploaded and not pasted.strip():
        st.error("Upload a resume file or paste the text.", icon=":material/error:")
    else:
        data: dict[str, Any] = {"use_ai": str(use_ai).lower()}
        if job_description.strip():
            data["job_description"] = job_description
        if job_title.strip():
            data["job_title"] = job_title
        if company.strip():
            data["company"] = company

        client = get_client(api_url)
        spinner_text = "Analysing... (AI coaching can take a few seconds)" if use_ai else "Analysing..."
        try:
            with st.spinner(spinner_text):
                if uploaded is not None:
                    response = client.post(
                        "/api/v1/analyses",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        data=data,
                    )
                else:
                    response = client.post(
                        "/api/v1/analyses/text", data={"resume_text": pasted, **data}
                    )
            if response.status_code == 201:
                st.session_state.result = response.json()
            else:
                st.session_state.result = None
                st.error(api_error_message(response), icon=":material/error:")
        except httpx.HTTPError as exc:
            st.session_state.result = None
            st.error(f"Could not reach the API at {api_url}: {exc}", icon=":material/cloud_off:")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

result = st.session_state.result
if result is None:
    st.info(
        "Upload a resume to begin. Adding a job description unlocks match scoring and gap analysis.",
        icon=":material/lightbulb:",
    )
    st.stop()

st.divider()

score = result["overall_score"]
band = result["band"]

with st.container(border=True):
    header, gauge = st.columns([2, 1])
    with header:
        st.subheader("Overall score", divider=False)
        st.badge(band.title(), color=BAND_COLOR.get(band, "gray"), icon=":material/verified:")
        st.markdown(f"### {score} / 100")
        st.write(result["summary"])
    with gauge:
        st.progress(int(score), text=f"{score}/100")
        document = result["document"]
        st.caption(
            f"{document['source_format'].upper()} · {document['page_count']} page(s) · "
            f"{document['word_count']} words · {result['years_of_experience']} yrs experience"
        )

with st.container(horizontal=True):
    for dimension in result["dimensions"]:
        st.metric(
            dimension["label"],
            f"{dimension['score']:.0f}",
            delta=f"weight {dimension['weight']:.0%}",
            delta_color="off",
            border=True,
            help=f"Severity: {dimension['severity']}",
        )

for warning in result["document"]["warnings"]:
    st.warning(warning, icon=":material/warning:")

tab_labels = ["Fix list", "Job match", "Skills", "Parsed resume", "AI coaching", "All checks"]
fixes_tab, match_tab, skills_tab, parsed_tab, ai_tab, checks_tab = st.tabs(tab_labels)


with fixes_tab:
    recommendations = result["recommendations"]
    if not recommendations:
        st.success("No issues found. This resume passes every check.", icon=":material/thumb_up:")
    else:
        st.caption(
            "Ranked by how many points on the overall score each fix recovers, "
            "so the top item is the best use of your time."
        )
        for index, item in enumerate(recommendations, start=1):
            severity = item["severity"]
            with st.container(border=True):
                head = st.container(horizontal=True)
                with head:
                    st.badge(
                        severity,
                        color=SEVERITY_COLOR.get(severity, "gray"),
                        icon=SEVERITY_ICON.get(severity),
                    )
                    st.badge(f"+{item['impact_points']:.2f} pts", color="violet")
                st.markdown(f"**{index}. {item['title']}**")
                st.write(item["action"])
                if item["evidence"]:
                    with st.expander("Evidence", icon=":material/find_in_page:"):
                        for line in item["evidence"]:
                            st.markdown(f"- {line}")

    if result["strengths"]:
        with st.expander("What this resume already does well", icon=":material/check_circle:"):
            for strength in result["strengths"]:
                st.markdown(f"- {strength}")


with match_tab:
    match = result.get("match")
    if not match:
        st.info(
            "Paste a job description above to see required-skill coverage and gap analysis.",
            icon=":material/work:",
        )
    else:
        job = result["job"]
        with st.container(horizontal=True):
            st.metric("Match score", f"{match['score']:.0f}", border=True)
            st.metric("Required coverage", f"{match['required_coverage']:.0%}", border=True)
            st.metric("Gaps", len(match["gaps"]), border=True)

        st.caption(
            f"Posting read as **{job['seniority']}** level"
            + (f", minimum {job['min_years_experience']} years" if job["min_years_experience"] else "")
            + (f" · {job['title']}" if job["title"] else "")
        )

        if match["gaps"]:
            st.subheader("Skill gaps", divider="gray")
            gaps = pd.DataFrame(match["gaps"])
            gaps["adjacent_owned"] = gaps["adjacent_owned"].apply(lambda names: ", ".join(names))
            st.dataframe(
                gaps,
                hide_index=True,
                width="stretch",
                column_config={
                    "name": st.column_config.TextColumn("Skill"),
                    "category": st.column_config.TextColumn("Category"),
                    "required": st.column_config.CheckboxColumn("Required"),
                    "priority": st.column_config.TextColumn("Priority"),
                    "adjacent_owned": st.column_config.TextColumn(
                        "You already have", help="Related skills you can use to frame the gap."
                    ),
                },
            )

        covered, absent = st.columns(2)
        with covered, st.container(border=True):
            st.markdown("**Required skills you evidence**")
            st.write(", ".join(match["matched_required"]) or "None found.")
        with absent, st.container(border=True):
            st.markdown("**Posting terms missing from your resume**")
            st.write(", ".join(match["missing_keywords"]) or "None - good coverage.")

        if match["surplus_skills"]:
            with st.expander("Skills you have that this posting does not ask for"):
                st.write(", ".join(match["surplus_skills"]))


with skills_tab:
    skills = result["skills"]
    if not skills:
        st.warning("No recognised skills were found.", icon=":material/warning:")
    else:
        frame = pd.DataFrame(skills)
        st.caption(
            "Confidence reflects *where* a skill appears: demonstrated inside a work "
            "bullet counts for more than listed in a keyword dump."
        )
        st.dataframe(
            frame[["name", "category", "confidence", "mentions", "demonstrated"]],
            hide_index=True,
            width="stretch",
            column_config={
                "name": st.column_config.TextColumn("Skill"),
                "category": st.column_config.TextColumn("Category"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "mentions": st.column_config.NumberColumn("Mentions"),
                "demonstrated": st.column_config.CheckboxColumn("Shown in context"),
            },
        )
        st.bar_chart(
            frame.groupby("category").size().reset_index(name="count"),
            x="category",
            y="count",
            horizontal=True,
        )


with parsed_tab:
    st.caption("What the parser extracted. If anything here is wrong, an ATS will get it wrong too.")
    contact = result["contact"]
    with st.container(border=True):
        st.markdown("**Contact**")
        st.table(
            pd.DataFrame(
                [{"Field": k.title(), "Value": v or "not found"} for k, v in contact.items()]
            ).set_index("Field")
        )

    if result["experience"]:
        st.subheader("Experience", divider="gray")
        for entry in result["experience"]:
            dates = entry.get("dates") or {}
            period = dates.get("raw") or "dates not parsed"
            with st.expander(
                f"{entry['title'] or 'Untitled role'} — {entry['organization'] or 'unknown employer'}"
                f"  ({period})",
                icon=":material/work_history:",
            ):
                for bullet in entry["bullets"]:
                    st.markdown(f"- {bullet}")
                if not entry["bullets"]:
                    st.caption("No bullet points detected for this role.")

    if result["education"]:
        st.subheader("Education", divider="gray")
        st.dataframe(pd.DataFrame(result["education"]).drop(columns=["dates"]),
                     hide_index=True, width="stretch")


with ai_tab:
    feedback = result["ai_feedback"]
    if not feedback["available"]:
        st.info(feedback["status"], icon=":material/smart_toy:")
    else:
        content = feedback["content"]
        st.markdown(f"### {content['headline']}")
        usage = content["usage"]
        st.caption(
            f"{content['model']} · {usage['input_tokens']} in / {usage['output_tokens']} out"
            f" · {usage['cache_read_tokens']} cached · {usage['latency_ms']:.0f} ms"
        )

        if content["integrity_notes"]:
            for note in content["integrity_notes"]:
                st.warning(note, icon=":material/policy:")
        if content["dropped_rewrites"]:
            for note in content["dropped_rewrites"]:
                st.error(note, icon=":material/block:")
        if content["injection_findings"]:
            with st.expander("Prompt-injection attempts detected in the document", icon=":material/security:"):
                for finding in content["injection_findings"]:
                    st.code(finding, language=None)

        if content["strengths"]:
            with st.container(border=True):
                st.markdown("**Strengths**")
                for strength in content["strengths"]:
                    st.markdown(f"- {strength}")

        if content["priority_fixes"]:
            st.subheader("Priority fixes", divider="gray")
            for fix in content["priority_fixes"]:
                with st.container(border=True):
                    st.markdown(f"**{fix['target']}**")
                    st.write(fix["problem"])
                    st.success(fix["fix"], icon=":material/edit:")
                    st.caption(fix["why_it_matters"])

        if content["bullet_rewrites"]:
            st.subheader("Suggested rewrites", divider="gray")
            for rewrite in content["bullet_rewrites"]:
                with st.container(border=True):
                    before, after = st.columns(2)
                    with before:
                        st.caption("Before")
                        st.write(rewrite["original"])
                    with after:
                        st.caption("After")
                        st.write(rewrite["improved"])
                    st.caption(rewrite["rationale"])

        if content["summary_rewrite"]:
            with st.container(border=True):
                st.markdown("**Suggested summary**")
                st.code(content["summary_rewrite"], language=None, wrap_lines=True)

        if content["interview_risks"]:
            with st.expander("Questions this resume invites", icon=":material/help:"):
                for risk in content["interview_risks"]:
                    st.markdown(f"- {risk}")


with checks_tab:
    st.caption("Every signal behind the scores, including the ones that passed.")
    for dimension in result["dimensions"]:
        with st.expander(
            f"{dimension['label']} — {dimension['score']:.0f}/100",
            icon=SEVERITY_ICON.get(dimension["severity"]),
        ):
            rows = [
                {
                    "Check": signal["label"],
                    "Score": signal["score"],
                    "Weight": signal["weight"],
                    "Severity": signal["severity"],
                    "Detail": signal["detail"] or "",
                }
                for signal in dimension["signals"]
            ]
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )
