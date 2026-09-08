"""Parsing layer: sections, contact details, dates and entities."""

from __future__ import annotations

import pytest

from app.core.parsing.contact import extract_contact
from app.core.parsing.entities import (
    extract_education,
    extract_experience,
    parse_date_range,
    total_experience_months,
)
from app.core.parsing.model import DateRange, SectionKind
from app.core.parsing.normalize import clean_text, is_bullet, strip_bullet, tokenize
from app.core.parsing.parser import parse_resume
from app.core.parsing.sections import classify_heading, looks_like_heading, segment


class TestNormalize:
    @pytest.mark.parametrize(
        "line",
        ["• Built the thing", "- Built the thing", "* Built the thing", "1. Built the thing"],
    )
    def test_recognises_bullet_forms(self, line):
        assert is_bullet(line)
        assert "Built the thing" in strip_bullet(line)

    def test_prose_is_not_a_bullet(self):
        assert not is_bullet("Senior Engineer at Acme Corp")

    def test_tokenizer_preserves_technology_punctuation(self):
        tokens = tokenize("Skilled in C++, Node.js, CI/CD and .NET")
        assert "c++" in tokens
        assert "node.js" in tokens
        assert "ci/cd" in tokens

    def test_clean_text_collapses_whitespace_but_keeps_lines(self):
        cleaned = clean_text("A  \t B\r\n\r\n\r\n\r\nC")
        assert cleaned == "A B\n\nC"


class TestSections:
    def test_classifies_standard_headings(self):
        assert classify_heading("PROFESSIONAL EXPERIENCE") is SectionKind.EXPERIENCE
        assert classify_heading("Technical Skills") is SectionKind.SKILLS
        assert classify_heading("Education") is SectionKind.EDUCATION

    def test_classifies_decorated_heading(self):
        assert classify_heading("TECHNICAL SKILLS & TOOLS") is SectionKind.SKILLS

    def test_body_line_mentioning_a_vocabulary_word_is_not_a_heading(self):
        """Regression: this line contains "technologies" and was being read as a
        Skills heading, which truncated the Experience section to nothing."""
        line = "Senior Software Engineer | Razorline Technologies | Mar 2021 - Present"
        assert classify_heading(line) is None

    def test_skills_body_line_is_not_a_languages_heading(self):
        assert classify_heading("Languages: Python, Go, SQL, JavaScript, Bash") is None

    def test_long_sentence_is_not_a_heading(self):
        assert not looks_like_heading(
            "Led the migration of thirty-eight services to Kubernetes last year."
        )

    def test_header_block_is_kept_as_contact(self, strong_resume_text):
        sections, _ = segment(clean_text(strong_resume_text))
        assert SectionKind.CONTACT in sections
        assert "priya.raghavan@example.com" in sections[SectionKind.CONTACT].body

    def test_segments_all_core_sections(self, strong_resume_text):
        sections, _ = segment(clean_text(strong_resume_text))
        for kind in (SectionKind.EXPERIENCE, SectionKind.EDUCATION, SectionKind.SKILLS):
            assert kind in sections, f"missing {kind}"
        assert "Razorline" in sections[SectionKind.EXPERIENCE].body

    def test_repeated_headings_are_merged_not_overwritten(self):
        text = "EXPERIENCE\n- First role bullet\n\nEXPERIENCE\n- Second role bullet"
        sections, _ = segment(text)
        body = sections[SectionKind.EXPERIENCE].body
        assert "First role bullet" in body and "Second role bullet" in body

    def test_unknown_heading_is_reported(self):
        text = "Jane Doe\njane@example.com\n\nMy Journey So Far\n- Did a thing here\n"
        _, unknown = segment(text)
        assert "My Journey So Far" in unknown


class TestContact:
    def test_extracts_full_contact_block(self, strong_resume_text):
        contact = extract_contact(clean_text(strong_resume_text))
        assert contact.name == "Priya Raghavan"
        assert contact.email == "priya.raghavan@example.com"
        assert contact.phone is not None
        assert "linkedin.com/in/priyaraghavan" in (contact.linkedin or "")
        assert "github.com/praghavan" in (contact.github or "")
        assert contact.location == "Bengaluru, India"
        assert contact.has_minimum

    def test_uppercase_name_is_title_cased(self):
        contact = extract_contact("JOHN SMITH\njohn@example.com\n")
        assert contact.name == "John Smith"

    def test_date_range_is_not_mistaken_for_a_phone_number(self):
        contact = extract_contact("Jane Doe\njane@example.com\nExperience 2019 - 2022\n")
        assert contact.phone is None

    def test_falls_back_to_email_local_part_for_name(self):
        contact = extract_contact("Curriculum Vitae\njane.doe@example.com\n")
        assert contact.name == "Jane Doe"

    def test_missing_contact_is_detected(self):
        contact = extract_contact("SOMEONE\nNo way to reach them.\n")
        assert not contact.has_minimum


class TestDates:
    @pytest.mark.parametrize(
        ("text", "start_year", "start_month", "current"),
        [
            ("Mar 2021 - Present", 2021, 3, True),
            ("January 2019 – December 2021", 2019, 1, False),
            ("05/2019 - 07/2021", 2019, 5, False),
            ("2018 to 2020", 2018, None, False),
        ],
    )
    def test_parses_common_date_formats(self, text, start_year, start_month, current):
        parsed = parse_date_range(text)
        assert parsed is not None
        assert parsed.start_year == start_year
        assert parsed.start_month == start_month
        assert parsed.is_current is current

    def test_single_year_is_treated_as_a_point(self):
        parsed = parse_date_range("B.Tech Computer Science, 2016")
        assert parsed is not None and parsed.start_year == 2016

    def test_no_date_returns_none(self):
        assert parse_date_range("Senior Engineer at Acme") is None

    def test_duration_counts_inclusive_months(self):
        assert DateRange(start_year=2020, start_month=1, end_year=2020, end_month=12).months == 12

    def test_overlapping_roles_are_not_double_counted(self):
        from app.core.parsing.model import ExperienceEntry

        entries = [
            ExperienceEntry(dates=DateRange(2020, 1, 2022, 12)),
            ExperienceEntry(dates=DateRange(2021, 1, 2022, 12)),  # fully overlapping
        ]
        # Three calendar years, not five.
        assert 34 <= total_experience_months(entries) <= 36


class TestEntities:
    def test_splits_roles_and_attaches_bullets(self, strong_resume_text):
        parsed = parse_resume(strong_resume_text)
        assert len(parsed.experience) == 3
        first = parsed.experience[0]
        assert first.dates is not None and first.dates.is_current
        assert len(first.bullets) == 4
        assert "Razorline" in (first.organization or "")
        assert "Engineer" in (first.title or "")

    def test_computes_years_of_experience(self, strong_resume_text):
        parsed = parse_resume(strong_resume_text)
        assert parsed.years_of_experience >= 8.0

    def test_undated_section_still_yields_one_entry(self):
        entries = extract_experience("Developer at TechCorp\n- Did some work here\n")
        assert len(entries) == 1
        assert entries[0].bullets == ["Did some work here"]

    def test_parses_degree_institution_and_gpa(self):
        entries = extract_education(
            "B.Tech in Computer Science, National Institute of Technology Trichy, 2016\nGPA: 8.6/10"
        )
        assert len(entries) == 1
        entry = entries[0]
        assert entry.degree is not None and "tech" in entry.degree.lower()
        assert "Institute of Technology" in (entry.institution or "")
        # 8.6 on a 10-point scale normalises onto the 4.0 scale.
        assert entry.gpa is not None and 3.3 <= entry.gpa <= 3.5

    def test_full_parse_reports_missing_sections(self, weak_resume_text):
        parsed = parse_resume(weak_resume_text)
        assert SectionKind.SKILLS in parsed.missing_core_sections
