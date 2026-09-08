"""Extraction layer: format detection, guards, and DOCX/PDF handling."""

from __future__ import annotations

import io
import zipfile

import pytest

from app.core.extraction.base import SourceFormat
from app.core.extraction.detector import detect_format, extract
from app.errors import EmptyDocument, ExtractionFailed, FileTooLarge, UnsupportedFileType


class TestDetection:
    def test_detects_pdf_by_magic_bytes(self):
        assert detect_format(b"%PDF-1.7\nrest") is SourceFormat.PDF

    def test_detects_plain_text(self):
        assert detect_format(b"Jane Doe\nSoftware Engineer\n", "resume.txt") is SourceFormat.TXT

    def test_content_beats_a_lying_extension(self):
        """A .txt name on real PDF bytes must still be read as a PDF."""
        assert detect_format(b"%PDF-1.4\n...", "resume.txt") is SourceFormat.PDF

    def test_rejects_legacy_doc(self):
        with pytest.raises(UnsupportedFileType, match="Legacy"):
            detect_format(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1payload", "old.doc")

    def test_rejects_rtf(self):
        with pytest.raises(UnsupportedFileType, match="RTF"):
            detect_format(rb"{\rtf1\ansi", "resume.rtf")

    def test_rejects_non_word_zip(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("photo.png", b"not a document")
        with pytest.raises(UnsupportedFileType, match="not a Word document"):
            detect_format(buffer.getvalue(), "archive.zip")

    def test_rejects_pdf_extension_on_binary_junk(self):
        with pytest.raises(UnsupportedFileType):
            detect_format(b"\x00\x01\x02\x03\x04binary", "resume.pdf")


class TestExtractGuards:
    def test_enforces_size_limit(self):
        with pytest.raises(FileTooLarge) as info:
            extract(b"x" * 2048, "big.txt", max_bytes=1024)
        assert info.value.details["limit_bytes"] == 1024

    def test_rejects_empty_upload(self):
        with pytest.raises(EmptyDocument):
            extract(b"   \n  ", "empty.txt")

    def test_truncates_to_max_chars_and_warns(self):
        document = extract(("word " * 5000).encode(), "long.txt", max_chars=200)
        assert len(document.text) == 200
        assert any("truncated" in w.lower() for w in document.warnings)

    def test_txt_extraction_reports_format_warning(self):
        document = extract(b"Jane Doe\nEngineer\nBuilt things.", "cv.txt")
        assert document.source_format is SourceFormat.TXT
        assert document.warnings, "plain text should carry a formatting caveat"

    def test_decodes_cp1252_bytes(self):
        """Windows-encoded resumes are common; they must not blow up."""
        document = extract("Café Manager – Paris".encode("cp1252"), "cv.txt")
        assert "Manager" in document.text


class TestDocx:
    @staticmethod
    def _docx_bytes(build) -> bytes:
        docx = pytest.importorskip("docx")
        document = docx.Document()
        build(document)
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()

    def test_reads_paragraphs_and_flags_tables(self):
        def build(document):
            document.add_paragraph("Jane Doe")
            document.add_paragraph("Senior Engineer")
            table = document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "Python"
            table.cell(0, 1).text = "Kubernetes"

        payload = self._docx_bytes(build)
        result = extract(payload, "resume.docx")

        assert result.source_format is SourceFormat.DOCX
        assert "Jane Doe" in result.text
        # Table content must still reach the matcher even though it is a defect.
        assert "Python" in result.text
        assert result.has_tables and result.table_count == 1
        assert any("table" in w.lower() for w in result.warnings)

    def test_flags_header_content(self):
        def build(document):
            document.add_paragraph("Body text that is long enough to survive.")
            document.sections[0].header.paragraphs[0].text = "jane@example.com"

        result = extract(self._docx_bytes(build), "resume.docx")
        assert result.has_text_in_headers_footers
        assert any("header" in w.lower() for w in result.warnings)

    def test_list_paragraphs_become_bullets(self):
        def build(document):
            document.add_paragraph("Experience")
            document.add_paragraph("Shipped the payments service", style="List Bullet")

        result = extract(self._docx_bytes(build), "resume.docx")
        assert "- Shipped the payments service" in result.text

    def test_corrupt_docx_raises_extraction_failed(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", b"<not-valid-ooxml>")
        with pytest.raises(ExtractionFailed):
            extract(buffer.getvalue(), "broken.docx")


class TestPdf:
    def test_image_only_pdf_is_rejected_with_actionable_message(self):
        """A scan has no text layer; the error must say what to do about it."""
        pypdf = pytest.importorskip("pypdf")
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buffer = io.BytesIO()
        writer.write(buffer)

        with pytest.raises(ExtractionFailed) as info:
            extract(buffer.getvalue(), "scan.pdf")
        assert "scan" in str(info.value).lower()
