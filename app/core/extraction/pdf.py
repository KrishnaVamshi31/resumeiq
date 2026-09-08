"""PDF text extraction with layout diagnostics.

We deliberately capture more than the text: how much text each page yielded,
whether pages carry images, and whether the line-length distribution suggests a
two-column layout. Those are exactly the traits that make a resume parse badly
in an applicant tracking system, and they are unrecoverable after flattening.
"""

from __future__ import annotations

from contextlib import suppress
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.extraction.base import ExtractedDocument, PageStats, SourceFormat
from app.errors import ExtractionFailed
from app.logging_conf import get_logger

logger = get_logger(__name__)

# A page of a normal single-column resume yields ~1500-3500 characters. Far
# less usually means the text is an image or trapped in vector art.
LOW_TEXT_PAGE_THRESHOLD = 200
SHORT_LINE_CHARS = 35
MULTI_COLUMN_SHORT_LINE_RATIO = 0.55


def extract_pdf(data: bytes) -> ExtractedDocument:
    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as exc:
        raise ExtractionFailed(f"The PDF could not be parsed: {exc}") from exc
    except Exception as exc:  # pypdf raises assorted low-level errors
        raise ExtractionFailed("The PDF appears to be corrupt or unsupported.") from exc

    warnings: list[str] = []
    encrypted = bool(getattr(reader, "is_encrypted", False))
    if encrypted:
        # Many resumes are "encrypted" with an empty owner password, which only
        # restricts editing. That case decrypts cleanly and is worth retrying.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ExtractionFailed(
                "The PDF is password protected. Upload an unprotected copy."
            ) from exc
        warnings.append(
            "PDF carries encryption/permission flags; some ATS parsers reject such files."
        )

    page_texts: list[str] = []
    pages: list[PageStats] = []
    image_count = 0

    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            logger.warning("pdf_page_extract_failed", extra={"page": index})
            text = ""

        text = _normalise(text)
        page_texts.append(text)

        lines = [ln for ln in text.splitlines() if ln.strip()]
        short_lines = sum(1 for ln in lines if len(ln.strip()) <= SHORT_LINE_CHARS)
        pages.append(
            PageStats(
                index=index,
                char_count=len(text),
                line_count=len(lines),
                short_line_ratio=(short_lines / len(lines)) if lines else 0.0,
                has_extractable_text=len(text.strip()) >= LOW_TEXT_PAGE_THRESHOLD,
            )
        )

        # Image enumeration depends on optional codecs; never fail the parse for it.
        with suppress(Exception):
            image_count += len(page.images)

    full_text = "\n".join(page_texts).strip()
    if not full_text:
        raise ExtractionFailed(
            "No selectable text found. The PDF is likely a scan or an exported image - "
            "ATS parsers will read it as blank. Export from the original document instead."
        )

    dead_pages = sum(1 for p in pages if not p.has_extractable_text)
    if dead_pages:
        warnings.append(
            f"{dead_pages} of {len(pages)} page(s) yielded almost no text; "
            "content may be embedded as an image."
        )

    # Multi-column layouts flatten into a stream of short fragments.
    text_pages = [p for p in pages if p.has_extractable_text]
    multi_column = bool(text_pages) and (
        sum(p.short_line_ratio for p in text_pages) / len(text_pages)
        >= MULTI_COLUMN_SHORT_LINE_RATIO
    )
    if multi_column:
        warnings.append(
            "Line lengths suggest a multi-column layout; ATS parsers frequently "
            "interleave columns into unreadable text."
        )

    return ExtractedDocument(
        text=full_text,
        source_format=SourceFormat.PDF,
        page_count=max(len(reader.pages), 1),
        pages=pages,
        warnings=warnings,
        has_images=image_count > 0,
        image_count=image_count,
        has_multi_column_hint=multi_column,
        is_encrypted=encrypted,
    )


def _normalise(text: str) -> str:
    """Repair the artefacts PDF text extraction reliably introduces."""
    replacements = {
        "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "•": "- ", " ": " ",
        "": "- ", "": "- ", "●": "- ", "▪": "- ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Collapse the ragged trailing whitespace pypdf leaves on every line.
    return "\n".join(line.rstrip() for line in text.splitlines())
