"""File-type detection and the single entry point into extraction.

The declared filename and the browser-supplied content type are both attacker-
controlled and routinely wrong, so detection is driven by magic bytes first and
only falls back to the extension.
"""

from __future__ import annotations

from app.core.extraction.base import ExtractedDocument, PageStats, SourceFormat
from app.core.extraction.docx import extract_docx
from app.core.extraction.pdf import extract_pdf
from app.errors import EmptyDocument, FileTooLarge, UnsupportedFileType
from app.observability import track

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
LEGACY_DOC_MAGIC = b"\xd0\xcf\x11\xe0"  # OLE2 compound file: .doc, .xls, .ppt
RTF_MAGIC = b"{\\rtf"

# Only what an ATS can reliably read. Rejecting the rest is itself useful advice.
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt", ".md"})


def detect_format(data: bytes, filename: str | None = None) -> SourceFormat:
    """Identify the document type, preferring content over the filename."""
    if data.startswith(PDF_MAGIC):
        return SourceFormat.PDF
    if data.startswith(ZIP_MAGIC):
        # Any OOXML file is a zip; confirm it is specifically a Word document.
        if b"word/document.xml" in data[:8192] or b"word/" in data:
            return SourceFormat.DOCX
        raise UnsupportedFileType(
            "The file is a zip archive but not a Word document. Upload a PDF, DOCX or TXT."
        )
    if data.startswith(LEGACY_DOC_MAGIC):
        raise UnsupportedFileType(
            "Legacy .doc files are not supported and parse poorly in most ATS. "
            "Re-save as .docx or PDF."
        )
    if data.startswith(RTF_MAGIC):
        raise UnsupportedFileType("RTF is not supported. Re-save as .docx or PDF.")

    suffix = _suffix(filename)
    if suffix in {".txt", ".md"} or _looks_like_text(data):
        return SourceFormat.TXT
    if suffix == ".pdf":
        raise UnsupportedFileType("The file has a .pdf extension but is not a valid PDF.")

    raise UnsupportedFileType(
        "Unrecognised file type. Supported formats: PDF, DOCX, TXT."
    )


def extract(data: bytes, filename: str | None = None, *, max_bytes: int | None = None,
            max_chars: int | None = None) -> ExtractedDocument:
    """Validate, detect and extract in one call. The only extraction entry point."""
    if max_bytes is not None and len(data) > max_bytes:
        raise FileTooLarge(
            f"File is {len(data) / 1_048_576:.1f} MiB; the limit is {max_bytes / 1_048_576:.1f} MiB.",
            details={"size_bytes": len(data), "limit_bytes": max_bytes},
        )
    if not data.strip():
        raise EmptyDocument("The uploaded file is empty.")

    fmt = detect_format(data, filename)

    with track("extract", format=str(fmt)) as ctx:
        if fmt is SourceFormat.PDF:
            document = extract_pdf(data)
        elif fmt is SourceFormat.DOCX:
            document = extract_docx(data)
        else:
            document = _extract_txt(data)

        if max_chars is not None and len(document.text) > max_chars:
            # Truncate defensively rather than reject: a 200-page CV dump is
            # still analysable, and the caller is told what happened.
            document.text = document.text[:max_chars]
            document.warnings.append(
                f"Document truncated to {max_chars} characters for analysis."
            )

        ctx["chars"] = document.char_count
        ctx["pages"] = document.page_count
        ctx["warnings"] = len(document.warnings)

    return document


def _extract_txt(data: bytes) -> ExtractedDocument:
    text = _decode(data).strip()
    if not text:
        raise EmptyDocument("The uploaded file contains no text.")
    word_count = len(text.split())
    return ExtractedDocument(
        text=text,
        source_format=SourceFormat.TXT,
        page_count=max(1, round(word_count / 500) or 1),
        pages=[
            PageStats(
                index=0,
                char_count=len(text),
                line_count=len(text.splitlines()),
                has_extractable_text=True,
            )
        ],
        warnings=[
            "Plain text has no formatting. It parses perfectly but reads as unpolished "
            "to human reviewers - submit a PDF where the employer accepts one."
        ],
    )


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _looks_like_text(data: bytes, sample: int = 2048) -> bool:
    """Heuristic: mostly printable and no NUL bytes in the first couple of KiB."""
    head = data[:sample]
    if b"\x00" in head:
        return False
    printable = sum(1 for b in head if b in (9, 10, 13) or 32 <= b < 127 or b >= 128)
    return bool(head) and printable / len(head) > 0.9


def _suffix(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()
