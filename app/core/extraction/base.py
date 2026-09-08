"""Shared types for the extraction layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SourceFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


@dataclass(slots=True)
class PageStats:
    """Per-page facts used later by ATS layout heuristics."""

    index: int
    char_count: int
    line_count: int
    # Fraction of lines that look like a second column fragment (short, indented).
    short_line_ratio: float = 0.0
    has_extractable_text: bool = True


@dataclass(slots=True)
class ExtractedDocument:
    """Normalised output of any extractor.

    `text` is plain UTF-8 with `\n` line breaks. Everything an ATS check might
    need about the *original* layout is captured here, because the layout is
    gone once we hand plain text to the rest of the pipeline.
    """

    text: str
    source_format: SourceFormat
    page_count: int = 1
    pages: list[PageStats] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Layout signals an ATS would trip over.
    has_tables: bool = False
    table_count: int = 0
    has_images: bool = False
    image_count: int = 0
    has_multi_column_hint: bool = False
    has_text_in_headers_footers: bool = False
    is_encrypted: bool = False

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def chars_per_page(self) -> float:
        return self.char_count / self.page_count if self.page_count else 0.0

    @property
    def pages_without_text(self) -> int:
        return sum(1 for p in self.pages if not p.has_extractable_text)
