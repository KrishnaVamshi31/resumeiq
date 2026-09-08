"""Minimal PDF writer for tests.

Building a real PDF byte-for-byte (rather than mocking pypdf) is the only way
to prove the extraction path actually works end to end: the text operators, the
xref offsets and the page tree all have to be right for `page.extract_text()`
to return anything.
"""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages: list[list[str]], *, font_size: int = 11, leading: int = 14) -> bytes:
    """Render `pages` (a list of pages, each a list of text lines) to PDF bytes."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # 1-indexed object number

    catalog_num = 1
    pages_num = 2
    objects.extend([b"", b""])  # placeholders, filled in once kids are known

    kids: list[int] = []
    for lines in pages:
        stream_lines = [b"BT", f"/F1 {font_size} Tf".encode(), b"1 0 0 1 56 736 Tm",
                        f"{leading} TL".encode()]
        for line in lines:
            stream_lines.append(f"({_escape(line)}) Tj".encode())
            stream_lines.append(b"T*")
        stream_lines.append(b"ET")
        stream = b"\n".join(stream_lines)

        content_num = add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                          + stream + b"\nendstream")
        page_num = add(
            b"<< /Type /Page /Parent " + str(pages_num).encode() + b" 0 R "
            b"/MediaBox [0 0 612 792] /Contents " + str(content_num).encode() + b" 0 R "
            b"/Resources << /Font << /F1 " + b"FONTREF" + b" 0 R >> >> >>"
        )
        kids.append(page_num)

    font_num = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Patch the font reference now that the font object number is known.
    objects = [obj.replace(b"FONTREF", str(font_num).encode()) for obj in objects]

    objects[catalog_num - 1] = (
        b"<< /Type /Catalog /Pages " + str(pages_num).encode() + b" 0 R >>"
    )
    objects[pages_num - 1] = (
        b"<< /Type /Pages /Kids ["
        + b" ".join(f"{k} 0 R".encode() for k in kids)
        + b"] /Count " + str(len(kids)).encode() + b" >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    count = len(objects) + 1
    out += f"xref\n0 {count}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {count} /Root {catalog_num} 0 R >>\nstartxref\n{xref_offset}\n".encode()
    )
    out += b"%%EOF\n"
    return bytes(out)
