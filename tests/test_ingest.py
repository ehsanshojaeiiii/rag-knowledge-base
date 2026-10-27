import pytest

from rag_kb.ingest import (
    CHUNK_SIZE,
    UnreadableDocument,
    UnsupportedDocument,
    chunk,
    media_type,
    normalize,
)


def make_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one line of Latin text per page."""
    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(len(pages)))}]"
        f" /Count {len(pages)} >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects[4 + 2 * i] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            f" /Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
        )
        objects[5 + 2 * i] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    pdf, offsets = b"%PDF-1.4\n", {}
    for number in sorted(objects):
        offsets[number] = len(pdf)
        pdf += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode()
    xref, size = len(pdf), max(objects) + 1
    pdf += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    pdf += "".join(f"{offsets[n]:010d} 00000 n \n" for n in range(1, size)).encode()
    pdf += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return pdf


class TestNormalize:
    def test_unifies_arabic_letters_and_digits(self) -> None:
        assert normalize("كتاب‌هاي ٣ و ۴") == "کتاب‌های 3 و 4"

    def test_removes_diacritics_tatweel_and_direction_marks(self) -> None:
        assert normalize("‫كَتَـــب‬") == "کتب"

    def test_keeps_half_space(self) -> None:
        assert normalize("می‌شود") == "می‌شود"

    def test_folds_pdf_ligatures_into_letters(self) -> None:
        assert normalize("ﻻ") == "لا"

    def test_leaves_technical_text_unchanged(self) -> None:
        text = "Version 2.1, 200,000 chars: https://example.com/a_b?x=1"
        assert normalize(text) == text


class TestMediaType:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [("a.PDF", "application/pdf"), ("b.md", "text/markdown"), ("c.txt", "text/plain")],
    )
    def test_detects_supported_types(self, filename: str, expected: str) -> None:
        assert media_type(filename) == expected

    def test_rejects_other_types(self) -> None:
        with pytest.raises(UnsupportedDocument):
            media_type("report.docx")


class TestChunk:
    def test_markdown_chunks_carry_their_heading_path(self) -> None:
        markdown = b"# CDN\nIntro.\n## Cache\nPurge the cache from the panel.\n"

        chunks = chunk(markdown, "text/markdown")

        assert [(c.section, c.content.splitlines()[-1]) for c in chunks] == [
            ("CDN", "Intro."),
            ("CDN > Cache", "Purge the cache from the panel."),
        ]

    def test_pdf_chunks_carry_their_page_number(self) -> None:
        chunks = chunk(make_pdf(["First page", "Second page"]), "application/pdf")

        assert [(c.page, c.content) for c in chunks] == [(1, "First page"), (2, "Second page")]

    def test_long_text_is_split_with_overlap(self) -> None:
        sentences = [f"Sentence number {i} about caching." for i in range(200)]

        chunks = chunk(" ".join(sentences).encode(), "text/plain")

        assert len(chunks) > 1
        assert all(len(c.content) <= CHUNK_SIZE for c in chunks)
        last_sentence_of_first = chunks[0].content.split(". ")[-1]
        assert last_sentence_of_first in chunks[1].content

    def test_text_is_normalised(self) -> None:
        assert chunk("كتاب".encode(), "text/plain")[0].content == "کتاب"

    @pytest.mark.parametrize(
        ("data", "kind"),
        [
            (b"not a pdf", "application/pdf"),
            (b"   \n\n ", "text/plain"),
            ("سلام".encode("utf-16"), "text/plain"),
        ],
        ids=["broken-pdf", "empty-text", "not-utf8"],
    )
    def test_rejects_unreadable_files(self, data: bytes, kind: str) -> None:
        with pytest.raises(UnreadableDocument):
            chunk(data, kind)
