"""Turn an uploaded file into normalised, citable chunks of text."""

import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import PurePath

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1200  # characters
CHUNK_OVERLAP = 150

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}

# Arabic letter forms and Persian/Arabic digits that look identical to readers but differ in bytes.
_CHARACTER_MAP = str.maketrans(
    {"ي": "ی", "ى": "ی", "ك": "ک", "ـ": None}
    | {chr(0x06F0 + d): str(d) for d in range(10)}
    | {chr(0x0660 + d): str(d) for d in range(10)}
)
_KEEP_FORMAT_CHARS = {"\u200c", "\u200d"}  # half-space (ZWNJ) and ZWJ carry meaning in Persian
_KEEP_CONTROL_CHARS = {"\n", "\r", "\t"}  # other control characters (e.g. NUL) cannot be stored

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "؟ ", "? ", "! ", "؛ ", "، ", ", ", " ", ""],
    keep_separator="end",
)
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")


class UnsupportedDocument(ValueError):
    """The file is not PDF, Markdown or plain text."""


class UnreadableDocument(ValueError):
    """The file could not be turned into text."""


type _Part = tuple[str | None, int | None, str]  # (section, page, text)


@dataclass(frozen=True)
class TextChunk:
    content: str
    section: str | None = None
    page: int | None = None


def media_type(filename: str) -> str:
    try:
        return MEDIA_TYPES[PurePath(filename).suffix.lower()]
    except KeyError:
        raise UnsupportedDocument(
            f"Unsupported file type {filename!r}: use PDF, Markdown (.md) or text (.txt)"
        ) from None


def normalize(text: str) -> str:
    """Give each character one spelling, so documents and questions match.

    NFKC folds presentation forms from PDFs (e.g. the ligature ﻻ) into plain letters. Diacritics
    and invisible direction marks are dropped; half-spaces are kept. Control characters other than
    line breaks and tabs are dropped: they carry no text, and Postgres rejects NUL.
    """
    text = unicodedata.normalize("NFKC", text).translate(_CHARACTER_MAP)
    return "".join(
        c
        for c in text
        if unicodedata.category(c) != "Mn"
        and (unicodedata.category(c) != "Cf" or c in _KEEP_FORMAT_CHARS)
        and (unicodedata.category(c) != "Cc" or c in _KEEP_CONTROL_CHARS)
    )


def chunk(data: bytes, media_type: str) -> list[TextChunk]:
    """Split a document into chunks that remember their section (Markdown) or page (PDF)."""
    parts: list[_Part]
    if media_type == "application/pdf":
        parts = [(None, number, text) for number, text in enumerate(_pdf_pages(data), start=1)]
    elif media_type == "text/markdown":
        parts = _markdown_sections(_decode(data))
    else:
        parts = [(None, None, _decode(data))]

    chunks = [
        TextChunk(piece.strip(), section, page)
        for section, page, text in parts
        for piece in _splitter.split_text(normalize(text))
        if piece.strip()
    ]
    if not chunks:
        raise UnreadableDocument("No text found; scanned PDFs (images) are not supported")
    return chunks


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UnreadableDocument("Text and Markdown files must be UTF-8 encoded") from None


def _pdf_pages(data: bytes) -> list[str]:
    try:
        result = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", "-", "-"],
            input=data,
            capture_output=True,
            check=True,
            timeout=60,
        )
    except subprocess.CalledProcessError:
        raise UnreadableDocument("The file is not a readable PDF") from None
    # pdftotext ends every page with a form feed.
    return result.stdout.decode("utf-8").split("\f")[:-1]


def _markdown_sections(text: str) -> list[_Part]:
    """Split Markdown at #, ## and ### headings; each part keeps its heading path as the section.

    Written here rather than using LangChain's MarkdownHeaderTextSplitter, which drops characters
    that are not "printable" — including the Persian half-space.
    """
    sections: list[tuple[list[str], list[str]]] = [([], [])]  # (heading path, lines)
    in_code_block = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
        heading = None if in_code_block else _HEADING.match(line)
        if heading:
            parent = sections[-1][0][: len(heading[1]) - 1]
            sections.append(([*parent, normalize(heading[2])], []))
        sections[-1][1].append(line)
    return [
        (" > ".join(path) or None, None, "\n".join(lines))
        for path, lines in sections
        if any(line.strip() and not _HEADING.match(line) for line in lines)
    ]
