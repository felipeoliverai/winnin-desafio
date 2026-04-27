"""Text splitting and section detection for scientific papers."""
import re
from collections.abc import Iterable

from app.models import Chunk

SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"^\s*abstract\s*$", re.IGNORECASE | re.MULTILINE)),
    ("introduction", re.compile(r"^\s*\d?\.?\s*introduction\s*$", re.IGNORECASE | re.MULTILINE)),
    ("related_work", re.compile(r"^\s*\d?\.?\s*(related work|background)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("method", re.compile(r"^\s*\d?\.?\s*(method|methods|approach|model architecture|our approach)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("experiments", re.compile(r"^\s*\d?\.?\s*(experiments|experimental setup|training)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("results", re.compile(r"^\s*\d?\.?\s*(results|evaluation)\s*$", re.IGNORECASE | re.MULTILINE)),
    ("conclusion", re.compile(r"^\s*\d?\.?\s*(conclusion|conclusions|discussion)\s*$", re.IGNORECASE | re.MULTILINE)),
]


def detect_section(text: str, current: str) -> str:
    """Return the section name when a heading is found at the top of `text`, else `current`."""
    head = text[:200]
    for name, pattern in SECTION_PATTERNS:
        if pattern.search(head):
            return name
    return current


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph then sentence boundaries."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(
    pages: Iterable[tuple[int, str]],
    paper_id: str,
    paper_title: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Turn `(page_number, page_text)` pairs into `Chunk`s with detected section metadata."""
    section = "abstract"
    chunks: list[Chunk] = []
    counter = 0
    for page_num, page_text in pages:
        section = detect_section(page_text, section)
        for piece in split_text(page_text, chunk_size, overlap):
            section = detect_section(piece, section)
            chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}-{counter}",
                    paper_id=paper_id,
                    paper_title=paper_title,
                    section=section,
                    text=piece,
                    page=page_num,
                )
            )
            counter += 1
    return chunks
