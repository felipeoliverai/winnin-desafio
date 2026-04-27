"""Download the 3 arXiv PDFs and index them into ChromaDB.

Usage:
    python ingest.py            # skip if already indexed
    python ingest.py --force    # rebuild from scratch
"""
import argparse
import logging
import sys
from pathlib import Path

import httpx
from pypdf import PdfReader

from app.chunking import build_chunks
from app.config import settings
from app.models import PAPERS
from app.rag import add_chunks, get_collection, reset_collection

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("ingest")


def download_pdf(arxiv_id: str, dest: Path) -> Path:
    """Download a PDF from arXiv to `dest`, skipping if a non-empty file already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("PDF already present: %s", dest)
        return dest
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    logger.info("Downloading %s", url)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)
    return dest


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Read a PDF and return `(page_number, text)` pairs, skipping pages with no extractable text."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Page %d failed: %s", i, exc)
            text = ""
        if text.strip():
            pages.append((i, text))
    return pages


def run(force: bool = False) -> None:
    """Run the full ingestion pipeline (download, parse, chunk, index). Idempotent unless `force=True`."""
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    collection = reset_collection() if force else get_collection()
    if not force and collection.count() > 0:
        logger.info("Collection already populated (%d chunks). Use --force to rebuild.", collection.count())
        return

    total = 0
    for arxiv_id, title in PAPERS.items():
        pdf_path = settings.pdf_dir / f"{arxiv_id}.pdf"
        download_pdf(arxiv_id, pdf_path)
        pages = extract_pages(pdf_path)
        logger.info("%s: %d pages with text", arxiv_id, len(pages))
        chunks = build_chunks(pages, paper_id=arxiv_id, paper_title=title)
        logger.info("%s: produced %d chunks", arxiv_id, len(chunks))
        add_chunks(collection, chunks)
        total += len(chunks)

    logger.info("Ingestion complete. Total chunks indexed: %d", total)


def main() -> None:
    """CLI entry point. Parses `--force` and runs the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest arXiv PDFs into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Drop and reindex the collection.")
    args = parser.parse_args()
    try:
        run(force=args.force)
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
