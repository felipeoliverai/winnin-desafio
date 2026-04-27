from app.chunking import build_chunks, detect_section, split_text


def test_split_text_respects_chunk_size():
    text = "abc. " * 500
    chunks = split_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 220 for c in chunks)


def test_split_text_overlaps():
    text = " ".join(f"token{i}" for i in range(200))
    chunks = split_text(text, chunk_size=200, overlap=60)
    assert len(chunks) >= 2
    tail = chunks[0][-40:].strip()
    assert tail and tail in chunks[1]


def test_detect_section_finds_headings():
    assert detect_section("Abstract\nThis paper...", "intro") == "abstract"
    assert detect_section("3 Method\nWe propose...", "intro") == "method"
    assert detect_section("Random middle text", "results") == "results"


def test_build_chunks_assigns_metadata():
    pages = [
        (1, "Abstract\nWe present a new model. " * 30),
        (2, "1 Introduction\nThis section motivates. " * 30),
    ]
    chunks = build_chunks(pages, paper_id="0000.0001", paper_title="Test Paper", chunk_size=200, overlap=20)
    assert len(chunks) > 0
    assert all(c.paper_id == "0000.0001" for c in chunks)
    sections = {c.section for c in chunks}
    assert {"abstract", "introduction"} <= sections
