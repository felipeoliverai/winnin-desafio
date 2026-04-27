from unittest.mock import patch

from app.models import RetrievedChunk
from app.tools import ExtractSection, SearchDocuments


def _fake_chunk(paper_id: str = "1706.03762", section: str = "abstract", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        paper_id=paper_id,
        paper_title="Attention Is All You Need",
        section=section,
        page=page,
        text="Self-attention is the core mechanism.",
        score=0.9,
    )


def test_search_documents_declaration_has_required_fields():
    tool = SearchDocuments()
    decl = tool.declaration()
    assert decl["name"] == "search_documents"
    assert "query" in decl["parameters"]["properties"]
    assert decl["parameters"]["required"] == ["query"]


def test_search_documents_returns_chunks():
    tool = SearchDocuments()
    with patch("app.tools.search", return_value=[_fake_chunk(), _fake_chunk(page=2)]) as mock_search:
        result = tool.run(query="what is attention?", top_k=2)
    mock_search.assert_called_once_with(query="what is attention?", top_k=2)
    assert result.success is True
    assert len(result.data) == 2
    assert result.data[0]["paper_id"] == "1706.03762"


def test_search_documents_handles_errors():
    tool = SearchDocuments()
    with patch("app.tools.search", side_effect=RuntimeError("boom")):
        result = tool.run(query="x")
    assert result.success is False
    assert "boom" in (result.error or "")


def test_extract_section_filters_by_metadata():
    tool = ExtractSection()
    chunks = [_fake_chunk(section="method", page=2), _fake_chunk(section="method", page=1)]
    with patch("app.tools.search", return_value=chunks) as mock_search:
        result = tool.run(paper_id="1706.03762", section="method")
    args, kwargs = mock_search.call_args
    assert kwargs["where"] == {"$and": [{"paper_id": "1706.03762"}, {"section": "method"}]}
    assert result.success is True
    assert result.data["found"] is True
    assert "Self-attention" in result.data["text"]


def test_extract_section_empty_result():
    tool = ExtractSection()
    with patch("app.tools.search", return_value=[]):
        result = tool.run(paper_id="1810.04805", section="conclusion")
    assert result.success is True
    assert result.data["found"] is False
    assert result.data["text"] == ""
