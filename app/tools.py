"""Tools the QA agent can call. Each tool is atomic, stateless, and returns a `ToolResult`."""
from typing import Any

from pydantic import BaseModel

from app.config import settings
from app.models import ToolResult
from app.rag import search

VALID_SECTIONS = ("abstract", "introduction", "related_work", "method", "experiments", "results", "conclusion")


class Tool(BaseModel):
    """Base contract for tools. Subclasses must implement `declaration` and `run`."""

    name: str
    description: str

    def declaration(self) -> dict[str, Any]:
        """Return the function-calling schema in Gemini's expected format."""
        raise NotImplementedError

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool synchronously and return a `ToolResult`."""
        raise NotImplementedError


class SearchDocuments(Tool):
    """Semantic search over all indexed papers."""

    name: str = "search_documents"
    description: str = (
        "Performs semantic search over the indexed scientific papers and returns the most "
        "relevant chunks for a query. Use this for any general question about the papers."
    )

    def declaration(self) -> dict[str, Any]:
        """Function-calling schema for Gemini."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to retrieve relevant passages.",
                    },
                },
                "required": ["query"],
            },
        }

    def run(self, query: str, top_k: int | None = None) -> ToolResult:
        """Run a semantic search; returns the top-k chunks ranked by cosine similarity."""
        try:
            k = int(top_k) if top_k else settings.top_k
            chunks = search(query=query, top_k=k)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=[c.model_dump() for c in chunks],
            )
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


class ExtractSection(Tool):
    """Returns the full text of a named section from a specific paper."""

    name: str = "extract_section"
    description: str = (
        "Returns the text of a specific section (abstract, introduction, method, "
        "experiments, results, conclusion, related_work) from a given paper. "
        "Use this when the user explicitly asks about a section."
    )

    def declaration(self) -> dict[str, Any]:
        """Function-calling schema for Gemini."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": (
                            "Identifier of an indexed paper (e.g. arXiv ID). "
                            "Discover valid IDs from the metadata returned by `search_documents`."
                        ),
                    },
                    "section": {
                        "type": "string",
                        "description": f"Section name. One of: {', '.join(VALID_SECTIONS)}.",
                    },
                },
                "required": ["paper_id", "section"],
            },
        }

    def run(self, paper_id: str, section: str) -> ToolResult:
        """Return all chunks tagged `paper_id + section`, joined in page order."""
        try:
            where = {"$and": [{"paper_id": paper_id}, {"section": section.lower()}]}
            chunks = search(query=section, top_k=10, where=where)
            if not chunks:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    data={"paper_id": paper_id, "section": section, "text": "", "found": False},
                )
            chunks.sort(key=lambda c: c.page)
            full_text = "\n\n".join(c.text for c in chunks)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "paper_id": paper_id,
                    "paper_title": chunks[0].paper_title,
                    "section": section,
                    "text": full_text,
                    "found": True,
                },
            )
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


def all_tools() -> list[Tool]:
    """Return one instance of every tool registered with the agent."""
    return [SearchDocuments(), ExtractSection()]
