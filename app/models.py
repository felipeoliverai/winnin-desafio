"""Pydantic v2 domain models shared across the app."""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

PAPERS: dict[str, str] = {
    "1706.03762": "Attention Is All You Need",
    "1810.04805": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
}

QuestionStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=500),
]


class Chunk(BaseModel):
    """A piece of paper text with metadata, ready to index in the vector store."""

    chunk_id: str
    paper_id: str
    paper_title: str
    section: str
    text: str
    page: int


class RetrievedChunk(BaseModel):
    """A chunk returned by a vector-store query, including its similarity score."""

    paper_id: str
    paper_title: str
    section: str
    page: int
    text: str
    score: float


class ToolResult(BaseModel):
    """Standard envelope returned by every tool — success flag plus data or error."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None


class AskRequest(BaseModel):
    """Body of `POST /ask`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "O que é RAG e quais problemas ele resolve segundo os autores?"},
            ]
        }
    )

    question: QuestionStr = Field(
        description="Pergunta em linguagem natural sobre os 3 papers indexados.",
    )


class AskResponse(BaseModel):
    """Body returned by `POST /ask`."""

    question: str = Field(description="Pergunta original (após sanitização).")
    answer: str = Field(description="Resposta gerada pelo agente, fundamentada nos papers.")
    sources: list[str] = Field(
        default_factory=list,
        description="Títulos dos papers consultados pelas tools para produzir a resposta.",
    )


class HealthResponse(BaseModel):
    """Body returned by `GET /health`."""

    status: Literal["ok"] = "ok"
    model: str = Field(description="Gemini model in use.")
    indexed_chunks: int = Field(ge=0, description="Number of chunks currently in the vector store.")


class ErrorResponse(BaseModel):
    """Generic error envelope."""

    detail: str
