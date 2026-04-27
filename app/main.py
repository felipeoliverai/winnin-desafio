"""FastAPI entry point exposing the QA agent over HTTP."""
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from google.api_core.exceptions import ResourceExhausted, TooManyRequests

from app.agent import QAAgent
from app.config import settings
from app.models import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
)
from app.rag import get_collection

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the QA agent and warm up the vector store on startup."""
    logger.info("Initializing QA agent (model=%s)", settings.gemini_model)
    collection = get_collection()
    chunk_count = collection.count()
    if chunk_count == 0:
        logger.warning("Vector store is empty — run `python ingest.py` before querying.")
    else:
        logger.info("Vector store ready with %d chunks.", chunk_count)
    app.state.agent = QAAgent()
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="Scientific Q&A Agent",
    description="RAG agent answering questions about Attention Is All You Need, BERT, and RAG papers.",
    version="0.1.0",
    lifespan=lifespan,
    responses={
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse, "description": "Upstream LLM rate limit."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Unexpected error."},
    },
)


async def _gemini_quota_handler(_: Request, exc: Exception) -> JSONResponse:
    """Translate Gemini rate-limit errors (gRPC or REST transport) to HTTP 429."""
    logger.warning("Gemini quota exhausted: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=ErrorResponse(detail="Gemini quota exhausted, please retry later.").model_dump(),
    )


app.add_exception_handler(ResourceExhausted, _gemini_quota_handler)
app.add_exception_handler(TooManyRequests, _gemini_quota_handler)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe and runtime info.",
)
def health() -> HealthResponse:
    """Return runtime state — model in use and number of chunks indexed."""
    return HealthResponse(
        model=settings.gemini_model,
        indexed_chunks=get_collection().count(),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the QA agent a question grounded on the 3 papers.",
)
def ask(request: AskRequest) -> AskResponse:
    """Send a validated question to the agent and return its grounded answer plus cited sources."""
    agent: QAAgent = app.state.agent
    try:
        return agent.ask(request.question)
    except (ResourceExhausted, TooManyRequests):
        raise
    except Exception as exc:
        logger.exception("Agent failed to answer")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {exc}",
        ) from exc
