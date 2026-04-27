"""ChromaDB wrapper: persistent client, embeddings, indexing and semantic search."""
import logging
from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from app.config import settings
from app.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> ClientAPI:
    """Return a process-wide persistent ChromaDB client (lazy-initialized)."""
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_path))


@lru_cache(maxsize=1)
def _embedding_function() -> embedding_functions.EmbeddingFunction:
    """Return the sentence-transformers embedding function configured in settings."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)


def get_collection() -> Collection:
    """Get (or create) the papers collection with cosine similarity."""
    return _client().get_or_create_collection(
        name=settings.collection_name,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> Collection:
    """Drop the existing collection if present and create a fresh one. Used by `--force` ingest."""
    client = _client()
    try:
        client.delete_collection(settings.collection_name)
    except (ValueError, chromadb.errors.NotFoundError):
        pass
    return get_collection()


def add_chunks(collection: Collection, chunks: list[Chunk]) -> None:
    """Index a batch of `Chunk`s into the given collection (no-op on empty input)."""
    if not chunks:
        return
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"paper_id": c.paper_id, "paper_title": c.paper_title, "section": c.section, "page": c.page}
            for c in chunks
        ],
    )
    logger.info("Indexed %d chunks", len(chunks))


def search(query: str, top_k: int = 4, where: dict | None = None) -> list[RetrievedChunk]:
    """Semantic search over the papers collection. `where` filters by metadata (e.g. paper_id, section)."""
    collection = get_collection()
    result = collection.query(query_texts=[query], n_results=top_k, where=where)
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    out: list[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, distances):
        out.append(
            RetrievedChunk(
                paper_id=str(meta["paper_id"]),
                paper_title=str(meta["paper_title"]),
                section=str(meta["section"]),
                page=int(meta["page"]),
                text=doc,
                score=1.0 - float(dist),
            )
        )
    return out
