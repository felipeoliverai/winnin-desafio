#!/bin/sh
set -e

# Auto-run ingestion on first boot if the vector store is empty.
if [ ! -f /app/data/chroma/chroma.sqlite3 ]; then
    echo "[entrypoint] Vector store empty — running ingestion..."
    python ingest.py
else
    echo "[entrypoint] Vector store already populated — skipping ingestion."
fi

exec "$@"
