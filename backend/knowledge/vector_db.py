"""
Vikas.ai — Vector Database (ChromaDB)
Provides semantic search over ingested medical/civic knowledge.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import get_settings

logger = logging.getLogger("vikas.knowledge.vector_db")
settings = get_settings()

# ── Lazy singletons ─────────────────────────────────────────
_chroma_client: chromadb.ClientAPI | None = None


def _get_chroma() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB persistent client initialized at %s", settings.chroma_persist_dir)
    return _chroma_client


def get_or_create_collection(name: str | None = None) -> chromadb.Collection:
    """Return the default knowledge collection, creating it if needed."""
    cname = name or settings.chroma_collection_name
    client = _get_chroma()
    collection = client.get_or_create_collection(
        name=cname,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Collection '%s' ready — %d documents", cname, collection.count())
    return collection


async def add_documents(
    documents: list[dict[str, Any]],
    collection_name: str | None = None,
) -> int:
    """
    Ingest a batch of documents into ChromaDB.

    Each document dict must have:
      - id: str
      - content: str
      - source: str  (e.g. "PubMed:12345" or "NIH Guidelines")
      - metadata: dict (optional)
    """
    collection = get_or_create_collection(collection_name)
    ids, texts, metadatas, embeddings = [], [], [], []

    for doc in documents:
        ids.append(doc["id"])
        texts.append(doc["content"])
        metadatas.append({
            "source": doc.get("source", "unknown"),
            **(doc.get("metadata", {})),
        })

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info("Upserted %d documents into collection '%s'", len(ids), collection.name)
    return len(ids)


async def query_knowledge_base(
    query: str,
    n_results: int = 5,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Perform a semantic search against the knowledge base.

    Returns a list of dicts with keys: content, source, relevance_score, metadata.
    """
    collection = get_or_create_collection(collection_name)

    if collection.count() == 0:
        logger.warning("Knowledge base is empty — returning no results")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs: list[dict[str, Any]] = []
    for i, doc_text in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        # ChromaDB cosine distance → similarity = 1 - distance
        similarity = 1.0 - distance
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        docs.append({
            "content": doc_text,
            "source": meta.get("source", "unknown"),
            "relevance_score": round(similarity, 4),
            "metadata": meta,
        })

    logger.info("Query returned %d documents (top score %.4f)", len(docs), docs[0]["relevance_score"] if docs else 0.0)
    return docs
