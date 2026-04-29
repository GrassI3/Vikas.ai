"""
Vikas.ai — PubMed Knowledge Ingestion
Uses NCBI E-utilities (free, no API key required) to bulk-ingest
peer-reviewed medical abstracts into the ChromaDB knowledge base.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from backend.knowledge.vector_db import add_documents

logger = logging.getLogger("vikas.knowledge.pubmed")

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ── Topics config file ─────────────────────────────────
_TOPICS_FILE = pathlib.Path(__file__).parent / "pubmed_topics.json"


def load_topics() -> list[str]:
    """
    Load PubMed search topics from pubmed_topics.json.
    Edit that file to add/remove topics without touching Python code.
    Falls back to a minimal list if the file is missing.
    """
    if _TOPICS_FILE.exists():
        with open(_TOPICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        topics = data.get("topics", [])
        logger.info("Loaded %d topics from %s", len(topics), _TOPICS_FILE.name)
        return topics
    else:
        logger.warning("pubmed_topics.json not found — using minimal fallback list")
        return [
            "headache diagnosis treatment",
            "fever differential diagnosis adults",
            "chest pain emergency triage",
        ]


async def search_pubmed(
    query: str,
    max_results: int = 20,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """
    Search PubMed and return a list of PMIDs.

    Args:
        query: Medical search term (e.g. "headache diagnosis treatment")
        max_results: Maximum number of results to return
        client: Optional shared httpx client

    Returns:
        List of PubMed IDs (PMIDs)
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30.0)

    try:
        resp = await client.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info("PubMed search '%s' returned %d PMIDs", query, len(pmids))
        return pmids
    except Exception as e:
        logger.error("PubMed search failed for '%s': %s", query, e)
        return []
    finally:
        if own_client:
            await client.aclose()


async def fetch_abstracts(
    pmids: list[str],
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch article details (title, abstract, authors, journal) for given PMIDs.

    Returns a list of dicts ready for ChromaDB ingestion.
    """
    if not pmids:
        return []

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=60.0)

    try:
        # Fetch in batches of 50 to respect rate limits
        all_articles = []
        for i in range(0, len(pmids), 50):
            batch = pmids[i:i + 50]
            resp = await client.get(
                f"{EUTILS_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "xml",
                    "retmode": "xml",
                },
            )
            resp.raise_for_status()
            articles = _parse_pubmed_xml(resp.text)
            all_articles.extend(articles)

            # Be nice to NCBI — 0.4s delay between batches (3 req/sec limit)
            if i + 50 < len(pmids):
                await asyncio.sleep(0.4)

        logger.info("Fetched %d abstracts from PubMed", len(all_articles))
        return all_articles
    except Exception as e:
        logger.error("PubMed fetch failed: %s", e)
        return []
    finally:
        if own_client:
            await client.aclose()


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed XML response into document dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error("Failed to parse PubMed XML: %s", e)
        return []

    for article_elem in root.findall(".//PubmedArticle"):
        try:
            medline = article_elem.find(".//MedlineCitation")
            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "unknown"

            article = medline.find(".//Article")
            if article is None:
                continue

            # Title
            title_elem = article.find(".//ArticleTitle")
            title = _get_text(title_elem) if title_elem is not None else "Untitled"

            # Abstract
            abstract_elem = article.find(".//Abstract")
            if abstract_elem is None:
                continue  # Skip articles without abstracts

            abstract_parts = []
            for text_elem in abstract_elem.findall(".//AbstractText"):
                label = text_elem.get("Label", "")
                text = _get_text(text_elem)
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            if len(abstract) < 100:
                continue  # Skip very short abstracts

            # Journal
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown Journal"

            # Publication year
            year_elem = article.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None else ""

            # Authors
            authors = []
            for author in article.findall(".//AuthorList/Author"):
                last = author.find("LastName")
                first = author.find("ForeName")
                if last is not None:
                    name = last.text
                    if first is not None:
                        name = f"{first.text} {name}"
                    authors.append(name)

            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            # MeSH terms (medical subject headings)
            mesh_terms = []
            for mesh in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
                mesh_terms.append(mesh.text)

            # Build document
            content = f"Title: {title}\n\n{abstract}"
            source = f"PubMed PMID:{pmid} — {journal}"
            if year:
                source += f" ({year})"

            articles.append({
                "id": f"pubmed-{pmid}",
                "content": content,
                "source": source,
                "metadata": {
                    "domain": "medical",
                    "topic": ", ".join(mesh_terms[:5]) if mesh_terms else "general medicine",
                    "authors": author_str,
                    "journal": journal,
                    "year": year,
                    "pmid": pmid,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                },
            })

        except Exception as e:
            logger.warning("Failed to parse article: %s", e)
            continue

    return articles


def _get_text(elem) -> str:
    """Extract all text content from an XML element, including nested tags."""
    return "".join(elem.itertext()).strip()


async def ingest_pubmed_topic(
    topic: str,
    max_articles: int = 20,
) -> int:
    """
    Search PubMed for a topic and ingest all found abstracts.

    Args:
        topic: Medical search query
        max_articles: Max articles to fetch per topic

    Returns:
        Number of documents ingested
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        pmids = await search_pubmed(topic, max_results=max_articles, client=client)
        if not pmids:
            return 0

        articles = await fetch_abstracts(pmids, client=client)
        if not articles:
            return 0

        count = await add_documents(articles)
        logger.info("Ingested %d PubMed articles for topic '%s'", count, topic)
        return count


async def ingest_all_default_topics(
    max_per_topic: int = 20,
) -> int:
    """
    Ingest articles for all topics in pubmed_topics.json.
    Edit that file to control what gets ingested — no code changes needed.
    """
    topics = load_topics()
    total = 0
    for i, topic in enumerate(topics):
        logger.info("[%d/%d] Ingesting topic: %s", i + 1, len(topics), topic)
        count = await ingest_pubmed_topic(topic, max_articles=max_per_topic)
        total += count
        # Rate limit: 3 requests/sec without API key
        await asyncio.sleep(0.5)

    logger.info("PubMed bulk ingestion complete: %d total articles", total)
    return total


# ── CLI entrypoint ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Vikas.ai — PubMed Ingestion")
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Single topic to search and ingest (e.g. 'malaria treatment')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all 20 default medical topics (~400 articles)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        help="Max articles per topic (default: 20)",
    )
    args = parser.parse_args()

    if not args.topic and not args.all:
        print("Usage:")
        print("  python -m backend.knowledge.pubmed --topic 'headache treatment'")
        print("  python -m backend.knowledge.pubmed --all")
        print("  python -m backend.knowledge.pubmed --all --max 50")
        sys.exit(0)

    async def _run():
        if args.all:
            total = await ingest_all_default_topics(max_per_topic=args.max)
        else:
            total = await ingest_pubmed_topic(args.topic, max_articles=args.max)
        print(f"\n[OK] Ingested {total} PubMed articles into the knowledge base.\n")

    asyncio.run(_run())
