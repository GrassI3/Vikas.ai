"""
Vikas.ai — Document Ingestion Script
Load medical/civic documents from text files, PDFs, or raw strings
and upsert them into the ChromaDB knowledge base.

Usage:
    python -m backend.knowledge.ingest --source ./backend/data/seed_docs.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from pathlib import Path

from backend.knowledge.vector_db import add_documents

logger = logging.getLogger("vikas.knowledge.ingest")


# ── Seed data for initial demonstration ─────────────────────
SEED_DOCUMENTS: list[dict] = [
    {
        "id": "nih-headache-001",
        "content": (
            "Tension-type headaches are the most common form of headache. They typically "
            "present as a dull, aching pain on both sides of the head. Over-the-counter "
            "analgesics such as ibuprofen or acetaminophen are first-line treatments. "
            "Patients should seek emergency care if the headache is sudden and severe, "
            "accompanied by fever, stiff neck, confusion, or follows a head injury."
        ),
        "source": "NIH — National Institute of Neurological Disorders and Stroke",
        "metadata": {"domain": "medical", "topic": "headache"},
    },
    {
        "id": "who-fever-002",
        "content": (
            "Fever is defined as a body temperature above 38°C (100.4°F). In adults, "
            "fever alone is usually not dangerous and often resolves with rest, hydration, "
            "and antipyretics. Emergency indicators include: temperature above 40°C (104°F), "
            "persistent fever lasting more than 3 days, difficulty breathing, chest pain, "
            "or severe abdominal pain."
        ),
        "source": "WHO — World Health Organization Clinical Guidelines",
        "metadata": {"domain": "medical", "topic": "fever"},
    },
    {
        "id": "nimhans-mental-003",
        "content": (
            "Anxiety disorders are characterised by excessive worry, restlessness, and "
            "physical symptoms such as rapid heartbeat and sweating. Cognitive behavioral "
            "therapy (CBT) is the gold-standard treatment. If a person expresses thoughts "
            "of self-harm or suicide, they should be connected immediately to a crisis "
            "helpline such as iCall (9152987821) or Vandrevala Foundation (1860-2662-345)."
        ),
        "source": "NIMHANS — National Institute of Mental Health and Neurosciences, India",
        "metadata": {"domain": "mental_health", "topic": "anxiety"},
    },
    {
        "id": "govt-disability-004",
        "content": (
            "Under the Rights of Persons with Disabilities Act, 2016 (India), individuals "
            "with benchmark disabilities (40% or above) are entitled to a disability "
            "certificate issued by a certified medical authority. This certificate enables "
            "access to government schemes including free bus passes, reservation in "
            "educational institutions, and housing subsidies. Applications can be submitted "
            "at the nearest District Disability Rehabilitation Centre (DDRC)."
        ),
        "source": "Department of Empowerment of Persons with Disabilities, Govt. of India",
        "metadata": {"domain": "civic", "topic": "disability_rights"},
    },
    {
        "id": "first-aid-burn-005",
        "content": (
            "For minor burns (first-degree): cool the burn under running water for at "
            "least 10 minutes. Do not apply ice, butter, or toothpaste. Cover with a "
            "sterile non-adhesive bandage. For severe burns with blistering, charring, "
            "or burns larger than 3 inches, call emergency services immediately. Chemical "
            "burns should be flushed continuously with water for at least 20 minutes."
        ),
        "source": "Red Cross — First Aid Guidelines",
        "metadata": {"domain": "medical", "topic": "burns"},
    },
]


async def ingest_from_json(path: Path) -> int:
    """Load and ingest documents from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Ensure each doc has an ID
    for doc in documents:
        if "id" not in doc:
            doc["id"] = str(uuid.uuid4())

    count = await add_documents(documents)
    logger.info("Ingested %d documents from %s", count, path)
    return count


async def ingest_seed_data() -> int:
    """Ingest the built-in seed documents for demonstration."""
    count = await add_documents(SEED_DOCUMENTS)
    logger.info("Ingested %d seed documents", count)
    return count


async def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Vikas.ai — Knowledge Ingestion")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to a JSON file containing documents to ingest",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Ingest built-in seed documents for demonstration",
    )
    args = parser.parse_args()

    total = 0
    if args.source:
        total += await ingest_from_json(Path(args.source))
    if args.seed or not args.source:
        total += await ingest_seed_data()

    print(f"\n✅ Ingestion complete — {total} documents added to knowledge base.\n")


if __name__ == "__main__":
    asyncio.run(main())
