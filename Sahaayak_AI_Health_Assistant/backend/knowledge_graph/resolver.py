"""
Knowledge Graph Resolver — Fuzzy-matches Indian cultural idioms to medical conditions.

Pipeline:
  1. Load the JSON-LD knowledge graph
  2. Build a flat index of (idiom_term → condition_id, clinical_name, severity)
  3. For each input text, fuzzy-match against the index
  4. Return normalized medical terms + any escalation flags
"""

import json
import os
from rapidfuzz import fuzz, process
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "graph.jsonld")
SAFETY_RULES_PATH = os.path.join(os.path.dirname(__file__), "safety_rules.json")
MATCH_THRESHOLD = 80  # Minimum fuzzy-match score (0-100)


class KnowledgeGraphResolver:
    """Resolves cultural idioms to standardized medical conditions."""

    def __init__(
        self,
        graph_path: str = GRAPH_PATH,
        safety_path: str = SAFETY_RULES_PATH,
        threshold: int = MATCH_THRESHOLD,
    ):
        self.threshold = threshold
        self.idiom_index: dict[str, dict] = {}
        self.safety_rules: list[dict] = []

        self._load_graph(graph_path)
        self._load_safety_rules(safety_path)

    def _load_graph(self, path: str):
        """Build a flat lookup from idiom terms to conditions."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for node in data.get("@graph", []):
            condition_id = node.get("@id", "")
            clinical_name = node.get("med:clinicalName", "")
            default_severity = node.get("med:defaultSeverity", "LOW")
            escalation_triggers = node.get("med:escalationTriggers", [])
            related = node.get("med:relatedSymptoms", [])

            for idiom in node.get("indic:idioms", []):
                term = idiom.get("indic:term", "").lower().strip()
                if term:
                    self.idiom_index[term] = {
                        "condition_id": condition_id,
                        "clinical_name": clinical_name,
                        "default_severity": default_severity,
                        "lang": idiom.get("indic:lang", ""),
                        "literal": idiom.get("indic:literal", ""),
                        "context": idiom.get("indic:context", ""),
                        "escalation_triggers": escalation_triggers,
                        "related_symptoms": related,
                    }

        print(f"[KG] Loaded {len(self.idiom_index)} idiom entries from knowledge graph")

    def _load_safety_rules(self, path: str):
        """Load the symbolic safety gate rules."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.safety_rules = data.get("rules", [])
        print(f"[KG] Loaded {len(self.safety_rules)} safety rules")

    def resolve_idioms(self, text: str) -> dict:
        """
        Scan input text for cultural idioms and return matches.

        Returns dict with:
          - matches: list of matched conditions
          - normalized_text: text with idioms replaced by clinical terms
          - escalation_flags: any triggered escalation rules
        """
        text_lower = text.lower()
        words = text_lower.split()
        matches = []
        seen_conditions = set()

        # Multi-word phrase matching (bigrams, trigrams)
        all_idiom_terms = list(self.idiom_index.keys())

        for n in range(4, 0, -1):  # Check 4-grams down to unigrams
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])

                # Exact match first
                if phrase in self.idiom_index:
                    info = self.idiom_index[phrase]
                    cid = info["condition_id"]
                    if cid not in seen_conditions:
                        seen_conditions.add(cid)
                        matches.append({
                            "idiom": phrase,
                            "match_score": 100,
                            **info,
                        })
                    continue

                # Fuzzy match
                result = process.extractOne(
                    phrase,
                    all_idiom_terms,
                    scorer=fuzz.ratio,
                    score_cutoff=self.threshold,
                )
                if result:
                    matched_term, score, _ = result
                    info = self.idiom_index[matched_term]
                    cid = info["condition_id"]
                    if cid not in seen_conditions:
                        seen_conditions.add(cid)
                        matches.append({
                            "idiom": phrase,
                            "matched_to": matched_term,
                            "match_score": score,
                            **info,
                        })

        # Build normalized text
        normalized = text
        for m in matches:
            normalized = normalized + f" [{m['clinical_name']}]"

        return {
            "matches": matches,
            "normalized_text": normalized,
            "condition_count": len(matches),
        }

    def check_safety_overrides(self, text: str) -> list[dict]:
        """
        Check text against the Symbolic Safety Gate.

        Returns a list of triggered rules (empty if none triggered).
        """
        text_lower = text.lower()
        triggered = []

        for rule in self.safety_rules:
            for keyword in rule["keywords"]:
                if keyword.lower() in text_lower:
                    triggered.append({
                        "rule_id": rule["id"],
                        "matched_keyword": keyword,
                        "override_to": rule["override_to"],
                        "flag": rule["flag"],
                        "action": rule["action"],
                    })
                    break  # One match per rule is enough

        return triggered

    def process(self, text: str) -> dict:
        """
        Full pipeline: resolve idioms + check safety overrides.

        Returns the combined result with final severity recommendation.
        """
        idiom_result = self.resolve_idioms(text)
        safety_overrides = self.check_safety_overrides(text)

        # Determine KG-suggested severity
        kg_severity = "LOW"
        severity_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "EMERGENCY": 3}

        for match in idiom_result["matches"]:
            s = match.get("default_severity", "LOW")
            if severity_rank.get(s, 0) > severity_rank.get(kg_severity, 0):
                kg_severity = s

        # Safety overrides always win
        final_override = None
        if safety_overrides:
            final_override = "EMERGENCY"

        return {
            "knowledge_graph": idiom_result,
            "safety_overrides": safety_overrides,
            "kg_suggested_severity": kg_severity,
            "final_override": final_override,
            "normalized_text": idiom_result["normalized_text"],
        }
