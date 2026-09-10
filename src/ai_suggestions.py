"""
ai_suggestions.py
------------------
Simulates the output of an AI/LLM-assisted normalization step: given a raw,
noisy technician-written issue description, propose (a) a canonical category
and (b) a cleaned-up normalized description, each with a confidence score.

In production this module's `suggest_for_text` would likely be swapped for
an actual LLM call (e.g. a classification prompt via the Anthropic API) or a
fine-tuned small model. It is implemented here with transparent, inspectable
rule-based + fuzzy-matching logic so the app is fully self-contained, fast,
and deterministic for a live demo -- while keeping the exact same interface
(text in -> category + normalized text + confidence out) that a real model
call would need to satisfy. That interface boundary is the point: swapping
the engine later should not require touching the UI or the curation queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

# Canonical categories and the keyword signatures the "model" looks for.
# Order matters: first sufficiently-confident match wins.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Overheating": ["temp", "temperature", "overheat", "hot", "heat"],
    "Vibration Anomaly": ["vibr", "vibration", "imbalance", "shaft"],
    "Oil Leak": ["leak", "oil", "pooling", "seal"],
    "Unusual Noise": ["noise", "noize", "grinding", "sound"],
    "Electrical Fault": ["elec", "electrical", "breaker", "power fluctuation", "fault"],
    "Bearing Wear": ["bearing", "worn", "wear"],
    "No Issue Found": ["no issue", "routine check ok", "ok"],
}

CANONICAL_LABEL_TEXT: dict[str, str] = {
    "Overheating": "Overheating detected",
    "Vibration Anomaly": "Abnormal vibration detected",
    "Oil Leak": "Oil leak detected",
    "Unusual Noise": "Unusual operating noise reported",
    "Electrical Fault": "Electrical fault suspected",
    "Bearing Wear": "Bearing wear detected",
    "No Issue Found": "No issue found",
}


@dataclass
class Suggestion:
    category: str
    normalized_text: str
    confidence: float  # 0.0 - 1.0


def _fuzzy_contains(haystack: str, needle: str, threshold: float = 0.8) -> bool:
    """
    Cheap typo-tolerant substring check: slides a window of the needle's
    length across the haystack and checks similarity ratio. Lets the engine
    match things like 'noize' -> 'noise' without a real NLP model.
    """
    if needle in haystack:
        return True
    window = len(needle)
    if window == 0 or window > len(haystack):
        return False
    for start in range(len(haystack) - window + 1):
        chunk = haystack[start:start + window]
        if SequenceMatcher(None, chunk, needle).ratio() >= threshold:
            return True
    return False


def suggest_for_text(raw_text: str | float) -> Suggestion:
    """
    Produce a simulated AI suggestion for a single raw issue description.

    Missing/blank input -> low-confidence 'Unknown' suggestion, since a real
    model can't categorize what isn't there either; this keeps the curation
    queue honest about what actually needs a human to go re-check the source.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return Suggestion(category="Unknown", normalized_text="", confidence=0.0)

    text = raw_text.strip().lower()

    best_category = "Unknown"
    best_score = 0.0

    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if _fuzzy_contains(text, kw))
        if hits == 0:
            continue
        # Confidence heuristic: more/stronger keyword hits -> higher
        # confidence, capped at 0.97 (a real model would rarely report 1.0
        # either -- this keeps the "accept/reject" workflow meaningful).
        score = min(0.97, 0.55 + 0.14 * hits)
        if score > best_score:
            best_score = score
            best_category = category

    normalized = CANONICAL_LABEL_TEXT.get(best_category, raw_text.strip().capitalize())
    return Suggestion(category=best_category, normalized_text=normalized, confidence=round(best_score, 2))


def suggest_batch(texts) -> list[Suggestion]:
    """Vectorized-friendly wrapper: apply suggest_for_text over an iterable."""
    return [suggest_for_text(t) for t in texts]
