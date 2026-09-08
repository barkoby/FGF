from __future__ import annotations

PAPER_SCENARIOS = [
    "cmt_renamed",
    "conference_renamed",
    "sigkdd_renamed",
    "cmt_structured",
    "conference_structured",
    "sigkdd_structured",
    "sigkdd_mixed",
    "conference_nofks",
    "cmt_denormalized",
]

# Reference scores are not distributed; comparison fields remain empty when absent.
PAPER_TARGET_F1: dict[str, float] = {}

SOURCE_BASE = "urn:coding-fgf:source:"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
REQUESTED_MATCH_MODEL = "GPT-5.4-nano"
REQUESTED_CODE_MODEL = REQUESTED_MATCH_MODEL
FALLBACK_MATCH_MODEL = "gpt-5-nano"
FALLBACK_CODE_MODEL = None
