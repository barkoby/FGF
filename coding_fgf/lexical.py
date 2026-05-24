from __future__ import annotations

import re


def words(value: object) -> set[str]:
    text = str(value or "")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    raw_words = [part.lower() for part in re.split(r"\W+", text) if part]
    normalized: set[str] = set()
    for word in raw_words:
        normalized.add(word)
        if word == "conf":
            normalized.add("conference")
        if word == "pc":
            normalized.update({"program", "committee"})
        if word.endswith("ies"):
            normalized.add(word[:-3] + "y")
        if word.endswith("s") and len(word) > 3:
            normalized.add(word[:-1])
    if "chairman" in normalized:
        normalized.add("chair")
    if "people" in normalized:
        normalized.add("person")
    return normalized
