"""safety.py - prompt-injection detection used on uploaded data AND on chat messages."""
from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier|your) (instructions|rules|prompt)",
    r"disregard (all |any )?(the )?(previous|prior|above|your) ",
    r"you are now",
    r"system prompt",
    r"reveal (your|the) (prompt|instructions)",
    r"act as (a|an|the) ",
    r"(recommend|rank|select|choose) (this|me|us|our) (vendor|supplier|company|option)?\s*(as )?(first|#?1|number one|top)",
    r"developer mode",
    r"jailbreak",
    r"pretend (to be|you are)",
]
_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def looks_like_injection(text) -> bool:
    return bool(text) and bool(_RE.search(str(text)))


def sanitise(text, max_len: int = 200) -> str:
    """Text that will be shown to the LLM: injection removed, length capped."""
    if text is None:
        return ""
    t = str(text)
    if looks_like_injection(t):
        return "[removed by safety filter]"
    return t if len(t) <= max_len else t[:max_len] + "..."
