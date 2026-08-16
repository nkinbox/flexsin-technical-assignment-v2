"""Query understanding: history-aware rewriting and multi-part decomposition.

Two problems this solves, both of which break retrieval if ignored:

1. **Follow-ups carry no searchable meaning.** "What about the second one?"
   embeds to nothing useful. Rewriting it against recent turns into a standalone
   question is the single highest-impact part of the chat-memory feature.

2. **Multi-part questions retrieve badly as one vector.** "Compare A's and B's
   margins" averaged into a single embedding matches neither A nor B well.
   Splitting into sub-queries and retrieving each independently fixes it.

The `# task:` first line is a dispatch tag read by DevProvider offline; Vertex
treats it as an ordinary comment.
"""

from __future__ import annotations

import json
import logging
import re

from backend.providers.base import ModelProvider

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6
MAX_SUB_QUERIES = 3

_REWRITE_SYSTEM = """\
# task: rewrite
You rewrite a follow-up question into a standalone search query.

Rules:
- Resolve every pronoun and ellipsis using the conversation history.
- Preserve the user's terminology exactly; do not introduce synonyms.
- Do not answer the question. Do not add information.
- If the question already stands alone, return it unchanged.
- Output the rewritten query only, with no quotes or preamble.
"""

_DECOMPOSE_SYSTEM = """\
# task: decompose
You split a question into the minimum set of independent search queries.

Rules:
- A question asking about one thing yields exactly one query.
- A comparison or multi-part question yields one query per part.
- Never produce more than 3 queries.
- Each query must stand alone and be searchable on its own.
- Respond with JSON only: {"parts": ["query one", "query two"]}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def rewrite_query(
    question: str, history: list[dict[str, str]], provider: ModelProvider
) -> str:
    """Resolve a follow-up into a standalone query."""
    if not history:
        return question

    recent = history[-MAX_HISTORY_TURNS:]
    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)
    user = f"<history>\n{transcript}\n</history>\n\n<question>{question}</question>"

    try:
        rewritten = provider.generate(_REWRITE_SYSTEM, user).strip()
    except Exception as exc:
        logger.warning("query rewrite failed, using original: %s", exc)
        return question

    if not rewritten:
        return question

    # A rewrite that balloons in length usually means the model answered
    # instead of rewriting; fall back rather than search with prose.
    if len(rewritten) > max(300, len(question) * 6):
        logger.warning("rewrite looked like an answer, using original question")
        return question

    return rewritten


def decompose_query(question: str, provider: ModelProvider) -> list[str]:
    """Split a multi-part question into independent sub-queries."""
    user = f"<question>{question}</question>"

    try:
        raw = provider.generate(_DECOMPOSE_SYSTEM, user)
    except Exception as exc:
        logger.warning("decomposition failed, using single query: %s", exc)
        return [question]

    parts = _parse_parts(raw)
    if not parts:
        return [question]

    # Guard against a model that "decomposes" into near-duplicates.
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts[:MAX_SUB_QUERIES]:
        key = part.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(part.strip())

    return unique or [question]


def _parse_parts(raw: str) -> list[str]:
    match = _JSON_BLOCK_RE.search(raw or "")
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    parts = payload.get("parts")
    if isinstance(parts, list):
        return [str(p) for p in parts if str(p).strip()]
    return []
