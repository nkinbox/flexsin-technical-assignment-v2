"""Offline provider -- runs the full pipeline with no cloud credentials.

Purpose: prove the plumbing (upload -> extract -> chunk -> store -> retrieve ->
cite) end to end before Google Cloud is configured, and keep the test suite
hermetic and free.

NOT a substitute for the real model. Embeddings here are feature-hashed
bag-of-words vectors, so retrieval is *lexical* -- it genuinely ranks by term
overlap, which makes offline demos meaningful, but it has no semantic
understanding. Answers are templates, not generated text. Evaluate quality only
with PROVIDER=vertex.

Provider dispatch convention
----------------------------
System prompts begin with a machine-readable task tag (``# task: rewrite``).
Vertex ignores it as a comment; this provider uses it to pick a stub behaviour.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from backend.config import get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopwords are dropped before hashing. Without this, "What was the ...?" and
# "Who won the ...?" share enough common tokens that an unrelated question
# scores similarly to a relevant one, which collapses the margin the grounding
# threshold depends on. A real embedding model handles this implicitly.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    from by with without into over under is are was were be been being do does
    did doing have has had having i you he she it we they me him her them my
    your his our their what which who whom whose when where why how all any
    both each few more most other some such no nor not only own same so too
    very can will just should now about as
    """.split()
)
_TASK_RE = re.compile(r"^#\s*task:\s*(\w+)", re.MULTILINE)
_QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL)
# Matches the numbered context entries built by rag/generate.py:
#   [1] (report.pdf, p.4) body text...
_CITATION_RE = re.compile(r"^\[(\d+)\]\s*\([^)]*\)\s*(.*)$", re.MULTILINE)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _feature_hash(text: str, dim: int) -> list[float]:
    """Deterministic sparse bag-of-words vector, L2-normalised.

    Unigrams carry full weight, bigrams half -- enough word-order sensitivity
    that phrase matches outrank bag matches. Cosine similarity over these
    vectors approximates lexical overlap, so hybrid search behaves sensibly
    offline.
    """
    vec = [0.0] * dim
    tokens = _tokenize(text)

    for token in tokens:
        idx = int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16)
        vec[idx % dim] += 1.0

    for first, second in zip(tokens, tokens[1:]):
        idx = int(
            hashlib.blake2b(f"{first}_{second}".encode(), digest_size=8).hexdigest(), 16
        )
        vec[idx % dim] += 0.5

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        # Empty/punctuation-only text: return a valid unit vector rather than
        # zeros, which would make cosine similarity undefined.
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


class DevProvider:
    """Deterministic offline stand-in for VertexProvider."""

    name = "dev"

    def __init__(self) -> None:
        self._dim = get_settings().embed_dim

    # --- Embeddings -------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_feature_hash(t, self._dim) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _feature_hash(text, self._dim)

    # --- Generation -------------------------------------------------------
    def generate(self, system: str, user: str) -> str:
        task_match = _TASK_RE.search(system)
        task = task_match.group(1) if task_match else "answer"

        if task == "rewrite":
            return self._stub_rewrite(user)
        if task == "decompose":
            return self._stub_decompose(user)
        return self._stub_answer(user)

    @staticmethod
    def _extract_question(user: str) -> str:
        match = _QUESTION_RE.search(user)
        return match.group(1).strip() if match else user.strip()

    def _stub_rewrite(self, user: str) -> str:
        # No coreference resolution offline -- pass the question through
        # unchanged. Follow-ups using pronouns will retrieve poorly in dev mode;
        # that is expected and is exactly what the real provider fixes.
        return self._extract_question(user)

    def _stub_decompose(self, user: str) -> str:
        return json.dumps({"parts": [self._extract_question(user)]})

    def _stub_answer(self, user: str) -> str:
        entries = _CITATION_RE.findall(user)
        if not entries:
            return (
                "I don't have enough information in the uploaded documents "
                "to answer that."
            )

        number, body = entries[0]
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        excerpt = " ".join(sentences[:2]).strip()
        return (
            f"[dev mode -- template answer, not generated] "
            f"The most relevant passage says: {excerpt} [{number}]"
        )

    # --- Vision -----------------------------------------------------------
    def extract_from_image(self, image_bytes: bytes, mime_type: str) -> str:
        digest = hashlib.blake2b(image_bytes, digest_size=6).hexdigest()
        return (
            f"> _[dev mode: vision extraction unavailable. "
            f"{len(image_bytes)} bytes, {mime_type}, digest {digest}. "
            f"Set PROVIDER=vertex to transcribe this page.]_"
        )

    # --- Tokenisation -----------------------------------------------------
    def count_tokens(self, text: str) -> int:
        # ~4 characters per token is a reasonable English approximation and
        # keeps chunk sizing consistent offline.
        return max(1, len(text) // 4)
