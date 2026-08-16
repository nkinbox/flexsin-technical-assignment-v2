"""Google Vertex AI provider -- Gemini for generation and vision, and the
Vertex embedding model for vectors.

Authentication is Application Default Credentials, so this file contains no
credential handling at all. Locally that means an impersonated service account
(`gcloud auth application-default login --impersonate-service-account=...`);
on Compute Engine it is the attached service account read from the metadata
server. Same code both places -- see README section 8.
"""

from __future__ import annotations

import logging
import time

from backend.config import get_settings

logger = logging.getLogger(__name__)

# Retry policy for transient failures (429 rate limit, 5xx). Deliberately small:
# ingest batches are large, and failing fast beats masking a misconfiguration.
_MAX_RETRIES = 4
_BACKOFF_BASE_SECONDS = 1.5

# Vertex embedding endpoints cap the number of instances per request.
_EMBED_BATCH_SIZE = 100

_VISION_PROMPT = """\
Transcribe this document page into clean Markdown.

Rules:
- Preserve heading hierarchy using # and ##.
- Render tables as Markdown tables, keeping the header row intact.
- Correct reading order for multi-column layouts (read each column fully, in order).
- Transcribe text verbatim. Do not summarise, correct, or infer missing content.
- For any chart, graph, or diagram, add a line starting with "Figure:" describing
  what it shows, including axis labels and the overall trend.
- Ignore page headers, footers, and page numbers.
- Output only the Markdown. No preamble, no commentary.
"""


class VertexProvider:
    """Gemini generation + vision and Vertex embeddings via the google-genai SDK."""

    name = "vertex"

    def __init__(self) -> None:
        from google import genai  # imported lazily so dev mode never needs it

        settings = get_settings()
        if not settings.gcp_project_id:
            raise RuntimeError(
                "GCP_PROJECT_ID is required when PROVIDER=vertex. "
                "Set it in .env, or use PROVIDER=dev to run offline."
            )

        self._settings = settings
        self._client = genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_region,
        )

    # --- Retry helper -----------------------------------------------------
    @staticmethod
    def _with_retry(operation, description: str):
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return operation()
            except Exception as exc:  # SDK raises provider-specific types
                message = str(exc).lower()
                transient = any(
                    marker in message
                    for marker in ("429", "resource_exhausted", "503", "500", "unavailable", "deadline")
                )
                last_error = exc
                if not transient or attempt == _MAX_RETRIES - 1:
                    raise
                delay = _BACKOFF_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                    description, attempt + 1, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
        raise last_error  # unreachable; satisfies type checkers

    # --- Embeddings -------------------------------------------------------
    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            response = self._with_retry(
                lambda b=batch: self._client.models.embed_content(
                    model=self._settings.embed_model,
                    contents=b,
                    config=types.EmbedContentConfig(task_type=task_type),
                ),
                f"embed_content[{task_type}]",
            )
            vectors.extend(list(e.values) for e in response.embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    # --- Generation -------------------------------------------------------
    def generate(self, system: str, user: str) -> str:
        from google.genai import types

        response = self._with_retry(
            lambda: self._client.models.generate_content(
                model=self._settings.gen_model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    # Grounded extraction and citation are not creative tasks;
                    # low temperature keeps answers close to the source text.
                    temperature=0.1,
                ),
            ),
            "generate_content",
        )
        return (response.text or "").strip()

    # --- Vision -----------------------------------------------------------
    def extract_from_image(self, image_bytes: bytes, mime_type: str) -> str:
        from google.genai import types

        response = self._with_retry(
            lambda: self._client.models.generate_content(
                model=self._settings.gen_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    _VISION_PROMPT,
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            ),
            "extract_from_image",
        )
        return (response.text or "").strip()

    # --- Tokenisation -----------------------------------------------------
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            response = self._client.models.count_tokens(
                model=self._settings.gen_model, contents=text
            )
            return int(response.total_tokens)
        except Exception as exc:
            # Token counting is a sizing hint, not correctness-critical. A
            # network blip during chunking should not fail the whole ingest.
            logger.warning("count_tokens failed, using heuristic: %s", exc)
            return max(1, len(text) // 4)
