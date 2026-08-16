import warnings

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

from backend.providers.dev import DevProvider


@pytest.fixture(scope="session")
def provider() -> DevProvider:
    """Offline provider -- the unit tests never touch Weaviate or Vertex."""
    return DevProvider()
