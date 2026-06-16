"""Shared test fixtures.

The ingest fixture ensures Milvus is populated before any tests that
do retrieval. Session-scoped so ingest runs once per pytest session,
not once per test.
"""

import pytest
from app.ingest import main as ingest_main


@pytest.fixture(scope="session", autouse=True)
def ingest():
    """Populate the vector store before any tests run."""
    ingest_main()
