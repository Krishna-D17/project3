"""
Global Pytest Configuration and Database Isolation Bootstrap for MediaProof.

Ensures that whenever pytest is executed (including during test collection and discovery):
1. The database path is redirected to an isolated sandbox database in a temporary directory
   BEFORE any test file or application module can trigger database initialization.
2. MEDIAPROOF_DB_PATH environment variable is configured so dynamic path resolvers point to sandbox.
3. Both backend.paths.get_db_path and provenance.database.get_db_path are safely patched.
4. The production database (database/mediaproof.db) is never connected to, initialized, or modified.
"""

import os
from pathlib import Path
import tempfile
import pytest

# Allocate isolated sandbox database path for the pytest session
_SESSION_TMP_DIR = tempfile.TemporaryDirectory()
_SANDBOX_DB_PATH = Path(_SESSION_TMP_DIR.name) / "pytest_session_sandbox.db"

# 1. Environment variable override (evaluated dynamically by backend.paths.get_db_path)
os.environ["MEDIAPROOF_DB_PATH"] = str(_SANDBOX_DB_PATH)

# 2. Patch backend.paths in memory before any other module imports it
import backend.paths
backend.paths.get_db_path = lambda: _SANDBOX_DB_PATH

# 3. Patch provenance.database and initialize schema in isolated sandbox
import provenance.database
provenance.database.get_db_path = lambda: _SANDBOX_DB_PATH
provenance.database.init_db()


@pytest.fixture(scope="session")
def session_sandbox_db() -> Path:
    """Provides path to the session-isolated temporary SQLite database."""
    return _SANDBOX_DB_PATH


@pytest.fixture(autouse=True)
def enforce_database_isolation(monkeypatch, tmp_path) -> None:
    """
    Enforces per-test database isolation.
    Ensures that any test accessing the database uses either an isolated test database
    or the session sandbox, guaranteeing zero interaction with the production database.
    """
    test_db = tmp_path / "test_sandbox_mediaproof.db"
    monkeypatch.setenv("MEDIAPROOF_DB_PATH", str(test_db))
    monkeypatch.setattr(backend.paths, "get_db_path", lambda: test_db)
    monkeypatch.setattr(provenance.database, "get_db_path", lambda: test_db)
    provenance.database.init_db()
