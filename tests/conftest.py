"""
Shared pytest fixtures.

Every test gets its own isolated in-memory database. Nothing here touches the git-tracked
`src/orbital_engine/data/planets.db`, and nothing should ever be added that does.
"""
from __future__ import annotations

from typing import Callable, Iterator, List, Tuple

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine.database import Base


@pytest.fixture
def db_session_factory() -> Iterator[Callable[[], Session]]:
    """
    Returns a callable that produces an independent, empty in-memory database on each call.

    Use this when one test needs several databases that must not see each other's rows - for example
    running the same scenario at three step sizes without the second run inheriting the first's
    bodies. For the ordinary single-database case use `db_session` below.

    Why StaticPool: every new connection to `sqlite:///:memory:` gets its *own* fresh database. With
    the default pool a second connection would silently open an empty one, so a session that looked
    connected would find no tables. StaticPool holds a single connection for the engine's lifetime,
    which is what makes the in-memory database stable across sessions built from it.
    """
    created: List[Tuple[Engine, Session]] = []

    def _make() -> Session:
        engine = create_engine(
            "sqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()
        created.append((engine, session))
        return session

    yield _make

    # Teardown runs even when the test fails, so a raised assertion cannot leak connections.
    for engine, session in created:
        session.close()
        engine.dispose()  # The in-memory database ceases to exist with its last connection.


@pytest.fixture
def db_session(db_session_factory: Callable[[], Session]) -> Session:
    """A single fresh in-memory database. The common case."""
    return db_session_factory()
