"""Neo4j connection management (Person 2).

Importing this module does not import or require the Neo4j driver; the driver is
loaded only when a connection is opened. A missing driver is an explicit,
actionable error, never a silent skip.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


class Neo4jUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jSettings":
        uri = os.environ.get("NEO4J_URI", "")
        user = os.environ.get("NEO4J_USER", "")
        password = os.environ.get("NEO4J_PASSWORD", "")
        database = os.environ.get("NEO4J_DATABASE", "neo4j")
        if not uri:
            raise Neo4jUnavailable(
                "NEO4J_URI is not set. Copy .env.example to .env and configure the connection, "
                "or run the pipeline without Neo4j (exports are produced from Python)."
            )
        return cls(uri=uri, user=user, password=password, database=database)


class Neo4jConnection:
    def __init__(self, settings: Neo4jSettings) -> None:
        self.settings = settings
        self._driver: Any | None = None

    @classmethod
    def from_env(cls) -> "Neo4jConnection":
        return cls(Neo4jSettings.from_env())

    def _ensure_driver(self) -> Any:
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - optional extra
            raise Neo4jUnavailable(
                "The neo4j driver is not installed. Run `pip install -r requirements/graph.txt` "
                "and `pip install neo4j`."
            ) from exc
        self._driver = GraphDatabase.driver(
            self.settings.uri,
            auth=(self.settings.user, self.settings.password),
        )
        return self._driver

    def verify(self) -> None:
        driver = self._ensure_driver()
        driver.verify_connectivity()

    @contextmanager
    def session(self) -> Iterator[Any]:
        driver = self._ensure_driver()
        session = driver.session(database=self.settings.database)
        try:
            yield session
        finally:
            session.close()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jConnection":
        self.verify()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
