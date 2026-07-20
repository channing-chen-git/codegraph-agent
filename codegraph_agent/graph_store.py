from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .models import CodeEdge, CodeSymbol, RepositoryIndex


class CodeGraph:
    """In-memory code graph with lightweight dependency queries."""

    def __init__(self, index: RepositoryIndex):
        self.index = index
        self._outgoing: Dict[str, List[CodeEdge]] = {}
        self._incoming: Dict[str, List[CodeEdge]] = {}
        for edge in index.edges:
            self._outgoing.setdefault(edge.source, []).append(edge)
            self._incoming.setdefault(edge.target, []).append(edge)

    @property
    def symbols(self) -> Dict[str, CodeSymbol]:
        return self.index.symbols

    def outgoing(self, symbol_id: str, kinds: Iterable[str] | None = None) -> List[CodeEdge]:
        edges = self._outgoing.get(symbol_id, [])
        if kinds is None:
            return list(edges)
        allowed = set(kinds)
        return [edge for edge in edges if edge.kind in allowed]

    def incoming(self, symbol_id: str, kinds: Iterable[str] | None = None) -> List[CodeEdge]:
        edges = self._incoming.get(symbol_id, [])
        if kinds is None:
            return list(edges)
        allowed = set(kinds)
        return [edge for edge in edges if edge.kind in allowed]

    def find_symbols(self, name: str) -> List[CodeSymbol]:
        lowered = name.lower()
        return [
            symbol
            for symbol in self.symbols.values()
            if symbol.name.lower() == lowered or lowered in symbol.name.lower()
        ]

    def impact_radius(self, symbol_id: str, depth: int = 2) -> List[CodeEdge]:
        """Return reverse call/import edges that may be affected by a change."""
        seen = {symbol_id}
        frontier = [symbol_id]
        result: List[CodeEdge] = []
        for _ in range(depth):
            next_frontier: List[str] = []
            for current in frontier:
                for edge in self.incoming(current, kinds=["calls", "imports", "inherits"]):
                    result.append(edge)
                    if edge.source not in seen:
                        seen.add(edge.source)
                        next_frontier.append(edge.source)
            frontier = next_frontier
            if not frontier:
                break
        return result

    def call_chain(self, symbol_id: str, depth: int = 3) -> List[CodeEdge]:
        seen = {symbol_id}
        frontier = [symbol_id]
        result: List[CodeEdge] = []
        for _ in range(depth):
            next_frontier: List[str] = []
            for current in frontier:
                for edge in self.outgoing(current, kinds=["calls"]):
                    result.append(edge)
                    if edge.target not in seen:
                        seen.add(edge.target)
                        next_frontier.append(edge.target)
            frontier = next_frontier
            if not frontier:
                break
        return result

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.index.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CodeGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(RepositoryIndex.from_dict(data))
