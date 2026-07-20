from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .graph_store import CodeGraph
from .models import CodeEdge, CodeSymbol, RepositoryIndex
from .parsers.cpp_parser import CppParser
from .parsers.python_parser import PythonParser


DEFAULT_EXCLUDES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
}


class RepositoryIndexer:
    """Build a symbol graph for Python and C/C++ repositories."""

    def __init__(self, excludes: Iterable[str] | None = None):
        self.excludes = set(excludes or DEFAULT_EXCLUDES)
        self.parsers = {
            ".py": PythonParser(),
            ".c": CppParser(),
            ".cc": CppParser(),
            ".cpp": CppParser(),
            ".h": CppParser(),
            ".hpp": CppParser(),
        }

    def build(self, repo_path: str | Path) -> CodeGraph:
        root = Path(repo_path).resolve()
        index = RepositoryIndex(root=str(root))
        raw_edges: List[CodeEdge] = []

        for path in self._iter_source_files(root):
            parser = self.parsers.get(path.suffix.lower())
            if parser is None:
                continue
            record, symbols, edges = parser.parse(root, path)
            index.files[record.path] = record
            for symbol in symbols:
                index.symbols[symbol.symbol_id] = symbol
            raw_edges.extend(edges)

        index.edges = self._resolve_edges(index.symbols, raw_edges)
        return CodeGraph(index)

    def _iter_source_files(self, root: Path):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            parts = set(path.relative_to(root).parts)
            if parts & self.excludes:
                continue
            if path.suffix.lower() in self.parsers:
                yield path

    def _resolve_edges(self, symbols: Dict[str, CodeSymbol], edges: List[CodeEdge]) -> List[CodeEdge]:
        by_name: Dict[str, List[str]] = {}
        by_short_qualified: Dict[str, List[str]] = {}
        for symbol in symbols.values():
            by_name.setdefault(symbol.name, []).append(symbol.symbol_id)
            by_short_qualified.setdefault(symbol.symbol_id.split("::")[-1].split("@")[0], []).append(symbol.symbol_id)

        resolved: List[CodeEdge] = []
        for edge in edges:
            if edge.kind in {"calls", "inherits"} and edge.target not in symbols:
                target = self._resolve_name(edge.target, edge.file_path, by_name, by_short_qualified)
                resolved.append(
                    CodeEdge(
                        source=edge.source,
                        target=target or edge.target,
                        kind=edge.kind,
                        evidence=edge.evidence,
                        file_path=edge.file_path,
                        line=edge.line,
                    )
                )
            else:
                resolved.append(edge)
        return resolved

    def _resolve_name(
        self,
        name: str,
        file_path: str,
        by_name: Dict[str, List[str]],
        by_short_qualified: Dict[str, List[str]],
    ) -> str:
        short = name.split(".")[-1].split("::")[-1]
        candidates = by_name.get(short) or by_short_qualified.get(short) or []
        if not candidates:
            return ""
        same_file = [candidate for candidate in candidates if candidate.startswith(f"{file_path}::")]
        if same_file:
            return same_file[0]
        return candidates[0]
