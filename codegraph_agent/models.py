from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CodeSymbol:
    symbol_id: str
    name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    language: str = "unknown"
    parent: Optional[str] = None


@dataclass
class CodeEdge:
    source: str
    target: str
    kind: str
    evidence: str = ""
    file_path: str = ""
    line: int = 0


@dataclass
class FileRecord:
    path: str
    language: str
    text: str
    imports: List[str] = field(default_factory=list)


@dataclass
class RepositoryIndex:
    root: str
    files: Dict[str, FileRecord] = field(default_factory=dict)
    symbols: Dict[str, CodeSymbol] = field(default_factory=dict)
    edges: List[CodeEdge] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "root": self.root,
            "files": {key: asdict(value) for key, value in self.files.items()},
            "symbols": {key: asdict(value) for key, value in self.symbols.items()},
            "edges": [asdict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RepositoryIndex":
        index = cls(root=data["root"])
        index.files = {
            key: FileRecord(**value) for key, value in data.get("files", {}).items()
        }
        index.symbols = {
            key: CodeSymbol(**value) for key, value in data.get("symbols", {}).items()
        }
        index.edges = [CodeEdge(**edge) for edge in data.get("edges", [])]
        return index


@dataclass
class ToolResult:
    tool: str
    summary: str
    data: Dict
    confidence: float = 1.0


@dataclass
class AgentTraceStep:
    step: str
    tool: str
    query: str
    summary: str
    confidence: float


@dataclass
class AgentResponse:
    answer: str
    tools_used: List[str]
    evidence: List[Dict]
    trace_path: Optional[str] = None
    confidence: float = 0.0


def normalize_path(path: str | Path) -> str:
    return str(Path(path).as_posix())
