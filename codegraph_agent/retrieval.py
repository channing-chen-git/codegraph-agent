from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .models import CodeSymbol, RepositoryIndex


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]+|[\u4e00-\u9fff]+")


@dataclass
class SearchHit:
    symbol_id: str
    score: float
    name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    snippet: str


class CodeRetriever:
    """BM25-like retriever over symbol signatures, docs and source snippets."""

    def __init__(self, index: RepositoryIndex):
        self.index = index
        self.documents: Dict[str, List[str]] = {}
        self.lengths: Dict[str, int] = {}
        self.df: Counter = Counter()
        self.avg_len = 1.0
        self._build()

    def search(self, query: str, top_k: int = 8) -> List[SearchHit]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scored = []
        total_docs = max(1, len(self.documents))
        for symbol_id, tokens in self.documents.items():
            tf = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                if token not in tf:
                    continue
                idf = math.log(1 + (total_docs - self.df[token] + 0.5) / (self.df[token] + 0.5))
                denom = tf[token] + 1.2 * (1 - 0.75 + 0.75 * self.lengths[symbol_id] / self.avg_len)
                score += idf * tf[token] * 2.2 / denom
            if score > 0:
                scored.append((score, symbol_id))
        scored.sort(reverse=True)
        return [self._hit(symbol_id, score) for score, symbol_id in scored[:top_k]]

    def _build(self) -> None:
        lengths = []
        for symbol in self.index.symbols.values():
            if symbol.kind in {"module", "translation_unit"}:
                continue
            text = " ".join(
                [
                    symbol.name,
                    symbol.kind,
                    symbol.signature,
                    symbol.docstring,
                    self._source_snippet(symbol),
                ]
            )
            tokens = self._tokenize(text)
            self.documents[symbol.symbol_id] = tokens
            self.lengths[symbol.symbol_id] = max(1, len(tokens))
            lengths.append(self.lengths[symbol.symbol_id])
            for token in set(tokens):
                self.df[token] += 1
        self.avg_len = sum(lengths) / max(1, len(lengths))

    def _hit(self, symbol_id: str, score: float) -> SearchHit:
        symbol = self.index.symbols[symbol_id]
        return SearchHit(
            symbol_id=symbol_id,
            score=round(score, 4),
            name=symbol.name,
            kind=symbol.kind,
            file_path=symbol.file_path,
            line_start=symbol.line_start,
            line_end=symbol.line_end,
            snippet=self._source_snippet(symbol, max_lines=8),
        )

    def _source_snippet(self, symbol: CodeSymbol, max_lines: int = 20) -> str:
        record = self.index.files.get(symbol.file_path)
        if not record:
            return ""
        lines = record.text.splitlines()
        start = max(0, symbol.line_start - 1)
        end = min(len(lines), symbol.line_end, start + max_lines)
        return "\n".join(lines[start:end])

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        for token in TOKEN_RE.findall(text.lower()):
            tokens.extend(self._split_identifier(token))
        return [token for token in tokens if len(token) > 1]

    def _split_identifier(self, token: str) -> List[str]:
        pieces = re.sub(r"([a-z])([A-Z])", r"\1 \2", token).replace("_", " ").split()
        return [piece.lower() for piece in pieces] or [token]
