from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import re
from typing import Dict, List

from ..graph_store import CodeGraph
from ..models import CodeEdge, CodeSymbol, ToolResult
from ..retrieval import CodeRetriever


class CodeIntelligenceTools:
    def __init__(self, graph: CodeGraph):
        self.graph = graph
        self.retriever = CodeRetriever(graph.index)

    def search_code(self, query: str, top_k: int = 8) -> ToolResult:
        hits = self.retriever.search(query, top_k=top_k)
        data = {"hits": [asdict(hit) for hit in hits]}
        summary = f"Found {len(hits)} relevant symbols for query: {query}"
        confidence = min(0.95, 0.2 + 0.1 * len(hits))
        return ToolResult("search_code", summary, data, confidence)

    def explain_symbol(self, symbol_query: str) -> ToolResult:
        symbol = self._best_symbol(symbol_query)
        if symbol is None:
            return ToolResult("explain_symbol", "No matching symbol found.", {"symbol": None}, 0.1)
        outgoing = self.graph.outgoing(symbol.symbol_id)
        incoming = self.graph.incoming(symbol.symbol_id)
        calls = [edge for edge in outgoing if edge.kind == "calls"]
        called_by = [edge for edge in incoming if edge.kind == "calls"]
        data = {
            "symbol": asdict(symbol),
            "calls": [self._edge_view(edge) for edge in calls[:20]],
            "called_by": [self._edge_view(edge) for edge in called_by[:20]],
            "defined_children": [self._edge_view(edge) for edge in outgoing if edge.kind == "defines"],
        }
        summary = (
            f"{symbol.name} is a {symbol.kind} in {symbol.file_path}:{symbol.line_start}. "
            f"It calls {len(calls)} symbols and is called by {len(called_by)} symbols."
        )
        return ToolResult("explain_symbol", summary, data, 0.85)

    def call_chain(self, symbol_query: str, depth: int = 3) -> ToolResult:
        symbol = self._best_symbol(symbol_query)
        if symbol is None:
            return ToolResult("call_chain", "No matching symbol found.", {"chains": []}, 0.1)
        edges = self.graph.call_chain(symbol.symbol_id, depth=depth)
        data = {"root": asdict(symbol), "edges": [self._edge_view(edge) for edge in edges]}
        summary = f"Call-chain analysis from {symbol.name} found {len(edges)} call edges within depth {depth}."
        return ToolResult("call_chain", summary, data, 0.8 if edges else 0.45)

    def impact_analysis(self, symbol_query: str, depth: int = 2) -> ToolResult:
        symbol = self._best_symbol(symbol_query)
        if symbol is None:
            return ToolResult("impact_analysis", "No matching symbol found.", {"impacted": []}, 0.1)
        edges = self.graph.impact_radius(symbol.symbol_id, depth=depth)
        impacted_files = sorted({edge.file_path for edge in edges if edge.file_path})
        impacted_symbols = []
        for edge in edges:
            source = self.graph.symbols.get(edge.source)
            if source:
                impacted_symbols.append(asdict(source))
        data = {
            "changed_symbol": asdict(symbol),
            "impacted_files": impacted_files,
            "impacted_symbols": impacted_symbols,
            "evidence_edges": [self._edge_view(edge) for edge in edges],
        }
        summary = (
            f"Changing {symbol.name} may affect {len(impacted_symbols)} symbols "
            f"across {len(impacted_files)} files."
        )
        return ToolResult("impact_analysis", summary, data, 0.82 if edges else 0.5)

    def test_recommendations(self, symbol_query: str) -> ToolResult:
        symbol = self._best_symbol(symbol_query)
        if symbol is None:
            return ToolResult("test_recommendations", "No matching symbol found.", {"tests": []}, 0.1)
        explain = self.explain_symbol(symbol.name)
        impact = self.impact_analysis(symbol.name)
        tests = [
            {
                "name": f"test_{symbol.name}_happy_path",
                "reason": f"Validate primary behavior of {symbol.name}.",
                "target": f"{symbol.file_path}:{symbol.line_start}",
            },
            {
                "name": f"test_{symbol.name}_edge_cases",
                "reason": "Cover empty input, invalid input and boundary values.",
                "target": f"{symbol.file_path}:{symbol.line_start}",
            },
        ]
        for file_path in impact.data.get("impacted_files", [])[:3]:
            tests.append(
                {
                    "name": f"test_integration_{symbol.name}_{file_path.replace('/', '_').replace('.', '_')}",
                    "reason": "Protect callers that depend on this symbol.",
                    "target": file_path,
                }
            )
        data = {"symbol": asdict(symbol), "tests": tests}
        summary = f"Generated {len(tests)} focused test recommendations for {symbol.name}."
        return ToolResult("test_recommendations", summary, data, 0.78)

    def repository_summary(self) -> ToolResult:
        languages = Counter(record.language for record in self.graph.index.files.values())
        kinds = Counter(symbol.kind for symbol in self.graph.symbols.values())
        edge_kinds = Counter(edge.kind for edge in self.graph.index.edges)
        data = {
            "files": len(self.graph.index.files),
            "symbols": len(self.graph.symbols),
            "edges": len(self.graph.index.edges),
            "languages": dict(languages),
            "symbol_kinds": dict(kinds),
            "edge_kinds": dict(edge_kinds),
        }
        summary = (
            f"Indexed {data['files']} files, {data['symbols']} symbols and "
            f"{data['edges']} dependency edges."
        )
        return ToolResult("repository_summary", summary, data, 0.9)

    def _best_symbol(self, query: str) -> CodeSymbol | None:
        exact = self._exact_symbol_from_query(query)
        if exact is not None:
            return exact
        direct = self.graph.find_symbols(query)
        if direct:
            return direct[0]
        hits = self.retriever.search(query, top_k=1)
        if not hits:
            return None
        return self.graph.symbols.get(hits[0].symbol_id)

    def _exact_symbol_from_query(self, query: str) -> CodeSymbol | None:
        query_identifiers = {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", query)
        }
        if not query_identifiers:
            return None
        matches = [
            symbol
            for symbol in self.graph.symbols.values()
            if symbol.name.lower() in query_identifiers
        ]
        if not matches:
            return None
        preferred_kinds = {"function": 0, "method": 1, "class": 2}
        matches.sort(
            key=lambda symbol: (
                preferred_kinds.get(symbol.kind, 9),
                symbol.file_path,
                symbol.line_start,
            )
        )
        return matches[0]

    def _edge_view(self, edge: CodeEdge) -> Dict:
        target = self.graph.symbols.get(edge.target)
        source = self.graph.symbols.get(edge.source)
        return {
            "source": edge.source,
            "source_name": source.name if source else edge.source,
            "target": edge.target,
            "target_name": target.name if target else edge.target,
            "kind": edge.kind,
            "evidence": edge.evidence,
            "file_path": edge.file_path,
            "line": edge.line,
        }
