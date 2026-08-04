from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import re
from typing import Dict, List

from ..graph_store import CodeGraph
from ..models import CodeEdge, CodeSymbol, ToolResult
from ..retrieval import CodeRetriever
from .change_intel import (
    changed_symbols,
    load_coverage,
    load_diff_file,
    load_runtime_traces,
    runtime_edges_for_symbol,
    symbol_coverage,
)


class CodeIntelligenceTools:
    def __init__(self, graph: CodeGraph):
        self.graph = graph
        self.retriever = CodeRetriever(graph.index)

    def function_schemas(self) -> List[Dict]:
        query_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language code question or symbol name.",
                }
            },
            "required": ["query"],
        }
        depth_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol name or code question to analyze.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum graph traversal depth.",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["query"],
        }
        return [
            self._function_schema(
                "repository_summary",
                "Summarize indexed files, symbols, languages and dependency edge counts.",
                {"type": "object", "properties": {}, "required": []},
            ),
            self._function_schema(
                "search_code",
                "Retrieve relevant symbols by name, signature, docstring and source snippet.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of symbol hits to return.",
                            "default": 8,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                },
            ),
            self._function_schema(
                "explain_symbol",
                "Explain one symbol and show direct incoming/outgoing call evidence.",
                query_schema,
            ),
            self._function_schema(
                "call_chain",
                "Trace outgoing call-chain edges from a symbol.",
                depth_schema,
            ),
            self._function_schema(
                "impact_analysis",
                "Find reverse dependencies and impacted files if a symbol changes.",
                depth_schema,
            ),
            self._function_schema(
                "test_recommendations",
                "Recommend unit and integration tests based on symbol impact evidence.",
                query_schema,
            ),
            self._function_schema(
                "pr_change_analysis",
                "Analyze changed symbols from a unified diff file and connect them to impact evidence.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural-language question."},
                        "diff_path": {
                            "type": "string",
                            "description": "Path to a unified diff file, relative to the repository root or absolute.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            self._function_schema(
                "coverage_gap_analysis",
                "Compare impacted symbols with coverage.py-style JSON data to find test gaps.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Symbol or code question."},
                        "coverage_path": {
                            "type": "string",
                            "description": "Path to coverage JSON, relative to the repository root or absolute.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            self._function_schema(
                "runtime_trace_analysis",
                "Use runtime call-trace evidence to supplement static CodeGraph impact analysis.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Symbol or code question."},
                        "trace_path": {
                            "type": "string",
                            "description": "Path to runtime trace JSON, relative to the repository root or absolute.",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

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

    def pr_change_analysis(self, query: str, diff_path: str = "changes.diff") -> ToolResult:
        path = self._resolve_artifact_path(diff_path)
        if not path.exists():
            return ToolResult(
                "pr_change_analysis",
                f"No diff file found at {path}.",
                {"diff_path": str(path), "changed_files": [], "changed_symbols": []},
                0.2,
            )
        changed_files = load_diff_file(path)
        symbols = changed_symbols(self.graph.symbols.values(), changed_files)
        impacted = []
        for symbol in symbols:
            impact = self.impact_analysis(symbol.name)
            impacted.append(
                {
                    "changed_symbol": symbol.name,
                    "file_path": symbol.file_path,
                    "line_start": symbol.line_start,
                    "impact": impact.data,
                }
            )
        data = {
            "diff_path": str(path),
            "changed_files": [
                {
                    "path": item.path,
                    "added_lines": item.added_lines,
                    "removed_lines": item.removed_lines,
                }
                for item in changed_files
            ],
            "changed_symbols": [self._symbol_view(symbol) for symbol in symbols],
            "impact_by_changed_symbol": impacted,
        }
        summary = (
            f"Diff analysis found {len(changed_files)} changed files and "
            f"{len(symbols)} changed symbols."
        )
        return ToolResult("pr_change_analysis", summary, data, 0.82 if symbols else 0.45)

    def coverage_gap_analysis(self, symbol_query: str, coverage_path: str = "coverage.json") -> ToolResult:
        path = self._resolve_artifact_path(coverage_path)
        if not path.exists():
            return ToolResult(
                "coverage_gap_analysis",
                f"No coverage file found at {path}.",
                {"coverage_path": str(path), "coverage": []},
                0.2,
            )
        symbol = self._best_symbol(symbol_query)
        if symbol is None:
            return ToolResult("coverage_gap_analysis", "No matching symbol found.", {"coverage": []}, 0.1)
        coverage = load_coverage(path)
        impact = self.impact_analysis(symbol.name)
        impacted_symbols = [
            self.graph.symbols[item["symbol_id"]]
            for item in impact.data.get("impacted_symbols", [])
            if item.get("symbol_id") in self.graph.symbols
        ]
        targets = [symbol] + impacted_symbols
        data = {
            "coverage_path": str(path),
            "root_symbol": self._symbol_view(symbol),
            "coverage": [symbol_coverage(target, coverage) for target in targets],
        }
        gaps = [
            item
            for item in data["coverage"]
            if item.get("coverage_status") != "covered"
        ]
        summary = (
            f"Coverage analysis checked {len(targets)} symbols and found "
            f"{len(gaps)} symbols with unknown or partial coverage."
        )
        return ToolResult("coverage_gap_analysis", summary, data, 0.8)

    def runtime_trace_analysis(self, symbol_query: str, trace_path: str = "runtime_traces.json") -> ToolResult:
        path = self._resolve_artifact_path(trace_path)
        if not path.exists():
            return ToolResult(
                "runtime_trace_analysis",
                f"No runtime trace file found at {path}.",
                {"trace_path": str(path), "runtime_edges": []},
                0.2,
            )
        symbol = self._best_symbol(symbol_query)
        if symbol is None:
            return ToolResult("runtime_trace_analysis", "No matching symbol found.", {"runtime_edges": []}, 0.1)
        traces = load_runtime_traces(path)
        edges = runtime_edges_for_symbol(symbol.name, traces)
        data = {
            "trace_path": str(path),
            "symbol": self._symbol_view(symbol),
            "runtime_edges": edges,
            "note": "Runtime evidence covers only executed paths and should complement static impact analysis.",
        }
        summary = f"Runtime trace analysis found {len(edges)} observed runtime edges for {symbol.name}."
        return ToolResult("runtime_trace_analysis", summary, data, 0.78 if edges else 0.45)

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

    def _symbol_view(self, symbol: CodeSymbol) -> Dict:
        return {
            "symbol_id": symbol.symbol_id,
            "name": symbol.name,
            "kind": symbol.kind,
            "file_path": symbol.file_path,
            "line_start": symbol.line_start,
            "line_end": symbol.line_end,
            "language": symbol.language,
        }

    def _resolve_artifact_path(self, path: str) -> object:
        from pathlib import Path

        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return Path(self.graph.index.root) / candidate

    def _function_schema(self, name: str, description: str, parameters: Dict) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
