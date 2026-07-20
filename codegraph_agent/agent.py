from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from .graph_store import CodeGraph
from .models import AgentResponse, ToolResult
from .repository_indexer import RepositoryIndexer
from .runtime.trace import TraceRecorder
from .tools.code_tools import CodeIntelligenceTools


class CodeGraphAgent:
    """Tool-using agent for repository-scale code understanding."""

    def __init__(self, repo_path: str | Path, trace_dir: str | Path = "runs/traces"):
        self.repo_path = Path(repo_path)
        self.graph: CodeGraph = RepositoryIndexer().build(self.repo_path)
        self.tools = CodeIntelligenceTools(self.graph)
        self.trace_dir = trace_dir

    def answer(self, query: str) -> AgentResponse:
        trace = TraceRecorder(self.trace_dir)
        plan = self._plan(query)
        evidence = []
        tools_used = []
        summaries = []
        confidence_values = []

        for step_name, tool_name, tool_fn in plan:
            result = tool_fn(query)
            trace.add(step_name, tool_name, query, result.summary, result.confidence)
            tools_used.append(tool_name)
            summaries.append(result.summary)
            evidence.append({"tool": result.tool, "data": result.data})
            confidence_values.append(result.confidence)

        answer = self._compose_answer(query, summaries, evidence)
        trace_path = trace.save(answer)
        confidence = sum(confidence_values) / max(1, len(confidence_values))
        return AgentResponse(answer, tools_used, evidence, trace_path, round(confidence, 3))

    def _plan(self, query: str) -> List[tuple[str, str, Callable[[str], ToolResult]]]:
        lowered = query.lower()
        plan: List[tuple[str, str, Callable[[str], ToolResult]]] = [
            ("observe_repository", "repository_summary", lambda _: self.tools.repository_summary())
        ]
        if any(keyword in lowered for keyword in ["impact", "影响", "change", "修改", "break"]):
            plan.append(("analyze_reverse_dependencies", "impact_analysis", self.tools.impact_analysis))
            plan.append(("recommend_tests", "test_recommendations", self.tools.test_recommendations))
        elif any(keyword in lowered for keyword in ["call", "调用", "链路", "chain", "flow"]):
            plan.append(("trace_call_chain", "call_chain", self.tools.call_chain))
            plan.append(("explain_entry_symbol", "explain_symbol", self.tools.explain_symbol))
        elif any(keyword in lowered for keyword in ["test", "测试", "coverage", "覆盖"]):
            plan.append(("recommend_tests", "test_recommendations", self.tools.test_recommendations))
        elif any(keyword in lowered for keyword in ["where", "find", "搜索", "在哪", "定位"]):
            plan.append(("retrieve_relevant_symbols", "search_code", self.tools.search_code))
        else:
            plan.append(("retrieve_relevant_symbols", "search_code", self.tools.search_code))
            plan.append(("explain_best_symbol", "explain_symbol", self.tools.explain_symbol))
        return plan

    def _compose_answer(self, query: str, summaries: List[str], evidence: List[Dict]) -> str:
        lines = [f"Query: {query}", "", "CodeGraphAgent findings:"]
        for summary in summaries:
            lines.append(f"- {summary}")
        lines.append("")
        lines.append("Key evidence:")
        for item in evidence:
            tool = item["tool"]
            data = item["data"]
            if tool == "search_code":
                for hit in data.get("hits", [])[:5]:
                    lines.append(
                        f"- {hit['name']} ({hit['kind']}) at {hit['file_path']}:{hit['line_start']} "
                        f"score={hit['score']}"
                    )
            elif tool == "impact_analysis":
                files = ", ".join(data.get("impacted_files", [])[:5]) or "no reverse dependencies found"
                lines.append(f"- impacted files: {files}")
            elif tool == "call_chain":
                for edge in data.get("edges", [])[:8]:
                    lines.append(f"- {edge['source_name']} -> {edge['target_name']} at {edge['file_path']}:{edge['line']}")
            elif tool == "test_recommendations":
                for test in data.get("tests", [])[:5]:
                    lines.append(f"- {test['name']}: {test['reason']}")
        return "\n".join(lines)
