from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

from .graph_store import CodeGraph
from .llm import LLMClient, LLMResult
from .models import AgentResponse, ToolResult
from .repository_indexer import RepositoryIndexer
from .runtime.trace import TraceRecorder
from .tools.code_tools import CodeIntelligenceTools


class CodeGraphAgent:
    """LLM-guided tool agent for repository-scale code understanding."""

    def __init__(
        self,
        repo_path: str | Path,
        trace_dir: str | Path = "runs/traces",
        use_llm: bool = True,
        require_llm: bool = True,
        llm_client: LLMClient | None = None,
        graph: CodeGraph | None = None,
    ):
        self.repo_path = Path(repo_path)
        self.graph: CodeGraph = graph or RepositoryIndexer().build(self.repo_path)
        self.tools = CodeIntelligenceTools(self.graph)
        self.trace_dir = trace_dir
        self.use_llm = use_llm
        self.require_llm = require_llm
        self.llm = llm_client or LLMClient()

    def answer(self, query: str) -> AgentResponse:
        trace = TraceRecorder(self.trace_dir)
        plan, planner_result = self._plan(query)
        trace.add(
            "plan_tools",
            "llm_planner" if planner_result.used_llm else "deterministic_planner",
            query,
            self._planner_summary(plan, planner_result),
            0.9 if planner_result.used_llm else 0.75,
        )
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

        answer, composer_result = self._compose_answer(query, summaries, evidence)
        trace.add(
            "compose_grounded_answer",
            "llm_composer" if composer_result.used_llm else "template_composer",
            query,
            self._composer_summary(composer_result),
            0.88 if composer_result.used_llm else 0.72,
        )
        trace_path = trace.save(answer)
        confidence = sum(confidence_values) / max(1, len(confidence_values))
        return AgentResponse(answer, tools_used, evidence, trace_path, round(confidence, 3))

    def _plan(self, query: str) -> tuple[List[tuple[str, str, Callable[[str], ToolResult]]], LLMResult]:
        if self.use_llm:
            llm_result = self._llm_plan(query)
            tool_names = self._parse_tool_plan(llm_result.content) if llm_result.used_llm else []
            if tool_names:
                return self._plan_from_tool_names(tool_names), llm_result
            if self.require_llm:
                raise RuntimeError(f"LLM planner failed: {llm_result.error}")
        fallback = LLMResult(
            content="",
            used_llm=False,
            provider=self.llm.provider,
            model=self.llm.model,
            error="Using deterministic planner fallback",
        )
        return self._deterministic_plan(query), fallback

    def _deterministic_plan(self, query: str) -> List[tuple[str, str, Callable[[str], ToolResult]]]:
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

    def _llm_plan(self, query: str) -> LLMResult:
        tool_schemas = self.tools.function_schemas()
        return self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a code-understanding agent planner. Select tools for the user query. "
                        "Use these OpenAI-compatible function schemas to decide which tools to call: "
                        f"{json.dumps(tool_schemas, ensure_ascii=False)}. Return only compact JSON like "
                        "{\"tools\":[\"repository_summary\",\"impact_analysis\"]}. Always include "
                        "repository_summary first."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )

    def _parse_tool_plan(self, content: str) -> List[str]:
        try:
            payload = json.loads(self._extract_json(content))
        except json.JSONDecodeError:
            return []
        tools = payload.get("tools", [])
        if not isinstance(tools, list):
            return []
        allowed = {
            "repository_summary",
            "search_code",
            "explain_symbol",
            "call_chain",
            "impact_analysis",
            "test_recommendations",
        }
        cleaned = []
        for tool in tools:
            if isinstance(tool, str) and tool in allowed and tool not in cleaned:
                cleaned.append(tool)
        if "repository_summary" not in cleaned:
            cleaned.insert(0, "repository_summary")
        return cleaned

    def _extract_json(self, content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    def _plan_from_tool_names(self, tool_names: List[str]) -> List[tuple[str, str, Callable[[str], ToolResult]]]:
        registry: Dict[str, tuple[str, Callable[[str], ToolResult]]] = {
            "repository_summary": ("observe_repository", lambda _: self.tools.repository_summary()),
            "search_code": ("retrieve_relevant_symbols", self.tools.search_code),
            "explain_symbol": ("explain_symbol", self.tools.explain_symbol),
            "call_chain": ("trace_call_chain", self.tools.call_chain),
            "impact_analysis": ("analyze_reverse_dependencies", self.tools.impact_analysis),
            "test_recommendations": ("recommend_tests", self.tools.test_recommendations),
        }
        return [(registry[name][0], name, registry[name][1]) for name in tool_names if name in registry]

    def _compose_answer(self, query: str, summaries: List[str], evidence: List[Dict]) -> tuple[str, LLMResult]:
        if self.use_llm:
            evidence_blob = json.dumps(evidence, ensure_ascii=False)[:12000]
            llm_result = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a senior AI coding assistant. Write a concise, evidence-grounded answer. "
                            "Do not invent files, symbols or tests. Use only the provided tool evidence. "
                            "Mention uncertainty or missing evidence when relevant."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Query:\n{query}\n\nTool summaries:\n"
                            + "\n".join(f"- {summary}" for summary in summaries)
                            + f"\n\nEvidence JSON:\n{evidence_blob}"
                        ),
                    },
                ],
                temperature=0.1,
            )
            if llm_result.used_llm and llm_result.content.strip():
                return llm_result.content.strip(), llm_result
            if self.require_llm:
                raise RuntimeError(f"LLM composer failed: {llm_result.error}")

        fallback = LLMResult(
            content="",
            used_llm=False,
            provider=self.llm.provider,
            model=self.llm.model,
            error="Using template composer fallback",
        )
        return self._compose_template_answer(query, summaries, evidence), fallback

    def _compose_template_answer(self, query: str, summaries: List[str], evidence: List[Dict]) -> str:
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

    def _planner_summary(self, plan: List[tuple[str, str, Callable[[str], ToolResult]]], result: LLMResult) -> str:
        tools = [tool_name for _, tool_name, _ in plan]
        if result.used_llm:
            return f"LLM selected tools from {len(self.tools.function_schemas())} function schemas: {tools}"
        return f"Deterministic planner selected tools: {tools}; reason={result.error}"

    def _composer_summary(self, result: LLMResult) -> str:
        if result.used_llm:
            return f"LLM composed grounded answer with model={result.model}"
        return f"Template composer fallback; reason={result.error}"
