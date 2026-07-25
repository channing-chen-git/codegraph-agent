from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .agent import CodeGraphAgent
from .runtime.config import resolve_runtime_path


def run_eval(
    repo_path: str | Path,
    tasks_path: str | Path,
    trace_dir: str | Path = "eval_traces",
) -> Dict:
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    agent = CodeGraphAgent(repo_path, trace_dir=trace_dir)
    results = []
    passed = 0
    for task in tasks:
        response = agent.answer(task["query"])
        answer_blob = json.dumps(
            {"answer": response.answer, "evidence": response.evidence},
            ensure_ascii=False,
        ).lower()
        matched = [term for term in task["expected_terms"] if term.lower() in answer_blob]
        ok = len(matched) == len(task["expected_terms"])
        passed += int(ok)
        results.append(
            {
                "id": task["id"],
                "query": task["query"],
                "passed": ok,
                "matched_terms": matched,
                "expected_terms": task["expected_terms"],
                "confidence": response.confidence,
                "trace_path": response.trace_path,
            }
        )
    return {
        "passed": passed,
        "total": len(tasks),
        "pass_rate": round(passed / max(1, len(tasks)), 4),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CodeGraphAgent offline evaluation.")
    parser.add_argument("--repo", default="examples/mini_repo")
    parser.add_argument("--tasks", default="eval/tasks.json")
    parser.add_argument("--out", default="eval_result.json")
    parser.add_argument("--trace-dir", default="eval_traces")
    args = parser.parse_args()
    result = run_eval(args.repo, args.tasks, trace_dir=args.trace_dir)
    output = resolve_runtime_path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
