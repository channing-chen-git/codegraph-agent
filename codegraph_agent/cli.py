from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import CodeGraphAgent
from .repository_indexer import RepositoryIndexer


def main() -> None:
    parser = argparse.ArgumentParser(description="CodeGraphAgent CLI")
    parser.add_argument("--repo", required=True, help="Repository path to analyze")
    parser.add_argument("--query", help="Natural-language code question")
    parser.add_argument("--session-id", default="default", help="Conversation session id for memory")
    parser.add_argument("--index-out", help="Write graph index JSON to this path")
    parser.add_argument("--trace-dir", default="runs/traces")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM planner/composer fallback path")
    args = parser.parse_args()

    if args.index_out:
        graph = RepositoryIndexer().build(args.repo)
        graph.save(args.index_out)
        print(f"Index written to {args.index_out}")
        return

    if not args.query:
        raise SystemExit("--query is required unless --index-out is provided")

    agent = CodeGraphAgent(
        args.repo,
        trace_dir=args.trace_dir,
        use_llm=not args.no_llm,
        require_llm=not args.no_llm,
    )
    response = agent.answer(args.query, session_id=args.session_id)
    print(response.answer)
    print("")
    print(
        json.dumps(
            {
                "confidence": response.confidence,
                "trace_path": response.trace_path,
                "session_id": response.session_id,
                "memory_path": response.memory_path,
                "rounds": response.rounds,
                "errors": response.errors,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
