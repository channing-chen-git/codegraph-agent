# Project Card: CodeGraphAgent

## One-liner

A repository-level code understanding agent with symbol graph construction,
cross-file dependency reasoning, code retrieval, impact analysis and offline
evaluation.

## Problem

Large codebases require more than single-file completion. A useful code agent
needs to understand symbols, call relationships, imports, dependency direction,
and the evidence behind an answer.

## Capabilities

- Build symbol graphs for Python and lightweight C/C++ code.
- Search repository context using code-aware lexical retrieval.
- Explain functions and classes with callers and callees.
- Trace call chains across files.
- Estimate impact radius for a changed function.
- Generate targeted test recommendations.
- Save traces and run offline evaluation tasks.

## Limitations

The current parser is intentionally lightweight. Production C/C++ support should
use Tree-sitter or clangd. The default planner is deterministic; it can be
replaced by an LLM planner while keeping the same tool boundary and evaluation
workflow.
