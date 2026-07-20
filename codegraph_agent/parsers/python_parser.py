from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

from ..models import CodeEdge, CodeSymbol, FileRecord


class PythonParser:
    language = "python"

    def parse(self, root: Path, path: Path) -> Tuple[FileRecord, List[CodeSymbol], List[CodeEdge]]:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        record = FileRecord(path=relative, language=self.language, text=text)
        symbols: List[CodeSymbol] = []
        edges: List[CodeEdge] = []
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return record, symbols, edges

        module_id = f"{relative}::module"
        symbols.append(
            CodeSymbol(
                symbol_id=module_id,
                name=relative,
                kind="module",
                file_path=relative,
                line_start=1,
                line_end=max(1, len(text.splitlines())),
                language=self.language,
            )
        )

        parent_stack: List[str] = [module_id]
        local_defs = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports = self._import_names(node)
                record.imports.extend(imports)
                for name in imports:
                    edges.append(
                        CodeEdge(
                            source=module_id,
                            target=f"external::{name}",
                            kind="imports",
                            evidence=name,
                            file_path=relative,
                            line=getattr(node, "lineno", 0),
                        )
                    )

        def visit_body(body, parent_id: str) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    symbol_id = f"{relative}::{node.name}"
                    symbol = CodeSymbol(
                        symbol_id=symbol_id,
                        name=node.name,
                        kind="class",
                        file_path=relative,
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno),
                        signature=f"class {node.name}",
                        docstring=ast.get_docstring(node) or "",
                        language=self.language,
                        parent=parent_id,
                    )
                    symbols.append(symbol)
                    local_defs[node.name] = symbol_id
                    edges.append(CodeEdge(parent_id, symbol_id, "defines", file_path=relative, line=node.lineno))
                    for base in node.bases:
                        base_name = self._name_of(base)
                        if base_name:
                            edges.append(
                                CodeEdge(
                                    source=symbol_id,
                                    target=base_name,
                                    kind="inherits",
                                    evidence=base_name,
                                    file_path=relative,
                                    line=node.lineno,
                                )
                            )
                    visit_body(node.body, symbol_id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = ", ".join(arg.arg for arg in node.args.args)
                    symbol_id = f"{relative}::{node.name}@{node.lineno}"
                    symbol = CodeSymbol(
                        symbol_id=symbol_id,
                        name=node.name,
                        kind="function",
                        file_path=relative,
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno),
                        signature=f"{node.name}({args})",
                        docstring=ast.get_docstring(node) or "",
                        language=self.language,
                        parent=parent_id,
                    )
                    symbols.append(symbol)
                    local_defs[node.name] = symbol_id
                    edges.append(CodeEdge(parent_id, symbol_id, "defines", file_path=relative, line=node.lineno))
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            call_name = self._name_of(child.func)
                            if call_name:
                                edges.append(
                                    CodeEdge(
                                        source=symbol_id,
                                        target=call_name,
                                        kind="calls",
                                        evidence=call_name,
                                        file_path=relative,
                                        line=getattr(child, "lineno", node.lineno),
                                    )
                                )
                    visit_body(node.body, symbol_id)

        visit_body(tree.body, module_id)
        return record, symbols, edges

    def _import_names(self, node: ast.AST) -> List[str]:
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            return [f"{prefix}.{alias.name}".strip(".") for alias in node.names]
        return []

    def _name_of(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = self._name_of(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        if isinstance(node, ast.Call):
            return self._name_of(node.func)
        return ""
