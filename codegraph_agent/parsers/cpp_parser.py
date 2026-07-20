from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from ..models import CodeEdge, CodeSymbol, FileRecord


class CppParser:
    language = "cpp"
    include_pattern = re.compile(r"^\s*#include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE)
    function_pattern = re.compile(
        r"(?P<signature>(?:[\w:<>&*\s]+)\s+(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\))\s*\{",
        re.MULTILINE,
    )
    call_pattern = re.compile(r"\b(?P<name>[A-Za-z_]\w*)\s*\(")
    control_words = {"if", "for", "while", "switch", "return", "sizeof", "catch"}

    def parse(self, root: Path, path: Path) -> Tuple[FileRecord, List[CodeSymbol], List[CodeEdge]]:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        record = FileRecord(path=relative, language=self.language, text=text)
        module_id = f"{relative}::translation_unit"
        symbols = [
            CodeSymbol(
                symbol_id=module_id,
                name=relative,
                kind="translation_unit",
                file_path=relative,
                line_start=1,
                line_end=max(1, len(text.splitlines())),
                language=self.language,
            )
        ]
        edges: List[CodeEdge] = []

        for match in self.include_pattern.finditer(text):
            include = match.group(1)
            line = text.count("\n", 0, match.start()) + 1
            record.imports.append(include)
            edges.append(CodeEdge(module_id, f"external::{include}", "imports", include, relative, line))

        for match in self.function_pattern.finditer(text):
            name = match.group("name")
            if name in self.control_words:
                continue
            line = text.count("\n", 0, match.start()) + 1
            end_line = self._find_block_end(text, match.end())
            symbol_id = f"{relative}::{name}@{line}"
            symbols.append(
                CodeSymbol(
                    symbol_id=symbol_id,
                    name=name,
                    kind="function",
                    file_path=relative,
                    line_start=line,
                    line_end=end_line,
                    signature=" ".join(match.group("signature").split()),
                    language=self.language,
                    parent=module_id,
                )
            )
            edges.append(CodeEdge(module_id, symbol_id, "defines", file_path=relative, line=line))
            body = text[match.end() : self._offset_for_line(text, end_line)]
            for call in self.call_pattern.finditer(body):
                call_name = call.group("name")
                if call_name not in self.control_words and call_name != name:
                    call_line = line + body.count("\n", 0, call.start())
                    edges.append(CodeEdge(symbol_id, call_name, "calls", call_name, relative, call_line))

        return record, symbols, edges

    def _find_block_end(self, text: str, start: int) -> int:
        depth = 1
        for offset in range(start, len(text)):
            if text[offset] == "{":
                depth += 1
            elif text[offset] == "}":
                depth -= 1
                if depth == 0:
                    return text.count("\n", 0, offset) + 1
        return text.count("\n") + 1

    def _offset_for_line(self, text: str, line: int) -> int:
        if line <= 1:
            return 0
        current = 1
        for offset, char in enumerate(text):
            if char == "\n":
                current += 1
                if current > line:
                    return offset
        return len(text)
