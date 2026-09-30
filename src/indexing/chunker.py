import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class CodeChunk:
    doc_id: str
    text: str
    language: str
    file_path: Optional[str] = None
    repository: Optional[str] = None
    version: Optional[str] = None
    symbol_name: Optional[str] = None  # function or class name
    symbol_type: Optional[str] = None  # "function" or "class"
    start_line: int = 1
    end_line: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Regex patterns for block / function / class declarations in supported languages
PATTERNS = {
    "python": [
        (r"^\s*(def|class)\s+([a-zA-Z0-9_]+)", "python"),
    ],
    "java": [
        (r"^\s*(public|protected|private|static|\s)+class\s+([a-zA-Z0-9_]+)", "class"),
        (r"^\s*(public|protected|private|static|\s)+[\w<>,\[\]]+\s+([a-zA-Z0-9_]+)\s*\(", "function"),
    ],
    "c": [
        (r"^\s*(struct|enum|union)\s+([a-zA-Z0-9_]+)", "struct"),
        (r"^\s*[\w\*\s]+\s+([a-zA-Z0-9_]+)\s*\([^;]*\)\s*\{", "function"),
    ],
    "cpp": [
        (r"^\s*(class|struct)\s+([a-zA-Z0-9_]+)", "class"),
        (r"^\s*[\w\*\:\<\>\s]+\s+([a-zA-Z0-9_]+)\s*\([^;]*\)\s*(const)?\s*\{", "function"),
    ],
    "javascript": [
        (r"^\s*(function|class)\s+([a-zA-Z0-9_]+)", "javascript"),
        (r"^\s*(const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(function|\([^)]*\)\s*=>)", "function"),
    ],
    "typescript": [
        (r"^\s*(function|class|interface|type)\s+([a-zA-Z0-9_]+)", "typescript"),
        (r"^\s*(const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(function|\([^)]*\)\s*=>)", "function"),
    ],
}


def detect_language(file_path: Optional[str] = None, text: str = "") -> str:
    """Detect programming language from file extension or code heuristics."""
    if file_path:
        ext = file_path.split(".")[-1].lower()
        mapping = {
            "py": "python",
            "java": "java",
            "c": "c",
            "h": "c",
            "cpp": "cpp",
            "hpp": "cpp",
            "cc": "cpp",
            "cxx": "cpp",
            "js": "javascript",
            "mjs": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "jsx": "javascript",
        }
        if ext in mapping:
            return mapping[ext]

    # Content heuristics
    if "def " in text or "import " in text and ":" in text:
        return "python"
    if "public class " in text or "system.out.println" in text.lower():
        return "java"
    if "#include" in text:
        if "std::" in text or "class " in text or "cout" in text:
            return "cpp"
        return "c"
    if "const " in text or "function " in text or "=>" in text:
        if "interface " in text or ": string" in text or ": number" in text:
            return "typescript"
        return "javascript"

    return "generic"


class CodeChunker:
    """Practical line- and symbol-aware code chunker."""

    def __init__(self, max_chunk_lines: int = 50, overlap_lines: int = 10):
        self.max_chunk_lines = max_chunk_lines
        self.overlap_lines = overlap_lines

    def chunk_code(
        self,
        text: str,
        doc_id: str,
        language: Optional[str] = None,
        file_path: Optional[str] = None,
        repository: Optional[str] = None,
        version: Optional[str] = None,
    ) -> List[CodeChunk]:
        """Split source code into chunks preserving line ranges and detected symbols."""
        if not text or not text.strip():
            return []

        lang = language or detect_language(file_path, text)
        lines = text.splitlines()
        total_lines = len(lines)

        if total_lines <= self.max_chunk_lines:
            symbol_name, symbol_type = self._extract_symbol(text, lang)
            return [
                CodeChunk(
                    doc_id=doc_id,
                    text=text,
                    language=lang,
                    file_path=file_path,
                    repository=repository,
                    version=version,
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    start_line=1,
                    end_line=total_lines,
                )
            ]

        chunks = []
        step = max(1, self.max_chunk_lines - self.overlap_lines)
        chunk_idx = 0

        for start_idx in range(0, total_lines, step):
            end_idx = min(start_idx + self.max_chunk_lines, total_lines)
            chunk_lines = lines[start_idx:end_idx]
            chunk_text = "\n".join(chunk_lines)

            symbol_name, symbol_type = self._extract_symbol(chunk_text, lang)
            sub_id = f"{doc_id}_chunk_{chunk_idx}" if chunk_idx > 0 else doc_id

            chunks.append(
                CodeChunk(
                    doc_id=sub_id,
                    text=chunk_text,
                    language=lang,
                    file_path=file_path,
                    repository=repository,
                    version=version,
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    start_line=start_idx + 1,
                    end_line=end_idx,
                )
            )
            chunk_idx += 1
            if end_idx == total_lines:
                break

        return chunks

    def _extract_symbol(self, code_text: str, language: str) -> tuple[Optional[str], Optional[str]]:
        """Extract top function or class name from code block."""
        lang_patterns = PATTERNS.get(language, [])
        for pattern, stype in lang_patterns:
            match = re.search(pattern, code_text, re.MULTILINE)
            if match:
                groups = match.groups()
                name = groups[-1]
                t_type = stype
                if stype in ("python", "javascript", "typescript"):
                    t_type = "class" if "class" in match.group(0) else "function"
                return name, t_type
        return None, None
