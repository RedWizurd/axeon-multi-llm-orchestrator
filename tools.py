"""Tooling for Axeon agent tool-calling loops."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

MAX_FILE_READ_BYTES = 200_000
MAX_FILE_WRITE_BYTES = 200_000
MAX_CODE_CHARS = 4000
DEFAULT_TIMEOUT_SECONDS = 12

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "web_search": "Search the web and return top snippets. Args: {query: str}",
    "calculator": "Evaluate a math expression safely. Args: {expression: str}",
    "code_execution": "Execute restricted Python code with timeout. Args: {code: str}",
    "file_read": "Read a UTF-8 text file within allowed roots. Args: {path: str}",
    "file_write": "Write a UTF-8 text file within allowed roots. Args: {path: str, content: str}",
    "api_call": "HTTP request wrapper. Args: {url: str, method: str='GET', data: dict|null}",
}


class ToolExecutionError(Exception):
    """Raised when a tool invocation is invalid or unsafe."""


class _SafeCalculator(ast.NodeVisitor):
    """Tiny safe arithmetic evaluator."""

    _BIN_OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
        ast.Mod: lambda a, b: a % b,
        ast.FloorDiv: lambda a, b: a // b,
    }
    _UNARY_OPS = {
        ast.UAdd: lambda x: +x,
        ast.USub: lambda x: -x,
    }

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolExecutionError("Only numeric constants are allowed in calculator.")

    def visit_Num(self, node: ast.Num) -> float:  # pragma: no cover (py<3.8 compatibility)
        return float(node.n)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        op_type = type(node.op)
        if op_type not in self._BIN_OPS:
            raise ToolExecutionError(f"Operator {op_type.__name__} is not allowed.")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self._BIN_OPS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        op_type = type(node.op)
        if op_type not in self._UNARY_OPS:
            raise ToolExecutionError(f"Unary operator {op_type.__name__} is not allowed.")
        operand = self.visit(node.operand)
        return self._UNARY_OPS[op_type](operand)

    def generic_visit(self, node: ast.AST) -> float:
        raise ToolExecutionError(f"Expression element {type(node).__name__} is not allowed.")


def _safe_calculate(expression: str) -> Dict[str, Any]:
    parsed = ast.parse(expression, mode="eval")
    evaluator = _SafeCalculator()
    value = evaluator.visit(parsed)
    return {"expression": expression, "value": value}


def _extract_duckduckgo_results(data: Dict[str, Any], max_results: int = 5) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []

    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        results.append(
            {
                "title": data.get("Heading") or "DuckDuckGo Abstract",
                "snippet": abstract,
                "url": data.get("AbstractURL") or "",
            }
        )

    def append_topic(topic: Dict[str, Any]) -> None:
        if len(results) >= max_results:
            return
        if "Topics" in topic and isinstance(topic["Topics"], list):
            for sub_topic in topic["Topics"]:
                append_topic(sub_topic)
            return
        text = (topic.get("Text") or "").strip()
        if not text:
            return
        results.append(
            {
                "title": text.split(" - ")[0][:120],
                "snippet": text,
                "url": topic.get("FirstURL") or "",
            }
        )

    for topic in data.get("RelatedTopics", []):
        append_topic(topic)
        if len(results) >= max_results:
            break

    return results[:max_results]


class ToolExecutor:
    """Executes a bounded set of safe tools."""

    def __init__(
        self,
        base_dir: str | Path,
        allowed_roots: Optional[List[str]] = None,
        request_timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        roots = allowed_roots or [str(self.base_dir)]
        resolved_roots: List[Path] = []
        for root in roots:
            root_path = Path(root)
            if not root_path.is_absolute():
                root_path = (self.base_dir / root_path).resolve()
            else:
                root_path = root_path.resolve()
            resolved_roots.append(root_path)
        self.allowed_roots = resolved_roots
        self.request_timeout = request_timeout

    def describe_tools(self) -> str:
        return "\n".join([f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()])

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = (tool_name or "").strip()
        args = args or {}

        handlers = {
            "web_search": self.web_search,
            "calculator": self.calculator,
            "code_execution": self.code_execution,
            "file_read": self.file_read,
            "file_write": self.file_write,
            "api_call": self.api_call,
        }

        handler = handlers.get(tool_name)
        if not handler:
            raise ToolExecutionError(f"Unknown tool '{tool_name}'.")

        try:
            return handler(**args)
        except TypeError as exc:
            raise ToolExecutionError(f"Invalid args for tool '{tool_name}': {exc}") from exc

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path:
            raise ToolExecutionError("Path is required.")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (self.base_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        for root in self.allowed_roots:
            try:
                candidate.relative_to(root)
                return candidate
            except ValueError:
                continue

        roots = ", ".join(str(root) for root in self.allowed_roots)
        raise ToolExecutionError(f"Path '{candidate}' is outside allowed roots: {roots}")

    def web_search(self, query: str) -> Dict[str, Any]:
        if not query or len(query.strip()) < 2:
            raise ToolExecutionError("Query must be at least 2 characters.")

        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1",
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = _extract_duckduckgo_results(data, max_results=5)

        return {
            "query": query,
            "results": results,
            "count": len(results),
        }

    def calculator(self, expression: str) -> Dict[str, Any]:
        if not expression or len(expression) > 300:
            raise ToolExecutionError("Expression is empty or too long.")
        return _safe_calculate(expression)

    def code_execution(self, code: str, timeout_seconds: int = 4) -> Dict[str, Any]:
        if not code:
            raise ToolExecutionError("Code cannot be empty.")
        if len(code) > MAX_CODE_CHARS:
            raise ToolExecutionError("Code too long.")

        lowered = code.lower()
        blocked_patterns = [
            r"\bimport\b",
            r"__",
            r"\bopen\(",
            r"\beval\(",
            r"\bexec\(",
            r"\bcompile\(",
            r"\bglobals\(",
            r"\blocals\(",
            r"\binput\(",
            r"\bos\.",
            r"\bsys\.",
            r"\bsubprocess\b",
            r"\bsocket\b",
            r"\bpathlib\b",
            r"\bshutil\b",
            r"\brequests\b",
        ]
        for pattern in blocked_patterns:
            if re.search(pattern, lowered):
                raise ToolExecutionError(f"Blocked pattern detected: {pattern}")

        safe_runner = f"""
import contextlib
import io
import json
import math

SAFE_BUILTINS = {{
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'enumerate': enumerate,
    'float': float,
    'int': int,
    'len': len,
    'list': list,
    'max': max,
    'min': min,
    'pow': pow,
    'print': print,
    'range': range,
    'round': round,
    'set': set,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'zip': zip,
}}

code = {json.dumps(code)}
stdout = io.StringIO()
locals_scope = {{}}
result = {{}}

try:
    with contextlib.redirect_stdout(stdout):
        compiled = compile(code, '<tool_code>', 'exec')
        exec(compiled, {{'__builtins__': SAFE_BUILTINS, 'math': math}}, locals_scope)
    result['stdout'] = stdout.getvalue()
    result['locals'] = {{k: repr(v) for k, v in locals_scope.items() if not k.startswith('_')}}
except Exception as exc:
    result['stdout'] = stdout.getvalue()
    result['error'] = str(exc)

print(json.dumps(result))
"""

        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", safe_runner],
            capture_output=True,
            text=True,
            timeout=max(1, min(timeout_seconds, 8)),
            env={"PYTHONIOENCODING": "utf-8"},
        )

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if proc.returncode != 0:
            return {
                "ok": False,
                "returncode": proc.returncode,
                "stderr": stderr[:1000],
                "stdout": stdout[:1000],
            }

        if not stdout:
            return {"ok": True, "stdout": "", "locals": {}}

        try:
            payload = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError:
            payload = {"stdout": stdout[:1000]}

        payload["ok"] = "error" not in payload
        return payload

    def file_read(self, path: str) -> Dict[str, Any]:
        target = self._resolve_path(path)
        if not target.exists() or not target.is_file():
            raise ToolExecutionError(f"File not found: {target}")

        if target.stat().st_size > MAX_FILE_READ_BYTES:
            raise ToolExecutionError(
                f"File too large ({target.stat().st_size} bytes). Max is {MAX_FILE_READ_BYTES}."
            )

        content = target.read_text(encoding="utf-8", errors="replace")
        return {
            "path": str(target),
            "content": content,
            "bytes": len(content.encode("utf-8")),
        }

    def file_write(self, path: str, content: str) -> Dict[str, Any]:
        target = self._resolve_path(path)
        encoded = (content or "").encode("utf-8")
        if len(encoded) > MAX_FILE_WRITE_BYTES:
            raise ToolExecutionError(
                f"Content too large ({len(encoded)} bytes). Max is {MAX_FILE_WRITE_BYTES}."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content or "", encoding="utf-8")
        return {
            "path": str(target),
            "bytes_written": len(encoded),
        }

    def api_call(
        self,
        url: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 10,
    ) -> Dict[str, Any]:
        if not url:
            raise ToolExecutionError("URL is required.")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolExecutionError("Only http/https URLs are allowed.")

        method = (method or "GET").upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ToolExecutionError(f"Method {method} is not allowed.")

        kwargs: Dict[str, Any] = {
            "method": method,
            "url": url,
            "timeout": max(1, min(timeout_seconds, 20)),
        }
        if method == "GET":
            kwargs["params"] = data or {}
        elif data is not None:
            kwargs["json"] = data

        response = requests.request(**kwargs)
        content_type = response.headers.get("content-type", "")
        body: Any
        if "application/json" in content_type:
            try:
                body = response.json()
            except ValueError:
                body = response.text
        else:
            body = response.text

        if isinstance(body, str):
            body = body[:2000]

        return {
            "url": url,
            "method": method,
            "status_code": response.status_code,
            "ok": response.ok,
            "body": body,
        }
