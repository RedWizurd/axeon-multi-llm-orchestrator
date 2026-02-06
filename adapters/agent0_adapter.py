"""Agent0 adapter (optional): hidden, one-shot tool execution with distilled output."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Agent0Result:
    status: str
    summary: str
    patch_plan: List[str]
    pitfalls: List[str]
    output_digest: str
    command: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "patch_plan": self.patch_plan,
            "pitfalls": self.pitfalls,
            "output_digest": self.output_digest,
            "command": self.command,
        }


class Agent0Adapter:
    """Wraps Agent0 as a hidden background executor.

    Design constraints:
    - One-shot execution only.
    - Never return raw command output to user-facing channel.
    - Return distilled summary + digest for auditability.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.timeout_s = int(cfg.get("timeout_s", 45))
        self.mock_mode = bool(cfg.get("mock_mode", True))
        self.command_template = cfg.get(
            "command_template",
            os.getenv("AGENT0_CMD", "./agent0ctl.sh run --task {task}"),
        )

    def run_once(self, task: str, trace_id: str) -> Dict[str, Any]:
        if not self.enabled:
            return Agent0Result(
                status="disabled",
                summary="Agent0 adapter is disabled; skipped tool execution.",
                patch_plan=["Enable adapters.agent0.enabled in config to execute one-shot tasks."],
                pitfalls=["No tool side effects were applied."],
                output_digest="",
                command="",
            ).to_dict()

        command = self.command_template.format(task=self._shell_safe(task), trace_id=trace_id)
        if self.mock_mode:
            pseudo_output = (
                "MOCK: Agent0 dry run\n"
                f"task={task}\n"
                "result=No external command executed in mock mode.\n"
                "suggestion=Switch mock_mode to false for real execution.\n"
            )
            return self._distill_output(command, pseudo_output, status="mock")

        try:
            proc = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            status = "ok" if proc.returncode == 0 else "error"
            return self._distill_output(command, combined, status=status)
        except Exception as exc:  # noqa: BLE001
            return Agent0Result(
                status="error",
                summary=f"Agent0 execution failed: {exc}",
                patch_plan=["Retry with a simpler one-shot tool task.", "Inspect Agent0 command template and runtime path."],
                pitfalls=["No deterministic tool output available."],
                output_digest="",
                command=command,
            ).to_dict()

    def _distill_output(self, command: str, raw_output: str, status: str) -> Dict[str, Any]:
        clean = raw_output.strip()
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest() if clean else ""
        lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
        top = lines[:6]
        summary = (
            "Agent0 ran hidden one-shot execution and returned distilled status. "
            f"Captured {len(lines)} output lines (digest={digest[:12]}...)."
        )
        patch_plan = [
            "Apply only deterministic changes from the distilled result.",
            "Run targeted tests before exposing output.",
            "Log digest for replay/audit comparisons.",
        ]
        pitfalls = [
            "Tool output can include non-deterministic logs; do not persist raw output.",
            "If command exits non-zero, fail-open to local writer response.",
        ]
        if top:
            snippet = textwrap.shorten(" | ".join(top), width=220, placeholder="...")
            summary = f"{summary} Key signal: {snippet}"

        return Agent0Result(
            status=status,
            summary=summary,
            patch_plan=patch_plan,
            pitfalls=pitfalls,
            output_digest=digest,
            command=command,
        ).to_dict()

    @staticmethod
    def _shell_safe(task: str) -> str:
        # Keep the prompt text compact and avoid unsafe command interpolation.
        return json.dumps(task)[1:-1]


def smoke_test() -> None:
    adapter = Agent0Adapter({"enabled": True, "mock_mode": True})
    out = adapter.run_once("list modified files and suggest fix plan", trace_id=f"t_{int(time.time())}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    smoke_test()
