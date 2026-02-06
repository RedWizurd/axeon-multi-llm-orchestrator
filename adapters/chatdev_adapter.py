"""ChatDev adapter: pruned role set + Medical Director audit for self-healing."""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_ROLES = ["Architect", "Implementer", "Tester"]


@dataclass
class ChatDevResult:
    status: str
    theme: str
    summary: str
    patch_plan: List[str]
    audit_findings: List[str]
    output_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "theme": self.theme,
            "summary": self.summary,
            "patch_plan": self.patch_plan,
            "audit_findings": self.audit_findings,
            "output_digest": self.output_digest,
        }


class ChatDevAdapter:
    """Minimal self-heal coordinator inspired by ChatDev with enforced compact role set."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.theme = str(cfg.get("theme", "clinic"))
        self.roles = list(cfg.get("roles", DEFAULT_ROLES))[:3]
        if len(self.roles) < 3:
            self.roles = DEFAULT_ROLES
        self.director_role = str(cfg.get("director_role", "Medical Director"))
        self.max_steps = int(cfg.get("max_steps", 5))
        self.workspace = Path(cfg.get("workspace", os.getcwd()))

    def run_self_heal(self, task: str, trace_id: str) -> Dict[str, Any]:
        if not self.enabled:
            return ChatDevResult(
                status="disabled",
                theme=self.theme,
                summary="ChatDev adapter is disabled; self-heal simulation skipped.",
                patch_plan=["Enable adapters.chatdev.enabled in config to run self-healing workflow."],
                audit_findings=["No code audit executed."],
                output_digest="",
            ).to_dict()

        role_notes = self._role_round(task)
        director = self._director_audit(role_notes)
        patch_plan = self._build_patch_plan(task, director)
        raw = json.dumps({"roles": role_notes, "director": director, "patch_plan": patch_plan}, sort_keys=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        summary = (
            f"ChatDev '{self.theme}' run complete with roles {', '.join(self.roles)} and {self.director_role} audit. "
            f"Distilled plan has {len(patch_plan)} steps (digest={digest[:12]}...)."
        )
        return ChatDevResult(
            status="ok",
            theme=self.theme,
            summary=summary,
            patch_plan=patch_plan,
            audit_findings=director,
            output_digest=digest,
        ).to_dict()

    def _role_round(self, task: str) -> Dict[str, str]:
        architect = (
            f"Architecture triage for '{self.theme}' theme: isolate blast radius, keep one-writer output, "
            "prefer deterministic fixes over broad refactors."
        )
        implementer = (
            "Implement smallest patch set first, gated by lint/tests; avoid unrelated file churn and avoid "
            "stateful side effects."
        )
        tester = (
            "Create focused regression checks for changed behavior and a smoke test for existing critical path."
        )
        base = {
            self.roles[0]: architect,
            self.roles[1]: implementer,
            self.roles[2]: tester,
        }
        if "database" in task.lower():
            base[self.roles[0]] += " Include migration rollback notes."
        if "api" in task.lower():
            base[self.roles[2]] += " Include contract test for response schema."
        return base

    def _director_audit(self, role_notes: Dict[str, str]) -> List[str]:
        findings = [
            "Confirm all changes are reversible and linked to explicit symptom.",
            "Reject speculative edits that do not map to a failing test or reproducible issue.",
            "Enforce concise incident note with root cause, fix, and verification evidence.",
        ]
        if self.theme.lower() == "clinic":
            findings.append("Apply triage severity labels (critical/high/medium/low) before implementation.")
        if len(role_notes) > 3:
            findings.append("Prune roles beyond the approved three-role workflow.")
        return findings

    def _build_patch_plan(self, task: str, director_findings: List[str]) -> List[str]:
        plan = [
            f"Triage issue: {textwrap.shorten(task, width=100, placeholder='...')}",
            "Patch minimal affected modules only.",
            "Run focused tests and one end-to-end smoke path.",
            "Publish a short post-fix audit summary for traceability.",
        ]
        if self.max_steps < len(plan):
            plan = plan[: self.max_steps]
        if "severity" in " ".join(director_findings).lower() and len(plan) < self.max_steps:
            plan.append("Tag incident severity and owner for follow-up hardening.")
        return plan


def smoke_test() -> None:
    adapter = ChatDevAdapter({"enabled": True, "theme": "clinic"})
    out = adapter.run_self_heal("self-heal failing API tests with flaky auth middleware", trace_id=f"t_{int(time.time())}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    smoke_test()
