#!/usr/bin/env python3
"""Minimal local-only Axeon orchestrator using Ollama."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from ollama import Client
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: ollama. Install with: pip install ollama"
    ) from exc


DEFAULT_CONFIG: dict[str, Any] = {
    "models": {
        "writer": "qwen2.5:7b-instruct-q4_K_M",
        "consult": "deepseek-coder:6.7b-instruct-q5_K_M",
    },
    "budgets": {
        "max_consult_ratio": 0.20,
        "max_consults_per_day": 20,
        "max_tokens_per_day": 999999999,
    },
    "adapters": {
        "chatdev": {"enabled": False},
    },
    "ollama": {
        "host": "http://127.0.0.1:11434",
    },
}

CONSULT_INTENTS = {"self_heal", "code", "debug"}
QUIT_COMMANDS = {"quit", "exit", ":q"}


@dataclass
class BudgetState:
    date: str
    consults: int
    consult_tokens: int
    total_tokens: int

    @classmethod
    def from_dict(cls, data: dict[str, Any], today: str) -> "BudgetState":
        if data.get("date") != today:
            return cls(date=today, consults=0, consult_tokens=0, total_tokens=0)
        return cls(
            date=today,
            consults=int(data.get("consults", 0)),
            consult_tokens=int(data.get("consult_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "consults": self.consults,
            "consult_tokens": self.consult_tokens,
            "total_tokens": self.total_tokens,
        }


class AxeonOrchestrator:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.base_dir = config_path.parent.resolve()
        self.config = self._load_config(config_path)

        host = self.config.get("ollama", {}).get("host", "http://127.0.0.1:11434")
        self.client = Client(host=host)

        self.writer_model = self.config["models"]["writer"]
        self.consult_model = self.config["models"]["consult"]

        budgets = self.config["budgets"]
        self.max_consult_ratio = float(budgets["max_consult_ratio"])
        self.max_consults_per_day = int(budgets["max_consults_per_day"])
        self.max_tokens_per_day = int(budgets["max_tokens_per_day"])

        self.chatdev_enabled = bool(
            self.config.get("adapters", {})
            .get("chatdev", {})
            .get("enabled", False)
        )

        self.state_path = self.base_dir / "budget_state.json"
        self.log_path = self.base_dir / "events.jsonl"

    def _load_config(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            path.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
            return json.loads(json.dumps(DEFAULT_CONFIG))

        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        self._deep_update(merged, loaded)
        return merged

    def _deep_update(self, base: dict[str, Any], incoming: dict[str, Any]) -> None:
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _today(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _load_budget_state(self) -> BudgetState:
        today = self._today()
        if not self.state_path.exists():
            return BudgetState(date=today, consults=0, consult_tokens=0, total_tokens=0)

        with self.state_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return BudgetState.from_dict(data, today=today)

    def _save_budget_state(self, state: BudgetState) -> None:
        self.state_path.write_text(
            json.dumps(state.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 1
        return max(1, len(text) // 4)

    def _detect_intent(self, prompt: str) -> str:
        text = prompt.lower()
        if "self_heal" in text or "self-heal" in text or "self heal" in text:
            return "self_heal"
        if "debug" in text:
            return "debug"
        code_markers = (
            "code",
            "python",
            "javascript",
            "typescript",
            "java",
            "golang",
            "rust",
            "sql",
            "bash",
            "function",
            "class",
            "stack trace",
            "traceback",
            "bug",
        )
        if any(marker in text for marker in code_markers):
            return "code"
        return "general"

    def _can_consult(self, state: BudgetState, estimated_tokens: int) -> bool:
        if state.consults >= self.max_consults_per_day:
            return False

        projected_total = state.total_tokens + estimated_tokens
        projected_consult = state.consult_tokens + estimated_tokens

        if projected_total > self.max_tokens_per_day:
            return False

        ratio = projected_consult / max(projected_total, 1)
        return ratio <= self.max_consult_ratio

    def _simulate_chatdev(self, user_text: str) -> str:
        triage = (
            "Triage Nurse: Symptoms captured. Identify failing components, recent changes, "
            "and expected healthy behavior before interventions."
        )
        diagnostician = (
            "Systems Diagnostician: Prioritize root-cause isolation with reproducible checks, "
            "log boundaries, and dependency verification."
        )
        recovery = (
            "Recovery Engineer: Apply smallest safe rollback/fix first, validate health checks, "
            "then harden with regression tests."
        )
        director = (
            "Medical Director: Approve treatment plan only if it includes risk containment, "
            "verification criteria, and post-incident prevention."
        )
        return (
            "[ChatDev Simulation]\n"
            f"User case: {user_text}\n"
            f"{triage}\n{diagnostician}\n{recovery}\n{director}"
        )

    def _chat(self, model: str, messages: list[dict[str, str]]) -> tuple[str, int]:
        response = self.client.chat(model=model, messages=messages)
        content = response.get("message", {}).get("content", "")

        prompt_eval_count = int(response.get("prompt_eval_count") or 0)
        eval_count = int(response.get("eval_count") or 0)
        token_count = prompt_eval_count + eval_count
        if token_count <= 0:
            token_count = self._estimate_tokens("\n".join(m["content"] for m in messages) + content)

        return content.strip(), token_count

    def _log_event(
        self,
        trace_id: str,
        intent: str,
        model: str,
        consult_used: bool,
        chatdev_used: bool,
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "intent": intent,
            "model": model,
            "consult_used": consult_used,
            "chatdev_used": chatdev_used,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")

    def handle_turn(self, user_text: str) -> str:
        trace_id = str(uuid.uuid4())
        intent = self._detect_intent(user_text)
        state = self._load_budget_state()

        consult_used = False
        selected_model = self.writer_model

        chatdev_context = ""
        if intent == "self_heal" and self.chatdev_enabled:
            chatdev_context = self._simulate_chatdev(user_text)

        consult_candidate = intent in CONSULT_INTENTS
        estimated_consult_tokens = self._estimate_tokens(user_text + chatdev_context)

        if consult_candidate and self._can_consult(state, estimated_consult_tokens):
            consult_used = True
            selected_model = self.consult_model
            system_content = (
                "You are a technical consult model. Return direct raw technical output. "
                "Do not format as numbered 1/2/3 lists."
            )
            user_content = user_text
            if chatdev_context:
                user_content = f"{user_text}\n\n{chatdev_context}"

            final_text, token_count = self._chat(
                model=self.consult_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            )

            state.consults += 1
            state.consult_tokens += token_count
            state.total_tokens += token_count
        else:
            system_content = "You are Axeon writer core. Be clear and concise."
            user_content = user_text
            if chatdev_context:
                user_content = (
                    f"{user_text}\n\nAdditional internal context:\n{chatdev_context}"
                )

            final_text, token_count = self._chat(
                model=self.writer_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            )
            state.total_tokens += token_count

        if state.total_tokens > self.max_tokens_per_day:
            final_text = "Daily token budget reached. Please try again tomorrow."

        self._save_budget_state(state)
        self._log_event(
            trace_id=trace_id,
            intent=intent,
            model=selected_model,
            consult_used=consult_used,
            chatdev_used=bool(chatdev_context),
        )

        raw_mode = consult_used or intent in {"code", "self_heal"}
        if raw_mode:
            return final_text

        return final_text

    def run_cli(self, once: str | None = None) -> None:
        if once is not None:
            reply = self.handle_turn(once)
            print(f"axeon> {reply}")
            return

        while True:
            try:
                user_text = input("you> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\naxeon> Session ended.")
                break

            if not user_text:
                continue

            if user_text.lower() in QUIT_COMMANDS:
                print("axeon> Session ended.")
                break

            reply = self.handle_turn(user_text)
            print(f"axeon> {reply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Axeon Ollama orchestrator")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config JSON (default: ./config.json)",
    )
    parser.add_argument(
        "--once",
        default=None,
        help="Run one prompt and exit",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    orchestrator = AxeonOrchestrator(config_path=config_path)
    orchestrator.run_cli(once=args.once)


if __name__ == "__main__":
    main()
