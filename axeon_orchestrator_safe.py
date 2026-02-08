#!/usr/bin/env python3
"""Axeon multi-LLM orchestrator (Ollama-local only).

- One-Writer rule: one final writer model response per turn.
- Consult budget guardrails: <20% consult ratio + per-day limits.
- Local-only model routing across two Ollama models.
- Optional adapters: Agent0, ChatDev, RDF Forensics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from adapters import Agent0Adapter, ChatDevAdapter, RDFForensicsAdapter

try:
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore

    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False

try:
    from langgraph.graph import END, StateGraph  # type: ignore

    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False


WRITER_ALIAS = "local_qwen"
CONSULT_ALIAS = "deepseek_coder"
WRITER_TAG = "qwen2.5:7b-instruct-q4_K_M"
CONSULT_TAG = "deepseek-coder-v2-lite-instruct-q4_k_m"

DEFAULT_CONFIG: Dict[str, Any] = {
    "engine": {
        "use_langgraph": True,
        "writer_model": WRITER_ALIAS,
        "consult_model": CONSULT_ALIAS,
        "max_writer_tokens": 1200,
        "max_consult_output_tokens": 900,
        "timeout_s": 30,
    },
    "budgets": {
        "max_consult_ratio": 0.20,
        "max_consults_per_day": 20,
        "max_total_consult_tokens_per_day": 20000,
    },
    "policy": {
        "one_writer": True,
        "allow_sending_user_content": False,
        "redact_secrets": True,
    },
    "audits": {
        "events_path": "./events.jsonl",
    },
    "models": {
        WRITER_ALIAS: {
            "provider": "ollama",
            "model": WRITER_TAG,
            "generate_endpoint": "http://127.0.0.1:11434/api/generate",
            "tags_endpoint": "http://127.0.0.1:11434/api/tags",
            "enabled": True,
        },
        CONSULT_ALIAS: {
            "provider": "ollama",
            "model": CONSULT_TAG,
            "generate_endpoint": "http://127.0.0.1:11434/api/generate",
            "tags_endpoint": "http://127.0.0.1:11434/api/tags",
            "enabled": True,
        },
    },
    "adapters": {
        "agent0": {
            "enabled": False,
            "mock_mode": True,
            "timeout_s": 45,
            "command_template": "./agent0ctl.sh run --task {task}",
        },
        "chatdev": {
            "enabled": False,
            "theme": "clinic",
            "roles": ["Architect", "Implementer", "Tester"],
            "director_role": "Medical Director",
            "max_steps": 5,
        },
        "rdf_forensics": {
            "enabled": False,
            "policy_path": "./policies/rdf_forensics_policy.json",
            "overrides": {},
        },
    },
}

INTENT_KEYWORDS = {
    "forensics": ["forensics", "rdf", "knowledge graph", "kg", "provenance", "corpora", "wikileaks"],
    "self_heal": ["self-heal", "self heal", "heal code", "fix code", "debug", "failing test"],
    "tool_exec": ["agent0", "run tool", "execute command", "tool execution"],
    "code": ["code", "refactor", "function", "class", "bug", "patch"],
    "architecture": ["architecture", "system design", "design tradeoff", "scale"],
    "build_feature": ["build", "implement", "feature", "add support", "create"],
    "research": ["research", "compare", "evaluate", "options", "study"],
}

CONSULT_INTENTS = {"code", "tool_exec", "self_heal", "forensics"}


class ModelNotFoundError(RuntimeError):
    def __init__(self, model_alias: str, model_tag: str, detail: str) -> None:
        super().__init__(detail)
        self.model_alias = model_alias
        self.model_tag = model_tag
        self.detail = detail


@dataclass
class RouterDecision:
    intent: str
    confidence: float
    needs_consult: bool
    consult_reason: str
    consult_model: str
    consult_question: str
    max_consult_output_tokens: int
    timeout_s: int
    data_policy: Dict[str, Any]


@dataclass
class Reduction:
    final_answer: str
    decisions: List[str]
    invariants: List[str]
    pitfalls: List[str]
    patch_plan: List[str]
    disagreements: List[str]
    learned_patterns: List[Dict[str, str]]
    confidence_update: float
    confidence_score: float


class AuditLedger:
    """Append-only audit/event logger with daily consult budget accounting."""

    def __init__(self, events_path: str) -> None:
        self.events_path = Path(events_path).expanduser().resolve()
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.events_path.exists():
            self.events_path.write_text("", encoding="utf-8")

    def log(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        line = json.dumps(payload, separators=(",", ":"))
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def today_counters(self) -> Dict[str, int]:
        today = datetime.now(timezone.utc).date().isoformat()
        consults = 0
        consult_tokens = 0
        turns = 0
        with self.events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = str(ev.get("timestamp", ""))
                if not ts.startswith(today):
                    continue
                if ev.get("event") == "turn_complete":
                    turns += 1
                if ev.get("event") == "consult_complete":
                    consults += 1
                    consult_tokens += int(ev.get("tokens", 0) or 0)
        return {"turns": turns, "consults": consults, "consult_tokens": consult_tokens}

    def allow_consult(
        self,
        planned_tokens: int,
        max_consult_ratio: float,
        max_consults_per_day: int,
        max_total_consult_tokens_per_day: int,
    ) -> Tuple[bool, str]:
        stats = self.today_counters()
        projected_turns = stats["turns"] + 1
        projected_consults = stats["consults"] + 1

        # Bootstrap: allow max 1 consult in first 5 turns, then strict ratio.
        if projected_turns <= 5 and projected_consults > 1:
            return False, "bootstrap consult cap exceeded (max 1 consult in first 5 turns)"
        if projected_turns > 5:
            projected_ratio = projected_consults / projected_turns
            if projected_ratio > max_consult_ratio:
                return False, f"consult ratio cap exceeded ({projected_ratio:.2f}>{max_consult_ratio:.2f})"

        if projected_consults > max_consults_per_day:
            return False, "max consults per day reached"
        if stats["consult_tokens"] + planned_tokens > max_total_consult_tokens_per_day:
            return False, "daily consult token budget exceeded"
        return True, "ok"


class ProviderClient:
    """Ollama-only model client with model-existence checks."""

    def __init__(self, models: Dict[str, Dict[str, Any]], timeout_s: int = 30) -> None:
        self.models = models
        self.timeout_s = timeout_s
        self._tags_cache: Optional[Tuple[float, List[str]]] = None

    def list_ollama_models(self, force: bool = False) -> List[str]:
        now = time.time()
        if not force and self._tags_cache and (now - self._tags_cache[0] < 10):
            return self._tags_cache[1]

        probe = self.models.get(WRITER_ALIAS) or self.models.get(CONSULT_ALIAS)
        if not probe:
            return []
        tags_endpoint = probe.get("tags_endpoint", "http://127.0.0.1:11434/api/tags")
        try:
            data = self._get_json(tags_endpoint, timeout_s=self.timeout_s)
            names = [str(m.get("name", "")) for m in data.get("models", []) if m.get("name")]
            self._tags_cache = (now, names)
            return names
        except Exception:
            return []

    def is_available(self, model_alias: str) -> bool:
        model = self.models.get(model_alias)
        if not model or not model.get("enabled", True):
            return False
        if model.get("provider") != "ollama":
            return False
        names = self.list_ollama_models()
        return model.get("model", "") in names

    def generate(self, model_alias: str, system_prompt: str, user_prompt: str, max_tokens: int, timeout_s: Optional[int] = None) -> str:
        model = self.models.get(model_alias)
        if not model:
            raise RuntimeError(f"unknown model alias '{model_alias}'")
        if model.get("provider") != "ollama":
            raise RuntimeError("only ollama provider is supported in this build")

        model_tag = str(model.get("model", ""))
        timeout = timeout_s or self.timeout_s
        payload = {
            "model": model_tag,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }

        try:
            data = self._post_json(model["generate_endpoint"], payload, timeout_s=timeout)
            return str(data.get("response", "")).strip()
        except RuntimeError as exc:
            lower = str(exc).lower()
            if "not found" in lower and "model" in lower:
                raise ModelNotFoundError(model_alias, model_tag, str(exc)) from exc
            if "network error calling" in lower:
                raise RuntimeError(
                    "Cannot reach Ollama at http://127.0.0.1:11434. "
                    "Is Ollama running? Try: ollama serve"
                ) from exc
            raise RuntimeError(
                f"Ollama request failed for model '{model_tag}'. "
                f"Model not pulled? Run: ollama pull {model_tag}. "
                f"Details: {exc}"
            ) from exc

    @staticmethod
    def _get_json(url: str, timeout_s: int = 30) -> Dict[str, Any]:
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code} {url}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error calling {url}: {exc}") from exc

    @staticmethod
    def _post_json(url: str, payload: Dict[str, Any], timeout_s: int = 30) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code} {url}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error calling {url}: {exc}") from exc


class IntentRouter:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def decide(self, user_text: str) -> RouterDecision:
        lower = user_text.lower()
        intent = self._detect_intent(lower)
        confidence = self._estimate_confidence(intent, user_text)

        needs_consult = False
        consult_reason = "No consult required by router rules."

        if intent in CONSULT_INTENTS:
            needs_consult = True
            consult_reason = f"Intent '{intent}' uses specialized consult path."
        elif intent in {"architecture", "build_feature", "research"} and confidence < 0.75:
            needs_consult = True
            consult_reason = f"Low confidence ({confidence:.2f}) for {intent}; bounded consult requested."

        return RouterDecision(
            intent=intent,
            confidence=confidence,
            needs_consult=needs_consult,
            consult_reason=consult_reason,
            consult_model=self.config["engine"].get("consult_model", CONSULT_ALIAS),
            consult_question=textwrap.shorten(user_text, width=240, placeholder="..."),
            max_consult_output_tokens=int(self.config["engine"].get("max_consult_output_tokens", 900)),
            timeout_s=int(self.config["engine"].get("timeout_s", 30)),
            data_policy={
                "allow_sending_user_content": bool(self.config["policy"].get("allow_sending_user_content", False)),
                "notes": "Never send full chat history. Redact secrets.",
            },
        )

    def _detect_intent(self, lower: str) -> str:
        for intent, words in INTENT_KEYWORDS.items():
            if any(w in lower for w in words):
                return intent
        return "chat"

    @staticmethod
    def _estimate_confidence(intent: str, user_text: str) -> float:
        base = {
            "chat": 0.90,
            "forensics": 0.72,
            "self_heal": 0.70,
            "tool_exec": 0.68,
            "code": 0.70,
            "architecture": 0.65,
            "build_feature": 0.70,
            "research": 0.66,
        }.get(intent, 0.70)
        complexity_penalty = min(len(user_text.split()) / 80.0, 0.2)
        return round(max(0.2, min(0.95, base - complexity_penalty)), 3)


class Reducer:
    """Distills consultant/adaptor output with a lightweight confidence guard."""

    LOW_CONF_KEYWORDS = ["uncertain", "possibly", "claim", "guess", "unverified", "likely", "maybe"]

    @classmethod
    def reduce(cls, text: str) -> Reduction:
        lines = [ln.strip("- ").strip() for ln in text.splitlines() if ln.strip()]
        decisions = lines[:3] if lines else ["Proceed with conservative local-first strategy."]
        invariants = [
            "One-Writer: only final writer response is user-visible.",
            "Consult outputs are distilled; raw content is not persisted.",
            "Budgets and audit trail must remain within configured policy.",
        ]
        pitfalls = [
            "Over-consulting can violate latency/cost constraints.",
            "Ungrounded claims without provenance should be flagged.",
        ]
        patch_plan = [
            "Apply the minimal deterministic change first.",
            "Run focused validation/tests.",
            "Log decisions and provenance digest.",
        ]
        disagreements: List[str] = []
        if any("contradict" in ln.lower() for ln in lines):
            disagreements.append("Consult output contains contradictions; prioritize explicit invariants.")

        learned_patterns = [
            {
                "type": "workflow",
                "content": "Use bounded one-shot consult + reducer before writer synthesis.",
                "when_to_use": "Low confidence architecture/research turns.",
                "when_not_to_use": "Simple chat where local writer confidence is high.",
            }
        ]

        summary = " ".join(decisions[:2])
        final_answer = textwrap.shorten(summary, width=220, placeholder="...")
        confidence_score = cls._confidence_score(text)
        if confidence_score < 0.6:
            final_answer = f"{final_answer} [Low confidence — verify claims manually]".strip()

        return Reduction(
            final_answer=final_answer,
            decisions=decisions,
            invariants=invariants,
            pitfalls=pitfalls,
            patch_plan=patch_plan,
            disagreements=disagreements,
            learned_patterns=learned_patterns,
            confidence_update=0.08,
            confidence_score=confidence_score,
        )

    @classmethod
    def _confidence_score(cls, text: str) -> float:
        clean = text.strip()
        if not clean:
            return 0.25

        n = len(clean)
        score = 0.68

        if n < 80:
            score -= 0.18
        elif n < 180:
            score -= 0.08
        elif n > 1200:
            score += 0.08
        elif n > 400:
            score += 0.04

        lower = clean.lower()
        penalties = sum(1 for w in cls.LOW_CONF_KEYWORDS if w in lower)
        score -= min(0.30, penalties * 0.06)

        if any(k in lower for k in ["sha256", "digest", "provenance", "source:"]):
            score += 0.08
        if any(k in lower for k in ["contradict", "however", "on the other hand", "inconsistent"]):
            score -= 0.10

        return round(max(0.0, min(1.0, score)), 3)


class AxeonOrchestrator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.audit = AuditLedger(config["audits"]["events_path"])
        self.providers = ProviderClient(config["models"], timeout_s=int(config["engine"].get("timeout_s", 30)))
        self.router = IntentRouter(config)

        self.agent0 = Agent0Adapter(config["adapters"].get("agent0", {}))
        self.chatdev = ChatDevAdapter(config["adapters"].get("chatdev", {}))
        self.rdf = RDFForensicsAdapter(config["adapters"].get("rdf_forensics", {}))

        self.writer_model = WRITER_ALIAS
        self.graph = self._build_langgraph() if bool(config["engine"].get("use_langgraph", True)) else None

        self._audit_adapter_disabled("agent0", self.agent0.enabled)
        self._audit_adapter_disabled("chatdev", self.chatdev.enabled)
        self._audit_adapter_disabled("rdf_forensics", self.rdf.enabled)

    @classmethod
    def from_config_path(cls, path: Optional[str]) -> "AxeonOrchestrator":
        cfg = deep_copy(DEFAULT_CONFIG)
        if path:
            p = Path(path).expanduser().resolve()
            if p.exists():
                loaded = json.loads(p.read_text(encoding="utf-8"))
                cfg = deep_merge(cfg, loaded)
        return cls(cfg)

    def handle_turn(self, user_input: str) -> str:
        trace_id = new_trace_id()
        if self.graph is not None:
            out = self.graph.invoke({"user_input": user_input, "trace_id": trace_id})
            return str(out.get("final_text", ""))

        state = {"user_input": user_input, "trace_id": trace_id}
        state = self._node_route(state)
        state = self._node_consult(state)
        state = self._node_reduce(state)
        state = self._node_write(state)
        return str(state.get("final_text", ""))

    def _build_langgraph(self) -> Any:
        if not HAS_LANGGRAPH:
            return None
        workflow = StateGraph(dict)
        workflow.add_node("route", self._node_route)
        workflow.add_node("consult", self._node_consult)
        workflow.add_node("reduce", self._node_reduce)
        workflow.add_node("write", self._node_write)
        workflow.set_entry_point("route")
        workflow.add_edge("route", "consult")
        workflow.add_edge("consult", "reduce")
        workflow.add_edge("reduce", "write")
        workflow.add_edge("write", END)
        return workflow.compile()

    def _node_route(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision = self.router.decide(state["user_input"])
        state["decision"] = asdict(decision)
        self.audit.log(
            {
                "event": "router_decision",
                "trace_id": state["trace_id"],
                "intent": decision.intent,
                "confidence": decision.confidence,
                "needs_consult": decision.needs_consult,
                "consult_reason": decision.consult_reason,
                "consult_model": decision.consult_model,
            }
        )
        return state

    def _node_consult(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision = RouterDecision(**state["decision"])
        user_input = state["user_input"]
        trace_id = state["trace_id"]
        consult_payload: Dict[str, Any] = {"used": False, "source": "none", "text": "", "meta": {}}

        if not decision.needs_consult:
            state["consult_payload"] = consult_payload
            return state

        allow, reason = self.audit.allow_consult(
            planned_tokens=decision.max_consult_output_tokens,
            max_consult_ratio=float(self.config["budgets"]["max_consult_ratio"]),
            max_consults_per_day=int(self.config["budgets"]["max_consults_per_day"]),
            max_total_consult_tokens_per_day=int(self.config["budgets"]["max_total_consult_tokens_per_day"]),
        )
        if not allow:
            consult_payload["meta"] = {"skipped": True, "reason": reason}
            self.audit.log({"event": "consult_skipped", "trace_id": trace_id, "reason": reason, "intent": decision.intent})
            state["consult_payload"] = consult_payload
            return state

        consult_alias = self._select_consult_model(decision, trace_id)

        self.audit.log(
            {
                "event": "consult_start",
                "trace_id": trace_id,
                "intent": decision.intent,
                "model_alias": consult_alias,
                "reason": decision.consult_reason,
                "max_tokens": decision.max_consult_output_tokens,
                "timeout_s": decision.timeout_s,
            }
        )

        started = time.time()
        try:
            if decision.intent == "forensics" and self.rdf.enabled:
                result = self.rdf.run_build_kg(user_input, trace_id)
                consult_payload = {
                    "used": True,
                    "source": "rdf_forensics_adapter",
                    "text": self._adapter_result_to_text(result),
                    "meta": result,
                }
            elif decision.intent == "self_heal" and self.chatdev.enabled:
                result = self.chatdev.run_self_heal(user_input, trace_id)
                consult_payload = {
                    "used": True,
                    "source": "chatdev_adapter",
                    "text": self._adapter_result_to_text(result),
                    "meta": result,
                }
            elif decision.intent == "tool_exec" and self.agent0.enabled:
                result = self.agent0.run_once(user_input, trace_id)
                consult_payload = {
                    "used": True,
                    "source": "agent0_adapter",
                    "text": self._adapter_result_to_text(result),
                    "meta": result,
                }
            else:
                if decision.intent == "forensics" and not self.rdf.enabled:
                    self._warn(trace_id, "adapter_disabled", "Adapter rdf_forensics disabled; using model consult")
                if decision.intent == "self_heal" and not self.chatdev.enabled:
                    self._warn(trace_id, "adapter_disabled", "Adapter chatdev disabled; using model consult")
                if decision.intent == "tool_exec" and not self.agent0.enabled:
                    self._warn(trace_id, "adapter_disabled", "Adapter agent0 disabled; using model consult")

                consult_text = self._consult_model(decision, user_input, consult_alias, trace_id)
                consult_payload = {
                    "used": True,
                    "source": f"model:{consult_alias}",
                    "text": consult_text,
                    "meta": {"status": "ok"},
                }

            elapsed_ms = int((time.time() - started) * 1000)
            digest = hashlib.sha256(consult_payload["text"].encode("utf-8")).hexdigest() if consult_payload["text"] else ""
            self.audit.log(
                {
                    "event": "consult_complete",
                    "trace_id": trace_id,
                    "intent": decision.intent,
                    "source": consult_payload["source"],
                    "elapsed_ms": elapsed_ms,
                    "tokens": estimate_tokens(consult_payload["text"]),
                    "output_digest": digest,
                    "status": consult_payload.get("meta", {}).get("status", "ok"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            consult_payload = {
                "used": False,
                "source": "error",
                "text": "",
                "meta": {"error": str(exc), "status": "error"},
            }
            self.audit.log(
                {
                    "event": "consult_complete",
                    "trace_id": trace_id,
                    "intent": decision.intent,
                    "source": "error",
                    "elapsed_ms": int((time.time() - started) * 1000),
                    "tokens": 0,
                    "output_digest": "",
                    "status": "error",
                    "error": str(exc),
                }
            )

        state["consult_payload"] = consult_payload
        return state

    def _node_reduce(self, state: Dict[str, Any]) -> Dict[str, Any]:
        consult_payload = state.get("consult_payload", {})
        if not consult_payload.get("used"):
            reduction = Reduction(
                final_answer="No consult used; proceed with local writer.",
                decisions=["Use direct writer response."],
                invariants=["One-Writer must remain true."],
                pitfalls=["Potentially lower confidence for complex tasks."],
                patch_plan=["Answer directly and log no-consult path."],
                disagreements=[],
                learned_patterns=[],
                confidence_update=0.0,
                confidence_score=0.6,
            )
        else:
            reduction = Reducer.reduce(consult_payload.get("text", ""))

        state["reduction"] = asdict(reduction)
        self.audit.log(
            {
                "event": "reduction_complete",
                "trace_id": state["trace_id"],
                "used_consult": bool(consult_payload.get("used", False)),
                "decisions": len(reduction.decisions),
                "pitfalls": len(reduction.pitfalls),
                "confidence_score": reduction.confidence_score,
            }
        )
        return state

    def _node_write(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision = RouterDecision(**state["decision"])
        reduction = Reduction(**state["reduction"])
        consult_payload = state.get("consult_payload", {})
        trace_id = state["trace_id"]

        system_prompt, user_prompt = self._writer_prompts(state["user_input"], decision, reduction, consult_payload)

        try:
            final_text = self.providers.generate(
                WRITER_ALIAS,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=int(self.config["engine"].get("max_writer_tokens", 1200)),
                timeout_s=int(self.config["engine"].get("timeout_s", 30)),
            )
            writer_source = WRITER_ALIAS
        except ModelNotFoundError:
            writer_source = "fallback"
            friendly = (
                f"Writer model '{WRITER_TAG}' not found in Ollama. "
                f"Model not pulled? Run: ollama pull {WRITER_TAG}"
            )
            final_text = self._fallback_writer(decision, reduction, consult_payload, friendly)
            self._warn(trace_id, "writer_missing", friendly)
        except Exception as exc:  # noqa: BLE001
            writer_source = "fallback"
            final_text = self._fallback_writer(decision, reduction, consult_payload, str(exc))

        if not final_text.strip():
            writer_source = "fallback"
            final_text = self._fallback_writer(decision, reduction, consult_payload, "empty_writer_output")

        state["final_text"] = final_text.strip()
        self.audit.log(
            {
                "event": "turn_complete",
                "trace_id": trace_id,
                "intent": decision.intent,
                "writer_model": writer_source,
                "consult_used": bool(consult_payload.get("used", False)),
                "final_chars": len(final_text),
            }
        )
        return state

    def _select_consult_model(self, decision: RouterDecision, trace_id: str) -> str:
        requested = CONSULT_ALIAS if decision.intent in CONSULT_INTENTS else self.config["engine"].get("consult_model", CONSULT_ALIAS)
        if self.providers.is_available(requested):
            return requested

        consult_tag = str(self.config["models"][requested]["model"])
        writer_tag = str(self.config["models"][WRITER_ALIAS]["model"])
        warning = (
            f"Consult model '{consult_tag}' missing; falling back to writer '{writer_tag}'. "
            f"Model not pulled? Run: ollama pull {consult_tag}"
        )
        self._warn(trace_id, "consult_model_fallback", warning)
        return WRITER_ALIAS

    def _consult_model(self, decision: RouterDecision, user_input: str, model_alias: str, trace_id: str) -> str:
        sanitized = redact_secrets(user_input) if self.config["policy"].get("redact_secrets", True) else user_input
        if not decision.data_policy.get("allow_sending_user_content", False):
            sanitized = build_minimized_goal(sanitized)

        current_approach = [
            f"intent={decision.intent}",
            f"confidence={decision.confidence}",
            "local writer synthesis after distillation",
            "bounded one-shot consult only",
        ]

        system_prompt = (
            "You are a one-shot specialist consultant. Return compact, structured output only. "
            "Axeon remains sole writer to user."
        )
        user_prompt = (
            "GOAL:\n"
            f"{build_minimized_goal(sanitized)}\n\n"
            "CONSTRAINTS:\n"
            "- One-writer rule (Axeon is only user-facing writer)\n"
            "- Deterministic and bounded output\n"
            "- No hidden assumptions\n\n"
            "CURRENT APPROACH:\n"
            + "\n".join(f"- {x}" for x in current_approach)
            + "\n\nQUESTION:\n"
            f"- {decision.consult_question}\n\n"
            "RETURN FORMAT (strict):\n"
            "1) Decision\n2) Rationale\n3) Pitfalls\n4) Minimal patch plan"
        )

        try:
            return self.providers.generate(
                model_alias,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=decision.max_consult_output_tokens,
                timeout_s=decision.timeout_s,
            )
        except ModelNotFoundError as exc:
            if model_alias != WRITER_ALIAS:
                warning = (
                    f"Consult model missing ({exc.model_tag}). Falling back to writer model {WRITER_TAG}. "
                    f"Model not pulled? Run: ollama pull {exc.model_tag}"
                )
                self._warn(trace_id, "consult_model_fallback", warning)
                return self.providers.generate(
                    WRITER_ALIAS,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=decision.max_consult_output_tokens,
                    timeout_s=decision.timeout_s,
                )
            raise

    @staticmethod
    def _adapter_result_to_text(result: Dict[str, Any]) -> str:
        keys = ["summary", "patch_plan", "audit_findings", "pitfalls", "provenance", "outputs"]
        lines: List[str] = []
        for key in keys:
            val = result.get(key)
            if not val:
                continue
            if isinstance(val, list):
                lines.append(f"{key}: " + "; ".join(str(v) for v in val[:5]))
            elif isinstance(val, dict):
                compact = json.dumps(val, sort_keys=True)
                lines.append(f"{key}: {textwrap.shorten(compact, width=280, placeholder='...')}")
            else:
                lines.append(f"{key}: {val}")
        return "\n".join(lines)

    def _writer_prompts(
        self,
        user_input: str,
        decision: RouterDecision,
        reduction: Reduction,
        consult_payload: Dict[str, Any],
    ) -> Tuple[str, str]:
        consult_used = bool(consult_payload.get("used", False))
        provenance_note = ""

        if decision.intent == "forensics" and consult_used:
            prov = consult_payload.get("meta", {}).get("provenance", [])
            sample = prov[:3] if isinstance(prov, list) else []
            citations = []
            for rec in sample:
                file_info = rec.get("file", {})
                path = file_info.get("path", "unknown")
                digest = str(file_info.get("sha256", ""))[:12]
                citations.append(f"- source: {path} sha256={digest}...")
            provenance_note = "\n".join(citations)

        context = {
            "intent": decision.intent,
            "consult_used": consult_used,
            "consult_source": consult_payload.get("source", "none"),
            "decisions": reduction.decisions,
            "invariants": reduction.invariants,
            "pitfalls": reduction.pitfalls,
            "patch_plan": reduction.patch_plan,
            "confidence_score": reduction.confidence_score,
            "provenance": provenance_note,
        }
        context_json = json.dumps(context, indent=2)

        system_prompt = (
            "You are Axeon, a concise orchestrator following strict One-Writer policy. "
            "Never expose raw consultant text. Use distilled artifacts only. "
            "If forensics intent is detected, ground key claims in provenance citations."
        )

        if HAS_LANGCHAIN:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "{system_prompt}"),
                    (
                        "user",
                        "User request:\n{user_input}\n\nDistilled context JSON:\n{context_json}\n\n"
                        "Output:\n1) Direct answer\n2) Action plan\n3) Risks/guards\n"
                        "Keep concise and actionable.",
                    ),
                ]
            )
            built = prompt.format_messages(system_prompt=system_prompt, user_input=user_input, context_json=context_json)
            user_prompt = "\n\n".join(msg.content for msg in built if hasattr(msg, "content"))
            return system_prompt, user_prompt

        user_prompt = (
            f"User request:\n{user_input}\n\n"
            f"Distilled context JSON:\n{context_json}\n\n"
            "Output format:\n"
            "1) Direct answer\n"
            "2) Action plan\n"
            "3) Risks/guards\n"
            "Keep concise and actionable."
        )
        return system_prompt, user_prompt

    @staticmethod
    def _fallback_writer(decision: RouterDecision, reduction: Reduction, consult_payload: Dict[str, Any], error: str) -> str:
        lines = [
            "1) Direct answer",
            f"Intent: {decision.intent}. Local fail-open path due to: {error}.",
            reduction.final_answer or "Using conservative one-writer response.",
            "",
            "2) Action plan",
        ]
        for step in reduction.patch_plan[:4]:
            lines.append(f"- {step}")
        lines.append("")
        lines.append("3) Risks/guards")
        for p in reduction.pitfalls[:3]:
            lines.append(f"- {p}")
        lines.append(f"- Reduction confidence score: {reduction.confidence_score}")
        if consult_payload.get("meta", {}).get("outputs"):
            lines.append("- Provenance outputs available in adapter metadata.")
        return "\n".join(lines)

    def _warn(self, trace_id: str, event: str, message: str) -> None:
        print(f"[warn] {message}", file=sys.stderr)
        self.audit.log({"event": event, "trace_id": trace_id, "level": "warning", "message": message})

    def _audit_adapter_disabled(self, adapter_name: str, enabled: bool) -> None:
        if enabled:
            return
        self.audit.log(
            {
                "event": "adapter_disabled",
                "trace_id": "startup",
                "adapter": adapter_name,
                "level": "warning",
                "message": f"Adapter {adapter_name} disabled",
            }
        )


def redact_secrets(text: str) -> str:
    redacted = text
    for pat in ["sk-", "Bearer "]:
        if pat in redacted:
            redacted = redacted.replace(pat, "[REDACTED]-")
    return redacted


def build_minimized_goal(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= 180:
        return stripped
    return textwrap.shorten(stripped, width=180, placeholder="...")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def new_trace_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = hashlib.sha256(str(time.time_ns()).encode("utf-8")).hexdigest()[:6]
    return f"axn_{ts}_{suffix}"


def deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def cli_loop(orchestrator: AxeonOrchestrator) -> None:
    print("Axeon local orchestrator. Type '/exit' to quit, '/stats' for budget counters.")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not user_input:
            continue
        if user_input in {"/exit", "exit", "quit"}:
            print("bye")
            return
        if user_input == "/stats":
            print(json.dumps(orchestrator.audit.today_counters(), indent=2))
            continue
        if user_input == "/config":
            print(json.dumps(orchestrator.config, indent=2))
            continue

        response = orchestrator.handle_turn(user_input)
        print(f"axeon> {response}\n")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Axeon local Ollama orchestrator")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--once", type=str, default=None, help="Run one turn and exit")
    parser.add_argument("--print-default-config", action="store_true", help="Print default config JSON")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if args.print_default_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2))
        return 0

    orchestrator = AxeonOrchestrator.from_config_path(args.config)

    if args.once:
        print(orchestrator.handle_turn(args.once))
        return 0

    cli_loop(orchestrator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
