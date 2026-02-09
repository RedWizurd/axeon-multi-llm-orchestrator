from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from tools import ToolExecutor

LOGGER = logging.getLogger("axeon_orchestrator")

_RECENT_TRACES: deque[Dict[str, Any]] = deque(maxlen=300)
_TRACE_LOCK = Lock()


@dataclass
class TurnResult:
    response: str
    swarm_trace: List[Dict[str, Any]] = field(default_factory=list)
    used_swarm: bool = False
    used_web: bool = False
    mode: str = "direct"


def _truncate_text(text: str, max_len: int = 500) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _normalize_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for msg in history or []:
        role = str(msg.get("role", "user")).strip()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        entry = {"role": role, "content": content}
        if msg.get("name"):
            entry["name"] = str(msg.get("name"))
        normalized.append(entry)
    return normalized


def _compress_history(
    history: List[Dict[str, str]],
    max_chars: int = 9000,
    keep_recent: int = 8,
) -> List[Dict[str, str]]:
    if not history:
        return []

    serialized = json.dumps(history, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return history

    recent = history[-keep_recent:]
    older = history[:-keep_recent]
    summary_lines = []
    for msg in older[-20:]:
        role = msg.get("role", "user")
        content = _truncate_text(msg.get("content", ""), 220)
        summary_lines.append(f"- {role}: {content}")

    summary = "Earlier conversation summary:\n" + "\n".join(summary_lines)
    return [{"role": "system", "content": summary}] + recent


# CHANGED: model-assisted history summarization trigger helpers.
def _estimate_tokens_for_messages(messages: List[Dict[str, str]]) -> int:
    total_words = sum(len((m.get("content") or "").split()) for m in messages)
    return int(total_words * 1.3) + 20


def _summarize_history_with_consult_model(
    history: List[Dict[str, str]],
    ollama_host: str,
    consult_model: str,
    temperature: float = 0.2,
) -> List[Dict[str, str]]:
    if not history:
        return []
    if len(history) <= 5:
        return history

    older = history[:-5]
    recent = history[-5:]
    summary_prompt = (
        "Summarize the key points of this conversation history in 300-500 tokens, "
        "keeping recent messages intact:\n\n"
        f"{json.dumps(older, ensure_ascii=False)}"
    )
    try:
        summary = _call_ollama(
            ollama_host=ollama_host,
            model=consult_model,
            prompt=summary_prompt,
            temperature=temperature,
            timeout=45,
        )
        if summary.strip():
            return [{"role": "system", "content": f"Summarized history:\n{summary.strip()}"}] + recent
    except Exception as exc:
        LOGGER.warning("History summarization via consult model failed: %s", exc)
    return history


def _history_to_text(history: List[Dict[str, str]], max_items: int = 20) -> str:
    lines: List[str] = []
    for msg in history[-max_items:]:
        role = msg.get("role", "user")
        name = msg.get("name")
        prefix = f"{role}/{name}" if name else role
        content = _truncate_text(msg.get("content", ""), 1200)
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _extract_tool_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    candidates.extend(fenced)

    brace_match = re.search(r"(\{[\s\S]*\})", text)
    if brace_match:
        candidates.append(brace_match.group(1))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        tool_name = payload.get("tool")
        if isinstance(tool_name, str) and tool_name.strip():
            args = payload.get("args")
            if not isinstance(args, dict):
                args = {"value": args}
            thought = payload.get("thought")
            return {
                "tool": tool_name.strip(),
                "args": args,
                "thought": thought if isinstance(thought, str) else "",
            }
    return None


def _extract_thought(text: str) -> str:
    tool_payload = _extract_tool_json(text)
    if tool_payload and tool_payload.get("thought"):
        return _truncate_text(tool_payload["thought"], 180)

    match = re.search(r"(?:^|\n)thought\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return _truncate_text(match.group(1), 180)

    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return _truncate_text(first_line, 180)


def _looks_bad_output(text: str) -> bool:
    clean = (text or "").strip()
    low = clean.lower()
    return not clean or len(clean) < 20 or "[error" in low or "failed" in low or "traceback" in low


def _is_complex_query(message: str) -> bool:
    low = (message or "").lower()
    keywords = [
        "build",
        "design",
        "architecture",
        "debug",
        "refactor",
        "plan",
        "multi-step",
        "orchestrate",
        "research",
        "analyze",
        "optimize",
        "improve",
        "implement",
    ]
    return len(low) > 240 or any(word in low for word in keywords)


def _is_self_improvement_request(message: str) -> bool:
    low = (message or "").lower()
    # CHANGED: explicit self-improvement trigger phrases.
    keywords = ["self-improve", "upgrade yourself", "axeon 2.0", "improve your code"]
    return any(word in low for word in keywords)


def _should_offer_direct_tools(message: str) -> bool:
    low = (message or "").lower()
    # CHANGED: broader direct-tool trigger list for aggressive tool availability.
    triggers = [
        "calculate",
        "compute",
        "math",
        "use tools",
        "tool",
        "search",
        "fetch",
        "current",
        "now",
        "live",
        "real-time",
        "look up",
        "latest",
        "weather",
        "news",
        "external data",
        "read file",
        "write file",
        "api",
        "execute",
        "run code",
    ]
    return any(trigger in low for trigger in triggers)


def _detect_categories(message: str) -> set[str]:
    low = (message or "").lower()
    # CHANGED: expanded routing categories for web-fallback matching.
    category_keywords = {
        "creative": ["story", "poem", "creative", "brainstorm", "tagline", "script"],
        "research": ["research", "compare", "sources", "citations", "investigate"],
        "current_events": ["today", "latest", "news", "current", "this week", "breaking"],
        "weather": ["weather", "forecast", "temperature", "rain", "snow", "humidity"],
        "news": ["news", "headline", "breaking", "current events"],
        "real-time": ["live", "now", "real-time", "currently", "today"],
    }
    hits = set()
    for category, keywords in category_keywords.items():
        if any(keyword in low for keyword in keywords):
            hits.add(category)
    return hits


def _append_run_trace(record: Dict[str, Any]) -> None:
    with _TRACE_LOCK:
        _RECENT_TRACES.append(record)


def get_recent_traces(limit: int = 50) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(limit, 200))
    with _TRACE_LOCK:
        return list(_RECENT_TRACES)[-safe_limit:]


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    # Load from .env first, fallback to config.
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _call_ollama(
    ollama_host: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = requests.post(f"{ollama_host}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return (response.json().get("response") or "").strip()


class Agent:
    def __init__(
        self,
        role: str,
        model: str,
        ollama_host: str,
        timeout: int = 120,
        summary_model: Optional[str] = None,
    ):
        self.role = role
        self.model = model
        self.ollama_host = ollama_host
        self.timeout = timeout
        self.summary_model = summary_model or model

    def _build_prompt(
        self,
        task: str,
        history: List[Dict[str, str]],
        tool_executor: Optional[ToolExecutor],
    ) -> str:
        tool_block = ""
        if tool_executor:
            tool_block = (
                "\n\nTool usage (ReAct style):\n"
                "- If a tool is required, reply ONLY with strict JSON: "
                "{\"tool\": \"name\", \"args\": {...}, \"thought\": \"why\"}.\n"
                "- Do not wrap tool JSON in prose.\n"
                "- Available tools:\n"
                f"{tool_executor.describe_tools()}\n"
            )

        return (
            f"You are Axeon agent '{self.role}'.\n"
            "Produce clear, technically correct output.\n"
            "If the user asks for codebase improvements, you may return patch/diff text, but never auto-apply changes.\n"
            f"Task:\n{task}\n\n"
            f"Conversation context:\n{_history_to_text(history)}"
            f"{tool_block}\n"
            "Final response:"
        )

    def act(
        self,
        task: str,
        history: List[Dict[str, str]],
        temperature: float = 0.7,
        tool_executor: Optional[ToolExecutor] = None,
        max_tool_calls: int = 3,
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        trace_list = trace if trace is not None else []
        working_history = list(history)
        max_total_tool_calls = max(0, min(max_tool_calls, 3))
        total_tool_calls = 0
        last_non_json_output = ""
        last_tool_result_text = ""
        exit_reason = "model_output"

        while True:
            prompt = self._build_prompt(task, working_history, tool_executor)
            try:
                output = _call_ollama(
                    ollama_host=self.ollama_host,
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    timeout=self.timeout,
                )
            except Exception as exc:
                output = f"[{self.role} ERROR] {exc}"

            thought = _extract_thought(output)
            trace_list.append(
                {
                    "timestamp": time.time(),
                    "agent": self.role,
                    "step": f"model:{total_tool_calls + 1}",
                    "thought": thought,
                    "output": output,
                }
            )
            LOGGER.info("[%s] Thought: %s | Output: %s", self.role, thought, _truncate_text(output, 260))

            if not tool_executor:
                return output

            tool_payload = _extract_tool_json(output)
            if not tool_payload:
                LOGGER.info("[%s] Tool loop exit reason: model produced final non-tool output.", self.role)
                return output

            if total_tool_calls >= max_total_tool_calls:
                exit_reason = "max_tool_calls"
                LOGGER.warning("Max tool calls reached for agent %s", self.role)
                break

            tool_name = tool_payload["tool"]
            tool_args = tool_payload.get("args", {})
            total_tool_calls += 1
            LOGGER.info("[Tool] Attempt %s: %s(%s)", total_tool_calls, tool_name, tool_args)
            try:
                tool_result = tool_executor.execute(tool_name, tool_args)
            except Exception as exc:
                tool_result = {"error": str(exc)}

            tool_result_text = json.dumps(tool_result, ensure_ascii=False)
            last_tool_result_text = tool_result_text
            trace_list.append(
                {
                    "timestamp": time.time(),
                    "agent": self.role,
                    "step": f"tool:{tool_name}",
                    "thought": tool_payload.get("thought", ""),
                    "output": tool_result_text,
                    "tool": tool_name,
                }
            )
            LOGGER.info("[Tool] Result: %s", _truncate_text(tool_result_text, 300))
            LOGGER.info("[%s] Tool: %s | Result: %s", self.role, tool_name, _truncate_text(tool_result_text, 260))

            working_history.append({"role": "assistant", "name": self.role, "content": output})
            working_history.append({"role": "tool", "name": tool_name, "content": tool_result_text})

            # Force use of tool result if available
            if tool_result_text and "error" not in tool_result_text.lower():
                final_prompt = (
                    "Using this tool result, give a concise, natural answer to the user:\n\n"
                    f"Task: {task}\nTool Result: {tool_result_text}"
                )
                try:
                    final_output = requests.post(
                        f"{self.ollama_host}/api/generate",
                        json={
                            "model": self.summary_model,
                            "prompt": final_prompt,
                            "stream": False,
                            "options": {"temperature": 0.5},
                        },
                        timeout=30,
                    ).json().get("response", "").strip()
                    if final_output:
                        return final_output
                except Exception as e:
                    print(f"[DEBUG] Synthesis failed: {e}")

            # Fixed: if tool returns empty/error, produce useful fallback response instead of raw JSON.
            tool_failed = bool(tool_result.get("error")) or int(tool_result.get("count", 1) or 0) == 0
            if tool_failed:
                LOGGER.info("[%s] Tool loop exit reason: tool returned error/empty result.", self.role)
                fallback_prompt = (
                    "Tool failed, here's general info. "
                    "Provide best-effort guidance for this user query in concise natural language.\n\n"
                    f"User task:\n{task}\n\nTool output:\n{tool_result_text}"
                )
                try:
                    fallback_nl = _call_ollama(
                        ollama_host=self.ollama_host,
                        model=self.summary_model,
                        prompt=fallback_prompt,
                        temperature=min(temperature, 0.6),
                        timeout=self.timeout,
                    ).strip()
                    if fallback_nl and not _extract_tool_json(fallback_nl):
                        return fallback_nl
                except Exception:
                    pass

            # Prevent repeated raw JSON loops by requesting final natural-language synthesis.
            synth_prompt = (
                "Tool result received. Provide a final natural-language answer for the user. "
                "Do not output tool JSON.\n\n"
                f"Original task:\n{task}\n\nLatest tool result:\n{tool_result_text}"
            )
            try:
                natural_output = _call_ollama(
                    ollama_host=self.ollama_host,
                    model=self.summary_model,
                    prompt=synth_prompt,
                    temperature=min(temperature, 0.6),
                    timeout=self.timeout,
                ).strip()
            except Exception as exc:
                natural_output = f"[{self.role} ERROR] summary synthesis failed: {exc}"

            if natural_output and not _extract_tool_json(natural_output):
                last_non_json_output = natural_output
                trace_list.append(
                    {
                        "timestamp": time.time(),
                        "agent": self.role,
                        "step": f"summary:{total_tool_calls}",
                        "thought": "Generated final natural-language response after tool call.",
                        "output": natural_output,
                    }
                )
                LOGGER.info("[%s] Tool loop exit reason: produced post-tool natural-language response.", self.role)
                return natural_output

            if total_tool_calls >= max_total_tool_calls:
                exit_reason = "max_tool_calls"
                LOGGER.warning("Max tool calls reached for agent %s", self.role)
                break

        LOGGER.info("[%s] Tool loop exit reason: %s", self.role, exit_reason)
        # After tool loop - force tool result into final answer
        if last_tool_result_text and "error" not in last_tool_result_text.lower():
            final_prompt = (
                "Summarize this tool result as a direct answer to the user query.\n"
                "Do not mention tools, JSON, or internal process.\n\n"
                f"User query: {task}\n"
                f"Result: {last_tool_result_text}"
            )
            try:
                final_output = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.summary_model,
                        "prompt": final_prompt,
                        "stream": False,
                        "options": {"temperature": 0.4}
                    },
                    timeout=20
                ).json().get("response", "").strip()
                if final_output:
                    return final_output
            except Exception as e:
                LOGGER.warning(f"Final synthesis failed: {e}")

        # Ultimate fallback
        if last_tool_result_text and "error" in last_tool_result_text.lower():
            try:
                general_prompt = (
                    "Tool failed, here's general info. "
                    "Answer the user query with best-effort static knowledge, concise and clear.\n\n"
                    f"User query: {task}\n"
                    f"Failure details: {last_tool_result_text}"
                )
                general_output = _call_ollama(
                    ollama_host=self.ollama_host,
                    model=self.summary_model,
                    prompt=general_prompt,
                    temperature=0.5,
                    timeout=20,
                ).strip()
                if general_output:
                    return general_output
            except Exception:
                pass
        return f"Current conditions: {last_tool_result_text}" if last_tool_result_text else "No response generated."


def _run_agent_with_healing(
    agent: Agent,
    task: str,
    history: List[Dict[str, str]],
    temperature: float,
    tool_executor: Optional[ToolExecutor],
    max_tool_calls: int,
    healer: Optional[Agent],
) -> Tuple[str, List[Dict[str, Any]]]:
    local_trace: List[Dict[str, Any]] = []
    result = agent.act(
        task=task,
        history=history,
        temperature=temperature,
        tool_executor=tool_executor,
        max_tool_calls=max_tool_calls,
        trace=local_trace,
    )

    if healer and _looks_bad_output(result):
        heal_task = (
            "Repair the weak output below. Return a clean corrected response.\n\n"
            f"Problematic output:\n{result}\n\nOriginal task:\n{task}"
        )
        healed = healer.act(
            task=heal_task,
            history=history + [{"role": "assistant", "name": agent.role, "content": result}],
            temperature=temperature,
            tool_executor=tool_executor,
            max_tool_calls=max_tool_calls,
            trace=local_trace,
        )
        if healed and not _looks_bad_output(healed):
            result = healed

    return result, local_trace


def _execute_swarm_sequential(
    user_message: str,
    base_history: List[Dict[str, str]],
    agents: Dict[str, Agent],
    healer: Agent,
    temperature: float,
    tool_executor: Optional[ToolExecutor],
    max_tool_calls: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    trace: List[Dict[str, Any]] = []
    task = user_message
    swarm_history = list(base_history) + [{"role": "user", "content": user_message}]

    for role in ["CEO", "CTO", "Programmer", "Tester"]:
        output, local_trace = _run_agent_with_healing(
            agent=agents[role],
            task=task,
            history=swarm_history,
            temperature=temperature,
            tool_executor=tool_executor,
            max_tool_calls=max_tool_calls,
            healer=healer,
        )
        trace.extend(local_trace)
        swarm_history.append({"role": "assistant", "name": role, "content": output})
        task = output

    for msg in reversed(swarm_history):
        content = msg.get("content", "")
        if msg.get("role") == "assistant" and content and "error" not in content.lower():
            return content, trace

    return "Swarm completed but no valid final response generated.", trace


def _tester_parallel_pipeline(
    tester: Agent,
    merged_task: str,
    swarm_history: List[Dict[str, str]],
    programmer_future: concurrent.futures.Future,
    temperature: float,
    tool_executor: Optional[ToolExecutor],
    max_tool_calls: int,
    healer: Agent,
) -> Tuple[str, List[Dict[str, Any]]]:
    trace: List[Dict[str, Any]] = []

    precheck_task = (
        "Create a QA checklist while implementation runs. Focus on correctness, edge cases, and test strategy.\n\n"
        f"Plan:\n{merged_task}"
    )
    checklist, checklist_trace = _run_agent_with_healing(
        agent=tester,
        task=precheck_task,
        history=swarm_history,
        temperature=temperature,
        tool_executor=tool_executor,
        max_tool_calls=max_tool_calls,
        healer=healer,
    )
    trace.extend(checklist_trace)

    programmer_output, _ = programmer_future.result()

    review_task = (
        "Review the Programmer output using the QA checklist. Identify defects, risks, and fixes.\n\n"
        f"QA Checklist:\n{checklist}\n\n"
        f"Programmer Output:\n{programmer_output}"
    )
    review_history = swarm_history + [{"role": "assistant", "name": "Programmer", "content": programmer_output}]
    review, review_trace = _run_agent_with_healing(
        agent=tester,
        task=review_task,
        history=review_history,
        temperature=temperature,
        tool_executor=tool_executor,
        max_tool_calls=max_tool_calls,
        healer=healer,
    )
    trace.extend(review_trace)

    combined = f"QA Checklist:\n{checklist}\n\nQA Review:\n{review}"
    return combined, trace


def _execute_swarm_parallel(
    user_message: str,
    base_history: List[Dict[str, str]],
    agents: Dict[str, Agent],
    healer: Agent,
    temperature: float,
    tool_executor: Optional[ToolExecutor],
    max_tool_calls: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    trace: List[Dict[str, Any]] = []
    swarm_history = list(base_history) + [{"role": "user", "content": user_message}]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        ceo_future = pool.submit(
            _run_agent_with_healing,
            agents["CEO"],
            user_message,
            swarm_history,
            temperature,
            tool_executor,
            max_tool_calls,
            healer,
        )
        cto_future = pool.submit(
            _run_agent_with_healing,
            agents["CTO"],
            user_message,
            swarm_history,
            temperature,
            tool_executor,
            max_tool_calls,
            healer,
        )

        ceo_output, ceo_trace = ceo_future.result()
        cto_output, cto_trace = cto_future.result()
        trace.extend(ceo_trace)
        trace.extend(cto_trace)

        merged_task = (
            "Merge these two strategic plans into a concrete implementation directive.\n\n"
            f"CEO Plan:\n{ceo_output}\n\n"
            f"CTO Plan:\n{cto_output}\n\n"
            "Deliver practical implementation details."
        )

        merged_history = swarm_history + [
            {"role": "assistant", "name": "CEO", "content": ceo_output},
            {"role": "assistant", "name": "CTO", "content": cto_output},
        ]

        programmer_future = pool.submit(
            _run_agent_with_healing,
            agents["Programmer"],
            merged_task,
            merged_history,
            temperature,
            tool_executor,
            max_tool_calls,
            healer,
        )

        tester_future = pool.submit(
            _tester_parallel_pipeline,
            agents["Tester"],
            merged_task,
            merged_history,
            programmer_future,
            temperature,
            tool_executor,
            max_tool_calls,
            healer,
        )

        programmer_output, programmer_trace = programmer_future.result()
        tester_output, tester_trace = tester_future.result()
        trace.extend(programmer_trace)
        trace.extend(tester_trace)

    final_output = programmer_output
    if tester_output:
        final_output = f"{programmer_output}\n\nTester Findings:\n{tester_output}"

    return final_output, trace


def _intent_check(
    user_message: str,
    chatdev_enabled: bool,
    ollama_host: str,
    consult_model: str,
) -> bool:
    if not chatdev_enabled:
        return False

    low = (user_message or "").lower()
    # CHANGED: force-intent keywords requested by user.
    force_keywords = [
        "use tools",
        "tool",
        "current",
        "live",
        "now",
        "fetch",
        "search",
        "external data",
        "real-time",
        "weather",
        "news",
        "api",
    ]
    if any(keyword in low for keyword in force_keywords):
        return True
    if _is_self_improvement_request(user_message):
        return True

    intent_prompt = (
        "Is this query likely to benefit from a multi-agent breakdown "
        "(software development, debugging, complex planning, research, multi-step reasoning, "
        "tool usage, current/live/real-time info, weather/news/API lookups, or external data fetch)? "
        "Answer only 'yes' or 'no'.\n\n"
        f"Query: {user_message}"
    )

    try:
        decision = _call_ollama(
            ollama_host=ollama_host,
            model=consult_model,
            prompt=intent_prompt,
            temperature=0.0,
            timeout=25,
        ).lower()
        return "yes" in decision
    except Exception:
        return _is_complex_query(user_message)


def _call_web_ai_api(user_message: str, history: List[Dict[str, str]], web_cfg: Dict[str, Any]) -> Optional[str]:
    api_url = web_cfg.get("api_url")
    if not api_url:
        return None

    payload = {
        "prompt": user_message,
        "history": history,
        "provider": web_cfg.get("provider", "gemini"),
    }
    headers = {"Content-Type": "application/json"}
    api_key = web_cfg.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = int(web_cfg.get("timeout_seconds", 45))
    response = requests.post(api_url, json=payload, headers=headers, timeout=max(5, min(timeout, 90)))
    response.raise_for_status()

    data = response.json() if "application/json" in response.headers.get("content-type", "") else {}
    for key in ["response", "text", "output", "content"]:
        if isinstance(data.get(key), str) and data[key].strip():
            return data[key].strip()

    if isinstance(response.text, str) and response.text.strip():
        return response.text.strip()
    return None


# === Gemini fallback using CURRENT google-genai package (2026 API) ===
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = None
    gemini_client = None
    if gemini_api_key:
        try:
            # Preferred package path.
            from google import genai as google_genai

            # New google.genai client API (no configure()).
            gemini_client = google_genai.Client(api_key=gemini_api_key)
            gemini_model = "gemini-1.5-flash"
            LOGGER.info("[Gemini] Initialized successfully with google.genai Client + gemini-1.5-flash")
            print("[GEMINI STARTUP] Initialized via google.genai Client (model=gemini-1.5-flash)")
        except Exception as primary_exc:
            try:
                # Legacy compatibility fallback.
                import google.generativeai as google_genai_legacy

                google_genai_legacy.configure(api_key=gemini_api_key)
                gemini_model = google_genai_legacy.GenerativeModel("gemini-1.5-flash")
                gemini_client = None
                LOGGER.info("[Gemini] Initialized with legacy google.generativeai + gemini-1.5-flash")
                print("[GEMINI STARTUP] Initialized via legacy google.generativeai (model=gemini-1.5-flash)")
            except Exception as legacy_exc:
                gemini_model = None
                gemini_client = None
                LOGGER.warning(
                    "google-genai package import failed - Gemini fallback disabled. "
                    "primary=%s legacy=%s",
                    primary_exc,
                    legacy_exc,
                )
                print(
                    "[GEMINI STARTUP] Disabled. "
                    f"primary={primary_exc} legacy={legacy_exc}"
                )
    else:
        LOGGER.warning("GEMINI_API_KEY not set in .env - Gemini fallback disabled")
        print("[GEMINI STARTUP] Disabled. GEMINI_API_KEY not set.")
except Exception as e:
    gemini_model = None
    gemini_client = None
    LOGGER.error(f"Gemini init failed: {str(e)}")
    print(f"[GEMINI STARTUP] Init failed: {str(e)}")


def _call_gemini_api(user_message: str, history: List[Dict[str, str]], web_cfg: Dict[str, Any]) -> Optional[str]:
    if not gemini_model:
        return None
    history_text = _history_to_text(history, max_items=10)
    prompt = (
        "You are a web-enabled assistant. Provide accurate, up-to-date output when possible.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"User request:\n{user_message}"
    )
    try:
        if gemini_client is not None:
            response = gemini_client.models.generate_content(model=gemini_model, contents=prompt)
        else:
            response = gemini_model.generate_content(prompt)
        text = getattr(response, "text", "") or ""
        if text.strip():
            print(f"[DEBUG] Gemini response: {text[:200]}...")
            return text.strip()
        return None
    except Exception as exc:
        error_text = str(exc)
        if "404" in error_text or "429" in error_text:
            LOGGER.warning("Gemini API returned %s; falling back to local path.", error_text)
            return None
        LOGGER.warning("Gemini API call failed: %s", error_text)
        return None


def _call_web_ai_selenium(user_message: str, web_cfg: Dict[str, Any]) -> Optional[str]:
    selenium_cfg = web_cfg.get("selenium", {})
    if selenium_cfg.get("enabled", False) is False:
        LOGGER.info("Selenium fallback disabled: ChromeDriver not configured")
        return None

    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except Exception as exc:
        LOGGER.warning("Selenium unavailable for web fallback: %s", exc)
        return None

    url = selenium_cfg.get("url") or "https://gemini.google.com/app"
    input_selector = selenium_cfg.get("input_selector", "textarea")
    submit_selector = selenium_cfg.get("submit_selector")
    response_selector = selenium_cfg.get(
        "response_selector",
        "div.markdown, div.response-content, model-response, message-content",
    )
    timeout_seconds = int(selenium_cfg.get("timeout_seconds", 45))

    options = webdriver.ChromeOptions()
    if selenium_cfg.get("headless", True):
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    chrome_profile = selenium_cfg.get("chrome_user_data_dir")
    if chrome_profile:
        options.add_argument(f"--user-data-dir={chrome_profile}")

    driver = None
    try:
        try:
            driver = webdriver.Chrome(options=options)
        except Exception as exc:
            LOGGER.warning("Selenium fallback disabled: ChromeDriver not configured (%s)", exc)
            return None
        wait = WebDriverWait(driver, timeout_seconds)
        driver.get(url)

        input_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, input_selector)))
        input_box.clear()
        input_box.send_keys(user_message)

        if submit_selector:
            submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, submit_selector)))
            submit.click()
        else:
            input_box.send_keys(Keys.ENTER)

        def _has_response(_driver: Any) -> bool:
            elements = _driver.find_elements(By.CSS_SELECTOR, response_selector)
            return bool(elements and elements[-1].text and elements[-1].text.strip())

        wait.until(_has_response)
        elements = driver.find_elements(By.CSS_SELECTOR, response_selector)
        if not elements:
            return None
        text = (elements[-1].text or "").strip()
        return text or None
    except Exception as exc:
        LOGGER.warning("Web selenium fallback failed: %s", exc)
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def _try_web_ai(user_message: str, history: List[Dict[str, str]], web_cfg: Dict[str, Any]) -> Optional[str]:
    if not web_cfg.get("enabled", False):
        return None

    preferred_for = set(web_cfg.get("preferred_for", []))
    categories = _detect_categories(user_message)
    if preferred_for:
        if not categories:
            return None
        if not (preferred_for & categories):
            return None

    # Fixed: prefer Gemini API first, then Selenium, then local writer fallback.
    try:
        gemini_response = _call_gemini_api(user_message, history, web_cfg)
        if gemini_response:
            return gemini_response
    except Exception as exc:
        LOGGER.warning("Gemini API fallback failed: %s", exc)

    try:
        selenium_response = _call_web_ai_selenium(user_message, web_cfg)
        if selenium_response:
            return selenium_response
    except Exception as exc:
        LOGGER.warning("Web Selenium fallback failed: %s", exc)

    return None


def build_orchestrator_runner(config: Dict[str, Any]):
    try:
        base_dir = Path(__file__).resolve().parent
        ollama_host = config.get("ollama_host", "http://localhost:11434")
        models = config.get("models", {})
        writer_model = models.get("writer", "qwen2.5:7b-instruct-q4_K_M")
        consult_model = models.get("consult", "deepseek-coder:6.7b-instruct-q5_K_M")

        adapters = config.get("adapters", {})
        chatdev_enabled = adapters.get("chatdev", {}).get("enabled", False)

        tools_cfg = config.get("tools", {})
        tools_enabled = tools_cfg.get("enabled", True)
        max_tool_calls = int(tools_cfg.get("max_tool_calls_per_agent", 3))
        allowed_roots = tools_cfg.get("allowed_roots") or [str(base_dir)]
        tool_timeout = int(tools_cfg.get("request_timeout_seconds", 12))

        tool_executor: Optional[ToolExecutor] = None
        if tools_enabled:
            tool_executor = ToolExecutor(
                base_dir=base_dir,
                allowed_roots=allowed_roots,
                request_timeout=tool_timeout,
            )

        swarm_mode = str(config.get("swarm_mode", "sequential")).lower().strip()
        if swarm_mode not in {"sequential", "parallel"}:
            swarm_mode = "sequential"

        web_cfg = dict(config.get("web_fallback", {}))
        # Load from .env first, fallback to config.
        gemini_api_key = os.getenv("GEMINI_API_KEY") or web_cfg.get("gemini_api_key") or config.get("gemini_api_key")
        openweather_api_key = os.getenv("OPENWEATHER_API_KEY") or config.get("openweather_api_key")
        web_cfg["gemini_api_key"] = gemini_api_key
        if openweather_api_key:
            web_cfg["openweather_api_key"] = openweather_api_key

        if web_cfg.get("enabled", False) and not gemini_api_key:
            LOGGER.warning(
                "Web fallback is enabled but GEMINI_API_KEY is not set; "
                "Gemini path disabled, local fallback remains active."
            )

        # CHANGED: default self-improvement iteration count now 3.
        max_iterations = max(1, int(config.get("max_iterations", 3)))
        history_cfg = config.get("history", {})
        history_max_chars = int(history_cfg.get("max_chars", 9000))
        history_keep_recent = int(history_cfg.get("keep_recent", 8))

        agents = {
            "CEO": Agent("CEO", consult_model, ollama_host, summary_model=writer_model),
            "CTO": Agent("CTO", consult_model, ollama_host, summary_model=writer_model),
            "Programmer": Agent("Programmer", writer_model, ollama_host, summary_model=writer_model),
            "Tester": Agent("Tester", consult_model, ollama_host, summary_model=writer_model),
            # CHANGED: ToolAdvisor agent for up-front tool planning.
            "ToolAdvisor": Agent("ToolAdvisor", consult_model, ollama_host, summary_model=writer_model),
            "DirectWriter": Agent("DirectWriter", writer_model, ollama_host, summary_model=writer_model),
        }
        healer = Agent("Error Healer", consult_model, ollama_host, summary_model=writer_model)

        def execute_swarm_once(
            prompt_text: str,
            base_history: List[Dict[str, str]],
            temperature: float,
        ) -> Tuple[str, List[Dict[str, Any]]]:
            if swarm_mode == "parallel":
                return _execute_swarm_parallel(
                    user_message=prompt_text,
                    base_history=base_history,
                    agents=agents,
                    healer=healer,
                    temperature=temperature,
                    tool_executor=tool_executor,
                    max_tool_calls=max_tool_calls,
                )
            return _execute_swarm_sequential(
                user_message=prompt_text,
                base_history=base_history,
                agents=agents,
                healer=healer,
                temperature=temperature,
                tool_executor=tool_executor,
                max_tool_calls=max_tool_calls,
            )

        def runner(
            user_message: str,
            history: Optional[List[Dict[str, str]]] = None,
            temperature: float = 0.7,
            include_trace: bool = False,
        ) -> TurnResult:
            run_id = f"axeon-{uuid.uuid4().hex[:12]}"
            norm_history = _normalize_history(history)
            compact_history = _compress_history(
                norm_history,
                max_chars=history_max_chars,
                keep_recent=history_keep_recent,
            )
            # CHANGED: consult-model summarization when context grows past ~6000 tokens.
            if _estimate_tokens_for_messages(compact_history) > 6000:
                compact_history = _summarize_history_with_consult_model(
                    compact_history,
                    ollama_host=ollama_host,
                    consult_model=consult_model,
                    temperature=min(temperature, 0.3),
                )

            trace: List[Dict[str, Any]] = []
            use_swarm = _intent_check(user_message, chatdev_enabled, ollama_host, consult_model)
            # Fixed: force tool/swarm mode on explicit user instruction.
            explicit_tool_request = ("use tools" in user_message.lower()) or ("tool" in user_message.lower())
            if explicit_tool_request:
                use_swarm = True
            direct_tool_bias = _should_offer_direct_tools(user_message)
            borderline_intent = (not use_swarm) and (_is_complex_query(user_message) or direct_tool_bias)

            used_web = False
            web_preferred_match = bool(set(web_cfg.get("preferred_for", [])) & _detect_categories(user_message))
            if use_swarm and web_cfg.get("enabled", False) and web_preferred_match:
                web_output = _try_web_ai(user_message, compact_history, web_cfg)
                if web_output:
                    used_web = True
                    trace.append(
                        {
                            "timestamp": time.time(),
                            "agent": "WebFallback",
                            "step": "web_ai",
                            "thought": "Web AI selected based on routing preference.",
                            "output": web_output,
                        }
                    )
                    result = TurnResult(
                        response=web_output,
                        swarm_trace=trace if include_trace else [],
                        used_swarm=True,
                        used_web=True,
                        mode="web",
                    )
                    _append_run_trace(
                        {
                            "run_id": run_id,
                            "timestamp": time.time(),
                            "mode": result.mode,
                            "used_swarm": result.used_swarm,
                            "used_web": result.used_web,
                            "input": _truncate_text(user_message, 500),
                            "response": _truncate_text(result.response, 1200),
                            "swarm_trace": trace,
                        }
                    )
                    return result
                trace.append(
                    {
                        "timestamp": time.time(),
                        "agent": "WebFallback",
                        "step": "web_ai_failed",
                        "thought": "Preferred web route failed, falling back to local writer model.",
                        "output": "Gemini/web fallback unavailable or failed.",
                    }
                )
                # CHANGED: explicit fallback to local writer model when preferred web route fails.
                direct_trace: List[Dict[str, Any]] = []
                direct_web_response = agents["DirectWriter"].act(
                    task=f"Respond directly and clearly.\n\nUser request:\n{user_message}",
                    history=compact_history + [{"role": "user", "content": user_message}],
                    temperature=temperature,
                    tool_executor=tool_executor if direct_tool_bias else None,
                    max_tool_calls=min(max_tool_calls, 3),
                    trace=direct_trace,
                )
                trace.extend(direct_trace)
                result = TurnResult(
                    response=direct_web_response or "No response generated.",
                    swarm_trace=trace if include_trace else [],
                    used_swarm=False,
                    used_web=False,
                    mode="direct_web_fallback",
                )
                _append_run_trace(
                    {
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "mode": result.mode,
                        "used_swarm": result.used_swarm,
                        "used_web": result.used_web,
                        "input": _truncate_text(user_message, 500),
                        "response": _truncate_text(result.response, 1200),
                        "swarm_trace": trace,
                    }
                )
                return result

            if use_swarm:
                # CHANGED: ToolAdvisor runs first to suggest tools/workflow before swarm execution.
                advisor_task = (
                    "Decide whether tools are needed. If yes, reply with tool JSON. "
                    "Otherwise provide a brief plan for CEO/CTO/Programmer/Tester."
                    f"\n\nUser request:\n{user_message}"
                )
                advisor_trace: List[Dict[str, Any]] = []
                advisor_output = agents["ToolAdvisor"].act(
                    task=advisor_task,
                    history=compact_history + [{"role": "user", "content": user_message}],
                    temperature=min(temperature, 0.5),
                    tool_executor=tool_executor,
                    max_tool_calls=max_tool_calls,
                    trace=advisor_trace,
                )
                trace.extend(advisor_trace)
                swarm_prompt = user_message
                if advisor_output.strip():
                    swarm_prompt = (
                        f"{user_message}\n\nToolAdvisor Guidance:\n{advisor_output}\n\n"
                        "Use the guidance above where useful."
                    )

                response, swarm_trace = execute_swarm_once(swarm_prompt, compact_history, temperature)
                trace.extend(swarm_trace)

                if _is_self_improvement_request(user_message):
                    iterative_response = response
                    for iteration in range(2, max_iterations + 1):
                        if iterative_response.strip().lower().startswith("done"):
                            break

                        refine_prompt = (
                            f"Iteration {iteration}/{max_iterations}. Swarm review the previous output, "
                            "suggest concrete changes, and have Programmer produce a diff/patch. "
                            "If no further improvements are needed, respond with DONE.\n\n"
                            f"Original user request:\n{user_message}\n\n"
                            f"Previous output:\n{iterative_response}"
                        )
                        next_output, next_trace = execute_swarm_once(
                            refine_prompt,
                            compact_history + [{"role": "assistant", "content": iterative_response}],
                            temperature,
                        )
                        for item in next_trace:
                            item["iteration"] = iteration
                        trace.extend(next_trace)

                        if next_output.strip().lower().startswith("done"):
                            break
                        iterative_response = next_output

                    response = iterative_response

                result = TurnResult(
                    response=response,
                    swarm_trace=trace if include_trace else [],
                    used_swarm=True,
                    used_web=used_web,
                    mode=swarm_mode,
                )
            else:
                direct_task = (
                    "Respond directly to the user's request. Keep it accurate and practical.\n\n"
                    f"User request:\n{user_message}"
                )

                # CHANGED: allow aggressive tool loop in direct mode for explicit/borderline tool intents.
                direct_tools_enabled = bool(tool_executor and (direct_tool_bias or borderline_intent))
                direct_trace: List[Dict[str, Any]] = []
                response = agents["DirectWriter"].act(
                    task=direct_task,
                    history=compact_history + [{"role": "user", "content": user_message}],
                    temperature=temperature,
                    tool_executor=tool_executor if direct_tools_enabled else None,
                    max_tool_calls=min(max_tool_calls, 3),
                    trace=direct_trace,
                )
                trace.extend(direct_trace)
                result = TurnResult(
                    response=response or "No response generated.",
                    swarm_trace=trace if include_trace else [],
                    used_swarm=False,
                    used_web=False,
                    mode="direct",
                )

            _append_run_trace(
                {
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "mode": result.mode,
                    "used_swarm": result.used_swarm,
                    "used_web": result.used_web,
                    "input": _truncate_text(user_message, 500),
                    "response": _truncate_text(result.response, 1200),
                    "swarm_trace": trace,
                }
            )
            return result

        return runner

    except Exception as exc:
        LOGGER.exception("Orchestrator build failed: %s", exc)

        def dummy_runner(
            user_message: str,
            history: Optional[List[Dict[str, str]]] = None,
            temperature: float = 0.7,
            include_trace: bool = False,
        ) -> TurnResult:
            return TurnResult(
                response="Axeon is starting up or encountered an init issue. Please retry in a moment.",
                swarm_trace=[],
                used_swarm=False,
                used_web=False,
                mode="init_error",
            )

        return dummy_runner


def handle_turn_with_meta(
    user_message: str,
    history: Optional[List[Dict[str, str]]],
    config_path: str = "config.json",
    temperature: float = 0.7,
    include_trace: bool = True,
) -> Dict[str, Any]:
    config = load_config(config_path)
    runner = build_orchestrator_runner(config)
    result = runner(user_message, history or [], temperature=temperature, include_trace=include_trace)
    return {
        "response": result.response,
        "swarm_trace": result.swarm_trace,
        "used_swarm": result.used_swarm,
        "used_web": result.used_web,
        "mode": result.mode,
    }


def handle_turn(
    user_message: str,
    history: Optional[List[Dict[str, str]]],
    config_path: str = "config.json",
    temperature: float = 0.7,
) -> str:
    result = handle_turn_with_meta(
        user_message=user_message,
        history=history,
        config_path=config_path,
        temperature=temperature,
        include_trace=False,
    )
    return result.get("response", "No response generated.")


if __name__ == "__main__":
    print(handle_turn("Hello, Axeon! Tell me about yourself.", []))
