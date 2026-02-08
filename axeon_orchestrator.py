import json
import requests
from typing import Dict, List, Any

class Agent:
    def __init__(self, role: str, model: str, ollama_host: str):
        self.role = role
        self.model = model
        self.ollama_host = ollama_host

    def act(self, task: str, history: List[Dict[str, str]]) -> str:
        prompt = f"You are an expert AI agent with role: {self.role}\n\nCurrent task: {task}\n\nConversation history so far:\n{json.dumps(history, indent=2)}\n\nYour response:"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        try:
            response = requests.post(f"{self.ollama_host}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"[{self.role} ERROR]: {str(e)}"

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def build_orchestrator_runner(config: Dict[str, Any]):
    try:
        ollama_host = config.get("ollama_host", "http://localhost:11434")
        writer_model = config["models"].get("writer", "qwen")
        consult_model = config["models"].get("consult", "deepseek-coder")
        chatdev_enabled = config.get("adapters", {}).get("chatdev", {}).get("enabled", False)

        # ChatDev-style agent swarm for complex / self-healing tasks
        agents = [
            Agent("CEO", consult_model, ollama_host),
            Agent("CTO", consult_model, ollama_host),
            Agent("Programmer", writer_model, ollama_host),
            Agent("Tester", consult_model, ollama_host)
        ]

        def runner(user_message: str, history: List[Dict[str, str]] = []) -> str:
            # Quick intent check: use swarm for anything that looks like dev, planning, debugging, complex reasoning
            intent_prompt = f"Is this query likely to benefit from a multi-agent breakdown (software development, debugging, complex planning, research, multi-step reasoning)? Answer only 'yes' or 'no'.\n\nQuery: {user_message}"
            try:
                intent_resp = requests.post(
                    f"{ollama_host}/api/generate",
                    json={"model": consult_model, "prompt": intent_prompt, "stream": False},
                    timeout=30
                ).json().get("response", "").strip().lower()
                use_swarm = chatdev_enabled and "yes" in intent_resp
            except:
                use_swarm = False  # fallback to direct if intent check fails

            if use_swarm:
                task = user_message
                swarm_history = [{"role": "user", "content": user_message}]
                for agent in agents:
                    result = agent.act(task, swarm_history)
                    swarm_history.append({"role": agent.role, "content": result})
                    task = result  # pass output forward

                    # Self-healing: if obvious error, add a quick repair step
                    if "ERROR" in result or "failed" in result.lower() or len(result) < 20:
                        healer = Agent("Error Healer", consult_model, ollama_host)
                        fix = healer.act(f"Analyze and repair this failure:\n{result}\nContext: {task}", swarm_history)
                        swarm_history.append({"role": "Healer", "content": fix})
                        task = fix

                # Final output is the last non-error result
                for entry in reversed(swarm_history):
                    if entry["role"] != "user" and "ERROR" not in entry["content"]:
                        return entry["content"]
                return "Swarm completed but no valid final response generated."

            else:
                # Direct single-model response (fast path)
                full_prompt = f"History:\n{json.dumps(history, indent=2)}\n\nUser: {user_message}\n\nAssistant:"
                try:
                    resp = requests.post(
                        f"{ollama_host}/api/generate",
                        json={"model": writer_model, "prompt": full_prompt, "stream": False},
                        timeout=90
                    ).json().get("response", "").strip()
                    return resp or "No response generated."
                except Exception as e:
                    return f"Direct model failed: {str(e)}. Try again."

        return runner

    except Exception as e:
        print(f"Orchestrator build failed: {e}")
        def dummy_runner(user_message: str, history: List[Dict[str, str]] = []) -> str:
            return "Axeon is starting up or encountered an init issue. Please retry in a moment."
        return dummy_runner

def handle_turn(user_message: str, history: List[Dict[str, str]], config_path: str = "config.json") -> str:
    config = load_config(config_path)
    runner = build_orchestrator_runner(config)
    return runner(user_message, history)

if __name__ == "__main__":
    print(handle_turn("Hello, Axeon! Tell me about yourself.", []))