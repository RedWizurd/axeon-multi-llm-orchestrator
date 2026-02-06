# axeon-multi-llm-orchestrator

LangChain/LangGraph-friendly wrapper for multi-LLM routing/distillation with strict governance:

- One-Writer rule (single visible writer response)
- Consult cap (<20% of turns + daily token/consult budgets)
- Audit trail (`events.jsonl`)
- Local-first fallback (Qwen via Ollama) when remote keys are missing
- Optional hidden adapters:
  - `agent0-adapter` (one-shot tool execution, distilled only)
  - `chatdev-adapter` (3 roles + Medical Director audits; configurable themes)
  - `rdf-forensics-adapter` (general/private/local corpora KG runs with provenance + cache/idempotency)

## Quick Start

```bash
cd /Users/eddie/Documents/axeon-multi-llm-orchestrator
python3 axeon_orchestrator.py --print-default-config > config.json
python3 axeon_orchestrator.py --config config.json
```

Optional env keys (remote consults):

- `XAI_API_KEY`
- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`
- `GEMINI_API_KEY`

If none are set, orchestrator remains local-first and uses `local_qwen`.

## Enable Adapters

Edit `config.json`:

```json
{
  "adapters": {
    "agent0": {"enabled": true, "mock_mode": true},
    "chatdev": {"enabled": true, "theme": "clinic"},
    "rdf_forensics": {
      "enabled": true,
      "policy_path": "./policies/rdf_forensics_policy.json"
    }
  }
}
```

## Governance Defaults

- `max_consult_ratio`: `0.20`
- `max_consults_per_day`: `20`
- `max_total_consult_tokens_per_day`: `20000`
- `allow_sending_user_content`: `false`
- `redact_secrets`: `true`

## Commands

- Interactive loop: `python3 axeon_orchestrator.py --config config.json`
- One-shot run: `python3 axeon_orchestrator.py --config config.json --once "build KG from my docs"`
- Session stats in loop: `/stats`
- Show live config in loop: `/config`

## Notes

- Raw consultant output is not logged or persisted.
- Audit events record digests, token estimates, and trace IDs.
- If LangGraph is installed, the route -> consult -> reduce -> write flow compiles into a graph automatically.
