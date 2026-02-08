from openai import OpenAI
client = OpenAI()

def ask_chatgpt(question):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful AI coder. Give direct fixes with exact line numbers and old/new code snippets only. No explanations or extra text."},
            {"role": "user", "content": question},
        ]
    )
    return response.choices[0].message.content

question = """Fix bug in axeon_orchestrator.py: DeepSeek tag 'deepseek-coder:6.7b-instruct-q5_K_M' is loaded, logs show 'consult_model': 'deepseek_coder' and 'source': 'model:deepseek_coder' on consult_complete, but responses are still in Qwen's numbered 1/2/3 format. ChatDev adapter warns 'disabled' even when enabled in config. Patch ProviderClient.is_available() for tag variations, bypass chatdev disabled warn when enabled, and skip numbered format in _writer_prompts for consult_used and code/self_heal intents. Keep One-Writer, budgets, audits. Output only 3 patches with exact line numbers + old/new code snippets, and one test command to verify."""

print(ask_chatgpt(question))
