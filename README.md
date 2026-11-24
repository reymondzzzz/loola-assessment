# Loola test task — agent harness

Minimal coding-agent loop (one `bash` tool, OpenAI-compatible API).

```bash
cp .env.example .env   # fill in LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
uv run --env-file .env agent.py "list files in the current directory"
```
