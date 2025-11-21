## OpenAI API Forwarder (Python SDK)

This service exposes two HTTP endpoints so that a Responses-only system and a Chat-Completions-only system can talk to each other:

| Endpoint | What it accepts | Forwarded to | Returned as |
| --- | --- | --- | --- |
| `POST /respond` (`/v1/responses`) | Responses payload | `/chat/completions` upstream | Responses-shape body |
| `POST /chat` (`/v1/chat/completions`) | Chat Completions payload | `/responses` upstream | Chat-Completions body |

Forwarding is performed directly by the official Python `openai` SDK (the repo vendors a current clone under `openai-python/`). Authorization headers from incoming requests are reused when present; otherwise the service falls back to environment variables.

### Requirements
- Python 3.10+
- `pip install -r requirements.txt` (installs Flask/Gunicorn plus the vendored OpenAI SDK)

### Running locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python src/server.py
```

Environment variables:

| Name | Description |
| --- | --- |
| `FORWARDER_PORT` | Port to listen on (default `3000`). |
| `OPENAI_API_KEY` | Default key for both directions (used only when a direction-specific key and incoming header are missing). |
| `OPENAI_BASE_URL` | Default upstream base URL. |
| `RESPONSES_BASE_URL`, `CHAT_COMPLETIONS_BASE_URL` | Optional overrides per direction (should include `/openai/v1`). |
| `RESPONSES_API_KEY`, `CHAT_COMPLETIONS_API_KEY` | Optional per-direction keys. |
| `LOG_LEVEL` | Python logging level (`INFO`, `DEBUG`, ...). |
| `MODEL_REASONING_EFFORT` | Optional default reasoning effort (`high`, `medium`, etc.) applied when chat requests do not specify one. |
| `GUNICORN_TIMEOUT` | Docker-only: gunicorn worker timeout in seconds (default `120`). |

### Docker
```bash
docker build -t openai-forwarder .
docker run -d --rm --name forwarder \
  -p 3000:3000 \
  -e RESPONSES_BASE_URL=https://project-a.example.com/openai/v1 \
  -e RESPONSES_API_KEY=123\
  -e CHAT_COMPLETIONS_BASE_URL=https://project-b.example.com/openai/v1 \
  -e CHAT_COMPLETIONS_API_KEY=123 \
  openai-forwarder
```

### Notes
- Incoming `Authorization: Bearer ...` headers are forwarded upstream; if absent, the service falls back to the configured keys above. Make sure at least one key is configured to avoid startup errors.
- `/config` and `/health` endpoints remain available for programmatic inspection. The former lets you tweak upstream URLs/keys at runtime.
- `/v1/chat/completions` now defaults to streaming responses (`stream: true`); disable streaming by explicitly passing `"stream": false` in the request body.
- The upstream Python SDK streams `/responses` requests under the hood and the forwarder still buffers the final result before responding downstream. See `DOCS/forwarder-enhancement-plan.md` for plans around exposing downstream streaming, reasoning blocks, tool calls, and file/image payloads.
