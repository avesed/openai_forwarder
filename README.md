## OpenAI Responses → Chat Forwarder

Need to talk to a Responses-only deployment but your app only speaks the Chat Completions API? Drop this forwarder in between. It takes your chat payloads, sends them upstream via the official `openai` Python SDK, and folds the Responses output (reasoning, tool calls, etc.) back into a regular Chat reply. 

### Endpoints

| Endpoint | Accepts | Sent upstream | Returned as |
| --- | --- | --- | --- |
| `POST /chat` (`/v1/chat/completions`) | Chat Completions payload | `/responses` | Chat Completions body (primary path) |

### Docker quick start

```bash
docker run -d --name forwarder \
  -p 3000:3000 \
  -e RESPONSES_BASE_URL=https://responses.example.com/openai/v1 \
  -e RESPONSES_API_KEY=sk-... \
  ghcr.io/avesed/openai-forwarder:latest
```

### Environment knobs

| Name | Description |
| --- | --- |
| `FORWARDER_PORT` | Default `3000`. | 
| `RESPONSES_BASE_URL` | Set your responses server url. |
| `RESPONSES_API_KEY` | Optional, overrides keys. |
| `LOG_LEVEL` | `INFO`, `DEBUG`, etc. |
| `GUNICORN_TIMEOUT` | Docker entrypoint: worker timeout (sec, default `120`). |
