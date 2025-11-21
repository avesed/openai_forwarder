import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, Response, jsonify, request, stream_with_context

# Ensure the vendored openai-python clone remains importable when not installed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENAI_SRC = PROJECT_ROOT / "openai-python" / "src"

try:
    from openai import OpenAI
except ImportError:
    if OPENAI_SRC.exists():
        sys.path.insert(0, str(OPENAI_SRC))
        from openai import OpenAI
    else:  # pragma: no cover - helps during misconfiguration
        raise RuntimeError(
            "Unable to import the openai SDK. Make sure openai-python/ exists with `src/`."
        )

PORT = int(os.getenv("FORWARDER_PORT") or os.getenv("PORT") or "3000")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("forwarder")

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_ORG_ID = os.getenv("OPENAI_ORG_ID")
DEFAULT_BASE_URL = (
    os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
).rstrip("/")
DEFAULT_REASONING_EFFORT = os.getenv("MODEL_REASONING_EFFORT")

SSE_DONE = "data: [DONE]\n\n"


def normalize_sdk_base_url(url: str) -> str:
    cleaned = (url or "").rstrip("/")
    if not cleaned:
        raise ValueError("Base URL cannot be empty.")

    for suffix in ("/responses", "/chat/completions"):
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip("/")

    if cleaned.lower().endswith("/openai/v1"):
        return cleaned

    if cleaned.lower().endswith("/v1"):
        return cleaned

    return f"{cleaned}/openai/v1"


def create_upstream(name: str, base_env: str, key_env: str, org_env: str) -> Dict[str, Any]:
    env_base = os.getenv(base_env) or DEFAULT_BASE_URL
    try:
        base_url = normalize_sdk_base_url(env_base)
    except ValueError:
        base_url = normalize_sdk_base_url(env_base.rstrip("/"))
    return {
        "name": name,
        "base_url": base_url,
        "api_key": os.getenv(key_env) or DEFAULT_API_KEY,
        "api_key_env": key_env,
        "org_id": os.getenv(org_env) or DEFAULT_ORG_ID,
    }


chat_upstream = create_upstream(
    name="chat completions upstream",
    base_env="CHAT_COMPLETIONS_BASE_URL",
    key_env="CHAT_COMPLETIONS_API_KEY",
    org_env="CHAT_COMPLETIONS_ORG_ID",
)

responses_upstream = create_upstream(
    name="responses upstream",
    base_env="RESPONSES_BASE_URL",
    key_env="RESPONSES_API_KEY",
    org_env="RESPONSES_ORG_ID",
)

if not any(filter(None, [DEFAULT_API_KEY, chat_upstream["api_key"], responses_upstream["api_key"]])):
    raise SystemExit(
        "Missing API key configuration. Set OPENAI_API_KEY or the direction-specific *_API_KEY variables."
    )

app = Flask(__name__)


class ForwarderError(Exception):
    """Raised for known bad requests to surface user-friendly errors."""


class UpstreamError(Exception):
    """Raised when OpenAI responds with an error."""


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/config")
def config():
    return jsonify(
        {
            "port": PORT,
            "chat": describe_upstream(chat_upstream),
            "responses": describe_upstream(responses_upstream),
        }
    )


@app.post("/config")
def update_config():
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:  # pragma: no cover
        return error_response("Invalid JSON body.", 400)

    updated = {}
    for key, upstream in (("responses", responses_upstream), ("chat", chat_upstream)):
        if key not in payload:
            continue

        body = payload[key]
        try:
            apply_upstream_updates(upstream, body)
        except ValueError as exc:
            return error_response(str(exc), 400)

        updated[key] = describe_upstream(upstream)

    if not updated:
        return error_response("No config updates provided.", 400)

    return jsonify(
        {
            "port": PORT,
            "chat": describe_upstream(chat_upstream),
            "responses": describe_upstream(responses_upstream),
            "updated": updated,
        }
    )


@app.get("/")
@app.get("/index.html")
def index():
    return jsonify(
        {
            "message": "Python SDK forwarder is running.",
            "routes": ["/respond", "/chat", "/health", "/config"],
        }
    )


@app.post("/respond")
@app.post("/v1/responses")
def respond_proxy():
    try:
        body = get_json_body()
    except ForwarderError as exc:
        return error_response(str(exc), 400)

    if body.get("stream"):
        return error_response("Streaming is not supported by this forwarder.", 400)

    log_request_details("/v1/responses", body)
    payload = build_chat_completion_payload(body)
    incoming_key = extract_incoming_api_key(request)

    logger.debug("Incoming /v1/responses Authorization: %s", mask_token(incoming_key))
    logger.debug("Incoming /v1/responses body: %s", truncate_dict(body))

    logger.info(
        "Forwarding Responses payload to chat upstream %s%s",
        chat_upstream["base_url"],
        "/chat/completions",
    )
    logger.debug("Payload: %s", truncate_dict(payload))

    try:
        chat_response = call_openai(chat_upstream, "chat", payload, api_key_override=incoming_key)
    except UpstreamError as exc:
        return error_response(str(exc), 502)

    return jsonify(translate_chat_to_respond(chat_response))


@app.post("/chat")
@app.post("/v1/chat/completions")
def chat_proxy():
    try:
        body = get_json_body()
    except ForwarderError as exc:
        return error_response(str(exc), 400)

    raw_stream_flag = body.get("stream")
    stream_requested = True if raw_stream_flag is None else bool(raw_stream_flag)
    body.setdefault("stream", stream_requested)

    log_request_details("/v1/chat/completions", body)
    payload = build_respond_payload(body)
    incoming_key = extract_incoming_api_key(request)

    logger.debug("Incoming /v1/chat/completions Authorization: %s", mask_token(incoming_key))
    logger.debug("Incoming /v1/chat/completions body: %s", truncate_dict(body))

    logger.info(
        "Forwarding Chat payload to responses upstream %s%s",
        responses_upstream["base_url"],
        "/responses",
    )
    logger.debug("Payload: %s", truncate_dict(payload))

    if stream_requested:
        return stream_chat_completion(payload, incoming_key)

    try:
        respond_response = call_openai(
            responses_upstream, "responses", payload, api_key_override=incoming_key
        )
    except UpstreamError as exc:
        return error_response(str(exc), 502)

    return jsonify(translate_respond_to_chat(respond_response))


def call_openai(
    upstream: Dict[str, Any], target: str, payload: Dict[str, Any], api_key_override: str | None = None
) -> Dict[str, Any]:
    client = build_openai_client(upstream, api_key_override)

    try:
        if target == "responses":
            payload_with_stream = dict(payload)
            payload_with_stream.pop("stream", None)
            with client.responses.stream(**payload_with_stream) as stream:
                stream.until_done()
                response = stream.get_final_response()
        else:
            response = client.chat.completions.create(**payload)
    except Exception as exc:
        logger.error("Upstream %s request failed: %s", upstream.get("name"), exc)
        raise UpstreamError(str(exc)) from exc

    data = response.model_dump()
    log_upstream_response(upstream.get("name"), data)
    return data


def stream_chat_completion(payload: Dict[str, Any], incoming_key: str | None):
    try:
        responses_stream = create_responses_stream(responses_upstream, payload, incoming_key)
    except UpstreamError as exc:
        return error_response(str(exc), 502)

    def generator():
        meta = {
            "id": None,
            "model": payload.get("model"),
            "created": int(time.time()),
        }
        sent_role = False
        reasoning_open = False
        reasoning_accum: List[str] = []
        try:
            with responses_stream as upstream_stream:
                for event in upstream_stream:
                    log_upstream_event(responses_upstream.get("name"), event)
                    event_type = getattr(event, "type", "")
                    if event_type == "response.created":
                        response_info = getattr(event, "response", None)
                        if response_info:
                            meta["id"] = getattr(response_info, "id", None) or meta["id"]
                            meta["model"] = getattr(response_info, "model", None) or meta["model"]
                            created = getattr(response_info, "created_at", None) or getattr(
                                response_info, "created", None
                            )
                            if created is not None:
                                meta["created"] = int(created)
                    elif event_type == "response.output_text.delta":
                        delta_text = getattr(event, "delta", "") or ""
                        if not delta_text:
                            continue
                        chunk = build_chat_stream_chunk(meta, delta_text, include_role=not sent_role)
                        sent_role = True
                        yield format_sse_data(chunk)
                    elif event_type == "response.reasoning_text.delta":
                        delta_text = getattr(event, "delta", "") or ""
                        if not delta_text and reasoning_open:
                            continue
                        if not reasoning_open:
                            prefix = build_chat_stream_chunk(
                                meta, "[thinking]\n", include_role=not sent_role
                            )
                            sent_role = True
                            reasoning_open = True
                            yield format_sse_data(prefix)
                        if delta_text:
                            reasoning_accum.append(delta_text)
                            chunk = build_chat_stream_chunk(meta, delta_text, include_role=False)
                            yield format_sse_data(chunk)
                    elif event_type == "response.reasoning_text.done":
                        if reasoning_open:
                            reasoning_open = False
                            suffix = build_chat_stream_chunk(
                                meta, "\n[/thinking]\n", include_role=False
                            )
                            yield format_sse_data(suffix)
                    elif event_type == "response.completed":
                        if reasoning_open:
                            reasoning_open = False
                            suffix = build_chat_stream_chunk(
                                meta, "\n[/thinking]\n", include_role=False
                            )
                            yield format_sse_data(suffix)
                        response_info = getattr(event, "response", None)
                        reasoning_text = "".join(reasoning_accum).strip() or None
                        chunk = build_chat_finish_chunk(meta, response_info, reasoning_text)
                        reasoning_accum.clear()
                        yield format_sse_data(chunk)
                yield SSE_DONE
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Streaming responses upstream failed: %s", exc)
            yield format_sse_data({"error": {"message": str(exc)}})
            yield SSE_DONE

    return build_sse_response(generator)


def build_openai_client(upstream: Dict[str, Any], api_key_override: str | None = None) -> OpenAI:
    api_key = api_key_override or upstream.get("api_key")
    if not api_key:
        hint = upstream.get("api_key_env")
        raise UpstreamError(
            f"Missing API key for {upstream.get('name')} ({hint} or OPENAI_API_KEY)."
        )

    client_kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": upstream["base_url"],
    }
    if upstream.get("org_id"):
        client_kwargs["organization"] = upstream["org_id"]

    return OpenAI(**client_kwargs)


def create_responses_stream(
    upstream: Dict[str, Any], payload: Dict[str, Any], api_key_override: str | None
):
    client = build_openai_client(upstream, api_key_override)
    payload_to_send = dict(payload)
    payload_to_send.pop("stream", None)
    try:
        return client.responses.stream(**payload_to_send)
    except Exception as exc:
        logger.error("Failed to open responses stream via %s: %s", upstream.get("name"), exc)
        raise UpstreamError(str(exc)) from exc


def build_chat_stream_chunk(meta: Dict[str, Any], delta_text: str, include_role: bool = False) -> Dict[str, Any]:
    chunk = {
        "id": meta.get("id") or "chatcmpl-temp",
        "object": "chat.completion.chunk",
        "created": meta.get("created") or int(time.time()),
        "model": meta.get("model") or "unknown",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": delta_text,
                },
                "finish_reason": None,
            }
        ],
    }
    if include_role:
        chunk["choices"][0]["delta"]["role"] = "assistant"
    return chunk


def build_chat_finish_chunk(
    meta: Dict[str, Any], response_info: Any | None, reasoning_text: str | None = None
) -> Dict[str, Any]:
    finish_reason = map_stop_reason(getattr(response_info, "stop_reason", None))
    chunk = {
        "id": meta.get("id") or "chatcmpl-temp",
        "object": "chat.completion.chunk",
        "created": meta.get("created") or int(time.time()),
        "model": meta.get("model") or "unknown",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    usage = getattr(response_info, "usage", None)
    if usage:
        chunk["usage"] = usage.model_dump()
    if reasoning_text:
        chunk["choices"][0]["metadata"] = {"reasoning": reasoning_text}
    return chunk


def map_stop_reason(value: str | None) -> str:
    mapping = {
        None: "stop",
        "stop": "stop",
        "end_turn": "stop",
        "max_tokens": "length",
        "length": "length",
    }
    return mapping.get(value, "stop")


def format_sse_data(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def build_sse_response(generator_fn):
    response = Response(stream_with_context(generator_fn()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


def get_json_body() -> Dict[str, Any]:
    if not request.data:
        return {}

    try:
        data = request.get_json(force=True, silent=False)
    except Exception as exc:  # pragma: no cover
        raise ForwarderError("Invalid JSON body.") from exc

    return data or {}


def error_response(message: str, status: int):
    return jsonify({"error": message}), status


def build_chat_completion_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    messages = convert_respond_input_to_chat_messages(body.get("input", []))

    payload = clean_dict(
        {
            "model": body.get("model") or "gpt-4o-mini",
            "messages": messages,
            "temperature": body.get("temperature"),
            "max_tokens": body.get("max_tokens"),
            "top_p": body.get("top_p"),
            "frequency_penalty": body.get("frequency_penalty"),
            "presence_penalty": body.get("presence_penalty"),
            "stop": body.get("stop"),
        }
    )

    add_optional_fields(
        payload,
        body,
        [
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "n",
            "user",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "parallel_tool_calls",
            "modalities",
            "audio",
            "prediction",
            "stream_options",
            "max_completion_tokens",
        ],
    )

    return payload


def build_respond_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    input_blocks = convert_chat_messages_to_respond_input(body.get("messages", []))
    reasoning = extract_reasoning_config(body)

    payload = clean_dict(
        {
            "model": body.get("model") or "gpt-4.1-mini",
            "input": input_blocks,
            "temperature": body.get("temperature"),
            "max_output_tokens": body.get("max_tokens"),
            "top_p": body.get("top_p"),
            "frequency_penalty": body.get("frequency_penalty"),
            "presence_penalty": body.get("presence_penalty"),
            "stop": body.get("stop"),
            "reasoning": reasoning,
        }
    )

    add_optional_fields(
        payload,
        body,
        [
            "metadata",
            "include",
            "max_tool_calls",
            "parallel_tool_calls",
            "service_tier",
            "store",
            "prompt_cache_key",
            "prompt_cache_retention",
            "conversation",
            "previous_response_id",
            "instructions",
            "safety_identifier",
            "background",
        ],
    )

    return payload


def convert_respond_input_to_chat_messages(input_payload: Any) -> List[Dict[str, Any]]:
    items = input_payload if isinstance(input_payload, list) else [input_payload]
    messages: List[Dict[str, Any]] = []

    for entry in items:
        if entry is None:
            continue

        if isinstance(entry, str):
            messages.append({"role": "user", "content": entry})
            continue

        role = entry.get("role") or "user"
        text = extract_text(entry.get("content") or entry.get("input") or "")
        messages.append({"role": role, "content": text})

    return messages


def convert_chat_messages_to_respond_input(messages_payload: Any) -> List[Dict[str, Any]]:
    items = messages_payload if isinstance(messages_payload, list) else [messages_payload]
    blocks: List[Dict[str, Any]] = []

    for message in items:
        if not message:
            continue

        role = message.get("role") or "user"
        text = extract_text(message.get("content"))
        content_type = "output_text" if role == "assistant" else "input_text"

        blocks.append(
            {
                "role": role,
                "content": [
                    {
                        "type": content_type,
                        "text": text,
                    }
                ],
            }
        )

    return blocks


def extract_reasoning_config(body: Dict[str, Any]) -> Dict[str, Any] | None:
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning:
        return reasoning
    if isinstance(reasoning, str) and reasoning.strip():
        return {"effort": reasoning.strip()}

    for key in ("reasoning_effort", "model_reasoning_effort"):
        reasoning_effort = body.get(key)
        if isinstance(reasoning_effort, str) and reasoning_effort.strip():
            return {"effort": reasoning_effort.strip()}
        if isinstance(reasoning_effort, dict) and reasoning_effort.get("effort"):
            return reasoning_effort

    if DEFAULT_REASONING_EFFORT:
        return {"effort": DEFAULT_REASONING_EFFORT}

    return None


def translate_chat_to_respond(chat: Dict[str, Any]) -> Dict[str, Any]:
    choices = chat.get("choices") or []
    choice = choices[0] if choices else {}
    message = (choice or {}).get("message") or {}

    message_block = {
        "id": choice.get("id") or f"{chat.get('id', 'chat')}-message-0",
        "type": "message",
        "role": message.get("role") or "assistant",
        "content": [
            {
                "type": "output_text",
                "text": extract_text(message.get("content")),
            }
        ],
    }

    return {
        "id": chat.get("id"),
        "object": "response",
        "created": chat.get("created"),
        "model": chat.get("model"),
        "usage": chat.get("usage"),
        "output": [message_block],
        "stop_reason": choice.get("finish_reason"),
    }


def translate_respond_to_chat(respond: Dict[str, Any]) -> Dict[str, Any]:
    output = respond.get("output") or []
    message_block = next(
        (block for block in output if block and block.get("type") == "message"), {}
    )
    content = message_block.get("content") or []
    reasoning_text = collect_reasoning_text(output)
    final_content = extract_text(content)
    metadata = None
    if reasoning_text:
        final_content = f"[thinking]\n{reasoning_text}\n[/thinking]\n{final_content}"
        metadata = {"reasoning": reasoning_text}

    message_payload = {
        "role": message_block.get("role") or "assistant",
        "content": final_content,
    }
    if metadata:
        message_payload["metadata"] = metadata

    return {
        "id": respond.get("id"),
        "object": "chat.completion",
        "created": respond.get("created"),
        "model": respond.get("model"),
        "usage": respond.get("usage"),
        "choices": [
            {
                "index": 0,
                "finish_reason": respond.get("stop_reason") or "stop",
                "message": message_payload,
            }
        ],
    }


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if not content:
        return ""

    if isinstance(content, list):
        parts = [extract_text(entry) for entry in content]
        return "\n".join(filter(None, parts))

    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "content" in content:
            return extract_text(content["content"])

    return ""


def collect_reasoning_text(output: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for block in output or []:
        if not block:
            continue
        block_type = (block.get("type") or "").lower()
        if block_type == "reasoning":
            parts.append(extract_text(block.get("content")))
            continue
        if block_type != "message":
            continue
        for entry in block.get("content") or []:
            entry_type = (entry.get("type") or "").lower() if isinstance(entry, dict) else ""
            if "reasoning" in entry_type:
                parts.append(extract_text(entry.get("content") or entry))
    return "\n".join([part for part in parts if part])


def log_request_details(endpoint: str, body: Dict[str, Any]) -> None:
    query_params = request.args.to_dict(flat=False)
    try:
        raw_body = request.get_data(as_text=True)
    except Exception:  # pragma: no cover - defensive
        raw_body = str(body)
    logger.debug("Incoming %s query params: %s", endpoint, query_params)
    logger.debug("Incoming %s raw body: %s", endpoint, raw_body)


def log_upstream_response(source: str | None, payload: Dict[str, Any]) -> None:
    name = source or "upstream"
    logger.debug("Upstream %s response: %s", name, truncate_dict(payload))


def log_upstream_event(source: str | None, event: Any) -> None:
    name = source or "upstream"
    try:
        event_payload = event.model_dump()
    except AttributeError:
        try:
            event_payload = vars(event)
        except TypeError:
            event_payload = str(event)
    logger.debug("Upstream %s event: %s", name, truncate_dict(event_payload))


def clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def add_optional_fields(target: Dict[str, Any], source: Dict[str, Any], keys: List[str]) -> None:
    for key in keys:
        if key in source:
            target[key] = source[key]


def describe_upstream(upstream: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": upstream.get("name"),
        "base_url": upstream.get("base_url"),
        "has_api_key": bool(upstream.get("api_key")),
        "org_id_present": bool(upstream.get("org_id")),
        "api_key_env": upstream.get("api_key_env"),
    }


def apply_upstream_updates(upstream: Dict[str, Any], data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"Config for {upstream.get('name')} must be an object.")

    if "base_url" in data:
        base = (data.get("base_url") or "").strip()
        if not base:
            raise ValueError(f"Base URL for {upstream.get('name')} cannot be empty.")
        upstream["base_url"] = normalize_sdk_base_url(base)

    if "api_key" in data:
        key_value = (data.get("api_key") or "").strip()
        upstream["api_key"] = key_value or None

    if "org_id" in data:
        org_value = (data.get("org_id") or "").strip()
        upstream["org_id"] = org_value or None

    logger.info("Updated upstream %s config: %s", upstream.get("name"), truncate_dict(upstream))


def extract_incoming_api_key(req) -> str | None:
    auth = req.headers.get("Authorization")
    if not auth:
        return None
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def mask_token(token: str | None) -> str:
    if not token:
        return "None"
    if len(token) <= 8:
        return token
    return token[:4] + "***" + token[-4:]


def truncate_dict(obj: Any, limit: int = 500) -> str:
    text = str(obj)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
