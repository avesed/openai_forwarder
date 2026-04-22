import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, Response, jsonify, request, stream_with_context, g, has_request_context

# Ensure the vendored openai-python clone remains importable when not installed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENAI_SRC = PROJECT_ROOT / "openai-python" / "src"

try:
    from openai import OpenAI
    from openai.chat_via_responses import patch_client
except ImportError:
    if OPENAI_SRC.exists():
        sys.path.insert(0, str(OPENAI_SRC))
        from openai import OpenAI
        from openai.chat_via_responses import patch_client
    else:  # pragma: no cover - helps during misconfiguration
        raise RuntimeError(
            "Unable to import the openai SDK. Make sure openai-python/ exists with `src/`."
        )

PORT = int(os.getenv("FORWARDER_PORT") or os.getenv("PORT") or "3000")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_STREAM_DELTAS = os.getenv("LOG_STREAM_DELTAS", "false").lower() not in {"false", "0", ""}
REQUEST_ID_HEADER = os.getenv("REQUEST_ID_HEADER", "X-Request-ID")

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
ALLOWED_INCLUDE_VALUES = {
    "file_search_call.results",
    "web_search_call.results",
    "web_search_call.action.sources",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
    "code_interpreter_call.outputs",
    "reasoning.encrypted_content",
    "message.output_text.logprobs",
}

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


@app.before_request
def assign_request_id():
    ensure_request_id()


@app.after_request
def attach_request_id_header(response):
    req_id = get_request_id()
    if req_id:
        response.headers.setdefault(REQUEST_ID_HEADER, req_id)
    return response


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


# ═══════════════════════════════════════════════════════════════════════════
# /v1/responses → chat completions upstream  (reverse path, kept as-is)
# ═══════════════════════════════════════════════════════════════════════════

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

    logger.debug(with_request_context("Incoming /v1/responses Authorization: %s"), mask_token(incoming_key))
    logger.debug(with_request_context("Incoming /v1/responses body: %s"), truncate_dict(body))

    logger.info(
        with_request_context("Forwarding Responses payload to chat upstream %s%s"),
        chat_upstream["base_url"],
        "/chat/completions",
    )
    logger.debug(with_request_context("Payload: %s"), truncate_dict(payload))

    try:
        chat_response = call_openai(chat_upstream, "chat", payload, api_key_override=incoming_key)
    except UpstreamError as exc:
        return error_response(str(exc), 502)

    return jsonify(translate_chat_to_respond(chat_response))


# ═══════════════════════════════════════════════════════════════════════════
# /v1/chat/completions → responses upstream  (main path, via SDK adapter)
# ═══════════════════════════════════════════════════════════════════════════

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
    incoming_key = extract_incoming_api_key(request)

    logger.debug(with_request_context("Incoming /v1/chat/completions Authorization: %s"), mask_token(incoming_key))
    logger.debug(with_request_context("Incoming /v1/chat/completions body: %s"), truncate_dict(body))

    logger.info(
        with_request_context("Forwarding Chat payload to responses upstream %s%s"),
        responses_upstream["base_url"],
        "/responses",
    )

    # Build a patched SDK client that routes chat → responses
    try:
        client = build_responses_client(responses_upstream, incoming_key)
    except UpstreamError as exc:
        return error_response(str(exc), 502)

    # Extract parameters from the incoming body for SDK call
    messages = body.get("messages", [])
    model = body.get("model") or "gpt-4.1-mini"

    # Collect all extra kwargs to pass through
    sdk_kwargs = _extract_sdk_kwargs(body)

    if stream_requested:
        return _stream_via_sdk(client, model, messages, sdk_kwargs)

    try:
        result = client.chat.completions.create(
            model=model, messages=messages, stream=False, **sdk_kwargs
        )
    except Exception as exc:
        logger.error(with_request_context("Upstream responses request failed: %s"), exc)
        return error_response(str(exc), 502)

    return jsonify(result.model_dump())


def _stream_via_sdk(client, model, messages, sdk_kwargs):
    """Stream chat completions via the SDK adapter, yielding SSE events."""

    def generator():
        try:
            chunk_iter = client.chat.completions.create(
                model=model, messages=messages, stream=True, **sdk_kwargs
            )
            for chunk in chunk_iter:
                yield format_sse_data(chunk.model_dump())
            yield SSE_DONE
        except Exception as exc:
            logger.error(with_request_context("Streaming responses upstream failed: %s"), exc)
            yield format_sse_data({"error": {"message": str(exc)}})
            yield SSE_DONE

    return build_sse_response(generator)


def _extract_sdk_kwargs(body: Dict[str, Any]) -> Dict[str, Any]:
    """Pull all recognized parameters from the incoming request body."""
    kwargs: Dict[str, Any] = {}

    # Direct scalar params
    for key in (
        "temperature", "top_p", "top_logprobs", "max_tokens", "max_completion_tokens",
        "parallel_tool_calls", "metadata", "store", "service_tier",
        "prompt_cache_key", "prompt_cache_retention", "safety_identifier", "user",
        "frequency_penalty", "presence_penalty", "logit_bias", "logprobs",
        "n", "seed", "stop", "audio", "modalities", "prediction", "verbosity",
        "stream_options",
    ):
        if key in body:
            kwargs[key] = body[key]

    # Tools
    if "tools" in body:
        kwargs["tools"] = body["tools"]
    if "functions" in body:
        kwargs["functions"] = body["functions"]
    if "tool_choice" in body:
        kwargs["tool_choice"] = body["tool_choice"]
    if "function_call" in body:
        kwargs["function_call"] = body["function_call"]

    # Reasoning
    for key in ("reasoning", "reasoning_effort", "model_reasoning_effort"):
        if key in body:
            kwargs[key] = body[key]

    # response_format
    if "response_format" in body:
        kwargs["response_format"] = body["response_format"]

    # web_search_options
    if "web_search_options" in body:
        kwargs["web_search_options"] = body["web_search_options"]

    # Responses-only passthrough (include, previous_response_id, etc.)
    for key in ("include", "max_tool_calls", "previous_response_id", "conversation",
                "instructions", "background"):
        if key in body:
            kwargs[key] = body[key]

    # Apply default reasoning effort from env if not specified
    if DEFAULT_REASONING_EFFORT and "reasoning" not in kwargs and "reasoning_effort" not in kwargs:
        kwargs["reasoning_effort"] = DEFAULT_REASONING_EFFORT

    return kwargs


def build_responses_client(upstream: Dict[str, Any], api_key_override: str | None = None) -> OpenAI:
    """Create an OpenAI client patched with ChatViaResponses adapter."""
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

    client = OpenAI(**client_kwargs)
    patch_client(client)
    return client


# ═══════════════════════════════════════════════════════════════════════════
# Reverse path helpers: Responses → Chat Completions
# (only used by /v1/responses endpoint)
# ═══════════════════════════════════════════════════════════════════════════

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
        logger.error(with_request_context("Upstream %s request failed: %s"), upstream.get("name"), exc)
        raise UpstreamError(str(exc)) from exc

    data = response.model_dump()
    log_upstream_response(upstream.get("name"), data)
    return data


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


def build_chat_completion_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Responses-format body to Chat Completions payload (reverse path)."""
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


def translate_chat_to_respond(chat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Chat Completions response to Responses format (reverse path)."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

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


def clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def add_optional_fields(target: Dict[str, Any], source: Dict[str, Any], keys: List[str]) -> None:
    for key in keys:
        if key in source:
            target[key] = source[key]


def log_request_details(endpoint: str, body: Dict[str, Any]) -> None:
    query_params = request.args.to_dict(flat=False)
    try:
        raw_body = request.get_data(as_text=True)
    except Exception:  # pragma: no cover - defensive
        raw_body = str(body)
    logger.debug(with_request_context("Incoming %s query params: %s"), endpoint, query_params)
    logger.debug(with_request_context("Incoming %s raw body: %s"), endpoint, raw_body)


def log_upstream_response(source: str | None, payload: Dict[str, Any]) -> None:
    name = source or "upstream"
    logger.debug(with_request_context("Upstream %s response: %s"), name, truncate_dict(payload))


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


def ensure_request_id() -> str | None:
    if not has_request_context():
        return None
    existing = getattr(g, "request_id", None)
    if existing:
        return existing
    incoming = request.headers.get(REQUEST_ID_HEADER)
    req_id = (incoming or "").strip() or str(uuid.uuid4())
    g.request_id = req_id
    return req_id


def get_request_id() -> str | None:
    if not has_request_context():
        return None
    return getattr(g, "request_id", None)


def with_request_context(message: str) -> str:
    req_id = get_request_id()
    if req_id:
        return f"[req_id={req_id}] {message}"
    return message


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
