"""
Adapter that transparently routes ``client.chat.completions.create()`` through
the Responses API (``/responses``), converting parameters and return values so
that callers see standard Chat Completions types.

Usage::

    from openai import OpenAI
    from openai.chat_via_responses import patch_client

    client = OpenAI(api_key=..., base_url=...)
    patch_client(client)

    # Now uses /responses under the hood, returns ChatCompletion as usual
    result = client.chat.completions.create(messages=..., model=...)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List

from .resources.chat.completions.completions import Completions
from .types.chat.chat_completion import ChatCompletion
from .types.chat.chat_completion_chunk import ChatCompletionChunk

logger = logging.getLogger("openai.chat_via_responses")

# ── Unsupported parameter names (Chat-only, no Responses equivalent) ─────
_UNSUPPORTED_PARAMS = frozenset({
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "logprobs",
    "n",
    "seed",
    "stop",
    "audio",
    "modalities",
    "prediction",
})


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def patch_client(client) -> None:
    """Replace *client*.chat.completions with the Responses-backed adapter."""
    client.chat.completions = ChatViaResponsesCompletions(client)


# ═══════════════════════════════════════════════════════════════════════════
# Completions subclass
# ═══════════════════════════════════════════════════════════════════════════

class ChatViaResponsesCompletions(Completions):
    """Drop-in replacement for ``Completions`` that routes every request
    through the Responses API, converting on the fly."""

    def create(self, *, messages, model, stream=None, **kwargs):
        payload = _build_responses_payload(messages, model, **kwargs)
        _warn_unsupported(kwargs)

        if stream:
            return self._stream_via_responses(payload, model, kwargs)

        response = self._client.responses.create(**payload)
        resp_data = response.model_dump()
        chat_dict = _response_to_chat_completion(resp_data, model)
        return ChatCompletion.model_validate(chat_dict)

    # ── streaming ────────────────────────────────────────────────────

    def _stream_via_responses(self, payload, model, kwargs):
        """Return an iterator of ``ChatCompletionChunk`` dicts wrapped in an
        SSE-compatible generator that the caller can iterate like a normal
        ``Stream[ChatCompletionChunk]``."""
        stream_mgr = self._client.responses.stream(**payload)
        return _ChatChunkIterator(stream_mgr, model)


# ═══════════════════════════════════════════════════════════════════════════
# Streaming adapter
# ═══════════════════════════════════════════════════════════════════════════

class _ChatChunkIterator:
    """Wraps a ``ResponseStreamManager`` and yields ``ChatCompletionChunk``
    objects by translating Responses stream events on the fly."""

    def __init__(self, stream_mgr, model: str):
        self._stream_mgr = stream_mgr
        self._model = model
        self._stream = None

    def __enter__(self):
        self._stream = self._stream_mgr.__enter__()
        return self

    def __exit__(self, *exc):
        if self._stream_mgr is not None:
            self._stream_mgr.__exit__(*exc)

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        meta: Dict[str, Any] = {
            "id": "chatcmpl-temp",
            "model": self._model,
            "created": int(time.time()),
        }
        sent_role = False
        reasoning_open = False
        reasoning_accum: List[str] = []
        tool_states: Dict[str, Dict[str, Any]] = {}
        next_tool_index = 0

        with self._stream_mgr as stream:
            for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "response.created":
                    resp = getattr(event, "response", None)
                    if resp:
                        meta["id"] = getattr(resp, "id", None) or meta["id"]
                        meta["model"] = getattr(resp, "model", None) or meta["model"]
                        created = getattr(resp, "created_at", None) or getattr(resp, "created", None)
                        if created is not None:
                            meta["created"] = int(created)

                elif event_type == "response.output_text.delta":
                    delta_text = getattr(event, "delta", "") or ""
                    if not delta_text:
                        continue
                    yield _text_chunk(meta, delta_text, role=not sent_role)
                    sent_role = True

                elif event_type == "response.reasoning_text.delta":
                    delta_text = getattr(event, "delta", "") or ""
                    if not reasoning_open:
                        yield _text_chunk(meta, "[thinking]\n", role=not sent_role)
                        sent_role = True
                        reasoning_open = True
                    if delta_text:
                        reasoning_accum.append(delta_text)
                        yield _text_chunk(meta, delta_text)

                elif event_type == "response.reasoning_text.done":
                    if reasoning_open:
                        reasoning_open = False
                        yield _text_chunk(meta, "\n[/thinking]\n")

                elif event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if not item or getattr(item, "type", "") != "function_call":
                        continue
                    item_id = getattr(item, "id", None) or getattr(item, "call_id", None) or f"tool_{len(tool_states)}"
                    call_id = getattr(item, "call_id", None) or item_id
                    name = getattr(item, "name", None) or ""
                    state = {"index": next_tool_index, "id": call_id, "name": name, "arguments": ""}
                    tool_states[item_id] = state
                    next_tool_index += 1
                    yield _tool_chunk(meta, [{
                        "index": state["index"], "id": state["id"], "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }], role=not sent_role)
                    sent_role = True

                elif event_type == "response.function_call_arguments.delta":
                    item_id = getattr(event, "item_id", None)
                    delta_text = getattr(event, "delta", "") or ""
                    if not item_id or not delta_text:
                        continue
                    state = tool_states.get(item_id)
                    if not state:
                        continue
                    state["arguments"] += delta_text
                    yield _tool_chunk(meta, [{
                        "index": state["index"],
                        "function": {"arguments": delta_text},
                    }])

                elif event_type == "response.function_call_arguments.done":
                    item_id = getattr(event, "item_id", None)
                    if not item_id:
                        continue
                    state = tool_states.get(item_id)
                    if not state:
                        continue
                    final_args = getattr(event, "arguments", None)
                    final_name = getattr(event, "name", None)
                    if isinstance(final_args, str) and final_args:
                        state["arguments"] = final_args
                    if final_name:
                        state["name"] = final_name

                elif event_type == "response.completed":
                    if reasoning_open:
                        reasoning_open = False
                        yield _text_chunk(meta, "\n[/thinking]\n")
                    resp = getattr(event, "response", None)
                    reasoning_text = "".join(reasoning_accum).strip() or None
                    yield _finish_chunk(meta, resp, tool_states, reasoning_text)
                    reasoning_accum.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Payload building  (Chat → Responses)
# ═══════════════════════════════════════════════════════════════════════════

def _build_responses_payload(messages, model: str, **kwargs) -> Dict[str, Any]:
    """Convert chat.completions.create() arguments into a responses.create() payload."""
    msg_list = list(messages)
    input_blocks = _messages_to_input(msg_list)

    # Extract system/developer message as instructions
    instructions = None
    for msg in msg_list:
        m = msg if isinstance(msg, dict) else dict(msg)
        if m.get("role") in ("system", "developer"):
            instructions = m.get("content", "")
            break

    payload: Dict[str, Any] = {"model": model, "input": input_blocks}

    if instructions:
        payload["instructions"] = instructions

    # Direct-mapped scalar params
    for chat_key, resp_key in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_logprobs", "top_logprobs"),
        ("parallel_tool_calls", "parallel_tool_calls"),
        ("metadata", "metadata"),
        ("store", "store"),
        ("service_tier", "service_tier"),
        ("prompt_cache_key", "prompt_cache_key"),
        ("prompt_cache_retention", "prompt_cache_retention"),
        ("safety_identifier", "safety_identifier"),
        ("user", "user"),
    ):
        if chat_key in kwargs and kwargs[chat_key] is not None:
            payload[resp_key] = kwargs[chat_key]

    # Renamed params
    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens

    # Tools
    tools = _convert_tools(kwargs.get("tools"), kwargs.get("functions"))
    if tools:
        payload["tools"] = tools

    # Tool choice
    tool_choice = _extract_tool_choice(kwargs)
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    # Reasoning
    reasoning = _extract_reasoning(kwargs)
    if reasoning:
        payload["reasoning"] = reasoning

    # response_format → text.format
    text_config = _convert_response_format(kwargs.get("response_format"), kwargs.get("verbosity"))
    if text_config:
        payload["text"] = text_config

    # web_search_options → web_search tool
    web_opts = kwargs.get("web_search_options")
    if web_opts:
        payload.setdefault("tools", []).append({"type": "web_search_preview"})

    # stream_options pass-through
    if "stream_options" in kwargs and kwargs["stream_options"] is not None:
        payload["stream_options"] = kwargs["stream_options"]

    # Responses-only params that callers may pass via extra_body
    for key in ("include", "max_tool_calls", "previous_response_id", "conversation",
                "background", "truncation", "prompt"):
        if key in kwargs and kwargs[key] is not None:
            payload[key] = kwargs[key]

    return payload


# ── message conversion ───────────────────────────────────────────────

def _messages_to_input(messages: list) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    tool_response_counter = 0

    for msg in messages:
        if not isinstance(msg, dict):
            msg = dict(msg) if hasattr(msg, "items") else {"role": "user", "content": str(msg)}

        role = (msg.get("role") or "user").lower()

        # Tool result
        if role == "tool":
            tool_response_counter += 1
            call_id = msg.get("tool_call_id") or msg.get("id") or f"tool_call_{tool_response_counter}"
            output = msg.get("content")
            if output is None:
                output = ""
            blocks.append({"type": "function_call_output", "call_id": call_id, "output": str(output)})
            continue

        # System/developer messages are handled as instructions; also keep as input
        content = msg.get("content")
        entries = _build_content_entries(content, role)
        if entries:
            blocks.append({"role": role, "content": entries})

        # Tool calls from assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                if not func.get("name"):
                    continue
                blocks.append({
                    "type": "function_call",
                    "name": func.get("name", ""),
                    "arguments": _stringify_args(func.get("arguments")),
                    "call_id": tc.get("id", "") if isinstance(tc, dict) else "",
                })

        # Legacy function_call on assistant
        fc = msg.get("function_call")
        if fc and isinstance(fc, dict) and fc.get("name"):
            blocks.append({
                "type": "function_call",
                "name": fc["name"],
                "arguments": _stringify_args(fc.get("arguments")),
                "call_id": msg.get("id", ""),
            })

    return blocks


def _build_content_entries(content: Any, role: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    if isinstance(content, str):
        if content:
            ctype = "output_text" if role == "assistant" else "input_text"
            entries.append({"type": ctype, "text": content})
    elif isinstance(content, list):
        for part in content:
            entry = _convert_content_part(part, role)
            if entry:
                entries.append(entry)
    elif isinstance(content, dict):
        entry = _convert_content_part(content, role)
        if entry:
            entries.append(entry)

    return entries


def _convert_content_part(part: Any, role: str) -> Dict[str, Any] | None:
    if isinstance(part, str):
        ctype = "output_text" if role == "assistant" else "input_text"
        return {"type": ctype, "text": part} if part else None

    if not isinstance(part, dict):
        return None

    ptype = (part.get("type") or "").lower()

    if ptype in ("text", "input_text", "output_text"):
        text = part.get("text") or ""
        if not text:
            return None
        ctype = "output_text" if role == "assistant" else "input_text"
        return {"type": ctype, "text": text}

    if ptype in ("image_url", "input_image"):
        if role == "assistant":
            return None
        img = part.get("image_url")
        url = None
        detail = "auto"
        if isinstance(img, dict):
            url = img.get("url")
            detail = img.get("detail") or "auto"
        elif isinstance(img, str):
            url = img
        if not url:
            url = part.get("url")
        if url:
            entry: Dict[str, Any] = {"type": "input_image", "image_url": url, "detail": detail}
            if part.get("file_id"):
                entry["file_id"] = part["file_id"]
            return entry

    return None


# ── tool conversion ──────────────────────────────────────────────────

def _convert_tools(tools: Any, legacy_functions: Any) -> List[Dict[str, Any]] | None:
    converted: List[Dict[str, Any]] = []

    if isinstance(tools, (list, tuple)):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and "function" in tool:
                t = _function_to_tool(tool["function"])
                if t:
                    converted.append(t)
            else:
                converted.append(tool)

    if isinstance(legacy_functions, (list, tuple)):
        for func_def in legacy_functions:
            t = _function_to_tool(func_def)
            if t:
                converted.append(t)

    return converted or None


def _function_to_tool(func: Any) -> Dict[str, Any] | None:
    if not isinstance(func, dict) or not func.get("name"):
        return None
    return {
        "type": "function",
        "name": func["name"],
        "description": func.get("description"),
        "parameters": func.get("parameters", {"type": "object", "properties": {}}),
        "strict": func.get("strict"),
    }


def _extract_tool_choice(kwargs: dict) -> Any:
    if "tool_choice" in kwargs:
        return kwargs["tool_choice"]
    if "function_call" not in kwargs:
        return None
    legacy = kwargs["function_call"]
    if isinstance(legacy, str):
        return legacy.lower() if legacy.lower() in ("none", "auto", "required") else None
    if isinstance(legacy, dict) and legacy.get("name"):
        return {"type": "function", "name": legacy["name"]}
    return None


# ── reasoning ────────────────────────────────────────────────────────

def _extract_reasoning(kwargs: dict) -> Dict[str, Any] | None:
    reasoning = kwargs.get("reasoning")
    if isinstance(reasoning, dict) and reasoning:
        return reasoning
    if isinstance(reasoning, str) and reasoning.strip():
        return {"effort": reasoning.strip()}

    for key in ("reasoning_effort", "model_reasoning_effort"):
        val = kwargs.get(key)
        if isinstance(val, str) and val.strip():
            return {"effort": val.strip()}
        if isinstance(val, dict) and val.get("effort"):
            return val

    return None


# ── response_format → text config ────────────────────────────────────

def _convert_response_format(response_format: Any, verbosity: Any) -> Dict[str, Any] | None:
    text_config: Dict[str, Any] = {}

    if isinstance(response_format, dict) and response_format:
        fmt_type = response_format.get("type", "")
        if fmt_type in ("json_object", "json_schema", "text"):
            text_config["format"] = response_format

    if isinstance(verbosity, str) and verbosity.strip():
        text_config["verbosity"] = verbosity.strip()

    return text_config or None


# ── helpers ──────────────────────────────────────────────────────────

def _stringify_args(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, separators=(",", ":"))
    except Exception:
        return str(value)


def _warn_unsupported(kwargs: dict) -> None:
    for key in _UNSUPPORTED_PARAMS:
        if key in kwargs and kwargs[key] is not None:
            logger.warning("Parameter '%s' is not supported by Responses API and will be ignored.", key)


# ═══════════════════════════════════════════════════════════════════════════
# Response conversion  (Responses → ChatCompletion)
# ═══════════════════════════════════════════════════════════════════════════

def _response_to_chat_completion(resp: Dict[str, Any], model: str) -> Dict[str, Any]:
    output = resp.get("output") or []
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    reasoning_parts: List[str] = []

    for block in output:
        if not isinstance(block, dict):
            continue
        btype = (block.get("type") or "").lower()

        if btype == "message":
            for entry in block.get("content") or []:
                etype = (entry.get("type") or "").lower() if isinstance(entry, dict) else ""
                if etype in ("output_text", "text"):
                    text_parts.append(entry.get("text", ""))
                elif "reasoning" in etype:
                    reasoning_parts.append(entry.get("text", ""))

        elif btype == "reasoning":
            text = _extract_text(block.get("content"))
            if text:
                reasoning_parts.append(text)

        elif btype == "function_call":
            name = block.get("name")
            if name:
                tool_calls.append({
                    "id": block.get("call_id") or block.get("id") or f"tool_call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _stringify_args(block.get("arguments")),
                    },
                })

    content = "\n".join(text_parts) if text_parts else None
    reasoning_text = "\n".join(p for p in reasoning_parts if p) or None

    if reasoning_text and content:
        content = f"[thinking]\n{reasoning_text}\n[/thinking]\n{content}"

    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        if len(tool_calls) == 1:
            message["function_call"] = tool_calls[0]["function"]

    finish_reason = "tool_calls" if tool_calls else _map_stop_reason(resp.get("stop_reason"))

    return {
        "id": resp.get("id", "chatcmpl-converted"),
        "object": "chat.completion",
        "created": resp.get("created_at") or resp.get("created") or int(time.time()),
        "model": resp.get("model") or model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": _convert_usage(resp.get("usage")),
    }


def _map_stop_reason(value: str | None) -> str:
    return {
        None: "stop", "stop": "stop", "end_turn": "stop",
        "max_tokens": "length", "length": "length",
        "tool_use": "tool_calls", "tool_calls": "tool_calls",
    }.get(value, "stop")


def _convert_usage(usage: Any) -> Dict[str, Any] | None:
    if not usage:
        return None
    u = usage if isinstance(usage, dict) else (usage.model_dump() if hasattr(usage, "model_dump") else {})
    return {
        "prompt_tokens": u.get("input_tokens", 0),
        "completion_tokens": u.get("output_tokens", 0),
        "total_tokens": u.get("total_tokens", 0),
    }


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (_extract_text(e) for e in content)))
    if isinstance(content, dict):
        if "text" in content:
            return content["text"]
        if "content" in content:
            return _extract_text(content["content"])
    return ""


# ── streaming chunk builders ─────────────────────────────────────────

def _text_chunk(meta: dict, text: str, role: bool = False) -> ChatCompletionChunk:
    delta: Dict[str, Any] = {"content": text}
    if role:
        delta["role"] = "assistant"
    return ChatCompletionChunk.model_validate({
        "id": meta.get("id", "chatcmpl-temp"),
        "object": "chat.completion.chunk",
        "created": meta.get("created", int(time.time())),
        "model": meta.get("model", "unknown"),
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    })


def _tool_chunk(meta: dict, tool_deltas: list, role: bool = False) -> ChatCompletionChunk:
    delta: Dict[str, Any] = {"tool_calls": tool_deltas}
    if role:
        delta["role"] = "assistant"
    return ChatCompletionChunk.model_validate({
        "id": meta.get("id", "chatcmpl-temp"),
        "object": "chat.completion.chunk",
        "created": meta.get("created", int(time.time())),
        "model": meta.get("model", "unknown"),
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    })


def _finish_chunk(meta: dict, response_info: Any, tool_states: dict, reasoning_text: str | None) -> ChatCompletionChunk:
    finish_reason = _map_stop_reason(getattr(response_info, "stop_reason", None) if response_info else None)
    if tool_states:
        finish_reason = "tool_calls"

    chunk_dict: Dict[str, Any] = {
        "id": meta.get("id", "chatcmpl-temp"),
        "object": "chat.completion.chunk",
        "created": meta.get("created", int(time.time())),
        "model": meta.get("model", "unknown"),
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }

    usage = getattr(response_info, "usage", None) if response_info else None
    if usage:
        chunk_dict["usage"] = _convert_usage(usage)

    return ChatCompletionChunk.model_validate(chunk_dict)
