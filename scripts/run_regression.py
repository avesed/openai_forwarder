#!/usr/bin/env python3
"""Quick regression checks for the Responses → Chat forwarder.

This script doesn't hit real network endpoints. It exercises the core
translation helpers to make sure we don't regress the Chat→Responses or
Responses→Chat transformations (reasoning blocks, tool calls, etc.).
It exits with code 0 on success and raises AssertionError otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure modules can be imported without real credentials
os.environ.setdefault("OPENAI_API_KEY", "test-key")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OPENAI_SRC = REPO_ROOT / "openai-python" / "src"
if str(OPENAI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENAI_SRC))

try:
    # SDK adapter (chat → responses direction)
    from openai.chat_via_responses import (
        _build_responses_payload,
        _response_to_chat_completion,
        _messages_to_input,
    )
    # Server helpers (responses → chat direction)
    from src.server import translate_chat_to_respond, build_chat_completion_payload
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    missing = str(exc)
    raise SystemExit(
        "Unable to import forwarder code. Did you install requirements? "
        "Run `pip install -r requirements.txt` and retry.\n"
        f"Original error: {missing}"
    ) from exc


def assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_chat_to_responses_case() -> None:
    """Test: Chat Completions body → Responses API payload (via SDK adapter)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "天气如何？"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/weather.png", "detail": "low"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "我去查一下天气。",
            "tool_calls": [
                {
                    "id": "call-weather-1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"shanghai"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-weather-1",
            "content": "多云转晴",
        },
    ]

    payload = _build_responses_payload(
        messages=messages,
        model="gpt-test",
        reasoning="medium",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather.",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    "strict": True,
                },
            }
        ],
    )

    assert_equal(payload.get("reasoning"), {"effort": "medium"}, "Reasoning config")

    first_block = payload["input"][0]
    user_content = first_block.get("content") or []
    assert_true(len(user_content) == 2, "User content should contain text and image entries")
    assert_equal(user_content[0]["type"], "input_text", "First user content should be text")
    assert_equal(user_content[1]["type"], "input_image", "Second user content should be image")
    assert_equal(user_content[1]["image_url"], "https://example.com/weather.png", "Image URL should be preserved")

    # Ensure tool call appeared as Responses input items
    function_calls: List[Dict[str, Any]] = [
        block for block in payload.get("input", []) if block.get("type") == "function_call"
    ]
    outputs: List[Dict[str, Any]] = [
        block for block in payload.get("input", []) if block.get("type") == "function_call_output"
    ]

    assert_true(bool(function_calls), "Tool call should be converted into function_call input")
    assert_true(bool(outputs), "Tool response should be converted into function_call_output input")

    call = function_calls[0]
    assert_equal(call.get("call_id"), "call-weather-1", "Function call id")
    assert_equal(call.get("name"), "get_weather", "Function call name")

    tool_output = outputs[0]
    assert_equal(tool_output.get("call_id"), "call-weather-1", "Tool output call id")
    assert_equal(tool_output.get("output"), "多云转晴", "Tool output text")


def run_responses_to_chat_case() -> None:
    """Test: Responses API output → Chat Completions format (via SDK adapter)."""
    respond_payload: Dict[str, Any] = {
        "id": "resp_123",
        "created_at": 111,
        "model": "gpt-test",
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "stop_reason": "tool_use",
        "output": [
            {
                "type": "reasoning",
                "content": [
                    {"type": "reasoning_text", "text": "先查看用户已使用的工具..."},
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "天气不错，出门别忘记带伞。"}],
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "call_id": "tool-1",
                "arguments": '{"city":"shanghai"}',
            },
        ],
    }

    chat_result = _response_to_chat_completion(respond_payload, "gpt-test")
    choice = chat_result["choices"][0]
    message = choice["message"]

    assert_true(
        "[thinking]" in message["content"],
        "Reasoning text should be wrapped with [thinking] tags in message content",
    )

    tool_calls = message.get("tool_calls")
    assert_true(bool(tool_calls), "Tool calls should be present in translated chat response")
    assert_equal(tool_calls[0]["function"]["name"], "get_weather", "Tool call name in chat response")
    assert_equal(choice["finish_reason"], "tool_calls", "Finish reason should reflect tool usage")


def run_reverse_path_case() -> None:
    """Test: Responses body → Chat Completions payload (reverse path in server.py)."""
    chat_response: Dict[str, Any] = {
        "id": "chatcmpl-abc",
        "created": 999,
        "model": "gpt-test",
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello!"},
            }
        ],
    }

    respond_result = translate_chat_to_respond(chat_response)
    assert_equal(respond_result["object"], "response", "Should be response object")
    assert_equal(respond_result["output"][0]["type"], "message", "Output should contain message block")
    assert_equal(
        respond_result["output"][0]["content"][0]["text"],
        "Hello!",
        "Text should be preserved",
    )


def main() -> None:
    run_chat_to_responses_case()
    run_responses_to_chat_case()
    run_reverse_path_case()
    print("Regression checks passed: chat<->responses translation looks good.")


if __name__ == "__main__":
    main()
