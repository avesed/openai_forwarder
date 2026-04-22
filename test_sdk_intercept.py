"""
Test the ChatViaResponses adapter: verifies that chat.completions.create()
is correctly intercepted and routed through the Responses API.

Runs in MOCK mode by default (no upstream needed).
Set RESPONSES_BASE_URL + RESPONSES_API_KEY for live test.
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent
OPENAI_SRC = PROJECT_ROOT / "openai-python" / "src"
if OPENAI_SRC.exists():
    sys.path.insert(0, str(OPENAI_SRC))

from openai import OpenAI
from openai.chat_via_responses import patch_client, ChatViaResponsesCompletions
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class FakeResponse:
    """Minimal mock for responses.create() return value."""
    def __init__(self, data):
        self._data = data
    def model_dump(self):
        return self._data


def make_fake_text_response(text="Four.", resp_id="resp_abc123"):
    return FakeResponse({
        "id": resp_id, "object": "response", "created_at": int(time.time()),
        "model": "gpt-4.1-mini",
        "output": [{"type": "message", "id": "msg_001", "role": "assistant",
                     "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22},
    })


def make_fake_tool_response():
    return FakeResponse({
        "id": "resp_tool_001", "object": "response", "created_at": int(time.time()),
        "model": "gpt-4.1-mini",
        "output": [{"type": "function_call", "id": "fc_001", "call_id": "call_abc",
                     "name": "get_weather", "arguments": '{"location":"Tokyo"}'}],
        "usage": {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45},
    })


def make_fake_reasoning_response():
    return FakeResponse({
        "id": "resp_reason_001", "object": "response", "created_at": int(time.time()),
        "model": "o3",
        "output": [
            {"type": "reasoning", "id": "r_001",
             "content": [{"type": "reasoning_text", "text": "Let me think about this..."}]},
            {"type": "message", "id": "msg_002", "role": "assistant",
             "content": [{"type": "output_text", "text": "The answer is 4."}]},
        ],
        "usage": {"input_tokens": 25, "output_tokens": 30, "total_tokens": 55},
    })


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    },
}]


def test_simple_message(client, use_real):
    print("=" * 60)
    print("TEST 1: Simple message")
    print("=" * 60)

    if not use_real:
        client.responses.create = MagicMock(return_value=make_fake_text_response())

    result = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are helpful. Reply concisely."},
            {"role": "user", "content": "What is 2+2? One word."},
        ],
        temperature=0,
    )

    assert isinstance(result, ChatCompletion), f"Expected ChatCompletion, got {type(result)}"
    assert result.choices[0].message.content is not None
    print(f"  content = {result.choices[0].message.content}")
    print(f"  finish  = {result.choices[0].finish_reason}")

    if not use_real:
        kw = client.responses.create.call_args.kwargs
        assert kw["model"] == "gpt-4.1-mini"
        assert kw["instructions"] == "You are helpful. Reply concisely."
        assert kw["temperature"] == 0
        print(f"  [verified] model={kw['model']}, instructions present, temperature={kw['temperature']}")

    print("  PASS\n")


def test_tool_call(client, use_real):
    print("=" * 60)
    print("TEST 2: Tool call")
    print("=" * 60)

    if not use_real:
        client.responses.create = MagicMock(return_value=make_fake_tool_response())

    result = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        tools=TOOLS,
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].finish_reason == "tool_calls"
    tc = result.choices[0].message.tool_calls
    assert tc and len(tc) > 0
    print(f"  tool = {tc[0].function.name}({tc[0].function.arguments})")

    if not use_real:
        kw = client.responses.create.call_args.kwargs
        assert kw["tools"][0]["name"] == "get_weather"
        print(f"  [verified] tools passed correctly")

    print("  PASS\n")


def test_multi_turn(client, use_real):
    print("=" * 60)
    print("TEST 3: Multi-turn (tool_call + tool result)")
    print("=" * 60)

    if not use_real:
        client.responses.create = MagicMock(return_value=make_fake_text_response("Tokyo is 22C and sunny."))

    result = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "Weather in Tokyo?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_abc", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_abc", "content": '{"temp":22,"condition":"sunny"}'},
        ],
        tools=TOOLS,
    )

    assert isinstance(result, ChatCompletion)
    print(f"  content = {result.choices[0].message.content}")

    if not use_real:
        kw = client.responses.create.call_args.kwargs
        block_types = [b.get("type", b.get("role", "?")) for b in kw["input"]]
        assert "function_call" in block_types
        assert "function_call_output" in block_types
        print(f"  [verified] input types = {block_types}")

    print("  PASS\n")


def test_reasoning(client, use_real):
    print("=" * 60)
    print("TEST 4: Reasoning model response")
    print("=" * 60)

    if not use_real:
        client.responses.create = MagicMock(return_value=make_fake_reasoning_response())

    result = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        reasoning_effort="medium",
    )

    assert isinstance(result, ChatCompletion)
    content = result.choices[0].message.content
    print(f"  content = {content[:80]}...")
    assert "[thinking]" in content
    assert "The answer is 4." in content

    if not use_real:
        kw = client.responses.create.call_args.kwargs
        assert kw["reasoning"] == {"effort": "medium"}
        print(f"  [verified] reasoning = {kw['reasoning']}")

    print("  PASS\n")


def test_usage_mapping(client, use_real):
    print("=" * 60)
    print("TEST 5: Usage field mapping")
    print("=" * 60)

    if not use_real:
        client.responses.create = MagicMock(return_value=make_fake_text_response())

    result = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert result.usage is not None
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 2
    assert result.usage.total_tokens == 22
    print(f"  prompt_tokens={result.usage.prompt_tokens}, completion_tokens={result.usage.completion_tokens}")
    print("  PASS\n")


def test_response_format(client, use_real):
    print("=" * 60)
    print("TEST 6: response_format -> text.format")
    print("=" * 60)

    if not use_real:
        client.responses.create = MagicMock(return_value=make_fake_text_response('{"answer": 4}'))

    result = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "What is 2+2? Reply in JSON."}],
        response_format={"type": "json_object"},
    )

    assert isinstance(result, ChatCompletion)

    if not use_real:
        kw = client.responses.create.call_args.kwargs
        assert "text" in kw
        assert kw["text"]["format"]["type"] == "json_object"
        print(f"  [verified] text.format = {kw['text']}")

    print("  PASS\n")


def test_legacy_functions(client, use_real):
    print("=" * 60)
    print("TEST 7: Legacy functions/function_call")
    print("=" * 60)

    if not use_real:
        client.responses.create = MagicMock(return_value=make_fake_tool_response())

    result = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        functions=[{
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
        }],
        function_call="auto",
    )

    assert isinstance(result, ChatCompletion)

    if not use_real:
        kw = client.responses.create.call_args.kwargs
        assert kw["tools"][0]["name"] == "get_weather"
        assert kw["tool_choice"] == "auto"
        print(f"  [verified] functions -> tools, function_call -> tool_choice")

    print("  PASS\n")


def test_patch_is_correct_type(client, use_real):
    print("=" * 60)
    print("TEST 8: Patched client type check")
    print("=" * 60)

    assert isinstance(client.chat.completions, ChatViaResponsesCompletions)
    print(f"  type = {type(client.chat.completions).__name__}")
    print("  PASS\n")


def main():
    base_url = os.getenv("RESPONSES_BASE_URL")
    api_key = os.getenv("RESPONSES_API_KEY") or os.getenv("OPENAI_API_KEY")
    use_real = bool(base_url and api_key)

    if use_real:
        base_url = base_url.rstrip("/")
        if not base_url.endswith(("/v1", "/openai/v1")):
            base_url += "/openai/v1"
        print(f"Mode: LIVE | {base_url}\n")
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        print("Mode: MOCK (set RESPONSES_BASE_URL + RESPONSES_API_KEY for live)\n")
        client = OpenAI(api_key="sk-fake", base_url="http://localhost:9999/v1")

    patch_client(client)

    test_patch_is_correct_type(client, use_real)
    test_simple_message(client, use_real)
    test_tool_call(client, use_real)
    test_multi_turn(client, use_real)
    test_reasoning(client, use_real)
    test_usage_mapping(client, use_real)
    test_response_format(client, use_real)
    test_legacy_functions(client, use_real)

    print("=" * 60)
    print(f"ALL 8 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
