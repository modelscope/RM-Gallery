#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for LiteLLMChatModel.

LiteLLM is an optional gateway backend that reuses OpenAIChatModel's response
handling while routing generation through ``litellm.acompletion``. These tests
inject a fake ``litellm`` module so the suite runs without the real dependency.
"""

import sys
import types

import pytest

from openjudge.models import LiteLLMChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel


def _install_litellm_stub(content="LITELLM_OK"):
    """Register a fake ``litellm`` module; return (module, calls list)."""
    fake = types.ModuleType("litellm")
    calls = []

    class _Message:
        def __init__(self):
            self.role = "assistant"
            self.content = content

        def model_dump(self):
            return {"role": self.role, "content": self.content}

    class _Choice:
        def __init__(self):
            self.message = _Message()

    class _Response:
        def __init__(self):
            self.choices = [_Choice()]

    async def _acompletion(**kwargs):
        calls.append(kwargs)
        return _Response()

    fake.acompletion = _acompletion
    sys.modules["litellm"] = fake
    return fake, calls


@pytest.mark.unit
class TestLiteLLMChatModelInit:
    """Verify constructor behavior."""

    def test_inherits_from_openai_chat_model(self):
        assert issubclass(LiteLLMChatModel, OpenAIChatModel)

    def test_exported_from_models_package(self):
        from openjudge.models import LiteLLMChatModel as Imported

        assert Imported is LiteLLMChatModel

    def test_no_persistent_client_built(self):
        # LiteLLM routes per call; unlike OpenAIChatModel it builds no client.
        model = LiteLLMChatModel(model="gpt-4o")
        assert not hasattr(model, "client")

    def test_defaults(self):
        model = LiteLLMChatModel(model="anthropic/claude-sonnet-4-6")
        assert model.model == "anthropic/claude-sonnet-4-6"
        assert model.stream is False
        assert model.drop_params is True
        assert model.api_key is None
        assert model.base_url is None

    def test_drop_params_opt_out(self):
        model = LiteLLMChatModel(model="gpt-4o", drop_params=False)
        assert model.drop_params is False


@pytest.mark.unit
class TestLiteLLMChatModelAchat:
    """Verify achat dispatches to litellm.acompletion correctly."""

    async def test_dispatch_sets_drop_params_and_omits_blank_creds(self):
        _, calls = _install_litellm_stub()
        model = LiteLLMChatModel(model="anthropic/claude-sonnet-4-6")
        resp = await model.achat(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
        )
        assert resp.content == "LITELLM_OK"
        call = calls[-1]
        assert call["model"] == "anthropic/claude-sonnet-4-6"
        assert call["drop_params"] is True
        assert call["temperature"] == 0.2
        assert call["stream"] is False
        # creds omitted when unset so litellm uses each provider's own env var
        assert "api_key" not in call
        assert "api_base" not in call

    async def test_forwards_proxy_creds_when_set(self):
        _, calls = _install_litellm_stub()
        model = LiteLLMChatModel(
            model="gpt-4o",
            api_key="sk-proxy",
            base_url="http://localhost:4000/v1",
            max_retries=3,
            timeout=30.0,
        )
        await model.achat(messages=[{"role": "user", "content": "hi"}])
        call = calls[-1]
        assert call["api_key"] == "sk-proxy"
        assert call["api_base"] == "http://localhost:4000/v1"
        assert call["num_retries"] == 3
        assert call["timeout"] == 30.0

    async def test_structured_model_sets_response_format_and_drops_tools(self):
        from pydantic import BaseModel

        class Verdict(BaseModel):
            score: int

        _, calls = _install_litellm_stub(content='{"score": 5}')
        model = LiteLLMChatModel(model="gpt-4o")
        await model.achat(
            messages=[{"role": "user", "content": "rate this"}],
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="auto",
            structured_model=Verdict,
        )
        call = calls[-1]
        assert call["response_format"] is Verdict
        assert "tools" not in call
        assert "tool_choice" not in call

    async def test_gemini_structured_uses_json_object(self):
        from pydantic import BaseModel

        class Verdict(BaseModel):
            score: int

        _, calls = _install_litellm_stub(content='{"score": 5}')
        model = LiteLLMChatModel(model="gemini/gemini-2.5-flash")
        await model.achat(
            messages=[{"role": "user", "content": "rate"}],
            structured_model=Verdict,
        )
        assert calls[-1]["response_format"] == {"type": "json_object"}

    async def test_invalid_tool_choice_raises(self):
        _install_litellm_stub()
        model = LiteLLMChatModel(model="gpt-4o")
        with pytest.raises(ValueError):
            await model.achat(
                messages=[{"role": "user", "content": "hi"}],
                tool_choice="no_such_function",
                tools=[{"type": "function", "function": {"name": "real_fn"}}],
            )

    async def test_non_list_messages_raises(self):
        _install_litellm_stub()
        model = LiteLLMChatModel(model="gpt-4o")
        with pytest.raises(ValueError):
            await model.achat(messages="not a list")

    async def test_assistant_function_call_without_content_is_accepted(self):
        _, calls = _install_litellm_stub()
        model = LiteLLMChatModel(model="gpt-4o")
        messages = [
            {
                "role": "assistant",
                "function_call": {
                    "name": "lookup",
                    "arguments": "{}",
                },
            },
        ]

        await model.achat(messages=messages)

        assert calls[-1]["messages"] == messages

    @pytest.mark.parametrize(
        "messages",
        [
            [{"content": "missing role"}],
            [{"role": "user"}],
            [{"role": "tool", "content": "missing tool call id"}],
            [42],
        ],
        ids=["missing_role", "missing_content", "missing_tool_call_id", "non_dict"],
    )
    async def test_invalid_message_format_raises(self, messages):
        _, calls = _install_litellm_stub()
        model = LiteLLMChatModel(model="gpt-4o")

        with pytest.raises(ValueError, match="Invalid message format"):
            await model.achat(messages=messages)
        assert calls == []

    @pytest.mark.parametrize(
        "instance_stream, call_stream, expected_handler",
        [
            (False, True, "streaming"),
            (True, False, "non_streaming"),
        ],
    )
    async def test_call_stream_override_selects_matching_response_handler(
        self,
        monkeypatch,
        instance_stream,
        call_stream,
        expected_handler,
    ):
        _, calls = _install_litellm_stub()
        model = LiteLLMChatModel(model="gpt-4o", stream=instance_stream)
        streaming_result = object()
        non_streaming_result = object()

        monkeypatch.setattr(
            model,
            "_handle_streaming_response",
            lambda *_args: streaming_result,
        )
        monkeypatch.setattr(
            model,
            "_handle_non_streaming_response",
            lambda *_args: non_streaming_result,
        )

        result = await model.achat(
            messages=[{"role": "user", "content": "hi"}],
            stream=call_stream,
        )

        assert calls[-1]["stream"] is call_stream
        expected_result = streaming_result if expected_handler == "streaming" else non_streaming_result
        assert result is expected_result
