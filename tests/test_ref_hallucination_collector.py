# -*- coding: utf-8 -*-
"""Offline coverage for the arena's target-model search and summary path.

Kept outside tests/cookbooks because pytest excludes directories named cookbooks.
"""

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from openai.types.chat import ChatCompletion

from cookbooks.ref_hallucination_arena.collectors import ResponseCollector
from cookbooks.ref_hallucination_arena.collectors import search_agent as search_module
from cookbooks.ref_hallucination_arena.schema import (
    OpenAIEndpoint,
    QueryItem,
    ToolConfig,
)

pytestmark = pytest.mark.unit

BIBTEX = "@article{verified, title={Verified paper}, author={Author}, year={2026}}"


def _collector(**tool_options):
    return ResponseCollector(
        {
            "target": OpenAIEndpoint(
                base_url="http://127.0.0.1:1/v1",
                api_key="offline-test",
                model="target-model",
                tool_config=ToolConfig(**{"enabled": True, "tavily_api_key": "search-test", **tool_options}),
            )
        }
    )


def _tool_call(call_id="search-1", query="verified paper", **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "web_search", "arguments": json.dumps({"query": query, **arguments})},
    }


def _mock_model(monkeypatch, collector, messages):
    """Stub only the API boundary, retaining OpenAIChatModel's message handling."""
    requests = []
    responses = iter(messages)

    async def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        message = next(responses)
        return ChatCompletion(
            id="offline",
            created=0,
            model="target-model",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    "message": {"role": "assistant", **message},
                }
            ],
        )

    monkeypatch.setattr(collector.models["target"].client.chat.completions, "create", create)
    return requests


def _mock_search_http(monkeypatch, handler):
    client_class = httpx.AsyncClient
    clients = []

    def create_client(**kwargs):
        client = client_class(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(search_module.httpx, "AsyncClient", create_client)
    return clients


def _query(text="Recommend a paper"):
    return QueryItem(query=text, discipline="computer_science", language="en", num_refs=1)


async def test_bare_mode_still_collects_without_search(monkeypatch):
    collector = _collector(enabled=False)
    calls = _mock_model(monkeypatch, collector, [{"content": BIBTEX}])
    assert not collector.agents
    result = await collector.collect([_query()])
    assert result[0]["responses"]["target"] == BIBTEX
    assert "tools" not in calls[0]


@pytest.mark.parametrize("model_depth", [None, "advanced"])
async def test_tool_mode_searches_with_target_model_and_returns_evidence(monkeypatch, model_depth):
    collector = _collector(search_depth="basic")
    assert collector.agents["target"].model is collector.models["target"]
    arguments = {"search_depth": model_depth} if model_depth else {}
    calls = _mock_model(
        monkeypatch, collector, [{"content": None, "tool_calls": [_tool_call(**arguments)]}, {"content": BIBTEX}]
    )
    searches = []

    def respond(request):
        searches.append(request)
        return httpx.Response(
            200,
            json={"results": [{"title": "Verified paper", "url": "https://example.test/paper", "content": "Evidence"}]},
        )

    clients = _mock_search_http(monkeypatch, respond)
    result = await collector._do_tool_call("target", _query(), timeout=5)
    assert result["response"] == BIBTEX
    assert result["metadata"] == {
        "tool_augmented": True,
        "tool_calls_count": 1,
        "iterations": 2,
        "used_fallback": False,
    }
    assert [call["model"] for call in calls] == ["target-model", "target-model"]
    assert str(searches[0].url) == "https://api.tavily.com/search"
    assert searches[0].headers["Authorization"] == "Bearer search-test"
    assert json.loads(searches[0].content) == {
        "query": "verified paper",
        "search_depth": model_depth or "basic",
        "max_results": 5,
    }
    tool_message = calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "search-1"
    assert "https://example.test/paper" in tool_message["content"]
    assert "Evidence" in tool_message["content"]
    assert all(client.is_closed for client in clients)


async def test_iteration_limit_summarizes_complete_search_history_without_tools(monkeypatch):
    collector = _collector(max_iterations=1)
    calls = _mock_model(
        monkeypatch,
        collector,
        [
            {"content": "Searching", "tool_calls": [_tool_call("one"), _tool_call("two")]},
            {"content": BIBTEX},
        ],
    )
    search = AsyncMock(return_value="Verified evidence")
    monkeypatch.setattr(collector.agents["target"], "_search", search)
    result = await collector._do_tool_call("target", _query(), timeout=5)
    assert result["response"] == BIBTEX
    assert result["metadata"]["used_fallback"] is True
    assert result["metadata"]["iterations"] == 1
    assert search.await_count == result["metadata"]["tool_calls_count"] == 2
    assert "tools" not in calls[1]
    assert calls[1]["messages"][1]["content"] == _query().query
    assert [message["tool_call_id"] for message in calls[1]["messages"] if message["role"] == "tool"] == ["one", "two"]
    assert calls[1]["messages"][-1]["role"] == "user"


@pytest.mark.parametrize(
    "arguments", ["{invalid", "[]", "{}", '{"query": 123}', '{"query": "q", "search_depth": "bad"}']
)
async def test_malformed_tool_arguments_are_reported_to_model(monkeypatch, arguments):
    collector = _collector()
    tool_call = _tool_call()
    tool_call["function"]["arguments"] = arguments
    calls = _mock_model(monkeypatch, collector, [{"tool_calls": [tool_call]}, {"content": BIBTEX}])
    search = AsyncMock()
    monkeypatch.setattr(collector.agents["target"], "_search", search)
    result = await collector._do_tool_call("target", _query(), timeout=5)
    assert result["success"] is True
    assert calls[1]["messages"][-1]["content"].startswith("Search error:")
    assert calls[1]["messages"][-1]["tool_call_id"] == "search-1"
    search.assert_not_awaited()


async def test_search_http_failure_is_reported_and_client_closed(monkeypatch):
    collector = _collector()
    calls = _mock_model(monkeypatch, collector, [{"tool_calls": [_tool_call()]}, {"content": BIBTEX}])
    clients = _mock_search_http(monkeypatch, lambda request: httpx.Response(429))
    await collector._do_tool_call("target", _query(), timeout=5)
    assert "429" in calls[1]["messages"][-1]["content"]
    assert calls[1]["messages"][-1]["content"].startswith("Search error:")
    assert all(client.is_closed for client in clients)


def test_search_uses_environment_key_when_not_configured(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "environment-test")
    collector = _collector(tavily_api_key=None)
    assert collector.agents["target"].api_key == "environment-test"


async def test_concurrent_queries_keep_separate_search_histories(monkeypatch):
    collector = _collector()
    arrived = 0
    both_searching = asyncio.Event()

    async def search(query, search_depth):
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both_searching.set()
        await asyncio.wait_for(both_searching.wait(), timeout=2)
        return f"Evidence for {query}"

    async def respond(messages, tools):
        from openjudge.models.schema.oai.response import ChatResponse

        query = messages[1]["content"]
        if messages[-1]["role"] == "user":
            return ChatResponse(role="assistant", tool_calls=[_tool_call(query, query)])
        assert messages[-1]["content"] == f"Evidence for {query}"
        assert messages[-1]["tool_call_id"] == query
        return ChatResponse(role="assistant", content=f"@article{{{query}}}")

    monkeypatch.setattr(collector.models["target"], "achat", respond)
    monkeypatch.setattr(collector.agents["target"], "_search", search)
    results = await collector.collect([_query("first"), _query("second")])
    assert [result["responses"]["target"] for result in results] == ["@article{first}", "@article{second}"]


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
async def test_search_cancellation_closes_http_client(monkeypatch, cancel):
    collector = _collector()
    calls = _mock_model(monkeypatch, collector, [{"tool_calls": [_tool_call()]}])
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def pending_search(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    clients = _mock_search_http(monkeypatch, pending_search)
    task = asyncio.create_task(collector._do_tool_call("target", _query(), timeout=5 if cancel else 0.1))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=2)
        assert stopped.is_set()
        assert all(client.is_closed for client in clients)
        assert len(calls) == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
