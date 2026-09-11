# -*- coding: utf-8 -*-
"""Cookbook-local search loop for collecting tool-augmented model responses.

This keeps the configured target model as the respondent. It is independent of
the grading harnesses, which judge already-collected candidate evidence.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx

from cookbooks.ref_hallucination_arena.schema import ToolConfig
from openjudge.models.openai_chat_model import OpenAIChatModel

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web to find and verify real academic papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "search_depth": {"type": "string", "enum": ["basic", "advanced"]},
            },
            "required": ["query"],
        },
    },
}


@dataclass
class SearchAgentResult:
    """Response and complete conversation used by the collector's summary step."""

    content: str
    messages: List[Dict[str, Any]]
    tool_calls_count: int
    iterations: int
    max_iterations_reached: bool = False


class ReferenceSearchAgent:
    """Run a bounded web-search conversation with one configured target model.

    Conversation state belongs to each invocation, so a collector may reuse this
    object for concurrent queries. HTTP clients are closed on completion or
    cancellation and search failures are returned to the model as tool errors.
    """

    def __init__(self, model: OpenAIChatModel, config: ToolConfig):
        self.model = model
        self.config = config
        self.api_key = config.tavily_api_key or os.getenv("TAVILY_API_KEY")

    async def _search(self, query: str, search_depth: str) -> str:
        """Fetch search evidence without blocking the collector's event loop.

        API contract: https://docs.tavily.com/documentation/api-reference/endpoint/search
        """
        if not self.api_key:
            raise ValueError("Set tool_config.tavily_api_key or TAVILY_API_KEY to use web search.")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"query": query, "search_depth": search_depth, "max_results": 5},
            )
            response.raise_for_status()
            results = response.json().get("results", [])

        evidence = [
            f"[{index}] {item.get('title', '')}\nURL: {item.get('url', '')}\nContent: {item.get('content', '')[:1500]}"
            for index, item in enumerate(results[:5], 1)
        ]
        return "\n\n".join(evidence) if evidence else "No results found"

    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Always pair an assistant tool call with its result, including errors."""
        try:
            function = tool_call["function"]
            if function["name"] != "web_search":
                raise ValueError(f"Unknown tool: {function['name']}. Available tool: web_search")
            arguments = json.loads(function.get("arguments", "{}"))
            if not isinstance(arguments, dict) or not isinstance(arguments.get("query"), str):
                raise ValueError("web_search requires an object with a string query.")
            depth = arguments.get("search_depth", self.config.search_depth)
            if depth not in ("basic", "advanced"):
                raise ValueError("search_depth must be basic or advanced.")
            content = await self._search(arguments["query"], depth)
        except Exception as exc:
            content = f"Search error: {exc}"
        if len(content) > 4000:
            content = content[:4000] + "\n... [truncated]"
        return {"role": "tool", "tool_call_id": tool_call["id"], "content": content}

    async def arun(self, messages: List[Dict[str, Any]]) -> SearchAgentResult:
        """Search until the model answers or reaches the configured iteration limit."""
        conversation = list(messages)
        tool_calls_count = 0
        content = ""
        for iteration in range(1, self.config.max_iterations + 1):
            response = await self.model.achat(messages=conversation, tools=[SEARCH_TOOL])
            content = response.get_text_content() or ""
            tool_calls = response.tool_calls or []
            assistant = {"role": "assistant", "content": content}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            conversation.append(assistant)
            if not tool_calls:
                return SearchAgentResult(content, conversation, tool_calls_count, iteration)
            for tool_call in tool_calls:
                conversation.append(await self._execute_tool_call(tool_call))
                tool_calls_count += 1

        return SearchAgentResult(content, conversation, tool_calls_count, self.config.max_iterations, True)
