from datetime import datetime
from typing import Any, Dict, Literal, Sequence

import shortuuid
from pydantic import BaseModel, Field

from rm_gallery.core.model.block import (
    AudioBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from rm_gallery.core.model.usage import ChatUsage


class ChatResponse(BaseModel):
    """The response of chat models, compatible with AgentScope ChatResponse."""

    content: Sequence[TextBlock | ToolUseBlock | ThinkingBlock | AudioBlock] = Field(
        default_factory=list,
        description="The content of the chat response, which can include text blocks, tool use blocks, or thinking blocks.",
    )

    id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        + "_"
        + shortuuid.uuid()[:6],
        description="The unique identifier",
    )

    created_at: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        description="When the response was created",
    )

    type: Literal["chat"] = Field(
        default="chat", description="The type of the response, which is always 'chat'."
    )

    usage: ChatUsage | None = Field(
        default=None,
        description="The usage information of the chat response, if available.",
    )

    metadata: Dict[str, Any] | None = Field(
        default=None, description="The metadata of the chat response"
    )

    @classmethod
    def from_dict(cls, json_data: dict) -> "ChatResponse":
        """Load a chat response object from the given JSON data."""
        return cls(**json_data)

    def to_dict(self) -> dict:
        """Convert the chat response into JSON dict data."""
        return self.model_dump()
