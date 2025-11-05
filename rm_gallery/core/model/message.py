from datetime import datetime
from typing import Any, Dict, Literal, Sequence

import shortuuid
from pydantic import BaseModel, Field

from rm_gallery.core.model.block import ContentBlock


class ChatMessage(BaseModel):
    """A message in a chat conversation, compatible with AgentScope Msg class."""

    name: str = Field(default="", description="The name of the message sender")
    content: str | Sequence[ContentBlock] = Field(
        default="",
        description="The content of the message, either a string or a list of content blocks",
    )
    role: Literal["user", "assistant", "system"] = Field(
        default="user", description="The role of the message sender"
    )
    metadata: Dict[str, Any] | None = Field(
        default=None, description="The metadata of the message"
    )
    timestamp: str | None = Field(
        default=None, description="The created timestamp of the message"
    )
    invocation_id: str | None = Field(
        default=None, description="The related API invocation id"
    )
    id: str = Field(
        default_factory=lambda: shortuuid.uuid(),
        description="Unique identifier for the message",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize ChatMessage with current timestamp if not provided.

        Args:
            **data: Message data
        """
        if "timestamp" not in data or data.get("timestamp") is None:
            data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        super().__init__(**data)

    def to_dict(self) -> dict:
        """Convert the message into JSON dict data.

        Returns:
            Dictionary representation of the message
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, json_data: dict) -> "ChatMessage":
        """Load a message object from the given JSON data.

        Args:
            json_data: JSON data to load from

        Returns:
            ChatMessage instance
        """
        return cls(**json_data)
