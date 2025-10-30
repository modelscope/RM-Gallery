"""
Multimodal message extensions for vision-language interactions.

This module extends the existing ChatMessage system to support multimodal content
while maintaining backward compatibility. Messages can contain text, images, or both.
"""

from typing import Any, List, Optional, Union

from pydantic import Field, field_validator

from rm_gallery.core.data.multimodal_content import ImageContent, MultimodalContent
from rm_gallery.core.model.message import ChatMessage, MessageRole


class MultimodalChatMessage(ChatMessage):
    """
    Extended ChatMessage supporting multimodal content (text + images).

    This class extends the standard ChatMessage to handle multimodal content
    while maintaining full backward compatibility with existing text-only messages.

    The content field can be:
    - str: Plain text (backward compatible)
    - MultimodalContent: Text + images

    Attributes:
        role: Message role (system/user/assistant/function)
        name: Optional name associated with the message
        content: Text string or MultimodalContent object
        reasoning_content: Internal reasoning information
        tool_calls: List of tools called in this message
        additional_kwargs: Extra metadata dictionary
        time_created: Timestamp of message creation

    Examples:
        >>> # Text-only message (backward compatible)
        >>> msg = MultimodalChatMessage(
        ...     role=MessageRole.USER,
        ...     content="Hello!"
        ... )

        >>> # Multimodal message
        >>> msg = MultimodalChatMessage(
        ...     role=MessageRole.USER,
        ...     content=MultimodalContent(
        ...         text="What's in this image?",
        ...         images=[ImageContent(type="url", data="https://...")]
        ...     )
        ... )

        >>> # Check if message has images
        >>> if msg.has_image():
        ...     images = msg.get_images()
    """

    # Override content field to accept MultimodalContent (also allow dict for backward compatibility)
    content: Optional[Union[str, MultimodalContent, dict]] = Field(
        default="",
        description="Message content: text string, MultimodalContent, or dict (backward compatible)",
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> Union[str, MultimodalContent, dict]:
        """
        Validate and convert content to appropriate type.

        Handles:
        - str: Keep as is
        - dict: Try to parse as MultimodalContent, keep as dict if fails (backward compatible)
        - MultimodalContent: Keep as is
        - None: Convert to empty string
        """
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, MultimodalContent):
            return v
        if isinstance(v, dict):
            # Only try to parse as MultimodalContent if dict explicitly has multimodal fields
            # and is not empty, AND only has multimodal-related keys
            if v and ("text" in v or "images" in v):
                # Check if dict has ONLY multimodal fields (plus allowed extras like metadata)
                multimodal_keys = {"text", "images"}
                dict_keys = set(v.keys())

                # If all keys are multimodal-related, try parsing
                # Allow extra fields like 'metadata', 'caption' etc.
                try:
                    result = MultimodalContent(**v)
                    # Only return MultimodalContent if it actually has content
                    if result.has_text() or result.has_image():
                        return result
                except Exception:
                    # If parsing fails, keep as dict (backward compatible)
                    pass
            # For other dicts (e.g., tool payloads, structured data, empty dict), keep as is
            return v
        # For any other type, convert to string
        return str(v)

    def has_image(self) -> bool:
        """
        Check if message contains any images.

        Returns:
            True if message contains one or more images
        """
        if isinstance(self.content, MultimodalContent):
            return self.content.has_image()
        return False

    def has_text(self) -> bool:
        """
        Check if message contains text.

        Returns:
            True if message contains text content
        """
        if isinstance(self.content, str):
            return bool(self.content.strip())
        elif isinstance(self.content, MultimodalContent):
            return self.content.has_text()
        return False

    def is_multimodal(self) -> bool:
        """
        Check if message is truly multimodal (contains both text and images).

        Returns:
            True if message has both text and images
        """
        if isinstance(self.content, MultimodalContent):
            return self.content.is_multimodal()
        return False

    def get_images(self) -> List[ImageContent]:
        """
        Get list of images from the message.

        Returns:
            List of ImageContent objects (empty if no images)
        """
        if isinstance(self.content, MultimodalContent):
            return self.content.get_images()
        return []

    def get_text(self) -> str:
        """
        Get text content from the message.

        Returns:
            Text string (empty if no text)
        """
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, MultimodalContent):
            return self.content.get_text()
        elif isinstance(self.content, dict):
            # For backward compatibility with dict content
            return str(self.content)
        return ""

    def get_content(self) -> Union[str, MultimodalContent, dict]:
        """
        Get the raw content object.

        Returns:
            Content as str, MultimodalContent, or dict
        """
        return self.content if self.content else ""

    def to_api_format(
        self, api_type: str = "openai", include_role: bool = True
    ) -> dict:
        """
        Convert message to API-compatible format.

        Args:
            api_type: Target API ("openai", "qwen", "anthropic")
            include_role: Whether to include role in output

        Returns:
            Dictionary formatted for the specified API

        Examples:
            >>> msg = MultimodalChatMessage(
            ...     role=MessageRole.USER,
            ...     content=MultimodalContent(
            ...         text="What's in this?",
            ...         images=[ImageContent(type="url", data="https://...")]
            ...     )
            ... )
            >>> msg.to_api_format("openai")
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': "What's in this?"},
                    {'type': 'image_url', 'image_url': {'url': 'https://...'}}
                ]
            }
        """
        result = {}

        if include_role:
            result["role"] = self.role.value

        # Handle content based on type
        if isinstance(self.content, str):
            result["content"] = self.content
        elif isinstance(self.content, MultimodalContent):
            result["content"] = self.content.to_api_format(api_type)
        elif isinstance(self.content, dict):
            # For dict content, serialize to string for API compatibility
            # Most APIs (OpenAI, Anthropic, etc.) expect string or structured array, not raw dict
            import json

            result["content"] = json.dumps(self.content, ensure_ascii=False)
        else:
            result["content"] = ""

        # Add name if present
        if self.name:
            result["name"] = self.name

        # Add tool calls if present (for function calling)
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tool.id,
                    "type": tool.type,
                    "function": {
                        "name": tool.function.name,
                        "arguments": tool.function.arguments,
                    },
                }
                for tool in self.tool_calls
            ]

        return result

    def to_text_only(self) -> str:
        """
        Convert message to text-only representation (for logging/display).

        Returns:
            Text representation of the message
        """
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, MultimodalContent):
            return self.content.to_string(include_image_info=True)
        elif isinstance(self.content, dict):
            # For backward compatibility with dict content
            return str(self.content)
        return ""

    def estimate_tokens(self, model: str = "gpt-4o") -> int:
        """
        Estimate token count for this message.

        Args:
            model: Model name for token estimation

        Returns:
            Estimated token count
        """
        if isinstance(self.content, str):
            # Rough estimate: 1 token ≈ 4 chars
            return len(self.content) // 4
        elif isinstance(self.content, MultimodalContent):
            return self.content.estimate_tokens(model)
        elif isinstance(self.content, dict):
            # For dict content, estimate by JSON string length
            import json

            return len(json.dumps(self.content)) // 4
        return 0

    def __str__(self) -> str:
        """String representation with role and content."""
        text = self.to_text_only()
        return f"{self.time_created.strftime('%Y-%m-%d %H:%M:%S')} {self.role.value}: {text}"

    @classmethod
    def from_chat_message(
        cls, message: ChatMessage, images: Optional[List[ImageContent]] = None
    ) -> "MultimodalChatMessage":
        """
        Convert a standard ChatMessage to MultimodalChatMessage.

        Args:
            message: Source ChatMessage
            images: Optional list of images to add

        Returns:
            New MultimodalChatMessage instance
        """
        content = message.content

        # If images provided, create MultimodalContent
        if images:
            if isinstance(content, str):
                content = MultimodalContent(text=content, images=images)
            elif isinstance(content, MultimodalContent):
                content.images.extend(images)

        return cls(
            role=message.role,
            name=message.name,
            content=content,
            reasoning_content=message.reasoning_content,
            tool_calls=message.tool_calls,
            additional_kwargs=message.additional_kwargs,
            time_created=message.time_created,
        )

    @classmethod
    def create_user_message(
        cls,
        text: Optional[str] = None,
        images: Optional[List[ImageContent]] = None,
        name: Optional[str] = None,
    ) -> "MultimodalChatMessage":
        """
        Convenience method to create a user message.

        Args:
            text: Text content
            images: List of images
            name: Optional user name

        Returns:
            New MultimodalChatMessage with role USER
        """
        if images:
            content = MultimodalContent(text=text, images=images)
        else:
            content = text or ""

        return cls(role=MessageRole.USER, name=name, content=content)

    @classmethod
    def create_assistant_message(
        cls, text: str, name: Optional[str] = None
    ) -> "MultimodalChatMessage":
        """
        Convenience method to create an assistant message.

        Args:
            text: Text content
            name: Optional assistant name

        Returns:
            New MultimodalChatMessage with role ASSISTANT
        """
        return cls(role=MessageRole.ASSISTANT, name=name, content=text)

    @classmethod
    def create_system_message(cls, text: str) -> "MultimodalChatMessage":
        """
        Convenience method to create a system message.

        Args:
            text: System message text

        Returns:
            New MultimodalChatMessage with role SYSTEM
        """
        return cls(role=MessageRole.SYSTEM, content=text)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        json_schema_extra = {
            "examples": [
                {"role": "user", "content": "Hello, how are you?"},
                {
                    "role": "user",
                    "content": {
                        "text": "What's in this image?",
                        "images": [
                            {"type": "url", "data": "https://example.com/image.jpg"}
                        ],
                    },
                },
            ]
        }


# Convenience function for backward compatibility
def convert_to_multimodal(messages: List[ChatMessage]) -> List[MultimodalChatMessage]:
    """
    Convert a list of ChatMessages to MultimodalChatMessages.

    Args:
        messages: List of ChatMessage objects

    Returns:
        List of MultimodalChatMessage objects
    """
    return [MultimodalChatMessage.from_chat_message(msg) for msg in messages]
