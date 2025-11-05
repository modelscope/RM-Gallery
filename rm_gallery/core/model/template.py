import asyncio
import re
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Type

from pydantic import BaseModel, Field, model_validator

from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.model.response import ChatResponse
from rm_gallery.core.model.utils import _json_loads_with_repair


class LanguageEnum(str, Enum):
    """Language enumeration for templates."""

    EN = "en"
    ZH = "zh"


class ChatTemplate(BaseModel):
    """Chat template for generating chat messages."""

    template: List[ChatMessage] | Dict[LanguageEnum, List[ChatMessage]] = Field(
        default=..., description="template for generating chat messages"
    )
    required: List[str] = Field(default_factory=list, description="required kwargs")
    model: Dict[str, Any] = Field(default=..., description="model parameters")

    @model_validator(mode="before")
    def validate_template(cls, values) -> dict:
        """Validate template and extract placeholders from messages.

        Args:
            values: Dictionary of values to validate

        Returns:
            Validated values
        """
        # Extract placeholders like {data.query}, {query}, {answer}, etc. from messages
        template = values.get("template", [])
        required = values.get("required", [])

        # Check if required is a list, if not initialize it
        if not isinstance(required, list):
            required = []

        # Pattern to match placeholders like {word}, {word.word}, {word.word.word}, etc.
        placeholder_pattern = (
            r"\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}"
        )

        # Extract messages from all language templates
        all_messages = []
        if isinstance(template, list):
            all_messages.extend(template)
        elif isinstance(template, dict):
            # Extract messages from all language templates
            for lang_messages in template.values():
                all_messages.extend(lang_messages)

        # Iterate through all messages to find placeholders
        for message in all_messages:
            content = (
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )
            # Find all placeholders in the content
            placeholders = re.findall(placeholder_pattern, content)
            for placeholder in placeholders:
                # Add to required if not already present
                if placeholder not in required:
                    required.append(placeholder)

        values["required"] = required
        return values

    def format(
        self, language: LanguageEnum = LanguageEnum.EN, **kwargs
    ) -> List[Dict[str, Any]]:
        """Format messages with provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to format with

        Returns:
            List of formatted message dictionaries
        """
        if isinstance(self.template, list):
            template = self.template
        elif isinstance(self.template, dict):
            assert language in self.template
            template = self.template.get(language, [])
        else:
            raise ValueError("Invalid template")

        messages = [message.to_dict() for message in template]
        for message in messages:
            message["content"] = message.get("content", "").format(**kwargs)
        return messages

    async def __call__(
        self, model_output: Callable | Type[BaseModel] | None = None, **kwargs
    ) -> ChatResponse:
        """Generate chat response using the template.

        Args:
            model_output: Optional model output type or callable
            **kwargs: Keyword arguments for formatting messages

        Returns:
            Chat response
        """
        messages = self.format(**kwargs)

        params = {}
        if isinstance(model_output, type) and issubclass(model_output, BaseModel):
            params["response_format"] = {"type": "json_object"}

        response = await OpenAIChatModel(**self.model)(messages=messages, **params)

        # Handle case where response might be an AsyncGenerator
        if isinstance(response, AsyncGenerator):
            # For streaming responses, collect all chunks
            content_parts = []
            metadata = {}
            usage = None

            async for chunk in response:
                content_parts.extend(chunk.content)
                if chunk.metadata:
                    metadata.update(chunk.metadata)
                if chunk.usage:
                    usage = chunk.usage

            # Create a consolidated response
            response = ChatResponse(
                content=content_parts, metadata=metadata or None, usage=usage
            )

        if model_output is not None:
            metadata = response.metadata if response.metadata else {}
            text_content = ""

            # Extract text from content blocks
            for content_block in response.content:
                if getattr(content_block, "type", "") == "text":
                    text_content = getattr(content_block, "text", "{}")
                    break

            if isinstance(model_output, type) and issubclass(model_output, BaseModel):
                try:
                    parsed_data = _json_loads_with_repair(text_content)
                    if isinstance(parsed_data, dict):
                        metadata.update(parsed_data)
                except Exception:
                    # If parsing fails, leave metadata as is
                    pass
            elif isinstance(model_output, Callable):
                try:
                    parsed_data = model_output(text_content)
                    if isinstance(parsed_data, dict):
                        metadata.update(parsed_data)
                except Exception:
                    # If parsing fails, leave metadata as is
                    pass

            response.metadata = metadata

        return response


if __name__ == "__main__":
    chat = ChatTemplate(
        template=[
            ChatMessage(
                role="system", content="You are a helpful assistant.", name="System"
            ),
            ChatMessage(role="user", content="{question}", name="User"),
        ],
        model={
            "model_name": "qwen-plus",
            "api_key": "sk-qS2yrmvJYAhsJN7xA4lZFJwYoOiQgglD5MukURFzMARrGlLJ",
            "stream": False,
            "client_args": {
                "timeout": 60,
                "base_url": "http://8.130.177.212:3000/v1",
            },
        },
    )
    result = asyncio.run(chat(question="What is the capital of France?"))
    print(result)
