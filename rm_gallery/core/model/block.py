from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """The text block."""

    type: Literal["text"] = Field(default="text", description="The type of the block")
    text: str = Field(default="", description="The text content")


class ThinkingBlock(BaseModel):
    """The thinking block."""

    type: Literal["thinking"] = Field(
        default="thinking", description="The type of the block"
    )
    thinking: str = Field(default="", description="The thinking content")


class Base64Source(BaseModel):
    """The base64 source"""

    type: Literal["base64"] = Field(
        default="base64", description="The type of the src, must be `base64`"
    )
    media_type: str = Field(
        ..., description="The media type of the data, e.g. `image/jpeg` or `audio/mpeg`"
    )
    data: str = Field(..., description="The base64 data, in format of RFC 2397")


class URLSource(BaseModel):
    """The URL source"""

    type: Literal["url"] = Field(
        default="url", description="The type of the src, must be `url`"
    )
    url: str = Field(..., description="The URL of the image or audio")


class ImageBlock(BaseModel):
    """The image block"""

    type: Literal["image"] = Field(default="image", description="The type of the block")
    source: Base64Source | URLSource = Field(..., description="The src of the image")


class AudioBlock(BaseModel):
    """The audio block"""

    type: Literal["audio"] = Field(default="audio", description="The type of the block")
    source: Base64Source | URLSource = Field(..., description="The src of the audio")


class VideoBlock(BaseModel):
    """The video block"""

    type: Literal["video"] = Field(default="video", description="The type of the block")
    source: Base64Source | URLSource = Field(..., description="The src of the audio")


class ToolUseBlock(BaseModel):
    """The tool use block."""

    type: Literal["tool_use"] = Field(
        default="tool_use", description="The type of the block, must be `tool_use`"
    )
    id: str = Field(..., description="The identity of the tool call")
    name: str = Field(..., description="The name of the tool")
    input: Dict[str, Any] = Field(..., description="The input of the tool")


class ToolResultBlock(BaseModel):
    """The tool result block."""

    type: Literal["tool_result"] = Field(
        default="tool_result", description="The type of the block"
    )
    id: str = Field(..., description="The identity of the tool call result")
    output: str | List[TextBlock | ImageBlock | AudioBlock] = Field(
        ..., description="The output of the tool function"
    )
    name: str = Field(..., description="The name of the tool function")


# The content block
ContentBlock = (
    ToolUseBlock
    | ToolResultBlock
    | TextBlock
    | ThinkingBlock
    | ImageBlock
    | AudioBlock
    | VideoBlock
)
