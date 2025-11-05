from typing import Literal

from pydantic import BaseModel, Field


class ChatUsage(BaseModel):
    """The usage of a chat model API invocation."""

    input_tokens: int = Field(..., description="The number of input tokens.")
    output_tokens: int = Field(..., description="The number of output tokens.")
    time: float = Field(..., description="The time used in seconds.")
    type: Literal["chat"] = Field(
        default="chat", description="The type of the usage, must be `chat`."
    )
