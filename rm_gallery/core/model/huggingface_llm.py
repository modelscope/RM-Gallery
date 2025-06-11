import json
from typing import Any, Dict, List, Optional, Type

from outlines.generate.choice import choice as outlines_choice
from outlines.generate.json import json as outlines_json
from outlines.models.transformers import transformers
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer

from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.model.message import (
    ChatMessage,
    ChatResponse,
    GeneratorChatResponse,
)


def _convert_messages_format(messages: List[ChatMessage] | str) -> list[dict[str, Any]]:
    """
    Converts a list of ChatMessage objects or a string into a list of dictionaries for model input.

    Args:
        messages (List[ChatMessage] | str): A list of ChatMessage objects or a single string.

    Returns:
        list[dict[str, Any]]: A formatted list of message dictionaries with 'role' and 'content'.
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    return [
        {"role": message.role.name.lower(), "content": message.content}
        for message in messages
    ]


def _get_tool_name(tools):
    """
    Extracts and returns the names of tools from a list of tool dictionaries.

    Args:
        tools (list): A list of dictionaries, each possibly containing a 'function' key with a 'name'.

    Returns:
        list: A list of tool names.

    Raises:
        ValueError: If any tool does not contain a valid 'name' in the 'function' key.
    """
    tool_names = []
    for tool in tools:
        # Check if the tool has a 'function' key and a 'name' in it
        if "function" in tool and "name" in tool["function"]:
            tool_names.append(tool["function"]["name"])
        else:
            raise ValueError(
                "Invalid tool format. 'name' field not found in 'function'."
            )

    return tool_names


def _get_tool_by_name(tools, function_name):
    """
    Finds and returns the tool from the list of tools that matches the specified function name.

    Args:
        tools (list): A list of dictionaries, where each dictionary may contain a "function" key.
        function_name (str): The name of the function to search for within the tools.

    Returns:
        dict: The dictionary representing the tool that contains the matching function name.

    """
    for tool in tools:
        # Check if the tool has a "function" key and if its "name" matches the function_name
        if "function" in tool and tool["function"].get("name") == function_name:
            return tool
    # Raise an exception if no tool with the given function name is found
    raise ValueError(f"Tool with function name '{function_name}' not found.")


class HuggingFaceLLM(BaseLLM):
    """
    Support transformers model(Qwen/Llama have been tested.)
    Notice:
    - device='mps' is not supported by structured outputs, but chat mode is still usable

    Input should be chatml format or string

    Usage:
    llm = HuggingFaceLLM(model = "/path/to/model/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    res = llm.chat([ChatMessage(role="user", content="Hello world!")])
    print(res)
    # "Hello! How can I assist you today?"

    """

    def __init__(
        self,
        model: str,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model)

        self.operator = transformers(model, device=device, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
        self.device = device

    def chat(
        self, messages: List[ChatMessage] | str, stream=False, **kwargs
    ) -> ChatResponse | GeneratorChatResponse:
        # Convert messages into appropriate input format
        messages = _convert_messages_format(messages)
        completed_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        if self.device is not None:
            input_ids = input_ids.to(
                self.operator.model.device
            )  # fix bad case when device is auto

        # Non-streaming mode
        if not stream:
            outputs = self.operator.model.generate(
                input_ids,
                max_length=self.max_tokens,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                top_k=kwargs.get("top_k", self.top_k),
                do_sample=True if self.temperature > 0 else False,
            )
            response_text = self.tokenizer.decode(
                outputs[:, input_ids.shape[-1] :][0], skip_special_tokens=True
            )

            return ChatResponse(
                message=ChatMessage(role="assistant", content=response_text)
            )

        # Streaming mode

        else:

            def generate_stream():
                # Initialize TextIteratorStreamer to incrementally receive the generated content
                streamer = TextIteratorStreamer(
                    self.tokenizer, skip_prompt=True, skip_special_tokens=True
                )

                # Start text generation, passing the streamer for streaming results
                self.operator.model.generate(
                    input_ids,
                    max_length=self.max_tokens,
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=kwargs.get("top_p", self.top_p),
                    top_k=kwargs.get("top_k", self.top_k),
                    do_sample=True if self.temperature > 0 else False,
                    return_dict_in_generate=True,
                    output_scores=False,
                    streamer=streamer,  # Use the built-in TextIteratorStreamer
                )

                response_stream = (
                    ""  # Initialize an empty string to store the accumulated response
                )

                # Retrieve generated content incrementally from the streamer
                for new_text in streamer:
                    response_stream += new_text

                    # Return the accumulated text in the required format
                    yield ChatResponse(
                        message=ChatMessage(role="assistant", content=response_stream)
                    )

            return generate_stream()

    def structured_output(
        self,
        messages: List[ChatMessage],
        schema: Type[BaseModel] | str | dict,
        method: str = "json_schema",
        **kwargs,
    ) -> BaseModel:
        """
        Use outlines to format the output based on json schema or pydantic model
        """
        # Convert messages to prompt format
        messages = _convert_messages_format(messages)
        prompt = self._convert_to_prompt(messages)

        if isinstance(schema, dict):
            schema = json.dumps(schema, indent=4)

        # JSON schema or Pydantic model
        generator = outlines_json(self.operator, schema)

        response = generator(prompt)

        return response

    def function_calls(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Supports calling external tools during chat, typically involving tool-use planning logic.

        tool schema follows openAI's example Format:
        {
          "type": "function",
          "function": {
            "name": "get_current_temperature",
            "description": "...",
            "parameters": {
              "type": "object",
              "properties": {
                "location": {
                  "type": "string",
                  "description": "..."
                },
                "unit": {
                  "type": "string",
                  "enum": ["Celsius", "Fahrenheit"],
                  "description": "..."
                }
              },
              "required": ["location", "unit"]
            }
          }

        """

        # choose tool from available tools:
        if len(tools) == 0:
            raise ValueError(
                "There is no tool provided for function calls. Please check your tools."
            )
        elif len(tools) == 1:
            chosen_tool = tools[0]
        else:
            tools_name = _get_tool_name(tools)
            chosen_tool_name = self.choice(messages, tools_name)
            chosen_tool = _get_tool_by_name(tools, chosen_tool_name)

        response = self.structured_output(
            messages, chosen_tool["function"]["parameters"]
        )

        return response

    def choice(self, messages: List[ChatMessage], choices: List[str]):
        """
        Process messages and generate a choice from the given options.

        Args:
            messages (List[ChatMessage]): List of chat messages to process.
            choices (List[str]): List of choice options.

        Returns:
            str: The selected choice.
        """
        messages = _convert_messages_format(messages)

        prompt = self._convert_to_prompt(messages)

        generator = outlines_choice(self.operator, choices)
        choice = generator(prompt)

        return choice

    def _convert_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """
        Convert a list of message dictionaries into a single string prompt.

        Args:
            messages (list[dict[str, Any]]): A list of message dictionaries, where each dictionary
            contains at least a "content" key representing the message text.

        Returns:
            str: The combined prompt string generated from the input messages.

        Raises:
            ValueError: If the list of messages is empty.
        """
        if len(messages) == 0:
            raise ValueError("please send correct messages before calling llm")

        if len(messages) == 1:
            return messages[0]["content"]
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return prompt
