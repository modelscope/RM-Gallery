import hashlib
from typing import Any, Dict, List

from loguru import logger

from rm_gallery.core.data.load import DataLoadStrategyRegistry, FileDataLoadStrategy
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataLoadStrategyRegistry.register("local", "chatmessage", "*")
class ChatMessageDataLoadStrategy(FileDataLoadStrategy):
    """
    Strategy for loading chat message data
    """

    def _convert_to_data_sample(self, data_dict: Dict[str, Any]) -> DataSample:
        """Convert chat message data to DataSample format"""
        # generate unique id
        content = str(data_dict)
        unique_id = hashlib.md5(content.encode()).hexdigest()

        try:
            # Create input from messages
            data_input = self._create_chat_input(data_dict)

            # Create output from response
            data_output = self._create_chat_output(data_dict)

            data_sample = DataSample(
                unique_id=unique_id,
                input=data_input,
                output=data_output,
                source="chatmessage",
                domain=data_dict.get("domain", "chat"),
                metadata={
                    "raw_data": data_dict,
                    "load_strategy": "ChatMessageDataLoadStrategy",
                },
            )

            return data_sample

        except Exception as e:
            logger.error(f"Error creating chat DataSample: {str(e)}")
            return None

    def _create_chat_input(self, data_dict: Dict[str, Any]) -> List[ChatMessage]:
        """Create DataInput from chat messages"""
        history = []

        # Handle messages field
        if "messages" in data_dict:
            messages = data_dict["messages"]
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", str(msg))
                        history.append(ChatMessage(role=role, content=content))
                    else:
                        history.append(ChatMessage(role="user", content=str(msg)))

        # Handle conversation field
        elif "conversation" in data_dict:
            conversation = data_dict["conversation"]
            if isinstance(conversation, list):
                for turn in conversation:
                    if isinstance(turn, dict):
                        role = turn.get("role", "user")
                        content = turn.get("content", str(turn))
                        history.append(ChatMessage(role=role, content=content))

        # Handle simple text field
        elif "text" in data_dict:
            history.append(ChatMessage(role="user", content=str(data_dict["text"])))

        return history

    def _create_chat_output(self, data_dict: Dict[str, Any]) -> List[DataOutput]:
        """Create DataOutput from chat response"""
        outputs = []

        # Handle response field
        if "response" in data_dict:
            outputs.append(
                DataOutput(
                    answer=Step(role="assistant", content=str(data_dict["response"]))
                )
            )

        # Handle assistant message in messages
        elif "messages" in data_dict:
            messages = data_dict["messages"]
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        outputs.append(
                            DataOutput(
                                answer=Step(
                                    role="assistant",
                                    content=str(msg.get("content", "")),
                                )
                            )
                        )

        return outputs
