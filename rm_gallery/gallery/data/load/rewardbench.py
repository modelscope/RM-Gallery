import hashlib
from typing import Any, Dict

from loguru import logger

from rm_gallery.core.data.load.base import DataConverter, DataConverterRegistry
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataConverterRegistry.register("rewardbench")
class RewardBenchConverter(DataConverter):
    """
    Unified converter for conversation data with prompt, chosen and rejected responses
    """

    def convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_info: Dict[str, Any]
    ) -> DataSample:
        """Convert conversation data to DataSample format"""
        # generate unique id
        content = str(data_dict.get("prompt", []))
        unique_id = hashlib.md5(content.encode()).hexdigest()

        # Create input from prompt
        data_input = self._create_conversation_input(data_dict)

        # Create outputs from chosen/rejected responses
        data_output = self._create_conversation_output(data_dict)

        try:
            # Build metadata based on source type
            metadata = {
                "raw_data": data_dict,
                "load_strategy": "RewardBenchConverter",
            }

            # Add source-specific metadata
            if source_info.get("load_type") == "local":
                metadata.update(
                    {
                        "source_file_path": source_info.get("source_file_path"),
                        "load_type": "local",
                    }
                )
            elif source_info.get("load_type") == "huggingface":
                metadata.update(
                    {
                        "dataset_name": source_info.get("dataset_name"),
                        "dataset_config": source_info.get("dataset_config"),
                        "split": source_info.get("split", "train"),
                        "load_type": "huggingface",
                    }
                )

            data_sample = DataSample(
                unique_id=unique_id,
                input=data_input,
                output=data_output,
                source="rewardbench",
                task_category="conversation",
                metadata=metadata,
            )

            return data_sample

        except Exception as e:
            logger.error(f"Error creating conversation DataSample: {str(e)}")
            return None

    def _create_conversation_input(
        self, data_dict: Dict[str, Any]
    ) -> list[ChatMessage]:
        """Create DataInput from conversation prompt"""
        history = []
        prompt = data_dict.get("prompt")

        # Convert single-turn conversation to list format
        if isinstance(prompt, dict):
            prompt = [prompt]

        if isinstance(prompt, list):
            for turn in prompt:
                if isinstance(turn, dict):
                    role = turn.get("role", "user")
                    content = turn.get("content", str(turn))
                    history.append(ChatMessage(role=role, content=content))
                else:
                    history.append(ChatMessage(role="user", content=str(turn)))
        elif isinstance(prompt, str):
            history.append(ChatMessage(role="user", content=prompt))

        return history

    def _create_conversation_output(
        self, data_dict: Dict[str, Any]
    ) -> list[DataOutput]:
        """Create DataOutput list from conversation responses"""
        outputs = []

        # Handle chosen response
        if "chosen" in data_dict:
            chosen_content = data_dict["chosen"]
            if isinstance(chosen_content, list):
                # Multi-turn chosen response
                for turn in chosen_content:
                    if isinstance(turn, dict):
                        content = turn.get("content", str(turn))
                    else:
                        content = str(turn)
                    outputs.append(
                        DataOutput(
                            answer=Step(
                                role="assistant",
                                content=content,
                                label={"preference": "chosen"},
                            ),
                        )
                    )
            else:
                outputs.append(
                    DataOutput(
                        answer=Step(
                            role="assistant",
                            content=str(chosen_content),
                            label={"preference": "chosen"},
                        ),
                    )
                )

        # Handle rejected response
        if "rejected" in data_dict:
            rejected_content = data_dict["rejected"]
            if isinstance(rejected_content, list):
                # Multi-turn rejected response
                for turn in rejected_content:
                    if isinstance(turn, dict):
                        content = turn.get("content", str(turn))
                    else:
                        content = str(turn)
                    outputs.append(
                        DataOutput(
                            answer=Step(
                                role="assistant",
                                content=content,
                                label={"preference": "rejected"},
                            ),
                        )
                    )
            else:
                outputs.append(
                    DataOutput(
                        answer=Step(
                            role="assistant",
                            content=str(rejected_content),
                            label={"preference": "rejected"},
                        ),
                    )
                )

        return outputs
