import hashlib
from typing import Any, Dict, List

from loguru import logger

from rm_gallery.core.data.load import DataLoadStrategyRegistry, FileDataLoadStrategy
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataLoadStrategyRegistry.register("local", "*", "*")
class NormalDataLoadStrategy(FileDataLoadStrategy):
    """
    Generic strategy for loading data and storing it directly in metadata
    """

    def _convert_to_data_sample(self, data_dict: Dict[str, Any]) -> DataSample:
        """Convert raw data to DataSample format with data stored in metadata"""
        # Generate unique id from data content
        content = str(data_dict)
        unique_id = hashlib.md5(content.encode()).hexdigest()

        try:
            # Create minimal input - use any text field available or fallback
            data_input = self._create_generic_input(data_dict)

            # Create minimal output - use any response field available or empty
            data_output = self._create_generic_output(data_dict)

            # Store all original data in metadata
            data_sample = DataSample(
                unique_id=unique_id,
                input=data_input,
                output=data_output,
                source=self.config.get("source", "generic"),
                domain=self.config.get("dimension", "generic"),
                metadata={
                    "raw_data": data_dict,  # Store all original data here
                    "load_strategy": "NormalDataLoadStrategy",
                    "data_keys": list(data_dict.keys())
                    if isinstance(data_dict, dict)
                    else [],
                },
            )

            return data_sample

        except Exception as e:
            logger.error(f"Error creating generic DataSample: {str(e)}")
            return None

    def _create_generic_input(self, data_dict: Dict[str, Any]) -> List[ChatMessage]:
        """Create generic input from any available text field"""
        # Try common input field names
        input_fields = ["prompt", "question", "input", "text", "query", "instruction"]

        for field in input_fields:
            if field in data_dict:
                content = data_dict[field]
                if isinstance(content, str):
                    return [ChatMessage(role="user", content=content)]
                elif isinstance(content, list):
                    # Handle list of messages
                    messages = []
                    for item in content:
                        if (
                            isinstance(item, dict)
                            and "role" in item
                            and "content" in item
                        ):
                            messages.append(
                                ChatMessage(role=item["role"], content=item["content"])
                            )
                        else:
                            messages.append(ChatMessage(role="user", content=str(item)))
                    return messages
                else:
                    return [ChatMessage(role="user", content=str(content))]

        # Fallback: use first string value found or empty
        for key, value in data_dict.items():
            if isinstance(value, str) and value.strip():
                return [ChatMessage(role="user", content=value)]

        return [ChatMessage(role="user", content="")]

    def _create_generic_output(self, data_dict: Dict[str, Any]) -> List[DataOutput]:
        """Create generic output from any available response field"""
        outputs = []

        # Try common output field names
        output_fields = [
            "response",
            "answer",
            "output",
            "completion",
            "chosen",
            "rejected",
        ]

        for field in output_fields:
            if field in data_dict:
                content = data_dict[field]
                if isinstance(content, str):
                    outputs.append(
                        DataOutput(
                            answer=Step(
                                role="assistant",
                                content=content,
                                label={"field": field},
                            )
                        )
                    )
                elif isinstance(content, list):
                    for item in content:
                        outputs.append(
                            DataOutput(
                                answer=Step(
                                    role="assistant",
                                    content=str(item),
                                    label={"field": field},
                                )
                            )
                        )
                else:
                    outputs.append(
                        DataOutput(
                            answer=Step(
                                role="assistant",
                                content=str(content),
                                label={"field": field},
                            )
                        )
                    )

        # If no outputs found, create empty output
        if not outputs:
            outputs.append(
                DataOutput(
                    answer=Step(role="assistant", content="", label={"field": "empty"})
                )
            )

        return outputs
