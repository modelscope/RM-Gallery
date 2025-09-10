import hashlib
from typing import Any, Dict

from loguru import logger

from rm_gallery.core.data.load.base import DataConverter, DataConverterRegistry
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataConverterRegistry.register("judgebench")
class JudgeBenchConverter(DataConverter):
    """
    Data converter: converts JudgeBench data format to DataSample format

    JudgeBench data format:
    {
        "pair_id": "unique identifier",
        "original_id": "original question ID",
        "source": "data source",
        "question": "question content",
        "response_model": "model used to generate responses",
        "response_A": "response A",
        "response_B": "response B",
        "label": "ground truth label (A>B, B>A, A=B)"
    }
    """

    def convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_info: Dict[str, Any]
    ) -> DataSample:
        """Convert JudgeBench data to DataSample format"""

        # Use pair_id as unique identifier
        unique_id = str(data_dict.get("pair_id", ""))
        if not unique_id:
            # If no pair_id, generate hash from question content
            content = str(data_dict.get("question", ""))
            unique_id = hashlib.md5(content.encode()).hexdigest()

        # Create input: question as user message
        data_input = [
            ChatMessage(
                role="user",
                content=str(data_dict.get("question", "")),
                additional_kwargs={"judgebench": {"label": data_dict.get("label", "")}},
            )
        ]

        # Create output: two responses as assistant answers
        data_output = []

        # Add response_A
        if "response_A" in data_dict:
            data_output.append(
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content=str(data_dict["response_A"]),
                        label={"response_type": "A"},
                    )
                )
            )

        # Add response_B
        if "response_B" in data_dict:
            data_output.append(
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content=str(data_dict["response_B"]),
                        label={"response_type": "B"},
                    )
                )
            )

        try:
            # Build metadata
            metadata = {
                "raw_data": data_dict,
                "load_strategy": "JudgeBenchConverter",
                "pair_id": data_dict.get("pair_id"),
                "original_id": data_dict.get("original_id"),
                "source": data_dict.get("source"),
                "response_model": data_dict.get("response_model"),
                "label": data_dict.get("label"),
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
                source="judgebench",
                task_category="pairwise_comparison",
                metadata=metadata,
            )

            return data_sample

        except Exception as e:
            logger.error(f"Error creating JudgeBench DataSample: {str(e)}")
            return None
