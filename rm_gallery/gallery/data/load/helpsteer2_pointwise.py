import hashlib
from typing import Any, Dict, List, Union

from loguru import logger

from rm_gallery.core.data.load.base import DataConverter, DataConverterRegistry
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataConverterRegistry.register("helpsteer2_pointwise")
class HelpSteer2PointwiseConverter(DataConverter):
    """
    Unified converter for HelpSteer2 data format
    Can handle data from both local files and HuggingFace Hub
    """

    def convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_info: Dict[str, Any]
    ) -> Union[DataSample, List[DataSample]]:
        """Convert HelpSteer2 data to DataSample format"""
        # Generate unique id
        content = str(data_dict)
        unique_id = hashlib.md5(content.encode()).hexdigest()

        try:
            # Create input from prompt
            data_input = [ChatMessage(role="user", content=data_dict["prompt"])]

            # Extract evaluation metrics for label
            label = {
                "helpfulness": data_dict.get("helpfulness"),
                "correctness": data_dict.get("correctness"),
                "coherence": data_dict.get("coherence"),
                "complexity": data_dict.get("complexity"),
                "verbosity": data_dict.get("verbosity"),
            }

            # Create output from response
            data_output = [
                DataOutput(
                    answer=Step(
                        role="assistant", content=data_dict["response"], label=label
                    )
                )
            ]

            # Build metadata based on source type
            metadata = {
                "raw_data": data_dict,
                "load_strategy": "HelpSteer2Converter",
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
                        "dataset_name": source_info.get(
                            "dataset_name", "nvidia/HelpSteer2"
                        ),
                        "dataset_config": source_info.get("dataset_config"),
                        "split": source_info.get("split", "train"),
                        "load_type": "huggingface",
                    }
                )

            data_sample = DataSample(
                unique_id=unique_id,
                input=data_input,
                output=data_output,
                source="helpsteer2",
                task_category="chat",
                metadata=metadata,
            )

            return [data_sample]

        except Exception as e:
            logger.error(f"Error creating HelpSteer2 DataSample: {str(e)}")
            return None
