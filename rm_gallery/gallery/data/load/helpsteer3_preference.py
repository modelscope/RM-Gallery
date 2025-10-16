import hashlib
from typing import Any, Dict, List, Union

from loguru import logger

from rm_gallery.core.data.load.base import DataConverter, DataConverterRegistry
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataConverterRegistry.register("helpsteer3_preference")
class HelpSteer3PreferenceConverter(DataConverter):
    """
    Converter for HelpSteer3 preference data format
    Handles multi-turn conversations with preference comparisons between two responses
    """

    def convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_info: Dict[str, Any]
    ) -> Union[DataSample, List[DataSample]]:
        """Convert HelpSteer3 preference data to DataSample format"""
        # Generate unique id
        content = str(data_dict)
        unique_id = hashlib.md5(content.encode()).hexdigest()

        try:
            # Create input from context (multi-turn conversation)
            data_input = self._create_conversation_input(data_dict)

            # Determine preference based on overall_preference
            # HelpSteer3 preference scoring:
            # -3: Response 1 is much better than Response 2
            # -2: Response 1 is better than Response 2
            # -1: Response 1 is slightly better than Response 2
            #  0: Response 1 is about the same as Response 2
            #  1: Response 2 is slightly better than Response 1
            #  2: Response 2 is better than Response 1
            #  3: Response 2 is much better than Response 1
            overall_preference = data_dict.get("overall_preference", 0)
            if overall_preference > 0:
                # Positive values: response2 is better
                preferred_response = "response2"
            elif overall_preference < 0:
                # Negative values: response1 is better
                preferred_response = "response1"
            else:
                # Zero: responses are about the same (tie)
                preferred_response = "tie"

            data_samples = []

            # Create first sample: response_A = response1, response_B = response2
            sample1_id = hashlib.md5(f"{str(data_dict)}_sample1".encode()).hexdigest()

            # Determine preferred for first sample
            if preferred_response == "response1":
                preferred_1 = "A"  # response_A (response1) is preferred
            elif preferred_response == "response2":
                preferred_1 = "B"  # response_B (response2) is preferred
            else:
                preferred_1 = "tie"

            # Create outputs for first sample
            output_1 = [
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content=data_dict["response1"],
                        label={
                            "response_type": "A",
                            "is_preferred": preferred_1 == "A",
                            "preference_score": overall_preference,
                            "original_response": "response1",
                        },
                    )
                ),
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content=data_dict["response2"],
                        label={
                            "response_type": "B",
                            "is_preferred": preferred_1 == "B",
                            "preference_score": overall_preference,
                            "original_response": "response2",
                        },
                    )
                ),
            ]

            # Build metadata for first sample
            metadata_1 = {
                "raw_data": data_dict,
                "load_strategy": "HelpSteer3PreferenceConverter",
                "domain": data_dict.get("domain"),
                "language": data_dict.get("language"),
                "response_A": data_dict["response1"],
                "response_B": data_dict["response2"],
                "preferred": preferred_1,
                "overall_preference": overall_preference,
                "individual_preference": data_dict.get("individual_preference", []),
                "sample_type": "original_order",
            }

            # Add source-specific metadata
            if source_info.get("load_type") == "local":
                metadata_1.update(
                    {
                        "source_file_path": source_info.get("source_file_path"),
                        "load_type": "local",
                    }
                )
            elif source_info.get("load_type") == "huggingface":
                metadata_1.update(
                    {
                        "dataset_name": source_info.get("dataset_name", "helpsteer3"),
                        "dataset_config": source_info.get("dataset_config"),
                        "split": source_info.get("split", "train"),
                        "load_type": "huggingface",
                    }
                )

            sample_1 = DataSample(
                unique_id=sample1_id,
                input=data_input,
                output=output_1,
                source="helpsteer3_preference",
                task_category="chat_preference",
                metadata=metadata_1,
            )
            data_samples.append(sample_1)

            # # Create second sample: response_A = response2, response_B = response1 (swapped)
            # sample2_id = hashlib.md5(f"{str(data_dict)}_sample2".encode()).hexdigest()

            # # Determine preferred for second sample (swapped)
            # if preferred_response == "response1":
            #     preferred_2 = "B"  # response_B (response1) is preferred
            # elif preferred_response == "response2":
            #     preferred_2 = "A"  # response_A (response2) is preferred
            # else:
            #     preferred_2 = "tie"

            # # Create outputs for second sample (swapped)
            # output_2 = [
            #     DataOutput(
            #         answer=Step(
            #             role="assistant",
            #             content=data_dict["response2"],
            #             label={
            #                 "response_type": "A",
            #                 "is_preferred": preferred_2 == "A",
            #                 "preference_score": overall_preference,
            #                 "original_response": "response2"
            #             },
            #         )
            #     ),
            #     DataOutput(
            #         answer=Step(
            #             role="assistant",
            #             content=data_dict["response1"],
            #             label={
            #                 "response_type": "B",
            #                 "is_preferred": preferred_2 == "B",
            #                 "preference_score": overall_preference,
            #                 "original_response": "response1"
            #             },
            #         )
            #     ),
            # ]

            # # Build metadata for second sample
            # metadata_2 = {
            #     "raw_data": data_dict,
            #     "load_strategy": "HelpSteer3PreferenceConverter",
            #     "domain": data_dict.get("domain"),
            #     "language": data_dict.get("language"),
            #     "response_A": data_dict["response2"],
            #     "response_B": data_dict["response1"],
            #     "preferred": preferred_2,
            #     "overall_preference": overall_preference,
            #     "individual_preference": data_dict.get("individual_preference", []),
            #     "sample_type": "swapped_order",
            # }

            # # Add source-specific metadata
            # if source_info.get("load_type") == "local":
            #     metadata_2.update(
            #         {
            #             "source_file_path": source_info.get("source_file_path"),
            #             "load_type": "local",
            #         }
            #     )
            # elif source_info.get("load_type") == "huggingface":
            #     metadata_2.update(
            #         {
            #             "dataset_name": source_info.get(
            #                 "dataset_name", "helpsteer3"
            #             ),
            #             "dataset_config": source_info.get("dataset_config"),
            #             "split": source_info.get("split", "train"),
            #             "load_type": "huggingface",
            #         }
            #     )

            # sample_2 = DataSample(
            #     unique_id=sample2_id,
            #     input=data_input,
            #     output=output_2,
            #     source="helpsteer3_preference",
            #     task_category="chat_preference",
            #     metadata=metadata_2,
            # )
            # data_samples.append(sample_2)

            return data_samples

        except Exception as e:
            logger.error(f"Error creating HelpSteer3 Preference DataSample: {str(e)}")
            return None

    def _create_conversation_input(
        self, data_dict: Dict[str, Any]
    ) -> List[ChatMessage]:
        """Create DataInput from context (multi-turn conversation)"""
        context = data_dict.get("context", [])
        if not isinstance(context, list):
            # Fallback for single message
            return [ChatMessage(role="user", content=str(context))]

        history = []
        for message in context:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                history.append(ChatMessage(role=role, content=content))
            else:
                # Fallback for non-dict messages
                history.append(ChatMessage(role="user", content=str(message)))

        return history
