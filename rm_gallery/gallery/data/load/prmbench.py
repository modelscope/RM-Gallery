import hashlib
from typing import Any, ClassVar, Dict

from loguru import logger

from rm_gallery.core.data.load.base import DataConverter, DataConverterRegistry
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataConverterRegistry.register("prmbench")
class PRMBenchConverter(DataConverter):
    """
    Unified converter for Process Reward Model (PRM) data
    Handles mathematical reasoning data with step-wise processes
    """

    # define as class attribute instead of instance attribute
    DIMENSION_CLASSIFICATION_MAPPING: ClassVar[Dict[str, str]] = {
        "confidence": "confidence",
        "*": None,  # wildcard, means no filtering
    }

    def convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_info: Dict[str, Any]
    ) -> DataSample:
        """Convert PRM data to DataSample format

        Expected input format:
        {
            "original_question": "...",
            "modified_question": "...",
            "original_process": ["step1", "step2", ...],
            "modified_process": ["step1", "step2", ...],
            "modified_steps": [5, 6],
            "error_steps": [5, 6],
            "reason": "...",
            "idx": "...",
            "question": "...",
            "classification": "confidence"
        }
        """

        # Generate unique id from idx or question
        unique_id = data_dict.get(
            "idx", hashlib.md5(str(data_dict.get("question", "")).encode()).hexdigest()
        )

        try:
            # Create input from question
            data_input = self._create_prm_input(data_dict)

            # Create outputs from processes
            data_output = self._create_prm_output(data_dict)

            # Build metadata based on source type
            metadata = {
                "classification": data_dict.get("classification"),
                "modified_steps": data_dict.get("modified_steps", []),
                "error_steps": data_dict.get("error_steps", []),
                "reason": data_dict.get("reason"),
                "idx": data_dict.get("idx"),
                "original_process_length": len(data_dict.get("original_process", [])),
                "modified_process_length": len(data_dict.get("modified_process", [])),
                "load_strategy": "PRMBenchConverter",
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

            # Create DataSample object
            data_sample = DataSample(
                unique_id=str(unique_id),
                input=data_input,
                output=data_output,
                source="prmbench",
                task_category=data_dict.get("classification", "reasoning"),
                metadata=metadata,
            )

            return data_sample

        except Exception as e:
            logger.error(f"Error creating DataSample from PRM data: {str(e)}")
            return None

    def _create_prm_input(self, data_dict: Dict[str, Any]) -> list[ChatMessage]:
        """Create DataInput from PRM question"""
        question = data_dict.get("question") or data_dict.get("original_question", "")
        return [ChatMessage(role="user", content=question)]

    def _create_prm_output(self, data_dict: Dict[str, Any]) -> list[DataOutput]:
        """Create DataOutput list from PRM processes"""
        outputs = []

        # Original process output
        if "original_process" in data_dict:
            original_steps = []
            for i, step_content in enumerate(data_dict["original_process"]):
                step = Step(
                    role="assistant",
                    content=step_content,
                    label={"correctness": "correct", "step_idx": i + 1},
                )
                original_steps.append(step)

            outputs.append(
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content="\n".join(data_dict["original_process"]),
                        label={"process_type": "original_correct"},
                    ),
                    steps=original_steps,
                )
            )

        # Modified process output (with errors)
        if "modified_process" in data_dict:
            modified_steps = []
            error_steps = set(data_dict.get("error_steps", []))

            for i, step_content in enumerate(data_dict["modified_process"]):
                step_idx = i + 1
                is_correct = step_idx not in error_steps

                step = Step(
                    role="assistant",
                    content=step_content,
                    label={
                        "correctness": "correct" if is_correct else "error",
                        "step_idx": step_idx,
                    },
                )
                modified_steps.append(step)

            # Calculate correctness score based on error ratio
            total_steps = len(data_dict["modified_process"])
            error_count = len(error_steps)

            outputs.append(
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content="\n".join(data_dict["modified_process"]),
                        label={
                            "process_type": f"Modified process with {error_count}/{total_steps} error steps"
                        },
                    ),
                    steps=modified_steps,
                )
            )

        return outputs
