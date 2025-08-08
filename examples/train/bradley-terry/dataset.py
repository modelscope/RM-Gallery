import json
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset
from loguru import logger
from transformers import AutoTokenizer

from rm_gallery.core.train.dataset import BaseBradleyTerryTrainDataset
from rm_gallery.core.utils.file import load_parquet


class HelpSteer3DataProcessor(BaseBradleyTerryTrainDataset):
    """Data processor for HelpSteer3 dataset format."""

    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)

    def _convert_to_preference_format(
        self, data_item: Dict[str, Any]
    ) -> Dict[str, List]:
        """
        Convert the new data format to preference format for training.

        Args:
            data_item: Single data item with 'input' and 'output' fields

        Returns:
            Dictionary with 'chosen' and 'rejected' conversation lists
        """
        input_conversation = data_item["input"]
        outputs = data_item["output"]

        # Handle string JSON parsing if needed
        if isinstance(outputs, str):
            outputs = json.loads(outputs)
        if isinstance(input_conversation, str):
            input_conversation = json.loads(input_conversation)

        # Extract responses
        response_a = outputs[0]["answer"]
        response_b = outputs[1]["answer"]

        try:
            # Determine preference
            is_a_preferred = response_a["label"]["is_preferred"]
            is_b_preferred = response_b["label"]["is_preferred"]
        except KeyError:
            is_a_preferred = response_a["label"].get("is_preferred", True)
            is_b_preferred = response_b["label"].get("is_preferred", False)

        if is_a_preferred and not is_b_preferred:
            chosen_response = response_a
            rejected_response = response_b
        elif is_b_preferred and not is_a_preferred:
            chosen_response = response_b
            rejected_response = response_a
        else:
            # Use preference scores as fallback
            score_a = outputs[0]["answer"]["label"].get("preference_score", 0)
            score_b = outputs[1]["answer"]["label"].get("preference_score", 0)

            if score_a > score_b:
                chosen_response = response_a
                rejected_response = response_b
            else:
                chosen_response = response_b
                rejected_response = response_a

        # Create conversation format
        chosen_conversation = input_conversation + [chosen_response]
        rejected_conversation = input_conversation + [rejected_response]

        return {"chosen": chosen_conversation, "rejected": rejected_conversation}

    def _build_dataset(self, train_path: str, eval_path: Optional[str] = None) -> tuple:
        """
        Build training and evaluation datasets from data files.

        Args:
            train_path: Path to training data file
            eval_path: Path to evaluation data file (optional)

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """

        def tokenize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
            """Tokenize a single sample with chosen and rejected responses."""
            # Apply chat template
            sample["positive"] = self.tokenizer.apply_chat_template(
                sample["chosen"], tokenize=False, add_generation_prompt=False
            )
            sample["negative"] = self.tokenizer.apply_chat_template(
                sample["rejected"], tokenize=False, add_generation_prompt=False
            )

            # Tokenize responses
            tokenized_pos = self.tokenizer(sample["positive"], truncation=True)
            tokenized_neg = self.tokenizer(sample["negative"], truncation=True)

            # Store tokenized data
            sample["input_ids_j"] = tokenized_pos["input_ids"]
            sample["attention_mask_j"] = tokenized_pos["attention_mask"]
            sample["input_ids_k"] = tokenized_neg["input_ids"]
            sample["attention_mask_k"] = tokenized_neg["attention_mask"]

            return sample

        # Load and process training data
        raw_train_data = load_parquet(train_path)

        train_preference_data = []
        for i, item in enumerate(raw_train_data):
            try:
                converted = self._convert_to_preference_format(item)
                train_preference_data.append(converted)
            except Exception as e:
                logger.warning(f"Error converting training sample {i}: {e}")
                continue

        # Create training dataset
        train_dataset = Dataset.from_list(train_preference_data)
        num_proc = os.cpu_count() or 1  # Number of processes for parallel processing
        train_dataset = train_dataset.map(tokenize_sample, num_proc=num_proc)

        # Create evaluation dataset
        if eval_path is None or eval_path == train_path:
            # Use subset of training data for evaluation
            eval_size = min(500, len(train_dataset))
            eval_dataset = train_dataset.select(range(eval_size))
        else:
            raw_eval_data = load_parquet(eval_path)

            eval_preference_data = []
            for i, item in enumerate(raw_eval_data):
                try:
                    converted = self._convert_to_preference_format(item)
                    eval_preference_data.append(converted)
                except Exception as e:
                    logger.warning(f"Error converting eval sample {i}: {e}")

                    continue

            eval_dataset = Dataset.from_list(eval_preference_data)
            eval_dataset = eval_dataset.map(tokenize_sample, num_proc=num_proc)

        return train_dataset, eval_dataset
