# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os

# Import base dataset
import sys
from typing import List

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .base_dataset import BaseChatRLDataset
except ImportError:
    try:
        from base_dataset import BaseChatRLDataset
    except ImportError:
        # If still failing, try importing from current directory
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "base_dataset", os.path.join(current_dir, "base_dataset.py")
        )
        base_dataset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base_dataset_module)
        BaseChatRLDataset = base_dataset_module.BaseChatRLDataset


class DataKeys:
    """Data field names configuration"""

    # Data field keys
    CHOSEN = "chosen"
    REJECTED = "rejected"
    CONVERSATIONS = "conversations"
    SOURCE = "source"

    # Default values
    DEFAULT_SOURCE = "alignment"


class AlignmentChatRLDataset(BaseChatRLDataset):
    """Alignment Chat RL Dataset

    Specialized for handling alignment data with chosen/rejected format
    """

    def __init__(self, data_files, tokenizer, config, processor=None):
        super().__init__(data_files, tokenizer, config, processor)
        print("Using Alignment mode")

    def _build_messages(self, example: dict) -> List[dict]:
        """Build chat messages from sample - Alignment mode"""
        messages = []

        # Priority: build messages from x field
        if "x" in example and example["x"] is not None:
            x_data = example["x"]
            # Handle numpy array format
            if hasattr(x_data, "tolist"):
                x_data = x_data.tolist()
            elif not isinstance(x_data, (list, tuple)):
                x_data = [x_data]

            for msg in x_data:
                if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        # If x field is empty, build messages from chosen field (excluding last assistant reply)
        elif DataKeys.CHOSEN in example and example[DataKeys.CHOSEN]:
            chosen_messages = example[DataKeys.CHOSEN]
            # Handle numpy array format
            if hasattr(chosen_messages, "tolist"):
                chosen_messages = chosen_messages.tolist()

            if isinstance(chosen_messages, (list, tuple)):
                for msg in chosen_messages:
                    if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                        # Only add user messages, not assistant messages (as they are the prediction target)
                        if msg.get("role") == "user":
                            messages.append(
                                {"role": msg["role"], "content": msg["content"]}
                            )

        # If still no messages found, try building from rejected field
        elif DataKeys.REJECTED in example and example[DataKeys.REJECTED]:
            rejected_messages = example[DataKeys.REJECTED]
            # Handle numpy array format
            if hasattr(rejected_messages, "tolist"):
                rejected_messages = rejected_messages.tolist()

            if isinstance(rejected_messages, (list, tuple)):
                for msg in rejected_messages:
                    if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                        if msg.get("role") == "user":
                            messages.append(
                                {"role": msg["role"], "content": msg["content"]}
                            )

        # If still no messages, create a default user message
        if len(messages) == 0:
            messages = [{"role": "user", "content": "Please help complete this task."}]

        return messages

    def _format_template(self, messages: List[dict], example: dict) -> str:
        """Format alignment template"""
        return messages

    def _extract_ground_truth(self, row_dict):
        """Extract alignment ground truth

        For alignment data, chosen can serve as a "better" reference
        """
        try:
            ground_truth_info = {}

            # Save both chosen and rejected to ground_truth for reward function use
            chosen_key = DataKeys.CHOSEN
            rejected_key = DataKeys.REJECTED
            source_key = DataKeys.SOURCE

            if chosen_key in row_dict and row_dict[chosen_key] is not None:
                chosen_data = row_dict[chosen_key]
                # Handle numpy array format
                if hasattr(chosen_data, "tolist"):
                    chosen_data = chosen_data.tolist()
                ground_truth_info[chosen_key] = chosen_data

            if rejected_key in row_dict and row_dict[rejected_key] is not None:
                rejected_data = row_dict[rejected_key]
                # Handle numpy array format
                if hasattr(rejected_data, "tolist"):
                    rejected_data = rejected_data.tolist()
                ground_truth_info[rejected_key] = rejected_data

            if source_key in row_dict:
                ground_truth_info[source_key] = row_dict[source_key]

            return ground_truth_info

        except Exception as e:
            print(f"Error extracting ground truth: {e}")
            return {}

    def __getitem__(self, item):
        """Get an item from the dataset"""
        row_dict = dict(self.dataframe[item])
        messages = self._build_messages(row_dict)

        # Format prompt
        raw_prompt_messages = self._format_template(messages, row_dict)

        # Apply chat template
        raw_prompt = self.tokenizer.apply_chat_template(
            raw_prompt_messages, add_generation_prompt=True, tokenize=False
        )

        # Tokenize
        model_inputs = self.tokenizer(
            raw_prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # Post-process
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask)

        # Prepare raw prompt IDs
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} exceeds {self.max_prompt_length}"
                )

        # Build x field (for passing to reward function)
        x_messages = []

        # Build complete conversation context from raw data
        chosen_key = DataKeys.CHOSEN
        if chosen_key in row_dict and row_dict[chosen_key]:
            # Use chosen as base for conversation context
            chosen_messages = row_dict[chosen_key]
            # Handle numpy array format
            if hasattr(chosen_messages, "tolist"):
                chosen_messages = chosen_messages.tolist()

            if isinstance(chosen_messages, (list, tuple)):
                for msg in chosen_messages:
                    if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                        x_messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )

        # If no messages obtained from chosen, use our built messages
        if not x_messages:
            x_messages = messages

        # Build result
        result = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "index": row_dict.get("index", item),
            "extra_info": copy.deepcopy(row_dict),
            "reward_model": {"ground_truth": self._extract_ground_truth(row_dict)},
            "data_source": row_dict.get("source", "alignment"),
        }

        # Add x field with conversation context
        result["extra_info"]["x"] = x_messages

        if self.return_raw_chat:
            result["raw_prompt"] = messages

        return result
