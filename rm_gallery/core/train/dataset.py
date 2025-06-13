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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import datasets
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask


class BaseTrainDataset(Dataset, ABC):
    """Base class for chat reinforcement learning datasets with common functionality"""

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor=None,  # keep for backward compatibility, but not used
    ):
        # initialize basic attributes
        self.data_files = self._normalize_data_files(data_files)
        self.original_data_files = copy.deepcopy(self.data_files)
        self.tokenizer = tokenizer
        self.config = config

        # load config settings
        self._load_config()

        # load and process data
        self._load_dataset()

    def _normalize_data_files(self, data_files):
        """Convert data files to list format"""
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]
        return copy.deepcopy(data_files)

    def _load_config(self):
        """Load config parameters - can be overridden by subclasses"""
        self.cache_dir = os.path.expanduser(
            self.config.get("cache_dir", "~/.cache/verl/rlhf")
        )
        self.prompt_key = self.config.get("prompt_key", "prompt")
        self.max_prompt_length = self.config.get("max_prompt_length", 1024)
        self.return_raw_chat = self.config.get("return_raw_chat", False)
        self.truncation = self.config.get("truncation", "error")
        self.filter_overlong_prompts = self.config.get("filter_overlong_prompts", True)
        self.num_workers = min(
            self.config.get(
                "filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)
            ),
            os.cpu_count(),
        )
        self.serialize_dataset = False

    def _download_files(self):
        """Download files to local cache"""
        from verl.utils.fs import copy_to_local

        for i, file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=file, cache_dir=self.cache_dir)

    def _load_dataset(self):
        """Load and process dataset"""
        self._download_files()

        # Load parquet files
        dataframes = []
        for file in self.data_files:
            df = datasets.load_dataset("parquet", data_files=file)["train"]
            dataframes.append(df)

        self.dataframe = datasets.concatenate_datasets(dataframes)
        print(f"dataset length: {len(self.dataframe)}")

        # Filter overlong prompts if enabled

        if self.filter_overlong_prompts:
            self._filter_long_prompts()

    def _filter_long_prompts(self):
        """Filter out overlong prompts using the same logic as runtime processing"""

        def is_prompt_valid(doc):
            try:
                # Use the same logic as runtime processing
                messages = self._build_messages(doc)
                raw_prompt = self._apply_chat_template(messages)
                raw_prompt_ids = self.tokenizer.encode(
                    raw_prompt, add_special_tokens=False
                )
                return len(raw_prompt_ids) <= self.max_prompt_length
            except Exception as e:
                print(f"Error processing sample during filtering: {e}")
                return False

        print("Starting prompt length filtering...")
        self.dataframe = self.dataframe.filter(
            is_prompt_valid,
            num_proc=self.num_workers,
            desc=f"filter out prompts longer than {self.max_prompt_length} tokens",
        )
        print(f"filtered dataset length: {len(self.dataframe)}")

    @abstractmethod
    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build chat messages from example - must be implemented by subclasses"""
        pass

    @abstractmethod
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template - can be overridden by subclasses"""
        pass

    @abstractmethod
    def _extract_ground_truth(self, row_dict: Dict[str, Any]) -> str:
        """Extract ground truth from row data - must be implemented by subclasses"""
        pass

    @abstractmethod
    def _get_data_source(self, row_dict: Dict[str, Any]) -> str:
        """Get data source - can be overridden by subclasses"""
        pass

    def __getitem__(self, item):
        """Get an item from dataset"""
        row_dict = dict(self.dataframe[item])
        messages = self._build_messages(row_dict)
        raw_prompt = self._apply_chat_template(messages)

        # Tokenize
        model_inputs = self.tokenizer(
            raw_prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # Postprocess
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Compute position ids
        position_ids = compute_position_id_with_mask(attention_mask)

        # Prepare raw prompt ids
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"prompt length {len(raw_prompt_ids)} exceeds {self.max_prompt_length}"
                )

        # Build result
        result = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "index": row_dict.get("index", item),
            "extra_info": copy.deepcopy(row_dict),
            "reward_model": {"ground_truth": self._extract_ground_truth(row_dict)},
            "data_source": self._get_data_source(row_dict),
        }

        if self.return_raw_chat:
            result["raw_prompt"] = messages

        return result

    def __len__(self):
        return len(self.dataframe)

    def resume_dataset_state(self):
        """Resume dataset state for checkpoint"""
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self.data_files = copy.deepcopy(self.original_data_files)
            self._load_dataset()
        else:
            print(
                "use old dataset loader checkpoint file, it is recommended to train from scratch"
            )

    def __getstate__(self):
        """Get state for serialization"""
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()
