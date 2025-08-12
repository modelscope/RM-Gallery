"""
Bradley-Terry Dataset for Reward Model Training
- Loads preference data from parquet files
- Each sample contains chosen and rejected responses
- Returns data in format suitable for Bradley-Terry loss
"""

from typing import Any, Dict, List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local


class BTDataset(Dataset):
    """
    Bradley-Terry Dataset for preference learning

    Expected data format in parquet:
    - chosen: text of chosen response (string)
    - rejected: text of rejected response (string)

    The dataset directly reads the chosen and rejected columns as text strings,
    without any complex message format conversion.
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        self.max_length = config.get("max_length", 4096)
        self.truncation = config.get("truncation", "left")
        self.use_shm = config.get("use_shm", False)

        # Keys for data columns
        self.chosen_key = config.get("chosen_key", "chosen")
        self.rejected_key = config.get("rejected_key", "rejected")

        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        """Download parquet files to local if needed"""
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_process(self):
        """Read and concatenate all parquet files"""
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)

        self.dataframe = pd.concat(dataframes, ignore_index=True)

        # Directly extract chosen and rejected text fields
        self.chosen_texts = self.dataframe[self.chosen_key].tolist()
        self.rejected_texts = self.dataframe[self.rejected_key].tolist()

        # Ensure all entries are strings
        self.chosen_texts = [str(text) for text in self.chosen_texts]
        self.rejected_texts = [str(text) for text in self.rejected_texts]

        print(
            f"Loaded {len(self.chosen_texts)} preference pairs from {len(self.parquet_files)} files"
        )

    def __len__(self):
        return len(self.chosen_texts)

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text and handle truncation/padding to fixed length"""
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        sequence_length = input_ids.shape[0]

        # Handle sequence length like SFT dataset
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )
            padded_input_ids = (
                torch.ones(
                    size=(self.max_length - sequence_length,), dtype=input_ids.dtype
                )
                * pad_token_id
            )
            padded_attention_mask = torch.zeros(
                size=(self.max_length - sequence_length,), dtype=attention_mask.dtype
            )

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # Keep the end of the conversation (including conclusion)
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(
                    f"Sequence length {sequence_length} > max_length {self.max_length}"
                )

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __getitem__(self, item) -> Dict[str, Any]:
        """
        Get a preference pair

        Returns:
            dict with keys:
            - input_ids_j: chosen response tokens
            - attention_mask_j: chosen response attention mask
            - input_ids_k: rejected response tokens
            - attention_mask_k: rejected response attention mask
        """
        chosen_text = self.chosen_texts[item]
        rejected_text = self.rejected_texts[item]

        # Tokenize both responses
        chosen_tokens = self._tokenize_text(chosen_text)
        rejected_tokens = self._tokenize_text(rejected_text)

        return {
            "input_ids_j": chosen_tokens["input_ids"],
            "attention_mask_j": chosen_tokens["attention_mask"],
            "input_ids_k": rejected_tokens["input_ids"],
            "attention_mask_k": rejected_tokens["attention_mask"],
        }
