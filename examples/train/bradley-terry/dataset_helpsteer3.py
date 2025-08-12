import json
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local


class HelpSteer3Dataset(Dataset):
    """
    Custom Bradley-Terry Dataset for complex preference data formats

    This dataset can handle nested JSON structures and complex preference formats
    like HelpSteer3, where preferences are embedded in complex structures.

    Expected data format in parquet:
    - input: List of conversation messages (JSON string or list)
    - output: List of response candidates with preference labels (JSON string or list)
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        self.max_length = config.get("max_length", 4096)
        self.truncation = config.get("truncation", "left")
        self.use_shm = config.get("use_shm", False)

        # Keys for data columns - for complex nested data
        self.input_key = config.get(
            "input_key", "input"
        )  # Column containing conversation
        self.output_key = config.get(
            "output_key", "output"
        )  # Column containing response candidates

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
            self.parquet_files[i] = copy_to_local(parquet_file)

    def _clean_conversation(
        self, conversation: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Clean conversation by keeping only 'role' and 'content' fields.
        This removes extra metadata, timestamps, IDs, and other fields that might
        interfere with reward model training by adding noise to the logits calculation.

        Args:
            conversation: List of message dictionaries that may contain extra fields

        Returns:
            Cleaned conversation with only essential 'role' and 'content' fields
        """
        cleaned_conversation = []
        for message in conversation:
            # Only keep role and content fields
            cleaned_message = {}
            if "role" in message:
                cleaned_message["role"] = message["role"]
            elif "user" in message:  # Handle different role field names
                cleaned_message["role"] = "user"

            if "content" in message:
                cleaned_message["content"] = message["content"]
            elif "text" in message:  # Handle different content field names
                cleaned_message["content"] = message["text"]

            # Only add message if it has both role and content
            if "role" in cleaned_message and "content" in cleaned_message:
                cleaned_conversation.append(cleaned_message)
            else:
                print(f"Warning: Skipping message without role/content: {message}")

        return cleaned_conversation

    def _convert_to_preference_format(
        self, data_item: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Convert complex nested data format to simple preference format.

        Args:
            data_item: Single data item with nested structure

        Returns:
            Dictionary with 'chosen' and 'rejected' text strings, or None if conversion fails
        """
        try:
            input_conversation = data_item["input"]
            outputs = data_item["output"]

            # Handle string JSON parsing if needed
            if isinstance(outputs, str):
                outputs = json.loads(outputs)
            if isinstance(input_conversation, str):
                input_conversation = json.loads(input_conversation)

            # Clean input conversation to keep only role and content fields
            input_conversation = self._clean_conversation(input_conversation)

            # Extract responses
            response_a = outputs[0]["answer"]
            response_b = outputs[1]["answer"]

            # Determine preference
            is_a_preferred = response_a["label"]["is_preferred"]
            is_b_preferred = response_b["label"]["is_preferred"]

            if is_a_preferred and not is_b_preferred:
                chosen_response = response_a
                rejected_response = response_b
            elif is_b_preferred and not is_a_preferred:
                chosen_response = response_b
                rejected_response = response_a
            else:
                # Use preference scores as fallback
                score_a = response_a["label"].get("preference_score", 0)
                score_b = response_b["label"].get("preference_score", 0)

                if score_a > score_b:
                    chosen_response = response_a
                    rejected_response = response_b
                else:
                    chosen_response = response_b
                    rejected_response = response_a

            # Clean and prepare response messages
            def clean_response(response):
                """Clean response to keep only role and content"""
                cleaned_response = {}
                if "role" in response:
                    cleaned_response["role"] = response["role"]
                else:
                    cleaned_response["role"] = "assistant"  # Default role for responses

                if "content" in response:
                    cleaned_response["content"] = response["content"]
                elif "text" in response:
                    cleaned_response["content"] = response["text"]
                else:
                    # If no content field, this might be the actual response content
                    cleaned_response["content"] = str(response)

                return cleaned_response

            # Create conversation format and convert to text
            cleaned_chosen = clean_response(chosen_response)
            cleaned_rejected = clean_response(rejected_response)

            chosen_conversation = input_conversation + [cleaned_chosen]
            rejected_conversation = input_conversation + [cleaned_rejected]

            # Convert to text using chat template
            chosen_text = self.tokenizer.apply_chat_template(
                chosen_conversation, tokenize=False, add_generation_prompt=False
            )
            rejected_text = self.tokenizer.apply_chat_template(
                rejected_conversation, tokenize=False, add_generation_prompt=False
            )

            return {"chosen": chosen_text, "rejected": rejected_text}

        except Exception as e:
            print(f"Error converting sample: {e}")
            return None

    def _read_files_and_process(self):
        """Read parquet files and process data into preference pairs."""

        all_data = []
        for parquet_file in self.parquet_files:
            df = pd.read_parquet(parquet_file)
            print(f"Loading {len(df)} samples from {parquet_file}")

            for idx, row in df.iterrows():
                # Convert complex nested data to simple preference format
                converted = self._convert_to_preference_format(row.to_dict())
                if converted is not None:
                    all_data.append(converted)
                else:
                    print(f"Skipping sample {idx} due to conversion error")

        print(f"Successfully processed {len(all_data)} preference pairs")

        # Tokenize and create tensors for each pair
        self.samples = []
        for i, sample in enumerate(all_data):
            try:
                tokenized_sample = self._tokenize_preference_pair(sample)
                if tokenized_sample is not None:
                    self.samples.append(tokenized_sample)
            except Exception as e:
                print(f"Error tokenizing sample {i}: {e}")
                continue

        print(f"Final dataset size: {len(self.samples)} samples")

    def _tokenize_preference_pair(
        self, sample: Dict[str, str]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Tokenize a preference pair (chosen vs rejected).

        Args:
            sample: Dict with 'chosen' and 'rejected' text strings

        Returns:
            Dict with tokenized tensors or None if tokenization fails
        """
        try:
            # Tokenize chosen response
            chosen_tokens = self.tokenizer(
                sample["chosen"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Tokenize rejected response
            rejected_tokens = self.tokenizer(
                sample["rejected"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids_j": chosen_tokens["input_ids"].squeeze(0),
                "attention_mask_j": chosen_tokens["attention_mask"].squeeze(0),
                "input_ids_k": rejected_tokens["input_ids"].squeeze(0),
                "attention_mask_k": rejected_tokens["attention_mask"].squeeze(0),
            }

        except Exception as e:
            print(f"Tokenization error: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
