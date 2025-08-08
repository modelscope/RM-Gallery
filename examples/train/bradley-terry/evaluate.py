"""
Reward Model Evaluation Script

This script evaluates a trained Bradley-Terry reward model on preference data.
It computes accuracy by checking if the model assigns higher rewards to preferred responses.

"""
import json
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoTokenizer, pipeline

from rm_gallery.core.utils.file import load_parquet


@dataclass
class RewardModelEvaluator:
    """Evaluator class for Bradley-Terry reward models."""

    model_path: str
    max_length: int = 8192
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        """Initialize tokenizer and model pipeline."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model_path,
            device_map=self.device_map,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": self.torch_dtype},
        )

        # Pipeline configuration
        self.pipe_kwargs = {"top_k": 1, "function_to_apply": "none", "batch_size": 1}

    def truncate_text(self, text: str) -> str:
        """Truncate text to fit model's maximum length.

        Args:
            text: Input text to truncate

        Returns:
            Truncated text that fits within model's context window
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
            text = self.tokenizer.decode(tokens)
        return text

    def get_reward(
        self, chat: List[Dict[str, str]]
    ) -> Tuple[Optional[float], Optional[str]]:
        """Get reward score for a conversation.

        Args:
            chat: List of conversation messages in chat format

        Returns:
            Tuple of (reward_score, error_message)
            - reward_score: Float score if successful, None if error
            - error_message: Error description if failed, None if successful
        """
        try:
            # Truncate each message
            truncated_chat = [
                {**msg, "content": self.truncate_text(msg["content"])} for msg in chat
            ]

            # Convert to model input format
            model_input = self.tokenizer.apply_chat_template(
                truncated_chat, tokenize=False, add_generation_prompt=False
            )

            # Get model prediction
            output = self.pipeline([model_input], **self.pipe_kwargs)
            reward = output[0][0]["score"]

            return reward, None

        except Exception as e:
            return None, str(e)

    def get_sample_scores(self, data: Dict) -> Dict[str, float]:
        """Get reward scores for a pair of responses.

        Args:
            data: Dictionary containing input and two responses in the required format:
                {
                    "input": [{"role": "user", "content": str}],
                    "output": [
                        {
                            "answer": {"role": "assistant", "content": str},
                            "label": {"is_preferred": bool}
                        },
                        {
                            "answer": {"role": "assistant", "content": str},
                            "label": {"is_preferred": bool}
                        }
                    ]
                }

        Returns:
            Dictionary with reward scores for both responses
        """
        try:
            # Extract conversations and parse JSON if needed
            input_msg = data["input"]
            if isinstance(input_msg, str):
                try:
                    input_msg = json.loads(input_msg)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse input JSON: {e}")
                    return {}

            outputs = data["output"]
            if isinstance(outputs, str):
                try:
                    outputs = json.loads(outputs)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse output JSON: {e}")
                    return {}

            # Extract answers safely
            try:
                answer_a = outputs[0]["answer"]
                answer_b = outputs[1]["answer"]
            except (IndexError, KeyError) as e:
                logger.error(f"Error extracting answers: {e}")
                return {}

            # Create chat contexts
            chat_a = input_msg + [answer_a]
            chat_b = input_msg + [answer_b]
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            return {}

        # Get rewards
        reward_a, error_a = self.get_reward(chat_a)
        reward_b, error_b = self.get_reward(chat_b)

        # Handle errors
        if error_a or error_b:
            if error_a:
                logger.error(f"Error A: {error_a}")
            if error_b:
                logger.error(f"Error B: {error_b}")
            return {}

        return {"response_a_score": reward_a, "response_b_score": reward_b}

    def process_dataset(self, data_path: str) -> None:
        """Process dataset and output reward scores for each sample.

        Args:
            data_path: Path to evaluation data file (parquet format)
        """
        try:
            # Load parquet data
            data_rows = load_parquet(data_path)
            # Process each row
            for idx, data in enumerate(data_rows):
                try:
                    # Get scores for sample
                    scores = self.get_sample_scores(data)
                    # Output scores
                    if scores:
                        print(
                            f"Score A: {scores['response_a_score']:.4f}, Score B: {scores['response_b_score']:.4f}"
                        )
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error loading parquet file: {str(e)}")
            raise


def main():
    """Main evaluation function."""
    parser = ArgumentParser(description="Evaluate Bradley-Terry reward model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained reward model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to evaluation data file (parquet format)",
    )
    parser.add_argument(
        "--max_length", type=int, default=8192, help="Maximum sequence length"
    )
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = RewardModelEvaluator(
        model_path=args.model_path, max_length=args.max_length
    )

    # Process dataset and output scores
    evaluator.process_dataset(args.data_path)


if __name__ == "__main__":
    main()
