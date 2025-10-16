"""
Simple Reward Model Evaluation Script

This script provides a basic test for a trained Bradley-Terry reward model.
It takes simple text inputs and returns reward scores.

"""
from argparse import ArgumentParser
from typing import Dict, List

import torch
from transformers import AutoTokenizer, pipeline


class SimpleRewardEvaluator:
    """Simple evaluator for Bradley-Terry reward models."""

    def __init__(self, model_path: str):
        """Initialize tokenizer and model pipeline."""
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
            "text-classification",
            model=model_path,
            device_map="auto",
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def get_reward(self, conversation: List[Dict[str, str]]) -> float:
        """Get reward score for a conversation.

        Args:
            conversation: List of messages in format [{"role": "user/assistant", "content": "text"}]

        Returns:
            Reward score as float
        """
        # Convert conversation to model input format
        model_input = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Get model prediction
        output = self.pipeline(model_input, top_k=1, function_to_apply="none")
        return output[0]["score"]

    def compare_responses(self, user_message: str, response_a: str, response_b: str):
        """Compare two responses to the same user message.

        Args:
            user_message: The user's question or prompt
            response_a: First response to compare
            response_b: Second response to compare

        Returns:
            Tuple of (score_a, score_b, preferred_response)
        """
        # Create conversations
        conv_a = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response_a},
        ]
        conv_b = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response_b},
        ]

        # Get scores
        score_a = self.get_reward(conv_a)
        score_b = self.get_reward(conv_b)

        # Determine preferred response
        preferred = "A" if score_a > score_b else "B"

        return score_a, score_b, preferred

    def simple_test(self, user_message: str, response: str) -> float:
        """Simple test: get reward score for a single response.

        Args:
            user_message: The user's question or prompt
            response: The assistant's response

        Returns:
            Reward score
        """
        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response},
        ]
        return self.get_reward(conversation)


def main():
    """Main function - simple test example."""
    parser = ArgumentParser(description="Simple Bradley-Terry reward model test")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained reward model"
    )
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = SimpleRewardEvaluator(args.model_path)

    # Simple test examples
    print("=== Simple Reward Model Test ===")

    # Test 1: Single response scoring
    user_msg = "What is the capital of France?"
    response1 = "The capital of France is Paris."
    response2 = "I don't know."

    score1 = evaluator.simple_test(user_msg, response1)
    score2 = evaluator.simple_test(user_msg, response2)

    print(f"User: {user_msg}")
    print(f"Response 1: {response1}")
    print(f"Score 1: {score1:.4f}")
    print(f"Response 2: {response2}")
    print(f"Score 2: {score2:.4f}")

    # Test 2: Compare two responses
    print("\n=== Response Comparison ===")
    score_a, score_b, preferred = evaluator.compare_responses(
        user_msg, response1, response2
    )
    print(f"Score A: {score_a:.4f}, Score B: {score_b:.4f}")
    print(f"Preferred response: {preferred}")


if __name__ == "__main__":
    main()
