"""
VLM Reward Usage Examples.

This module demonstrates how to use VLM reward base classes to create
custom multimodal reward models.
"""

import os

from rm_gallery.core.data.multimodal_content import ImageContent, MultimodalContent
from rm_gallery.core.data.multimodal_message import MultimodalChatMessage
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, MessageRole
from rm_gallery.core.model.vlm_api_base import BaseVLMAPI, VLMResponse
from rm_gallery.core.reward.vlm_reward import (
    BaseListWiseVLMReward,
    BasePointWiseVLMReward,
)


# Example 1: Simple Mock VLM API (for development/testing)
class MockQwenVLAPI(BaseVLMAPI):
    """
    Mock Qwen VL API for development and testing.

    Replace this with actual QwenVLAPI implementation in production.
    """

    api_key: str = "mock_key"
    model_name: str = "qwen-vl-plus"

    async def call_api(self, messages, **kwargs):
        """Mock API call that returns simulated response."""
        # In real implementation, this would call Qwen VL API
        return VLMResponse(
            content="8",  # Simulated score
            score=0.8,
            token_usage={"prompt_tokens": 150, "completion_tokens": 5},
        )

    def format_messages(self, text=None, images=None, system_prompt=None, history=None):
        """Format messages for API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if text:
            user_content.append({"type": "text", "text": text})
        if images:
            for img in images:
                user_content.append(img.to_api_format())

        messages.append({"role": "user", "content": user_content})
        return messages


# Example 2: Custom Point-wise Reward - Image-Text Alignment
class ImageTextAlignmentReward(BasePointWiseVLMReward):
    """
    Evaluate alignment between image and text description.

    Uses VLM to assess how well a text description matches an image.
    Score range: 0-1 (higher = better alignment)
    """

    name: str = "image_text_alignment"
    prompt_template: str = "请评估这段文字与图片的匹配程度（0-10分）：{text}\n只回答数字。"

    def _compute_reward(self, images, texts, **kwargs):
        """
        Compute alignment score using VLM API.

        Args:
            images: List of images
            texts: List of text descriptions

        Returns:
            Alignment score in [0, 1]
        """
        if not images or not texts:
            return self.fallback_score

        # Use first image and first text
        image = images[0]
        text = texts[0]

        # Call VLM API synchronously (wrapped in async by base class)
        # In real implementation, use: await self.vlm_api.compute_similarity(image, text)
        # For this mock, we'll simulate
        score = 0.85  # Mock score

        return float(score)


# Example 3: Custom Point-wise Reward - Visual Helpfulness
class VisualHelpfulnessReward(BasePointWiseVLMReward):
    """
    Evaluate helpfulness of a response given image context.

    Assesses whether the text response is helpful and relevant
    given the visual context.
    """

    name: str = "visual_helpfulness"

    def _compute_reward(self, images, texts, **kwargs):
        """
        Compute helpfulness score.

        Uses contrastive prompts to evaluate quality.
        """
        if not images or not texts:
            return self.fallback_score

        # In real implementation, call VLM with quality evaluation prompt
        # For now, mock the score
        score = 0.75

        return float(score)


# Example 4: Custom List-wise Reward - Multimodal Ranking
class MultimodalRankingReward(BaseListWiseVLMReward):
    """
    Rank multiple candidate responses based on image relevance.

    Evaluates and ranks all candidates to select the best response
    given visual context.
    """

    name: str = "multimodal_ranking"

    def _compute_reward(self, images, texts, **kwargs):
        """
        Compute score for one candidate.

        This will be called multiple times (once per candidate).
        """
        if not images or not texts:
            return self.fallback_score

        # Mock scoring based on text length (for demo)
        # In real implementation, use VLM API
        score = min(len(texts[0]) / 100.0, 1.0)

        return float(score)


# Example 5: Usage demonstration
def main():
    """Demonstrate VLM reward usage."""

    print("=" * 60)
    print("VLM Reward Base Classes - Usage Examples")
    print("=" * 60)

    # Initialize mock VLM API
    vlm_api = MockQwenVLAPI(
        api_key=os.getenv("DASHSCOPE_API_KEY", "mock_key"), model_name="qwen-vl-plus"
    )

    # Example 1: Point-wise Image-Text Alignment
    print("\n1. Point-wise Image-Text Alignment Reward")
    print("-" * 60)

    alignment_reward = ImageTextAlignmentReward(
        vlm_api=vlm_api, enable_cache=True, enable_cost_tracking=True
    )

    # Create multimodal sample
    sample = DataSample(
        unique_id="example_001",
        input=[
            MultimodalChatMessage(
                role=MessageRole.USER,
                content=MultimodalContent(
                    text="What animal is in this image?",
                    images=[
                        ImageContent(type="url", data="https://example.com/cat.jpg")
                    ],
                ),
            )
        ],
        output=[
            DataOutput(
                answer=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="This image shows a cat sitting on a couch.",
                )
            )
        ],
    )

    # Evaluate
    result = alignment_reward.evaluate(sample)
    score = result.output[0].answer.reward.score

    print(f"Sample ID: {result.unique_id}")
    print(f"Alignment Score: {score:.3f}")
    print(f"Details: {result.output[0].answer.reward.details[0].reason}")

    # Show cost stats
    stats = alignment_reward.get_cost_stats()
    print("\nCost Statistics:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Cache Hits: {stats['cache_hits']}")
    print(f"  Estimated Cost: {stats['estimated_cost_usd']}")

    # Example 2: List-wise Multimodal Ranking
    print("\n2. List-wise Multimodal Ranking Reward")
    print("-" * 60)

    ranking_reward = MultimodalRankingReward(vlm_api=vlm_api, enable_cache=True)

    # Create sample with multiple candidates
    multi_sample = DataSample(
        unique_id="example_002",
        input=[
            MultimodalChatMessage(
                role=MessageRole.USER,
                content=MultimodalContent(
                    text="Describe what you see",
                    images=[
                        ImageContent(type="url", data="https://example.com/scene.jpg")
                    ],
                ),
            )
        ],
        output=[
            DataOutput(
                answer=ChatMessage(role=MessageRole.ASSISTANT, content="A cat.")
            ),
            DataOutput(
                answer=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="A domestic cat sitting comfortably on a grey couch.",
                )
            ),
            DataOutput(
                answer=ChatMessage(role=MessageRole.ASSISTANT, content="An animal.")
            ),
        ],
    )

    # Evaluate and rank
    result = ranking_reward.evaluate(multi_sample)

    print(f"Sample ID: {result.unique_id}")
    print(f"Number of candidates: {len(result.output)}")

    for i, output in enumerate(result.output):
        score = output.answer.reward.score
        content = output.answer.content[:50]  # Truncate for display
        print(f'  Candidate {i+1}: {score:.3f} - "{content}..."')

    # Example 3: Batch Evaluation
    print("\n3. Batch Evaluation")
    print("-" * 60)

    # Create multiple samples
    samples = [
        DataSample(
            unique_id=f"batch_{i}",
            input=[
                MultimodalChatMessage(
                    role=MessageRole.USER,
                    content=MultimodalContent(
                        text=f"Question {i}",
                        images=[
                            ImageContent(
                                type="url", data=f"https://example.com/img{i}.jpg"
                            )
                        ],
                    ),
                )
            ],
            output=[
                DataOutput(
                    answer=ChatMessage(
                        role=MessageRole.ASSISTANT, content=f"Answer {i}"
                    )
                )
            ],
        )
        for i in range(3)
    ]

    # Batch evaluate
    results = alignment_reward.evaluate_batch(samples, max_workers=2)

    print(f"Evaluated {len(results)} samples")
    for result in results:
        score = result.output[0].answer.reward.score
        print(f"  {result.unique_id}: {score:.3f}")

    # Final cost summary
    print("\n4. Final Cost Summary")
    print("-" * 60)

    final_stats = alignment_reward.get_cost_stats()
    print(f"Total Requests: {final_stats['total_requests']}")
    print(f"Cache Rate: {final_stats['cache_rate']}")
    print(f"Total Tokens: {final_stats['total_tokens']}")
    print(f"Estimated Cost: {final_stats['estimated_cost_usd']}")
    print(f"Saved Cost (from cache): {final_stats['saved_cost_usd']}")

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
