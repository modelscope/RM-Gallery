"""
Comprehensive tests for Qwen VL reward models.

Tests all three reward models with positive and negative examples to validate:
1. QwenImageTextAlignmentReward - Image-text alignment scoring
2. QwenVisualHelpfulnessReward - Response quality evaluation
3. QwenMultimodalRankingReward - Multi-candidate ranking

Each model is tested with:
- Positive examples (high-quality matches)
- Negative examples (poor matches/quality)
- Edge cases (empty inputs, missing data)
"""

import os
from typing import List

from loguru import logger

from rm_gallery.core.data.multimodal_content import ImageContent, MultimodalContent
from rm_gallery.core.data.multimodal_message import MultimodalChatMessage
from rm_gallery.core.data.schema import DataOutput, DataSample
from rm_gallery.core.model.message import ChatMessage, MessageRole
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.gallery.rm.multimodal.qwen_alignment import QwenImageTextAlignmentReward
from rm_gallery.gallery.rm.multimodal.qwen_helpfulness import (
    QwenVisualHelpfulnessReward,
)
from rm_gallery.gallery.rm.multimodal.qwen_ranking import QwenMultimodalRankingReward

# Test image URLs (using public accessible images)
TEST_IMAGES = {
    "cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",  # Cat image
    "dog": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400",  # Dog image
    "landscape": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # Mountain landscape
}


def create_multimodal_sample(
    unique_id: str, image_url: str, question: str, answer: str
) -> DataSample:
    """
    Create a DataSample with multimodal content.

    Args:
        unique_id: Sample identifier
        image_url: URL of the image
        question: Question text
        answer: Answer text

    Returns:
        DataSample with multimodal input and output
    """
    # Create multimodal input message
    input_message = MultimodalChatMessage(
        role=MessageRole.USER,
        content=MultimodalContent(
            text=question, images=[ImageContent(type="url", data=image_url)]
        ),
    )

    # Create answer output
    answer_message = ChatMessage(role=MessageRole.ASSISTANT, content=answer)

    output = DataOutput(answer=answer_message)

    return DataSample(unique_id=unique_id, input=[input_message], output=[output])


def create_ranking_sample(
    unique_id: str, image_url: str, question: str, answers: List[str]
) -> DataSample:
    """
    Create a DataSample for ranking multiple answers.

    Args:
        unique_id: Sample identifier
        image_url: URL of the image
        question: Question text
        answers: List of candidate answers

    Returns:
        DataSample with multiple outputs for ranking
    """
    # Create multimodal input message
    input_message = MultimodalChatMessage(
        role=MessageRole.USER,
        content=MultimodalContent(
            text=question, images=[ImageContent(type="url", data=image_url)]
        ),
    )

    # Create multiple answer outputs
    outputs = [
        DataOutput(answer=ChatMessage(role=MessageRole.ASSISTANT, content=ans))
        for ans in answers
    ]

    return DataSample(unique_id=unique_id, input=[input_message], output=outputs)


class TestQwenRewards:
    """Test suite for Qwen VL reward models."""

    def __init__(self):
        """Initialize test suite with API key."""
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")

        # Initialize VLM API (shared across rewards)
        self.vlm_api = QwenVLAPI(
            api_key=self.api_key, model_name="qwen-vl-plus", enable_cache=True
        )

        self.results = {"alignment": [], "helpfulness": [], "ranking": []}

    def test_alignment_positive(self):
        """Test alignment reward with positive examples (good matches)."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 1: Image-Text Alignment - Positive Examples")
        logger.info("=" * 60)

        reward = QwenImageTextAlignmentReward(
            vlm_api=self.vlm_api, use_english_prompt=True
        )

        positive_cases = [
            {
                "id": "align_pos_1",
                "image": TEST_IMAGES["cat"],
                "question": "What animal is in this image?",
                "answer": "This is a cat.",
                "expected": "high",
            },
            {
                "id": "align_pos_2",
                "image": TEST_IMAGES["landscape"],
                "question": "Describe the scenery",
                "answer": "A beautiful mountain landscape with clear sky.",
                "expected": "high",
            },
        ]

        for case in positive_cases:
            sample = create_multimodal_sample(
                case["id"], case["image"], case["question"], case["answer"]
            )

            result = reward.evaluate(sample)
            score = result.output[0].answer.reward.score

            logger.info(f"\n✅ {case['id']}")
            logger.info(f"   Question: {case['question']}")
            logger.info(f"   Answer: {case['answer']}")
            logger.info(f"   Score: {score:.3f} (Expected: {case['expected']})")

            self.results["alignment"].append(
                {
                    "case": case["id"],
                    "type": "positive",
                    "score": score,
                    "answer": case["answer"],
                }
            )

    def test_alignment_negative(self):
        """Test alignment reward with negative examples (poor matches)."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Image-Text Alignment - Negative Examples")
        logger.info("=" * 60)

        reward = QwenImageTextAlignmentReward(
            vlm_api=self.vlm_api, use_english_prompt=True
        )

        negative_cases = [
            {
                "id": "align_neg_1",
                "image": TEST_IMAGES["cat"],
                "question": "What animal is in this image?",
                "answer": "This is a dog.",
                "expected": "low",
            },
            {
                "id": "align_neg_2",
                "image": TEST_IMAGES["landscape"],
                "question": "Describe the scenery",
                "answer": "A crowded city street with many buildings.",
                "expected": "low",
            },
        ]

        for case in negative_cases:
            sample = create_multimodal_sample(
                case["id"], case["image"], case["question"], case["answer"]
            )

            result = reward.evaluate(sample)
            score = result.output[0].answer.reward.score

            logger.info(f"\n❌ {case['id']}")
            logger.info(f"   Question: {case['question']}")
            logger.info(f"   Answer: {case['answer']}")
            logger.info(f"   Score: {score:.3f} (Expected: {case['expected']})")

            self.results["alignment"].append(
                {
                    "case": case["id"],
                    "type": "negative",
                    "score": score,
                    "answer": case["answer"],
                }
            )

    def test_helpfulness_positive(self):
        """Test helpfulness reward with positive examples (helpful responses)."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Visual Helpfulness - Positive Examples")
        logger.info("=" * 60)

        reward = QwenVisualHelpfulnessReward(
            vlm_api=self.vlm_api, use_english_prompt=True, use_detailed_rubric=True
        )

        positive_cases = [
            {
                "id": "help_pos_1",
                "image": TEST_IMAGES["cat"],
                "question": "What is the cat doing?",
                "answer": "The cat is sitting calmly and looking directly at the camera with an alert expression.",
                "expected": "high",
            },
            {
                "id": "help_pos_2",
                "image": TEST_IMAGES["dog"],
                "question": "Describe the dog's appearance",
                "answer": "This is a golden retriever with beautiful golden fur, looking happy and friendly.",
                "expected": "high",
            },
        ]

        for case in positive_cases:
            sample = create_multimodal_sample(
                case["id"], case["image"], case["question"], case["answer"]
            )

            result = reward.evaluate(sample)
            score = result.output[0].answer.reward.score

            logger.info(f"\n✅ {case['id']}")
            logger.info(f"   Question: {case['question']}")
            logger.info(f"   Answer: {case['answer']}")
            logger.info(f"   Score: {score:.3f} (Expected: {case['expected']})")

            self.results["helpfulness"].append(
                {
                    "case": case["id"],
                    "type": "positive",
                    "score": score,
                    "answer": case["answer"],
                }
            )

    def test_helpfulness_negative(self):
        """Test helpfulness reward with negative examples (unhelpful responses)."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: Visual Helpfulness - Negative Examples")
        logger.info("=" * 60)

        reward = QwenVisualHelpfulnessReward(
            vlm_api=self.vlm_api, use_english_prompt=True, use_detailed_rubric=True
        )

        negative_cases = [
            {
                "id": "help_neg_1",
                "image": TEST_IMAGES["cat"],
                "question": "What is the cat doing?",
                "answer": "I don't know.",
                "expected": "low",
            },
            {
                "id": "help_neg_2",
                "image": TEST_IMAGES["dog"],
                "question": "Describe the dog's appearance",
                "answer": "It's an animal.",
                "expected": "low",
            },
        ]

        for case in negative_cases:
            sample = create_multimodal_sample(
                case["id"], case["image"], case["question"], case["answer"]
            )

            result = reward.evaluate(sample)
            score = result.output[0].answer.reward.score

            logger.info(f"\n❌ {case['id']}")
            logger.info(f"   Question: {case['question']}")
            logger.info(f"   Answer: {case['answer']}")
            logger.info(f"   Score: {score:.3f} (Expected: {case['expected']})")

            self.results["helpfulness"].append(
                {
                    "case": case["id"],
                    "type": "negative",
                    "score": score,
                    "answer": case["answer"],
                }
            )

    def test_ranking_comparison(self):
        """Test ranking reward with multiple candidates."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: Multimodal Ranking - Candidate Comparison")
        logger.info("=" * 60)

        reward = QwenMultimodalRankingReward(
            vlm_api=self.vlm_api,
            ranking_metric="combined",
            use_parallel_evaluation=True,
        )

        test_cases = [
            {
                "id": "rank_1",
                "image": TEST_IMAGES["cat"],
                "question": "What animal is this?",
                "answers": [
                    "This is a cat.",  # Good
                    "This is a dog.",  # Wrong
                    "This is a beautiful domestic cat with striking eyes.",  # Best (detailed)
                ],
                "expected_best": 2,  # Index of best answer
            },
            {
                "id": "rank_2",
                "image": TEST_IMAGES["landscape"],
                "question": "Describe this scene",
                "answers": [
                    "Mountains.",  # Minimal
                    "A scenic mountain landscape with blue sky and clouds.",  # Best
                    "A city.",  # Wrong
                ],
                "expected_best": 1,
            },
        ]

        for case in test_cases:
            sample = create_ranking_sample(
                case["id"], case["image"], case["question"], case["answers"]
            )

            result = reward.evaluate(sample)
            ranks = result.details[0][0].rank
            best_idx = ranks.index(max(ranks))

            logger.info(f"\n🏆 {case['id']}")
            logger.info(f"   Question: {case['question']}")
            for i, (answer, rank) in enumerate(zip(case["answers"], ranks)):
                marker = "👑" if i == best_idx else "  "
                logger.info(f"   {marker} [{i}] Rank: {rank:.3f} - {answer}")
            logger.info(f"   Expected best: {case['expected_best']}, Got: {best_idx}")

            self.results["ranking"].append(
                {
                    "case": case["id"],
                    "ranks": ranks,
                    "best_idx": best_idx,
                    "expected_best": case["expected_best"],
                    "correct": best_idx == case["expected_best"],
                }
            )

    def print_summary(self):
        """Print test summary and statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)

        # Alignment summary
        logger.info("\n📊 Image-Text Alignment Results:")
        pos_scores = [
            r["score"] for r in self.results["alignment"] if r["type"] == "positive"
        ]
        neg_scores = [
            r["score"] for r in self.results["alignment"] if r["type"] == "negative"
        ]

        if pos_scores:
            logger.info(
                f"   Positive examples: Avg score = {sum(pos_scores)/len(pos_scores):.3f}"
            )
        if neg_scores:
            logger.info(
                f"   Negative examples: Avg score = {sum(neg_scores)/len(neg_scores):.3f}"
            )

        # Helpfulness summary
        logger.info("\n📊 Visual Helpfulness Results:")
        pos_scores = [
            r["score"] for r in self.results["helpfulness"] if r["type"] == "positive"
        ]
        neg_scores = [
            r["score"] for r in self.results["helpfulness"] if r["type"] == "negative"
        ]

        if pos_scores:
            logger.info(
                f"   Positive examples: Avg score = {sum(pos_scores)/len(pos_scores):.3f}"
            )
        if neg_scores:
            logger.info(
                f"   Negative examples: Avg score = {sum(neg_scores)/len(neg_scores):.3f}"
            )

        # Ranking summary
        logger.info("\n📊 Multimodal Ranking Results:")
        correct_rankings = sum(1 for r in self.results["ranking"] if r["correct"])
        total_rankings = len(self.results["ranking"])

        if total_rankings > 0:
            accuracy = correct_rankings / total_rankings
            logger.info(
                f"   Ranking accuracy: {accuracy:.1%} ({correct_rankings}/{total_rankings})"
            )

        # API usage statistics
        logger.info("\n💰 API Usage Statistics:")
        cost_stats = self.vlm_api.get_cost_stats()
        logger.info(f"   Total requests: {cost_stats['total_requests']}")
        logger.info(f"   Cache rate: {cost_stats['cache_rate']}")
        logger.info(f"   Estimated cost: {cost_stats['estimated_cost_usd']}")

        return self.results


def main():
    """Run all tests."""
    logger.info("Starting Qwen VL Reward Models Test Suite")
    logger.info("=" * 60)

    # Check API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        logger.error("❌ DASHSCOPE_API_KEY environment variable not set!")
        logger.info("Please set your DashScope API key:")
        logger.info("  export DASHSCOPE_API_KEY='your-api-key'")
        return

    # Initialize test suite
    test_suite = TestQwenRewards()

    try:
        # Run all tests
        test_suite.test_alignment_positive()
        test_suite.test_alignment_negative()
        test_suite.test_helpfulness_positive()
        test_suite.test_helpfulness_negative()
        test_suite.test_ranking_comparison()

        # Print summary
        results = test_suite.print_summary()

        logger.info("\n✅ All tests completed successfully!")

        return results

    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
