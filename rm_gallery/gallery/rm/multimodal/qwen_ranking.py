"""
Qwen VL Multimodal Ranking Reward Model.

Ranks multiple candidate responses based on their quality given visual context.
Uses Qwen VL to evaluate and compare multiple responses, selecting the best one
for Best-of-N sampling or response reranking.
"""

import asyncio
import os
from typing import List, Optional

from loguru import logger
from pydantic import Field

from rm_gallery.core.data.multimodal_content import ImageContent
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.reward.vlm_reward import BaseListWiseVLMReward


@RewardRegistry.register("qwen_multimodal_ranking")
class QwenMultimodalRankingReward(BaseListWiseVLMReward):
    """
    Qwen VL Multimodal Ranking Reward.

    Evaluates and ranks multiple candidate responses based on visual context.
    Each candidate is scored independently, then ranked from best to worst.

    Key Features:
    - List-wise evaluation (ranks multiple candidates)
    - Parallel/concurrent API calls for efficiency
    - Normalized ranking scores in [0, 1] range
    - Built-in caching to avoid redundant evaluations
    - Cost-efficient batch processing

    Attributes:
        name: Reward model identifier
        vlm_api: Qwen VL API client
        ranking_metric: Metric to use for ranking ("similarity", "quality", "combined")
        use_parallel_evaluation: Whether to evaluate candidates in parallel
        max_concurrent: Maximum concurrent API calls

    Use Cases:
        - Best-of-N sampling for multimodal generation
        - Response reranking for visual QA
        - Multimodal chatbot response selection
        - Preference data collection for RLHF

    Examples:
        >>> # Initialize ranking reward
        >>> reward = QwenMultimodalRankingReward(
        ...     vlm_api=QwenVLAPI(
        ...         api_key=os.getenv("DASHSCOPE_API_KEY"),
        ...         model_name="qwen-vl-plus"
        ...     ),
        ...     use_parallel_evaluation=True
        ... )
        >>>
        >>> # Rank multiple candidate responses
        >>> sample = DataSample(
        ...     unique_id="ranking_001",
        ...     input=[MultimodalChatMessage(
        ...         role=MessageRole.USER,
        ...         content=MultimodalContent(
        ...             text="What animal is in this image?",
        ...             images=[ImageContent(type="url", data="https://...")]
        ...         )
        ...     )],
        ...     output=[
        ...         DataOutput(answer=ChatMessage(content="A cat")),
        ...         DataOutput(answer=ChatMessage(content="A dog")),
        ...         DataOutput(answer=ChatMessage(content="A feline animal, specifically a domestic cat"))
        ...     ]
        ... )
        >>> result = reward.evaluate(sample)
        >>> ranks = result.details[0][0].rank
        >>> print(f"Rankings: {ranks}")  # e.g., [0.5, 0.0, 1.0] - third is best
        >>>
        >>> # Get best candidate index
        >>> best_idx = ranks.index(max(ranks))
        >>> print(f"Best response: {sample.output[best_idx].answer.content}")
    """

    name: str = Field(
        default="qwen_multimodal_ranking", description="Reward model name"
    )

    vlm_api: QwenVLAPI = Field(
        default_factory=lambda: QwenVLAPI(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            model_name="qwen-vl-plus",
            enable_cache=True,
        ),
        description="Qwen VL API client",
    )

    ranking_metric: str = Field(
        default="combined",
        description="Ranking metric: 'similarity', 'quality', or 'combined'",
    )

    similarity_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for similarity score (when metric='combined')",
    )

    quality_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for quality score (when metric='combined')",
    )

    use_parallel_evaluation: bool = Field(
        default=True, description="Whether to evaluate candidates in parallel"
    )

    max_concurrent: int = Field(
        default=5, gt=0, description="Maximum concurrent API calls"
    )

    similarity_prompt_template: str = Field(
        default=("请评估以下回答与图片内容的匹配程度（0-10分）：\n\n" "回答：{text}\n\n" "只回答一个0-10的数字。"),
        description="Prompt for similarity scoring",
    )

    quality_prompt_template: str = Field(
        default=(
            "请基于图片内容，评估以下回答的质量（0-10分）：\n\n"
            "回答：{text}\n\n"
            "评估标准：准确性、详细性、有用性\n"
            "只回答一个0-10的数字。"
        ),
        description="Prompt for quality scoring",
    )

    def _evaluate(self, sample, **kwargs):
        """
        Override base class _evaluate to support parallel evaluation.

        This method handles both parallel and sequential evaluation based on
        use_parallel_evaluation flag.
        """
        from rm_gallery.core.data.multimodal_content import MultimodalContent
        from rm_gallery.core.reward.schema import RewardDimensionWithRank, RewardResult

        # Extract multimodal content from input
        input_texts, images = self._extract_multimodal_content(sample)

        # Handle missing images
        if not images:
            n = len(sample.output)
            return RewardResult(
                name=self.name,
                details=[
                    [
                        RewardDimensionWithRank(
                            name=self.name,
                            rank=[1.0 / n] * n,  # Equal ranking
                            reason="No image content found",
                        )
                    ]
                ],
            )

        # Extract all candidate answers
        candidate_texts = []
        for output in sample.output:
            answer = output.answer.content
            if isinstance(answer, str):
                answer_text = answer
            elif isinstance(answer, MultimodalContent):
                answer_text = answer.text or ""
            else:
                answer_text = str(answer)
            candidate_texts.append(answer_text)

        try:
            # Choose parallel or sequential evaluation
            if self.use_parallel_evaluation:
                # Use parallel evaluation with asyncio
                try:
                    loop = asyncio.get_running_loop()
                    # Already in async context
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._evaluate_candidates_parallel(images, candidate_texts),
                        )
                        scores = future.result()
                except RuntimeError:
                    # No running loop
                    scores = asyncio.run(
                        self._evaluate_candidates_parallel(images, candidate_texts)
                    )
            else:
                # Use sequential evaluation
                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._evaluate_candidates_sequential(
                                images, candidate_texts
                            ),
                        )
                        scores = future.result()
                except RuntimeError:
                    scores = asyncio.run(
                        self._evaluate_candidates_sequential(images, candidate_texts)
                    )

            # Convert scores to ranks
            ranks = self._scores_to_ranks(scores)

            return RewardResult(
                name=self.name,
                details=[
                    [
                        RewardDimensionWithRank(
                            name=self.name,
                            rank=ranks,
                            reason=f"VLM-based ranking scores: {[f'{s:.3f}' for s in scores]}",
                        )
                    ]
                ],
            )

        except Exception as e:
            # Return equal ranking on error
            n = len(sample.output)
            logger.error(f"List-wise evaluation error: {str(e)}")
            return RewardResult(
                name=self.name,
                details=[
                    [
                        RewardDimensionWithRank(
                            name=self.name,
                            rank=[1.0 / n] * n,
                            reason=f"Error during ranking: {str(e)[:100]}",
                        )
                    ]
                ],
            )

    async def _compute_reward(
        self, images: List[ImageContent], texts: List[str], **kwargs
    ) -> float:
        """
        Compute score for a single candidate response.

        Args:
            images: List of images (uses first image)
            texts: List of texts (uses first text as the candidate)
            **kwargs: Additional parameters

        Returns:
            Score in range [0, 1]
        """
        # Validate inputs
        if not images:
            logger.warning("No images provided for ranking evaluation")
            return self.fallback_score

        if not texts or len(texts) == 0:
            logger.warning("No candidate text provided")
            return self.fallback_score

        # Use first image and first text
        image = images[0]
        candidate_text = texts[0].strip()

        if not candidate_text:
            logger.warning("Empty candidate text")
            return 0.0  # Empty responses get zero score

        try:
            # Compute score based on selected metric
            if self.ranking_metric == "similarity":
                score = await self._compute_similarity_score(image, candidate_text)

            elif self.ranking_metric == "quality":
                score = await self._compute_quality_score(image, candidate_text)

            elif self.ranking_metric == "combined":
                # Compute both scores
                similarity_score = await self._compute_similarity_score(
                    image, candidate_text
                )
                quality_score = await self._compute_quality_score(image, candidate_text)

                # Weighted combination
                score = (
                    self.similarity_weight * similarity_score
                    + self.quality_weight * quality_score
                )

                logger.debug(
                    f"Combined score: {score:.3f} (similarity: {similarity_score:.3f}, "
                    f"quality: {quality_score:.3f})"
                )

            else:
                logger.warning(
                    f"Unknown ranking metric: {self.ranking_metric}, using similarity"
                )
                score = await self._compute_similarity_score(image, candidate_text)

            # Ensure score is in valid range
            score = max(0.0, min(1.0, float(score)))

            return score

        except Exception as e:
            logger.error(f"Failed to compute ranking score: {str(e)}")
            return self.fallback_score

    async def _compute_similarity_score(self, image: ImageContent, text: str) -> float:
        """
        Compute image-text similarity score.

        Args:
            image: Image to compare
            text: Text to compare

        Returns:
            Similarity score in [0, 1]
        """
        try:
            score = await self.vlm_api.compute_similarity(
                image=image, text=text, prompt_template=self.similarity_prompt_template
            )
            return float(score)

        except Exception as e:
            logger.error(f"Similarity scoring failed: {str(e)}")
            return self.fallback_score

    async def _compute_quality_score(self, image: ImageContent, text: str) -> float:
        """
        Compute response quality score.

        Args:
            image: Reference image
            text: Response text to evaluate

        Returns:
            Quality score in [0, 1]
        """
        try:
            # Build quality evaluation prompt
            evaluation_prompt = self.quality_prompt_template.format(text=text)

            # Call VLM API directly to avoid prompt double-wrapping
            response = await self.vlm_api.generate(
                text=evaluation_prompt, images=[image], temperature=0.1, max_tokens=10
            )

            # Parse score
            score = self.vlm_api._parse_score(response.content)
            return float(score)

        except Exception as e:
            logger.error(f"Quality scoring failed: {str(e)}")
            return self.fallback_score

    async def _evaluate_candidates_parallel(
        self, images: List[ImageContent], candidates: List[str]
    ) -> List[float]:
        """
        Evaluate multiple candidates in parallel.

        Args:
            images: List of images
            candidates: List of candidate texts

        Returns:
            List of scores for each candidate
        """
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def evaluate_with_limit(candidate: str) -> float:
            async with semaphore:
                return await self._compute_reward(images, [candidate])

        # Evaluate all candidates concurrently
        tasks = [evaluate_with_limit(candidate) for candidate in candidates]
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_scores = []
        for i, score in enumerate(scores):
            if isinstance(score, Exception):
                logger.error(f"Candidate {i} evaluation failed: {score}")
                processed_scores.append(self.fallback_score)
            else:
                processed_scores.append(score)

        return processed_scores

    async def _evaluate_candidates_sequential(
        self, images: List[ImageContent], candidates: List[str]
    ) -> List[float]:
        """
        Evaluate multiple candidates sequentially.

        Args:
            images: List of images
            candidates: List of candidate texts

        Returns:
            List of scores for each candidate
        """
        scores = []
        for i, candidate in enumerate(candidates):
            try:
                score = await self._compute_reward(images, [candidate])
                scores.append(score)
                logger.debug(f"Candidate {i+1}/{len(candidates)} scored: {score:.3f}")

            except Exception as e:
                logger.error(f"Candidate {i} evaluation failed: {str(e)}")
                scores.append(self.fallback_score)

        return scores

    def set_ranking_metric(
        self, metric: str, similarity_weight: float = 0.5, quality_weight: float = 0.5
    ):
        """
        Configure ranking metric and weights.

        Args:
            metric: One of "similarity", "quality", or "combined"
            similarity_weight: Weight for similarity (when metric="combined")
            quality_weight: Weight for quality (when metric="combined")

        Examples:
            >>> # Use only similarity for ranking
            >>> reward.set_ranking_metric("similarity")
            >>>
            >>> # Use combined metric with custom weights
            >>> reward.set_ranking_metric(
            ...     "combined",
            ...     similarity_weight=0.3,
            ...     quality_weight=0.7
            ... )
        """
        valid_metrics = ["similarity", "quality", "combined"]
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of {valid_metrics}"
            )

        self.ranking_metric = metric

        if metric == "combined":
            # Validate weights
            total = similarity_weight + quality_weight
            if total <= 0:
                raise ValueError(
                    f"Sum of weights must be positive, got {total} "
                    f"(similarity_weight={similarity_weight}, quality_weight={quality_weight})"
                )

            # Normalize weights
            self.similarity_weight = similarity_weight / total
            self.quality_weight = quality_weight / total

            logger.info(
                f"Ranking metric set to 'combined' with weights: "
                f"similarity={self.similarity_weight:.2f}, quality={self.quality_weight:.2f}"
            )
        else:
            logger.info(f"Ranking metric set to '{metric}'")


# Convenience factory functions
def create_qwen_ranking_reward(
    api_key: Optional[str] = None,
    model_name: str = "qwen-vl-plus",
    ranking_metric: str = "combined",
    use_parallel: bool = True,
    max_concurrent: int = 5,
    **kwargs,
) -> QwenMultimodalRankingReward:
    """
    Factory function to create Qwen ranking reward with custom configuration.

    Args:
        api_key: DashScope API key (defaults to env var)
        model_name: Qwen model to use
        ranking_metric: Metric for ranking ("similarity", "quality", "combined")
        use_parallel: Whether to evaluate candidates in parallel
        max_concurrent: Maximum concurrent API calls
        **kwargs: Additional parameters for QwenMultimodalRankingReward

    Returns:
        Configured QwenMultimodalRankingReward instance

    Examples:
        >>> # Create with defaults (parallel, combined metric)
        >>> reward = create_qwen_ranking_reward()
        >>>
        >>> # Create with custom configuration
        >>> reward = create_qwen_ranking_reward(
        ...     model_name="qwen-vl-max",
        ...     ranking_metric="quality",
        ...     use_parallel=True,
        ...     max_concurrent=10
        ... )
    """
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Set DASHSCOPE_API_KEY environment variable "
            "or pass api_key parameter."
        )

    vlm_api = QwenVLAPI(api_key=api_key, model_name=model_name, enable_cache=True)

    return QwenMultimodalRankingReward(
        vlm_api=vlm_api,
        ranking_metric=ranking_metric,
        use_parallel_evaluation=use_parallel,
        max_concurrent=max_concurrent,
        **kwargs,
    )
