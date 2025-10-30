"""
Qwen VL Visual Helpfulness Reward Model.

Evaluates the helpfulness and quality of text responses given visual context.
Uses Qwen VL to assess accuracy, relevance, detail, and usefulness of responses
based on image content.
"""

import os
from typing import List, Optional

from loguru import logger
from pydantic import Field

from rm_gallery.core.data.multimodal_content import ImageContent
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.reward.vlm_reward import BasePointWiseVLMReward


@RewardRegistry.register("qwen_visual_helpfulness")
class QwenVisualHelpfulnessReward(BasePointWiseVLMReward):
    """
    Qwen VL Visual Helpfulness Reward.

    Evaluates the quality and helpfulness of text responses given visual context.
    Assesses multiple dimensions: accuracy, detail, relevance, and usefulness.

    Key Features:
    - Multi-criteria evaluation (accuracy, detail, relevance, usefulness)
    - Customizable evaluation criteria and prompts
    - Supports both Chinese and English evaluation
    - Built-in caching and cost tracking
    - Async API calls for performance

    Attributes:
        name: Reward model identifier
        vlm_api: Qwen VL API client
        evaluation_criteria: List of criteria for evaluation
        evaluation_prompt_template: Prompt template for quality assessment
        use_detailed_rubric: Whether to include detailed scoring rubric

    Use Cases:
        - Visual Question Answering (VQA) quality assessment
        - Image captioning evaluation
        - Multimodal chatbot response rating
        - Visual instruction following validation

    Examples:
        >>> # Initialize reward model
        >>> reward = QwenVisualHelpfulnessReward(
        ...     vlm_api=QwenVLAPI(
        ...         api_key=os.getenv("DASHSCOPE_API_KEY"),
        ...         model_name="qwen-vl-max"  # Use max for better quality assessment
        ...     ),
        ...     use_detailed_rubric=True
        ... )
        >>>
        >>> # Evaluate a visual Q&A response
        >>> sample = DataSample(
        ...     unique_id="vqa_001",
        ...     input=[MultimodalChatMessage(
        ...         role=MessageRole.USER,
        ...         content=MultimodalContent(
        ...             text="What is the person doing in this image?",
        ...             images=[ImageContent(type="url", data="https://...")]
        ...         )
        ...     )],
        ...     output=[DataOutput(
        ...         answer=ChatMessage(
        ...             role=MessageRole.ASSISTANT,
        ...             content="The person is riding a bicycle in the park on a sunny day."
        ...         )
        ...     )]
        ... )
        >>> result = reward.evaluate(sample)
        >>> print(f"Helpfulness score: {result.output[0].answer.reward.score:.3f}")
    """

    name: str = Field(
        default="qwen_visual_helpfulness", description="Reward model name"
    )

    vlm_api: QwenVLAPI = Field(
        default_factory=lambda: QwenVLAPI(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            model_name="qwen-vl-plus",
            enable_cache=True,
        ),
        description="Qwen VL API client",
    )

    evaluation_criteria: List[str] = Field(
        default=[
            "准确性：回答是否正确描述了图片内容",
            "详细性：回答是否提供了足够的细节信息",
            "相关性：回答是否切合问题的核心",
            "有用性：回答是否对用户有实际帮助",
        ],
        description="Evaluation criteria for helpfulness assessment",
    )

    evaluation_criteria_en: List[str] = Field(
        default=[
            "Accuracy: Does the answer correctly describe the image content",
            "Detail: Does the answer provide sufficient details",
            "Relevance: Does the answer address the core of the question",
            "Usefulness: Is the answer practically helpful to the user",
        ],
        description="English evaluation criteria",
    )

    evaluation_prompt_template: str = Field(
        default=(
            "请基于图片内容，评估以下回答的质量（0-10分）：\n\n"
            "问题：{question}\n"
            "回答：{answer}\n\n"
            "评估标准：\n"
            "{criteria}\n\n"
            "{rubric}"
            "只回答一个0-10的数字，不要有任何其他内容。"
        ),
        description="Prompt template for quality evaluation",
    )

    evaluation_prompt_template_en: str = Field(
        default=(
            "Based on the image content, evaluate the quality of the following answer (0-10 score):\n\n"
            "Question: {question}\n"
            "Answer: {answer}\n\n"
            "Evaluation Criteria:\n"
            "{criteria}\n\n"
            "{rubric}"
            "Only respond with a single number between 0-10, no other content."
        ),
        description="English prompt template",
    )

    detailed_rubric: str = Field(
        default=(
            "评分细则：\n"
            "- 9-10分：优秀 - 准确、详细、相关且非常有用\n"
            "- 7-8分：良好 - 大部分准确和有用，细节充分\n"
            "- 5-6分：及格 - 基本正确，但缺少细节或相关性不足\n"
            "- 3-4分：较差 - 部分错误或不够相关\n"
            "- 0-2分：很差 - 严重错误或完全不相关\n\n"
        ),
        description="Detailed scoring rubric",
    )

    detailed_rubric_en: str = Field(
        default=(
            "Scoring Rubric:\n"
            "- 9-10: Excellent - Accurate, detailed, relevant, and very helpful\n"
            "- 7-8: Good - Mostly accurate and useful, sufficient details\n"
            "- 5-6: Fair - Basically correct, but lacks details or relevance\n"
            "- 3-4: Poor - Partially incorrect or not very relevant\n"
            "- 0-2: Very Poor - Seriously incorrect or completely irrelevant\n\n"
        ),
        description="English scoring rubric",
    )

    use_detailed_rubric: bool = Field(
        default=True, description="Whether to include detailed scoring rubric in prompt"
    )

    use_english_prompt: bool = Field(
        default=False, description="Whether to use English prompt"
    )

    async def _compute_reward(
        self, images: List[ImageContent], texts: List[str], **kwargs
    ) -> float:
        """
        Compute visual helpfulness score using Qwen VL API.

        Args:
            images: List of images (uses first image)
            texts: List of texts - expects [answer, question, ...]
                   First text is the answer to evaluate
                   Second text (optional) is the original question
            **kwargs: Additional parameters

        Returns:
            Helpfulness score in range [0, 1]
        """
        # Validate inputs
        if not images:
            logger.warning("No images provided for helpfulness evaluation")
            return self.fallback_score

        if not texts or len(texts) == 0:
            logger.warning("No answer text provided for evaluation")
            return self.fallback_score

        # Extract answer and optional question
        answer = texts[0].strip()
        question = texts[1].strip() if len(texts) > 1 else "描述这张图片"  # Default question

        if not answer:
            logger.warning("Empty answer provided")
            return self.fallback_score

        # Use first image
        image = images[0]

        try:
            # Select language-specific templates
            if self.use_english_prompt:
                criteria_list = self.evaluation_criteria_en
                prompt_template = self.evaluation_prompt_template_en
                rubric = self.detailed_rubric_en if self.use_detailed_rubric else ""
            else:
                criteria_list = self.evaluation_criteria
                prompt_template = self.evaluation_prompt_template
                rubric = self.detailed_rubric if self.use_detailed_rubric else ""

            # Format criteria
            criteria_str = "\n".join(
                [f"{i+1}. {criterion}" for i, criterion in enumerate(criteria_list)]
            )

            # Build complete evaluation prompt
            evaluation_prompt = prompt_template.format(
                question=question, answer=answer, criteria=criteria_str, rubric=rubric
            )

            # Call VLM API directly with custom prompt (not using evaluate_quality)
            # to avoid double-wrapping of prompts
            response = await self.vlm_api.generate(
                text=evaluation_prompt,
                images=[image],
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=10,
            )

            # Parse score from response
            score = self.vlm_api._parse_score(response.content)

            # Ensure score is in valid range
            score = max(0.0, min(1.0, float(score)))

            logger.debug(
                f"Computed helpfulness score: {score:.3f} for answer length: {len(answer)}, "
                f"question length: {len(question)}"
            )

            return score

        except Exception as e:
            logger.error(f"Failed to compute helpfulness score: {str(e)}")
            return self.fallback_score

    def set_custom_criteria(
        self, criteria: List[str], criteria_en: Optional[List[str]] = None
    ):
        """
        Set custom evaluation criteria.

        Args:
            criteria: Chinese evaluation criteria
            criteria_en: Optional English evaluation criteria

        Examples:
            >>> reward.set_custom_criteria([
            ...     "准确性：是否正确识别了图片中的物体",
            ...     "完整性：是否描述了所有重要元素",
            ...     "清晰度：表达是否清晰易懂"
            ... ])
        """
        self.evaluation_criteria = criteria
        if criteria_en:
            self.evaluation_criteria_en = criteria_en
        logger.info(f"Updated evaluation criteria: {len(criteria)} criteria set")


# Convenience factory functions
def create_qwen_helpfulness_reward(
    api_key: Optional[str] = None,
    model_name: str = "qwen-vl-max",  # Default to max for better evaluation
    custom_criteria: Optional[List[str]] = None,
    use_detailed_rubric: bool = True,
    **kwargs,
) -> QwenVisualHelpfulnessReward:
    """
    Factory function to create Qwen helpfulness reward with custom configuration.

    Args:
        api_key: DashScope API key (defaults to env var)
        model_name: Qwen model to use ("qwen-vl-plus" or "qwen-vl-max")
        custom_criteria: Custom evaluation criteria (optional)
        use_detailed_rubric: Whether to include detailed scoring rubric
        **kwargs: Additional parameters for QwenVisualHelpfulnessReward

    Returns:
        Configured QwenVisualHelpfulnessReward instance

    Examples:
        >>> # Create with defaults
        >>> reward = create_qwen_helpfulness_reward()
        >>>
        >>> # Create with custom criteria
        >>> reward = create_qwen_helpfulness_reward(
        ...     model_name="qwen-vl-max",
        ...     custom_criteria=[
        ...         "是否准确识别了场景",
        ...         "是否描述了关键细节",
        ...         "表达是否自然流畅"
        ...     ],
        ...     use_detailed_rubric=True
        ... )
    """
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Set DASHSCOPE_API_KEY environment variable "
            "or pass api_key parameter."
        )

    vlm_api = QwenVLAPI(api_key=api_key, model_name=model_name, enable_cache=True)

    reward = QwenVisualHelpfulnessReward(
        vlm_api=vlm_api, use_detailed_rubric=use_detailed_rubric, **kwargs
    )

    if custom_criteria:
        reward.set_custom_criteria(custom_criteria)

    return reward
