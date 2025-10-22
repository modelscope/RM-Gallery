"""
Alignment Reward for RL Training

Provides compute_score interface for alignment tasks compatible with RL frameworks (VERL, OpenRLHF, etc.)
Note: For ELO rating, use AlignmentELOReward from pairwise_elo module
"""

from typing import Any, Callable, Dict, List, Optional

from rm_gallery.gallery.evaluation.llm_judge.evaluators.listwise import (
    ListwiseEvaluator,
)
from rm_gallery.gallery.evaluation.llm_judge.evaluators.pairwise import (
    PairwiseEvaluator,
)
from rm_gallery.gallery.evaluation.llm_judge.evaluators.pointwise import (
    PointwiseEvaluator,
)
from rm_gallery.gallery.evaluation.llm_judge.templates.alignment import (
    AlignmentPairwiseTemplate,
)
from rm_gallery.gallery.evaluation.llm_judge.templates.base import (
    ListwisePromptTemplate,
    PointwisePromptTemplate,
)


class AlignmentReward:
    """
    Alignment task reward function for RL training

    Directly uses LLM Judge evaluators.
    Supports:
    - Pairwise (winrate/copeland/dgr/elo)
    - Pointwise
    - Listwise
    """

    def __init__(
        self,
        model_name: str = "qwen3-32b",
        api_url: str = None,
        base_url: str = None,
        api_key: str = None,
        eval_mode: str = "pairwise",
        pairwise_mode: str = "dgr",  # 'dgr', 'copeland', 'winrate', 'elo'
        custom_prompt_fn: Optional[Callable] = None,
        custom_parse_fn: Optional[Callable] = None,
        result_tag: str = "result",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_workers: int = 10,
        verbose: bool = False,
        llm_timeout: float = 30.0,  # LLM调用超时（秒）
        max_retries: int = 2,  # LLM调用失败时的重试次数
        **kwargs,
    ):
        """
        Initialize Alignment Reward

        Parameters:
            model_name: LLM model name
            api_url: API endpoint URL (backward compatibility)
            base_url: API endpoint URL (preferred)
            api_key: API key
            eval_mode: 'pairwise', 'pointwise', or 'listwise'
            pairwise_mode: 'dgr' (TFAS), 'copeland' (net wins), 'winrate', or 'elo' (ELO Rating)
            custom_prompt_fn: Custom prompt generation function
            custom_parse_fn: Custom result parsing function
            result_tag: XML tag for result extraction
            temperature: LLM temperature
            max_tokens: Max tokens for LLM
            max_workers: Concurrency level
            verbose: Enable verbose logging
        """
        # Create LLM instance (using RM-Gallery's BaseLLM)
        from rm_gallery.core.model.openai_llm import OpenaiLLM

        # Support both api_url and base_url for backward compatibility
        effective_base_url = base_url or api_url

        # Prepare LLM kwargs
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": llm_timeout,  # 添加超时配置
            "max_retries": max_retries,  # 添加重试配置
        }

        if api_key:
            llm_kwargs["openai_api_key"] = api_key
        if effective_base_url:
            llm_kwargs["base_url"] = effective_base_url

        # Add other kwargs, but avoid duplicates
        for k, v in kwargs.items():
            if k not in llm_kwargs:
                llm_kwargs[k] = v

        self.llm = OpenaiLLM(**llm_kwargs)

        # Create template based on eval_mode
        if eval_mode == "pointwise":
            template = PointwisePromptTemplate(
                prompt_function=custom_prompt_fn,
                parse_function=custom_parse_fn,
                result_tag=result_tag,
            )
        elif eval_mode == "listwise":
            template = ListwisePromptTemplate(
                prompt_function=custom_prompt_fn,
                parse_function=custom_parse_fn,
                result_tag=result_tag,
            )
        else:  # pairwise
            template = AlignmentPairwiseTemplate(
                prompt_function=custom_prompt_fn,
                parse_function=custom_parse_fn,
                result_tag=result_tag,
            )

        # Create evaluator directly based on eval_mode
        if eval_mode == "pairwise":
            self.evaluator = PairwiseEvaluator(
                llm=self.llm,
                template=template,
                mode=pairwise_mode,
                max_workers=max_workers,
                verbose=verbose,
                llm_timeout=llm_timeout,  # 传递超时配置
                max_retries=max_retries,  # 传递重试配置
                **kwargs,
            )
        elif eval_mode == "pointwise":
            self.evaluator = PointwiseEvaluator(
                llm=self.llm,
                template=template,
                max_workers=max_workers,
                verbose=verbose,
                **kwargs,
            )
        elif eval_mode == "listwise":
            self.evaluator = ListwiseEvaluator(
                llm=self.llm, template=template, verbose=verbose, **kwargs
            )
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

        self.eval_mode = eval_mode
        self.pairwise_mode = pairwise_mode
        self.verbose = verbose

    def compute_score(
        self,
        data_source: str,
        solution_str: Any,
        ground_truth: Dict,
        extra_info: Dict = None,
        group_evaluation: bool = False,
        prompt_str: str = None,
        all_responses: List = None,
        all_extra_infos: List = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        VERL-compatible compute_score interface

        This interface maintains full compatibility with original DGR,
        user code requires no modification

        Parameters:
            data_source: Data source type
            solution_str: Model generated response (single or list)
            ground_truth: Ground truth label (contains chosen/rejected info)
            extra_info: Extra information (contains original data)
            group_evaluation: Whether in group evaluation mode
            prompt_str: Prompt string
            all_responses: All response list
            all_extra_infos: All extra info list

        Returns:
            For group evaluation or multiple responses:
                {
                    'group_scores': [0.8, 0.5, -0.3, ...],
                    'mode': 'pairwise_dgr',
                    ...
                }
            For single response (non-group):
                {
                    'score': 0.8,
                    'mode': 'pairwise_dgr',
                    ...
                }
        """
        try:
            # Standardize input: ensure solution_str is list format
            if isinstance(solution_str, str):
                solution_strs = [solution_str]
                is_single_response = True
            elif isinstance(solution_str, (list, tuple)):
                solution_strs = list(solution_str)
                is_single_response = len(solution_strs) == 1
            else:
                solution_strs = [str(solution_str)]
                is_single_response = True

            # Filter thinking parts (if present)
            solution_strs = [self._filter_thinking_parts(sol) for sol in solution_strs]

            # Extract prompt
            if prompt_str is None and extra_info and "x" in extra_info:
                prompt_str = self._extract_prompt_from_extra_info(extra_info)

            if prompt_str is None:
                prompt_str = ""

            # Extract reference (chosen response)
            reference = self._extract_reference(extra_info, ground_truth)

            # Call evaluator for evaluation
            result = self.evaluator.evaluate(
                prompt=prompt_str,
                responses=solution_strs,
                reference=reference,
                **kwargs,
            )

            # Add mode identifier
            if "mode" not in result or result["mode"] == "unknown":
                if self.eval_mode == "pairwise":
                    result["mode"] = f"pairwise_{self.pairwise_mode}"
                else:
                    result["mode"] = self.eval_mode

            # Format result for VERL
            return self._format_verl_result(
                result, group_evaluation, is_single_response
            )

        except Exception as e:
            if self.verbose:
                print(f"Error in alignment compute_score: {e}")
                import traceback

                traceback.print_exc()

            # Return default value - decide format based on evaluation mode
            num_responses = (
                len(solution_str) if isinstance(solution_str, (list, tuple)) else 1
            )
            if group_evaluation or num_responses > 1:
                return {"group_scores": [0.0] * num_responses, "error": str(e)}
            else:
                return {"score": 0.0, "error": str(e)}

    def _filter_thinking_parts(self, text: str) -> str:
        """Filter out thinking parts from text (e.g., <think>...</think>)"""
        if not isinstance(text, str):
            return text

        import re

        # Define thinking part patterns
        thinking_patterns = [
            r"<think>.*?</think>",
            r"<thinking>.*?</thinking>",
            r"【思考】.*?【/思考】",
            r"\[思考\].*?\[/思考\]",
        ]

        # Apply all patterns
        filtered_text = text
        for pattern in thinking_patterns:
            filtered_text = re.sub(
                pattern, "", filtered_text, flags=re.DOTALL | re.IGNORECASE
            )

        # Clean up extra whitespace
        filtered_text = re.sub(r"\n\s*\n", "\n\n", filtered_text)
        filtered_text = filtered_text.strip()

        return filtered_text

    def _extract_prompt_from_extra_info(self, extra_info: Dict) -> str:
        """Extract user prompt from extra_info"""
        if "x" in extra_info and extra_info["x"]:
            # Find user message
            for message in extra_info["x"]:
                if isinstance(message, dict) and message.get("role") == "user":
                    return message.get("content", "")

            # If no user message found, return first message content
            if extra_info["x"]:
                first_msg = extra_info["x"][0]
                if isinstance(first_msg, dict):
                    return first_msg.get("content", "")

        return ""

    def _extract_reference(self, extra_info: Dict, ground_truth: Dict) -> str:
        """Extract reference response (chosen) from extra_info or ground_truth"""
        # Try extra_info first
        if extra_info and "chosen" in extra_info:
            chosen_messages = extra_info["chosen"]
            if isinstance(chosen_messages, (list, tuple)):
                for msg in chosen_messages:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return msg.get("content", "")

        # Try ground_truth
        if ground_truth and "chosen" in ground_truth:
            chosen_messages = ground_truth["chosen"]
            if isinstance(chosen_messages, (list, tuple)):
                for msg in chosen_messages:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return msg.get("content", "")

        return ""

    def _format_verl_result(
        self, result: Dict, group_evaluation: bool, is_single_response: bool
    ) -> Dict:
        """
        Format evaluator result for RL framework (VERL, OpenRLHF, etc.)

        VERL expects:
        - group_evaluation=True or multiple responses: return 'group_scores' field
        - single response (non-group): return 'score' field (scalar)
        """
        scores = result.get("scores", [])

        # Convert to list of floats (pure Python implementation)
        if isinstance(scores, (list, tuple)):
            scores_list = [float(s) for s in scores]
        else:
            # Handle other iterable types
            scores_list = list(float(s) for s in scores)

        if group_evaluation or not is_single_response:
            # Group evaluation mode: return group_scores
            verl_result = {
                "group_scores": scores_list,
                "mode": result.get("mode", "unknown"),
                "n_responses": result.get("n_responses", len(scores_list)),
            }

            # Add extra info from evaluator
            if "removed_edges_count" in result:
                verl_result["conflicts_removed"] = result["removed_edges_count"]
            if "net_wins" in result:
                verl_result["net_wins"] = result["net_wins"]
            if "winrates" in result:
                verl_result["winrates"] = result["winrates"]

        else:
            # Single response mode: return scalar score
            verl_result = {
                "score": float(scores_list[0]) if scores_list else 0.0,
                "mode": result.get("mode", "unknown"),
            }

        return verl_result
