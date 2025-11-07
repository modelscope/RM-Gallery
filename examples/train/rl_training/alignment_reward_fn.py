"""
Alignment Reward Function for RL Training

Provides reward calculation for RL training (VERL, OpenRLHF, etc.) using RM-Gallery's
LLM as Judge evaluation. Supports 6 evaluation modes:

Pairwise (4 variants):
- winrate: Simple win rate
- copeland: Copeland (net wins)
- dgr: DGR (TFAS algorithm, resolve cycles)
- elo: ELO Rating

Other modes:
- pointwise: Direct scoring (1-10)
- listwise: Ranking

Compatible with RL frameworks' compute_score interface.
"""

import os
import sys
from typing import Any, Dict

# Add RM-Gallery to path if needed
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rm_gallery.gallery.evaluation.llm_judge import AlignmentReward

# ============================================================================
# Configuration
# ============================================================================

# API Configuration (can be overridden by environment variables)
MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "qwen3-32b")
API_URL = os.getenv("JUDGE_API_URL", "http://your-api-url/v1/chat/completions")
API_KEY = os.getenv("JUDGE_API_KEY", "your-api-key")

# Evaluation Mode Configuration
EVAL_MODE = os.getenv("EVAL_MODE", "pairwise")  # pairwise, pointwise, listwise
PAIRWISE_MODE = os.getenv("PAIRWISE_MODE", "dgr")  # dgr, copeland, winrate, elo

# Performance Configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30.0"))  # LLM call timeout (seconds)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # LLM call retry attempts on failure

# Debug Configuration
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


# ============================================================================
# Custom Prompt (Optional)
# ============================================================================


def custom_alignment_prompt(user_query, response_a, response_b, reference="", **kwargs):
    """
    Custom prompt for alignment evaluation

    This function generates the prompt for LLM judge to compare two responses.
    You can customize this to match your evaluation criteria.
    """
    reference_section = ""
    if reference and reference.strip():
        reference_section = f"""
## Reference Response
{reference}

"""

    return f"""# AI Assistant Response Quality Evaluation

Please evaluate the quality of the following two AI assistant responses based on safety, helpfulness, accuracy, and clarity.

## User Query
{user_query}

## Response A
{response_a}

## Response B
{response_b}
{reference_section}

## Instructions
**Prefer Clear Winners**: Look for meaningful differences in:
- **Safety**: Avoid harmful, biased, or inappropriate content
- **Helpfulness**: Provide useful and relevant information
- **Accuracy**: Ensure factual correctness
- **Clarity**: Clear and well-structured communication

Even small but clear advantages should lead to a winner, not a tie. A tie should be reserved for genuinely equivalent responses.

**Provide only your final judgment without any analysis or reasoning process.**

## Evaluation Result
Please provide your evaluation result in the <result></result> tags, choosing only one of the following three options:
- "A": If Response A is better
- "B": If Response B is better
- "tie": If both responses are of similar quality


<result>Your evaluation result</result>
"""


# ============================================================================
# Create Reward Instance
# ============================================================================

# Create unified reward instance
alignment_reward = AlignmentReward(
    model_name=MODEL_NAME,
    api_url=API_URL,
    api_key=API_KEY,
    eval_mode=EVAL_MODE,
    pairwise_mode=PAIRWISE_MODE,
    custom_prompt_fn=custom_alignment_prompt,
    result_tag="result",
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    max_workers=MAX_WORKERS,
    verbose=VERBOSE,
    llm_timeout=LLM_TIMEOUT,  # Add timeout configuration
    max_retries=MAX_RETRIES,  # Add retry configuration
)

# Log configuration
if VERBOSE:
    reward_type = (
        f"{EVAL_MODE}_{PAIRWISE_MODE}" if EVAL_MODE == "pairwise" else EVAL_MODE
    )
    print("[Reward Function] Configuration:")
    print(f"  Reward Type: {reward_type}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API URL: {API_URL}")
    print(f"  Eval Mode: {EVAL_MODE}")
    if EVAL_MODE == "pairwise":
        print(f"  Pairwise Mode: {PAIRWISE_MODE}")
    print(f"  Max Workers: {MAX_WORKERS}")
    print(f"  LLM Timeout: {LLM_TIMEOUT}s")
    print(f"  Max Retries: {MAX_RETRIES}")


# ============================================================================
# VERL-Compatible Interface
# ============================================================================


def compute_score(
    data_source: str,
    solution_str: Any,
    ground_truth: Dict,
    extra_info: Dict = None,
    group_evaluation: bool = False,
    prompt_str: str = None,
    all_responses: list = None,
    all_extra_infos: list = None,
    **kwargs,
) -> Dict:
    """
    VERL-compatible compute_score interface

    This function is called by VERL framework for reward calculation.
    Keep this interface unchanged to maintain compatibility.

    Parameters:
        data_source: Data source identifier
        solution_str: Model generated response(s)
        ground_truth: Ground truth data (contains chosen/rejected)
        extra_info: Additional information
        group_evaluation: Whether in group evaluation mode
        prompt_str: User prompt
        all_responses: All responses (for batch evaluation)
        all_extra_infos: All extra info (for batch evaluation)
        **kwargs: Additional arguments

    Returns:
        Dict containing scores and metadata
    """
    return alignment_reward.compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        group_evaluation=group_evaluation,
        prompt_str=prompt_str,
        all_responses=all_responses,
        all_extra_infos=all_extra_infos,
        **kwargs,
    )


# ============================================================================
# Testing (Optional)
# ============================================================================

if __name__ == "__main__":
    # Simple test
    print("Testing alignment reward function...")

    test_data = {
        "data_source": "test",
        "solution_str": [
            "This is a detailed and helpful response with accurate information.",
            "This is a brief response.",
        ],
        "ground_truth": {
            "chosen": [
                {"role": "user", "content": "What is Python?"},
                {
                    "role": "assistant",
                    "content": "Python is a high-level programming language.",
                },
            ]
        },
        "extra_info": {"x": [{"role": "user", "content": "What is Python?"}]},
    }

    result = compute_score(**test_data, group_evaluation=True)

    print("\n=== Test Result ===")
    print(f"Scores: {result.get('group_scores', [])}")
    print(f"Mode: {result.get('mode', 'unknown')}")
    print(f"Number of responses: {result.get('n_responses', 0)}")

    if "conflicts_removed" in result:
        print(f"Conflicts removed (DGR): {result['conflicts_removed']}")

    print("\n✓ Test completed successfully!")
