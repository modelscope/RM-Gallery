# Tutorial: Using RM-Gallery Reward Models in Post Training

This tutorial provides a detailed guide on how to use RM-Gallery reward models for post training within the VERL framework. We will focus on implementing custom reward managers, including asynchronous processing of prompt groups and support for pairwise rewards.

## 1. Overview

In Reinforcement Learning from Human Feedback (RLHF) and other post training methods, reward models play a crucial role. This tutorial will demonstrate how to:

1. **Integrate RM-Gallery into VERL Framework**: Create custom reward managers to support complex reward computations
2. **Asynchronous Prompt Group Processing**: Improve computational efficiency and support batch processing of multiple candidate responses for the same prompt
3. **Support Pairwise Rewards**: Implement more precise preference learning in algorithms like GRPO

### Key Features

- **Asynchronous Parallel Computing**: Support parallel processing of multiple prompt groups, significantly improving efficiency
- **Flexible Reward Composition**: Support combination of multiple reward functions (rubric-based rewards, format rewards, length rewards, etc.)
- **Pairwise Comparison**: Support pairwise comparisons to provide more precise preference signals for algorithms like GRPO
- **Statistical Information Tracking**: Automatically calculate and record reward distribution statistics for training monitoring


## 2. Environment Setup

First, ensure that the necessary dependencies are installed:



```python
# Install necessary dependencies
%pip install rm-gallery
%pip install verl

# Import necessary libraries
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import torch
from verl import DataProto

# Import RM-Gallery components
from rm_gallery.core.reward import RewardRegistry
from rm_gallery.core.reward.composition import RewardComposition
from rm_gallery.gallery.rm.general import GeneralReward

```

## 3. Core Implementation of Custom Reward Manager

### 3.1 Asynchronous Single Group Reward Computation Function

First, implement asynchronous processing for reward computation of a single prompt group:



```python
async def single_compute_score(compute_score, prompt, responses, extras, reward_kwargs, meta_info, executor, timeout=300.0):
    """
    Asynchronous task for single group reward computation

    Args:
        compute_score: Reward computation function
        prompt: Input prompt
        responses: List of candidate responses
        extras: Additional information
        reward_kwargs: Reward computation parameters
        meta_info: Meta information
        executor: Thread pool executor
        timeout: Timeout duration

    Returns:
        Computed reward scores and detailed information
    """
    loop = asyncio.get_running_loop()
    task = asyncio.wait_for(
        loop.run_in_executor(
            executor,
            partial(compute_score, prompt=prompt, responses=responses, extras=extras, **reward_kwargs, **meta_info),
        ),
        timeout=timeout,
    )
    return await task

```

### 3.2 Custom Reward Manager Class

This is the core Reward Manager implementation, including asynchronous parallel processing and pairwise comparison functionality:



```python
class RMGalleryRewardManager:
    """
    Custom reward manager based on RM-Gallery

    Core Features:
    1. Asynchronous parallel processing: Support parallel computation of multiple prompt groups
    2. Pairwise comparison: Provide pairwise comparison reward signals for algorithms like GRPO
    3. Flexible reward composition: Support combination of multiple reward functions
    4. Statistical tracking: Automatically compute reward distribution statistics
    """

    def __init__(self, tokenizer, num_examine=3, is_val_mode=False, compute_score=None,
                 reward_fn_key="data_source", **reward_kwargs):
        """
        Initialize Reward Manager

        Args:
            tokenizer: Tokenizer for decoding
            num_examine: Number of samples to print during debugging
            is_val_mode: Whether in validation mode (supports pairwise comparison)
            compute_score: Reward computation function
            reward_fn_key: Data source key name
            **reward_kwargs: Additional parameters for reward computation
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.is_val_mode = is_val_mode
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.max_workers = reward_kwargs.get("max_workers", 8)
        self.timeout = reward_kwargs.get("timeout", 300.0)
        self.meta_info = {}

        # Initialize RM-Gallery reward components
        if compute_score is None:
            self._init_rm_gallery_components()

    def _init_rm_gallery_components(self):
        """Initialize RM-Gallery reward components"""
        # Get reward functions from registry
        registry = RewardRegistry()

        # Combine multiple reward functions
        self.reward_composition = RewardComposition([
            registry.get("general"),  # General reward
            registry.get("format"),   # Format reward
            registry.get("length"),   # Length reward
        ])

        self.compute_score = self.reward_composition

```


```python
# Continue RMGalleryRewardManager class with asynchronous parallel computation methods
def extend_reward_manager():
    """Extend RMGalleryRewardManager class by adding parallel computation methods"""

    async def parallel_compute_scores(self, prompt_to_indices, responses_str, extras_info):
        """
        Parallel computation of reward scores for multiple groups

        This is the core function for asynchronous processing, which groups candidate responses
        with the same prompt and computes them in parallel across different groups,
        significantly improving computational efficiency.
        """
        all_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create asynchronous tasks for each group
            tasks = []
            for prompt, indices in prompt_to_indices.items():
                group_responses = [responses_str[i] for i in indices]
                group_extras = [extras_info[i] for i in indices]

                # In validation mode, add reference answer for pairwise comparison
                if self.is_val_mode:
                    reference_response = group_extras[0]["y"][0].get("content", "")
                    group_responses.append(reference_response)
                    group_extras.append(group_extras[0])

                # Create asynchronous task
                task = single_compute_score(
                    self.compute_score, prompt, group_responses, group_extras,
                    self.reward_kwargs, self.meta_info, executor, timeout=self.timeout
                )
                tasks.append((task, indices))

            # Execute all tasks in parallel
            results = await asyncio.gather(*(task for task, _ in tasks))

            # Process pairwise comparison results
            for (result, indices) in zip(results, [indices for _, indices in tasks]):
                if self.is_val_mode:
                    scores, reward_info = result[0], result[1]
                    scores = scores[:-1]  # Remove reference answer score

                    # Calculate win rate statistics (key metric for pairwise comparison)
                    comparison_scores = reward_info["comparison_score"]
                    win_rate = [1.0 if comparison_scores[0] > comparison_scores[1] else 0.0]
                    win_and_rate = [1.0 if comparison_scores[0] >= comparison_scores[1] else 0.0]

                    # Update reward information
                    for key, vals in reward_info.items():
                        reward_info[key] = vals[:-1]
                    reward_info.update({"win": win_rate, "win_and": win_and_rate})

                    print(f"Pairwise results: scores={scores}, win_rate={win_rate}")
                    result = (scores, reward_info)

                all_results.append((result, indices))

        return all_results

    # Add method to the class
    RMGalleryRewardManager.parallel_compute_scores = parallel_compute_scores

extend_reward_manager()

```


```python
# Add main call method to RMGalleryRewardManager class
def add_call_method():
    """Add main __call__ method"""

    def __call__(self, data: DataProto, return_dict=False):
        """
        Calculate reward values for input data, supports batch processing and async parallel computation

        Args:
            data: Data object containing model inputs and outputs
            return_dict: Whether to return results as dictionary

        Returns:
            Reward tensor or dictionary containing reward information
        """
        # If reward scores already exist, return directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Initialize reward tensor
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        # Update meta info (for statistical tracking)
        if data.meta_info.get("last_reward_info", None) is not None:
            self.meta_info.update({"last_mean_std": np.mean(data.meta_info["last_reward_info"]["reward_std"])})

        # Decode prompt and response
        responses_str = []
        prompts_str = []
        extras_info = []

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
            prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
            extra_info = data.non_tensor_batch['extra_info'][i]

            responses_str.append(response_str)
            prompts_str.append(prompt_str)
            extras_info.append(extra_info)

        # Group by prompt (key for async parallel processing)
        prompt_to_indices = defaultdict(list)
        for i, prompt in enumerate(prompts_str):
            prompt_to_indices[prompt].append(i)

        # Validate consistent sample count per group
        group_sizes = [len(indices) for indices in prompt_to_indices.values()]
        if len(set(group_sizes)) > 1:
            raise AssertionError(f"Sample count must be same per group, current group_sizes: {group_sizes}")

        print(f"Total {len(prompt_to_indices)} groups, {group_sizes[0]} samples per group, starting async parallel computation...")

        # Run async parallel computation
        all_results = asyncio.run(
            self.parallel_compute_scores(prompt_to_indices, responses_str, extras_info)
        )

        # Process results
        all_rewards = [0.0] * len(data)
        all_reward_infos = defaultdict(list)

        for result, indices in all_results:
            scores, reward_info = result[0], result[1]

            # Map scores back to original indices
            for score, idx in zip(scores, indices):
                all_rewards[idx] = score

            # Process reward info
            if reward_info and isinstance(reward_info, dict):
                for key, values in reward_info.items():
                    if key not in all_reward_infos:
                        all_reward_infos[key] = [0.0] * len(data)
                    for value, idx in zip(values, indices):
                        all_reward_infos[key][idx] = value

        # Populate reward tensor
        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            reward = all_rewards[i]
            reward_tensor[i, length - 1] = reward

            # Debug output
            if i < self.num_examine:
                print(f"[Sample {i}] Prompt: {prompts_str[i]}")
                print(f"[Sample {i}] Response: {responses_str[i]}")
                print(f"[Sample {i}] Score: {reward}")

        # Add accuracy info
        data.batch["acc"] = torch.tensor(all_rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(all_reward_infos)}
        else:
            return reward_tensor

    # Add method to class
    RMGalleryRewardManager.__call__ = __call__

add_call_method()

```

## 4. RM-Gallery Reward Function Implementation

Next, we implement the RM-Gallery-based reward computation function that supports combination of multiple reward types:



```python
def create_rm_gallery_reward_function(use_group_reward=True, return_details=False, return_statistics=True):
    """
    Create RM-Gallery-based reward computation function

    Args:
        use_group_reward: Whether to use group reward (supports pairwise comparison)
        return_details: Whether to return detailed information
        return_statistics: Whether to return statistical information

    Returns:
        Configured reward computation function
    """

    def reward_func(prompt, responses, extras=None, **kwargs):
        """
        Comprehensive reward computation function that combines multiple reward types

        Reward combination includes:
        1. Rubric-based rewards (95% weight): Based on helpfulness, harmlessness, honesty rubrics
        2. Format rewards (5% weight): Ensure output format correctness
        3. Length rewards: Control appropriate response length
        4. N-gram rewards: Reduce penalties for repetitive content
        """
        details = []

        # Ensure responses is in list format
        if not isinstance(responses, list):
            responses = [responses]
            if prompt and not isinstance(prompt, list):
                prompt = [prompt]

        # 1. Rubric-based reward computation (core reward)
        if use_group_reward:
            # Group reward supporting pairwise comparison
            scores_rubric, details = group_rm_gallery_grader(prompt, responses, extras, **kwargs)
        else:
            # Individual scoring reward
            scores_rubric, details = rm_gallery_grader(prompt, responses, extras, **kwargs)

        # 2. Format reward computation
        scores_format = compute_format_reward(responses)

        # 3. N-gram repetition penalty
        ngram_penalty_fn = create_ngram_penalty_reward(ngram_size=5, max_penalty=-1.0, min_scaling=0.1)
        scores_ngram = ngram_penalty_fn(responses)

        # 4. Length reward computation
        scores_thought_length, thought_lengths = compute_thought_length_reward(responses)
        scores_total_length, total_lengths = compute_total_length_reward(responses)

        # Convert to tensor format
        scores_rubric = torch.tensor(scores_rubric)
        scores_format = torch.tensor(scores_format)
        scores_ngram = torch.tensor(scores_ngram)
        scores_thought_length = torch.tensor(scores_thought_length)
        scores_total_length = torch.tensor(scores_total_length)
        thought_lengths = torch.tensor(thought_lengths, dtype=torch.float32)

        # Weighted reward combination
        scores = (0.95 * scores_rubric +
                 0.05 * scores_format +
                 scores_total_length +
                 scores_ngram)

        # Handle invalid rewards (e.g., HTTP errors)
        INVALID_REWARD = -999.0
        scores[scores_rubric == INVALID_REWARD] = INVALID_REWARD
        scores = scores.tolist()

        # Build reward information dictionary
        reward_info = {
            "reward_rubric": scores_rubric.tolist(),
            "reward_format": scores_format.tolist(),
            "reward_ngram": scores_ngram.tolist(),
            "thought_lengths": thought_lengths.tolist(),
            "scores_thought_length": scores_thought_length.tolist(),
            "scores_total_lengths": scores_total_length.tolist(),
        }

        if return_details:
            return scores, reward_info, details
        return scores, reward_info

    return reward_func

# Create reward function instance
rm_gallery_reward_function = create_rm_gallery_reward_function(
    use_group_reward=True,  # Enable pairwise comparison
    return_details=False,
    return_statistics=True
)

```

## 5. Registering Custom Reward Manager in VERL

To use our custom Reward Manager in the VERL framework, we need to register it in VERL's module system:



```python
# Register custom manager in VERL's reward manager initialization file
# File path: verl/workers/reward_manager/__init__.py

registration_code = '''
from .batch import BatchRewardManager
from .dapo import DAPORewardManager
from .naive import NaiveRewardManager
from .prime import PrimeRewardManager
from .rm_gallery import RMGalleryRewardManager  # Add our reward manager

__all__ = [
    "BatchRewardManager",
    "DAPORewardManager",
    "NaiveRewardManager",
    "PrimeRewardManager",
    "RMGalleryRewardManager"  # Add to export list
]
'''

print("Need to add the following registration code to the VERL project:")
print(registration_code)

# Create reward manager configuration example
reward_manager_config = {
    "reward_manager": {
        "type": "RMGalleryRewardManager",
        "args": {
            "num_examine": 3,
            "is_val_mode": True,  # Enable pairwise validation mode
            "compute_score": rm_gallery_reward_function,
            "max_workers": 8,
            "timeout": 300.0,
            "use_group_reward": True,
            "return_details": False,
            "return_statistics": True
        }
    }
}

print("\nConfiguration example:")
import json
print(json.dumps(reward_manager_config, indent=2, ensure_ascii=False))

```

## 6. Core Feature Detailed Explanation

### 6.1 Asynchronous Processing of Prompt Groups

One core innovation of our Reward Manager is **asynchronous parallel processing by prompt grouping**:

#### Why do we need prompt grouping?
During post training, typically multiple candidate responses (e.g., 4-8) are generated for each prompt, and these candidate responses need to be compared with each other to provide preference signals. The traditional approach is to compute rewards for each response individually, but this approach has several problems:

1. **Low efficiency**: Cannot leverage the advantages of batch processing
2. **Lack of comparison**: Cannot perform pairwise comparisons
3. **Resource waste**: Repeated computation of the same prompt's context

#### Our solution:
```python
# Group by prompt
prompt_to_indices = defaultdict(list)
for i, prompt in enumerate(prompts_str):
    prompt_to_indices[prompt].append(i)
```

**Advantages of asynchronous parallel processing:**
- **Intra-group batch processing**: Multiple candidate responses for the same prompt are processed together, supporting pairwise comparison
- **Inter-group parallelism**: Groups with different prompts can be computed in parallel, significantly improving efficiency
- **Resource optimization**: Avoid repeated computation of prompt embeddings, etc.

