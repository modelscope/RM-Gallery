# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DGR Reward Manager Template File

Usage Instructions:
1. Copy this file to <verl_root>/verl/workers/reward_manager/dgr.py
2. Ensure it's registered in <verl_root>/verl/workers/reward_manager/__init__.py
"""

from collections import defaultdict

import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dgr")
class DGRRewardManager(AbstractRewardManager):
    """
    DGR Reward Manager for alignment training with RLAIF evaluation.
    Supports Pointwise, Pairwise, and Listwise evaluation modes.

    This manager evaluates multiple responses generated from the same prompt as a group,
    using LLM as judge through compute_score function to obtain quality scores.
    Supports three RLAIF evaluation modes: Pointwise, Pairwise, and Listwise
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        **kwargs,
    ) -> None:
        """
        Initialize DGR Reward Manager

        Args:
            tokenizer: Tokenizer
            num_examine: Number of batches to print to console
            compute_score: Custom scoring function
            reward_fn_key: Key for obtaining data_source
            **kwargs: Additional parameters passed to compute_score
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.kwargs = kwargs

    def __call__(self, data: DataProto, return_dict=False):
        """
        Compute reward values

        Args:
            data: DataProto object containing batch data
            return_dict: Whether to return dict format (with extra info)

        Returns:
            reward_tensor: Reward tensor
            or
            dict: Dictionary containing reward_tensor and extra info
        """

        # If rm_scores already exist, return directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Initialize reward tensor and extra info
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Group by prompt
        prompt_to_indices = defaultdict(list)
        for idx, prompt_ids in enumerate(data.batch["prompts"]):
            prompt_key = tuple(prompt_ids.tolist())
            prompt_to_indices[prompt_key].append(idx)

        # Evaluate each group
        for prompt_key, indices in prompt_to_indices.items():
            if len(indices) == 0:
                continue

            # Prepare data for group evaluation
            queries = []
            prompts = []
            extras = []

            for idx in indices:
                # Decode response
                response_ids = data.batch["responses"][idx]
                response_text = self.tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )

                # Decode prompt
                prompt_ids = data.batch["prompts"][idx]
                prompt_text = self.tokenizer.decode(
                    prompt_ids, skip_special_tokens=True
                )

                # Build complete query (prompt + response)
                query = prompt_text + response_text
                queries.append(query)
                prompts.append(prompt_text)

                # Extract extra info
                prompt_raw_list = data.non_tensor_batch.get("prompt_raw", [])
                if len(prompt_raw_list) > 0 and idx < len(prompt_raw_list):
                    x_data = prompt_raw_list[idx]
                else:
                    x_data = {}

                extra_info = {
                    "x": x_data,
                }

                # Add ground truth info (if available)
                if "reward_model" in data.non_tensor_batch:
                    rm_list = data.non_tensor_batch["reward_model"]
                    if len(rm_list) > 0 and idx < len(rm_list):
                        rm_data = rm_list[idx]
                        if isinstance(rm_data, dict) and "ground_truth" in rm_data:
                            extra_info.update(rm_data["ground_truth"])

                extras.append(extra_info)

            # Get data_source (safe access, avoid numpy array ambiguous truth value)
            data_source_list = data.non_tensor_batch.get(
                self.reward_fn_key, ["alignment"]
            )
            try:
                if len(data_source_list) > 0 and indices[0] < len(data_source_list):
                    data_source = data_source_list[indices[0]]
                else:
                    data_source = "alignment"
            except (TypeError, ValueError):
                # If data_source_list is numpy array or other special type
                data_source = "alignment"

            # Call compute_score for group evaluation
            result = self.compute_score(
                data_source=data_source,
                solution_str=[q.split(prompts[0])[-1].strip() for q in queries],
                ground_truth=extras[0] if extras else {},
                extra_info=extras[0] if extras else {},
                group_evaluation=True,  # Important: enable group evaluation mode
                prompt_str=prompts[0],
                all_responses=[
                    q.split(prompts[i])[-1].strip() for i, q in enumerate(queries)
                ],
                all_extra_infos=extras,
                **self.kwargs,
            )

            # Extract scores
            if isinstance(result, dict):
                if "group_scores" in result:
                    scores = result["group_scores"]
                elif "score" in result:
                    scores = [result["score"]] * len(indices)
                else:
                    scores = [0.0] * len(indices)

                # Save extra info
                for key, value in result.items():
                    if key not in ["group_scores", "score"]:
                        if isinstance(value, list) and len(value) == len(indices):
                            reward_extra_info[key].extend(value)
                        else:
                            reward_extra_info[key].extend([value] * len(indices))
            else:
                scores = [float(result)] * len(indices)

            # Assign reward values
            for i, idx in enumerate(indices):
                reward_tensor[idx] = scores[i] if i < len(scores) else 0.0

        # Convert extra info to tensor (if possible)
        batch_size = len(data.batch["responses"])
        for key, values in reward_extra_info.items():
            if len(values) == batch_size:
                try:
                    reward_extra_info[key] = torch.tensor(values, dtype=torch.float32)
                except:
                    pass  # Keep as list if conversion fails

        # Return result
        if return_dict:
            return {"reward_tensor": reward_tensor, **reward_extra_info}
        else:
            return reward_tensor
