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
DGR Reward Manager 模板文件

使用说明：
1. 将本文件复制到 <verl_root>/verl/workers/reward_manager/dgr.py
2. 确保在 <verl_root>/verl/workers/reward_manager/__init__.py 中注册
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

    本管理器将同一prompt生成的多个response作为一组进行评估，
    通过调用compute_score函数使用LLM作为judge进行组评估，获取质量分数。
    支持三种RLAIF评估模式：Pointwise（逐点）、Pairwise（成对）、Listwise（列表排序）
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
        初始化DGR Reward Manager

        Args:
            tokenizer: 分词器
            num_examine: 打印到控制台的批次数
            compute_score: 自定义的评分函数
            reward_fn_key: 用于获取data_source的key
            **kwargs: 额外参数传递给compute_score
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.kwargs = kwargs

    def __call__(self, data: DataProto, return_dict=False):
        """
        计算奖励值

        Args:
            data: DataProto对象，包含batch数据
            return_dict: 是否返回字典格式（包含额外信息）

        Returns:
            reward_tensor: 奖励张量
            或
            dict: 包含reward_tensor和额外信息的字典
        """

        # 如果已有rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # 初始化奖励张量和额外信息
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # 按prompt分组
        prompt_to_indices = defaultdict(list)
        for idx, prompt_ids in enumerate(data.batch["prompts"]):
            prompt_key = tuple(prompt_ids.tolist())
            prompt_to_indices[prompt_key].append(idx)

        # 对每组进行评估
        for prompt_key, indices in prompt_to_indices.items():
            if len(indices) == 0:
                continue

            # 准备组评估的数据
            queries = []
            prompts = []
            extras = []

            for idx in indices:
                # 解码response
                response_ids = data.batch["responses"][idx]
                response_text = self.tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )

                # 解码prompt
                prompt_ids = data.batch["prompts"][idx]
                prompt_text = self.tokenizer.decode(
                    prompt_ids, skip_special_tokens=True
                )

                # 构建完整query（prompt + response）
                query = prompt_text + response_text
                queries.append(query)
                prompts.append(prompt_text)

                # 提取额外信息
                prompt_raw_list = data.non_tensor_batch.get("prompt_raw", [])
                if len(prompt_raw_list) > 0 and idx < len(prompt_raw_list):
                    x_data = prompt_raw_list[idx]
                else:
                    x_data = {}

                extra_info = {
                    "x": x_data,
                }

                # 添加ground truth信息（如果有）
                if "reward_model" in data.non_tensor_batch:
                    rm_list = data.non_tensor_batch["reward_model"]
                    if len(rm_list) > 0 and idx < len(rm_list):
                        rm_data = rm_list[idx]
                        if isinstance(rm_data, dict) and "ground_truth" in rm_data:
                            extra_info.update(rm_data["ground_truth"])

                extras.append(extra_info)

            # 获取data_source（安全访问，避免numpy数组ambiguous truth value）
            data_source_list = data.non_tensor_batch.get(
                self.reward_fn_key, ["alignment"]
            )
            try:
                if len(data_source_list) > 0 and indices[0] < len(data_source_list):
                    data_source = data_source_list[indices[0]]
                else:
                    data_source = "alignment"
            except (TypeError, ValueError):
                # 如果data_source_list是numpy数组或其他特殊类型
                data_source = "alignment"

            # 调用compute_score进行组评估
            result = self.compute_score(
                data_source=data_source,
                solution_str=[q.split(prompts[0])[-1].strip() for q in queries],
                ground_truth=extras[0] if extras else {},
                extra_info=extras[0] if extras else {},
                group_evaluation=True,  # 重要：启用组评估模式
                prompt_str=prompts[0],
                all_responses=[
                    q.split(prompts[i])[-1].strip() for i, q in enumerate(queries)
                ],
                all_extra_infos=extras,
                **self.kwargs,
            )

            # 提取评分
            if isinstance(result, dict):
                if "group_scores" in result:
                    scores = result["group_scores"]
                elif "score" in result:
                    scores = [result["score"]] * len(indices)
                else:
                    scores = [0.0] * len(indices)

                # 保存额外信息
                for key, value in result.items():
                    if key not in ["group_scores", "score"]:
                        if isinstance(value, list) and len(value) == len(indices):
                            reward_extra_info[key].extend(value)
                        else:
                            reward_extra_info[key].extend([value] * len(indices))
            else:
                scores = [float(result)] * len(indices)

            # 分配奖励值
            for i, idx in enumerate(indices):
                reward_tensor[idx] = scores[i] if i < len(scores) else 0.0

        # 转换额外信息为tensor（如果可能）
        batch_size = len(data.batch["responses"])
        for key, values in reward_extra_info.items():
            if len(values) == batch_size:
                try:
                    reward_extra_info[key] = torch.tensor(values, dtype=torch.float32)
                except:
                    pass  # 保持为list如果无法转换

        # 返回结果
        if return_dict:
            return {"reward_tensor": reward_tensor, **reward_extra_info}
        else:
            return reward_tensor
