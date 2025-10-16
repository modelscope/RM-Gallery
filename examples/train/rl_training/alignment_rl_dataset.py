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

import copy
import os
from typing import List

import verl.utils.torch_functional as verl_F
from recipe.rm_gallery.chat_rl_dataset import BaseChatRLDataset
from verl.utils.model import compute_position_id_with_mask

# 导入配置类
try:
    from .alignment_evaluator import DataKeys
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from alignment_evaluator import DataKeys


class AlignmentChatRLDataset(BaseChatRLDataset):
    """Alignment任务聊天强化学习数据集

    专门处理包含chosen/rejected格式的alignment数据
    """

    def __init__(self, data_files, tokenizer, config, processor=None):
        super().__init__(data_files, tokenizer, config, processor)
        print("使用 Alignment 模式")

    def _build_messages(self, example: dict) -> List[dict]:
        """从样本构建聊天消息 - Alignment模式"""
        messages = []

        # 优先从x字段构建消息
        if "x" in example and example["x"] is not None:
            x_data = example["x"]
            # 处理numpy数组格式
            if hasattr(x_data, "tolist"):
                x_data = x_data.tolist()
            elif not isinstance(x_data, (list, tuple)):
                x_data = [x_data]

            for msg in x_data:
                if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        # 如果x字段为空，从chosen字段构建消息（取前面的对话，不包括最后的assistant回复）
        elif DataKeys.CHOSEN in example and example[DataKeys.CHOSEN]:
            chosen_messages = example[DataKeys.CHOSEN]
            # 处理numpy数组格式
            if hasattr(chosen_messages, "tolist"):
                chosen_messages = chosen_messages.tolist()

            if isinstance(chosen_messages, (list, tuple)):
                for msg in chosen_messages:
                    if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                        # 只添加user消息，不添加assistant消息（因为那是要预测的目标）
                        if msg.get("role") == "user":
                            messages.append(
                                {"role": msg["role"], "content": msg["content"]}
                            )

        # 如果还是没有找到消息，尝试从rejected字段构建
        elif DataKeys.REJECTED in example and example[DataKeys.REJECTED]:
            rejected_messages = example[DataKeys.REJECTED]
            # 处理numpy数组格式
            if hasattr(rejected_messages, "tolist"):
                rejected_messages = rejected_messages.tolist()

            if isinstance(rejected_messages, (list, tuple)):
                for msg in rejected_messages:
                    if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                        if msg.get("role") == "user":
                            messages.append(
                                {"role": msg["role"], "content": msg["content"]}
                            )

        # 如果还是没有消息，创建一个默认的用户消息
        if len(messages) == 0:
            messages = [{"role": "user", "content": "请协助完成这个任务。"}]

        return messages

    def _format_template(self, messages: List[dict], example: dict) -> str:
        """格式化alignment模板"""
        return messages

    def _extract_ground_truth(self, row_dict):
        """提取alignment真实标签

        对于alignment数据，chosen可以作为一种"更好"的参考
        """
        try:
            ground_truth_info = {}

            # 将chosen和rejected都保存到ground_truth中，供奖励函数使用
            chosen_key = DataKeys.CHOSEN
            rejected_key = DataKeys.REJECTED
            source_key = DataKeys.SOURCE

            if chosen_key in row_dict and row_dict[chosen_key] is not None:
                chosen_data = row_dict[chosen_key]
                # 处理numpy数组格式
                if hasattr(chosen_data, "tolist"):
                    chosen_data = chosen_data.tolist()
                ground_truth_info[chosen_key] = chosen_data

            if rejected_key in row_dict and row_dict[rejected_key] is not None:
                rejected_data = row_dict[rejected_key]
                # 处理numpy数组格式
                if hasattr(rejected_data, "tolist"):
                    rejected_data = rejected_data.tolist()
                ground_truth_info[rejected_key] = rejected_data

            if source_key in row_dict:
                ground_truth_info[source_key] = row_dict[source_key]

            return ground_truth_info

        except Exception as e:
            print(f"Error extracting ground truth: {e}")
            return {}

    def __getitem__(self, item):
        """获取数据集中的一个项目"""
        row_dict = dict(self.dataframe[item])
        messages = self._build_messages(row_dict)

        # 格式化提示
        raw_prompt_messages = self._format_template(messages, row_dict)

        # 应用聊天模板
        raw_prompt = self.tokenizer.apply_chat_template(
            raw_prompt_messages, add_generation_prompt=True, tokenize=False
        )

        # 分词
        model_inputs = self.tokenizer(
            raw_prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # 后处理
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # 计算位置ID
        position_ids = compute_position_id_with_mask(attention_mask)

        # 准备原始提示ID
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"提示长度 {len(raw_prompt_ids)} 超过 {self.max_prompt_length}"
                )

        # 构建x字段（用于传递给奖励函数）
        x_messages = []

        # 从原始数据构建完整的对话上下文
        chosen_key = DataKeys.CHOSEN
        if chosen_key in row_dict and row_dict[chosen_key]:
            # 使用chosen作为基础构建对话上下文
            chosen_messages = row_dict[chosen_key]
            # 处理numpy数组格式
            if hasattr(chosen_messages, "tolist"):
                chosen_messages = chosen_messages.tolist()

            if isinstance(chosen_messages, (list, tuple)):
                for msg in chosen_messages:
                    if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                        x_messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )

        # 如果没有从chosen获取到消息，使用我们构建的messages
        if not x_messages:
            x_messages = messages

        # 构建结果
        result = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "index": row_dict.get("index", item),
            "extra_info": copy.deepcopy(row_dict),
            "reward_model": {"ground_truth": self._extract_ground_truth(row_dict)},
            "data_source": row_dict.get("source", "alignment"),
        }

        # 添加x字段，包含对话上下文
        result["extra_info"]["x"] = x_messages

        if self.return_raw_chat:
            result["raw_prompt"] = messages

        return result
