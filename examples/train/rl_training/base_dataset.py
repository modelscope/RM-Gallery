"""
Base Dataset Class for Alignment RL Training

Based on VERL's chat_rl_dataset.py BaseChatRLDataset class.
"""

import copy
import os
from typing import List, Union

import datasets
from torch.utils.data import Dataset

try:
    import verl.utils.torch_functional as verl_F
    from verl.utils.model import compute_position_id_with_mask
except ImportError:
    raise ImportError(
        "This dataset requires VERL to be installed. "
        "Please install VERL first: https://github.com/volcengine/verl"
    )

# Try to import omegaconf types for better compatibility
try:
    from omegaconf import DictConfig, ListConfig
except ImportError:
    # Fallback if omegaconf is not available
    DictConfig = dict
    ListConfig = list


class BaseChatRLDataset(Dataset):
    """聊天强化学习数据集基类"""

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer,
        config,
        processor=None,  # 保持向后兼容性，但不使用
    ):
        # Initialize basic attributes
        self.data_files = self._normalize_data_files(data_files)
        self.original_data_files = copy.deepcopy(self.data_files)
        self.tokenizer = tokenizer
        self.config = config

        # Load configuration settings
        self._load_config()

        # Load and process data
        self._load_dataset()

    def _normalize_data_files(self, data_files):
        """将数据文件转换为列表格式"""
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]
        return copy.deepcopy(data_files)

    def _load_config(self):
        """加载配置参数"""
        self.cache_dir = os.path.expanduser(
            self.config.get("cache_dir", "~/.cache/verl/rlhf")
        )
        self.prompt_key = self.config.get("prompt_key", "prompt")
        self.max_prompt_length = self.config.get("max_prompt_length", 1024)
        self.return_raw_chat = self.config.get("return_raw_chat", False)
        self.truncation = self.config.get("truncation", "error")
        self.filter_overlong_prompts = self.config.get("filter_overlong_prompts", True)
        self.num_workers = min(
            self.config.get(
                "filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)
            ),
            os.cpu_count(),
        )
        self.serialize_dataset = False

    def _download_files(self):
        """下载文件到本地缓存"""
        from verl.utils.fs import copy_to_local

        for i, file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=file, cache_dir=self.cache_dir)

    def _load_dataset(self):
        """加载和处理数据集"""
        self._download_files()

        # 加载parquet文件
        dataframes = []
        for file in self.data_files:
            df = datasets.load_dataset("parquet", data_files=file)["train"]
            dataframes.append(df)

        self.dataframe = datasets.concatenate_datasets(dataframes)
        print(f"数据集长度: {len(self.dataframe)}")

        # 过滤过长的提示
        if self.filter_overlong_prompts:
            self._filter_long_prompts()

    def _filter_long_prompts(self):
        """过滤掉过长的提示"""

        def is_prompt_valid(doc):
            try:
                prompt = self._extract_prompt(doc)
                return len(self.tokenizer.encode(prompt)) <= self.max_prompt_length
            except:
                return False

        self.dataframe = self.dataframe.filter(
            is_prompt_valid,
            num_proc=self.num_workers,
            desc=f"过滤长度超过 {self.max_prompt_length} tokens的提示",
        )
        print(f"过滤后数据集长度: {len(self.dataframe)}")

    def _extract_prompt(self, example):
        """从样本中提取提示"""
        # 首先尝试新的数据结构
        if "input" in example and example["input"]:
            for msg in example["input"]:
                if msg.get("role") == "user" and msg.get("content"):
                    return msg["content"]

        # 回退到旧的数据结构
        prompt = example.get(self.prompt_key)
        if prompt is None:
            prompt = example.get("x", [])
            if prompt:
                return prompt[-1].get("content", "")

        if isinstance(prompt, str):
            return prompt[: self.max_prompt_length]
        elif isinstance(prompt, list) and prompt:
            return (
                prompt[0].get("content", "")
                if isinstance(prompt[0], dict)
                else str(prompt[0])
            )

        return ""

    def _build_messages(self, example: dict) -> List[dict]:
        """从样本构建聊天消息 - 子类需要重写"""
        raise NotImplementedError("Subclasses must implement _build_messages")

    def _format_template(self, messages: List[dict], example: dict) -> str:
        """格式化模板 - 子类需要重写"""
        raise NotImplementedError("Subclasses must implement _format_template")

    def _extract_ground_truth(self, row_dict):
        """提取真实标签 - 子类需要重写"""
        raise NotImplementedError("Subclasses must implement _extract_ground_truth")

    def __getitem__(self, item):
        """获取数据集中的一个项目"""
        row_dict = dict(self.dataframe[item])
        messages = self._build_messages(row_dict)

        # 格式化提示
        raw_prompt_messages = self._format_template(messages, row_dict)

        raw_prompt = self.tokenizer.apply_chat_template(
            raw_prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
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

        # 构建结果
        result = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "index": row_dict.get("index", item),
            "extra_info": copy.deepcopy(row_dict),
            "reward_model": {"ground_truth": self._extract_ground_truth(row_dict)},
            "data_source": row_dict.get("source", "helpsteer2"),
        }

        if self.return_raw_chat:
            result["raw_prompt"] = messages

        return result

    def __len__(self):
        return len(self.dataframe)

    def resume_dataset_state(self):
        """恢复数据集状态用于检查点"""
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self.data_files = copy.deepcopy(self.original_data_files)
            self._load_dataset()
        else:
            print("使用旧的数据加载器检查点文件，建议从头开始训练")

    def __getstate__(self):
        """获取用于序列化的状态"""
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()
