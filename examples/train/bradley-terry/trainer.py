import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

from rm_gallery.core.train.dataset import BaseBradleyTerryTrainDataset


@dataclass
class ScriptArguments:
    """
    Training arguments for RLHF reward modeling.
    These arguments vary depending on hardware capacity and model size requirements.
    """

    # Infrastructure settings
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu training"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. "
            "You may need this if the model doesn't fit on a single GPU."
        },
    )

    # Training hyperparameters
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "Batch size per device during training"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "Batch size per device during evaluation"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=64,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass"
        },
    )
    learning_rate: Optional[float] = field(
        default=2e-6, metadata={"help": "The initial learning rate for AdamW"}
    )
    weight_decay: Optional[float] = field(
        default=0.001, metadata={"help": "Weight decay for AdamW if applied"}
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "Total number of training epochs to perform"}
    )

    # Model settings
    model_name: Optional[str] = field(
        default="Qwen/Qwen3-1.7B",
        metadata={
            "help": "The model name from Hugging Face hub (e.g., gpt2, bert, etc.)"
        },
    )
    max_length: Optional[int] = field(
        default=4096, metadata={"help": "Maximum sequence length for tokenization"}
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to save memory"},
    )

    # Optimizer settings
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use"},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The learning rate scheduler type"},
    )
    warmup_ratio: Optional[float] = field(
        default=0, metadata={"help": "Warmup ratio for learning rate scheduler"}
    )

    # Data paths
    train_set_path: Optional[str] = field(
        default="./data/train.parquet",
        metadata={"help": "Path to training data file (supports .jsonl and .parquet)"},
    )
    eval_set_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to eval data file (supports .jsonl and .parquet). "
            "If None, will use training data for evaluation."
        },
    )
    output_path: Optional[str] = field(
        default="./models/reward_model",
        metadata={"help": "Output directory for the trained model"},
    )

    # Data processing
    custom_bt_dataset_path: Optional[str] = field(
        default="./dataset.py",
        metadata={
            "help": "Path to custom BT dataset Python file or built-in type (e.g., './dataset.py')"
        },
    )
    custom_bt_dataset_name: Optional[str] = field(
        default="HelpSteer3DataProcessor",
        metadata={"help": "Name of the custom BT dataset class"},
    )
    # logging settings
    report_to: Optional[str] = field(
        default="swanlab",
        metadata={"help": "Report to wandb or swanlab"},
    )
    logging_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "Logging strategy"},
    )
    logging_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Logging steps"},
    )

    # Training control
    do_eval: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to evaluate the model"},
    )
    eval_delay: Optional[int] = field(
        default=0,
        metadata={"help": "Delay for evaluation"},
    )
    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Evaluate the model every x steps"},
    )
    run_name: Optional[str] = field(
        default="reward_model",
        metadata={"help": "Name of the training run"},
    )


def load_processor_from_file(file_path: str, class_name: str):
    """
    Load processor class from Python file.

    Args:
        file_path: Path to Python file
        class_name: Name of the class to load

    Returns:
        Processor class or factory function
    """

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Python file not found: {file_path}")

        # Create module spec from file
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from {file_path}")

        # Load the module
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Get the class from module
        if not hasattr(module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in {file_path}")

        processor_cls = getattr(module, class_name)

        # Check if it's a DataProcessor or BaseBradleyTerryTrainDataset

        if issubclass(processor_cls, BaseBradleyTerryTrainDataset):
            # Return the class itself
            return processor_cls
        else:
            raise TypeError(
                f"Class '{class_name}' in {file_path} must inherit from BaseBradleyTerryTrainDataset"
            )

    except Exception as e:
        raise ImportError(
            f"Failed to load processor {class_name} from {file_path}: {e}"
        )


@dataclass
class RewardDataCollatorWithPadding:
    """
    Data collator that batches data in preference format (j vs k).
    Handles padding and creates appropriate batch structure for reward model training.
    """

    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate features into a batch suitable for reward model training.

        Args:
            features: List of feature dictionaries

        Returns:
            Batched features dictionary
        """
        merged_features = []

        # Merge chosen and rejected responses into single batch
        for feature in features:
            # Add chosen response (j)
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            # Add rejected response (k)
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )

        # Apply padding (data already tokenized and truncated in preprocessing)
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }


class RewardTrainer(Trainer):
    """
    Custom trainer for reward model using Bradley-Terry preference learning.
    Implements the reward model loss function based on preference comparisons.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute the Bradley-Terry loss for preference learning.

        Args:
            model: The reward model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (unused)

        Returns:
            Loss tensor (and optionally outputs dict)
        """
        # Get reward scores
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]

        # Split rewards into chosen (j) and rejected (k) pairs
        batch_size = rewards.size(0)
        chosen_idx = torch.arange(0, batch_size, 2)
        rejected_idx = chosen_idx + 1

        rewards_chosen = rewards[chosen_idx]
        rewards_rejected = rewards[rejected_idx]

        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {"rewards_j": rewards_chosen, "rewards_k": rewards_rejected}
        return loss


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for reward model.

    Args:
        eval_pred: Evaluation predictions object

    Returns:
        Dictionary containing computed metrics
    """
    chosen_scores = eval_pred.predictions[0]
    rejected_scores = eval_pred.predictions[1]

    # Calculate accuracy: chosen should have higher scores than rejected
    accuracy = np.sum(chosen_scores > rejected_scores) / len(chosen_scores)

    return {"accuracy": accuracy}


def main():
    """Main training function that orchestrates the entire reward modeling process."""

    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False)

    # Configure tokenizer
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = script_args.max_length

    # Check training file exists
    if not os.path.exists(script_args.train_set_path):
        raise FileNotFoundError(
            f"Training data file not found: {script_args.train_set_path}"
        )

    # Build datasets using data processor
    # Use custom_bt_dataset_path and custom_bt_dataset_name to determine processor
    processor_class = load_processor_from_file(
        script_args.custom_bt_dataset_path,
        script_args.custom_bt_dataset_name,
    )

    # Instantiate the processor with tokenizer
    data_processor = processor_class(tokenizer)

    # Call the build_dataset method
    train_dataset, eval_dataset = data_processor._build_dataset(
        script_args.train_set_path, script_args.eval_set_path
    )

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_path,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        do_eval=script_args.do_eval,
        eval_delay=script_args.eval_delay,
        eval_steps=script_args.eval_every_steps,
        save_steps=script_args.save_every_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        bf16=script_args.bf16,
        logging_strategy=script_args.logging_strategy,
        logging_steps=script_args.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=True,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        report_to=script_args.report_to,
        run_name=script_args.run_name,
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16
    )

    # Configure model
    model.config.use_cache = not script_args.gradient_checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # Initialize trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=script_args.max_length
        ),
    )

    # Start training
    trainer.train()

    final_output_path = os.path.join(script_args.output_path, "last_checkpoint")
    trainer.save_model(final_output_path)
    tokenizer.save_pretrained(final_output_path)


if __name__ == "__main__":
    main()
