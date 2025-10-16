#!/usr/bin/env python3
"""
Rubric Generator Runner Script

Simple script to run rubric generation on a dataset.
This is useful for:
1. Testing rubric generation on new datasets
2. Quick prototyping and experimentation
3. Generating rubrics without the full MCR pipeline

Features:
- Incremental saving: Save progress periodically
- Resume support: Continue from last checkpoint
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

from loguru import logger

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.rubric.generator import create_simple_generator
from rm_gallery.core.utils.file import read_jsonl, write_json


def get_sample_hash(sample: DataSample) -> str:
    """Generate unique hash for a sample to identify it"""
    # Use input content as unique identifier
    # Convert Pydantic models to dict using model_dump with mode='json' to handle datetime
    input_data = [
        msg.model_dump(mode="json") if hasattr(msg, "model_dump") else msg
        for msg in sample.input
    ]
    content = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


def load_checkpoint(output_dir: Path) -> Tuple[List[DataSample], set]:
    """Load checkpoint from previous run

    Returns:
        Tuple of (processed_samples, processed_hashes)
    """
    checkpoint_file = output_dir / "checkpoint_samples.jsonl"

    if not checkpoint_file.exists():
        logger.info("No checkpoint found, starting from scratch")
        return [], set()

    logger.info(f"📥 Loading checkpoint from {checkpoint_file}")
    processed_samples = []
    processed_hashes = set()

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample_dict = json.loads(line)
                sample = DataSample(**sample_dict)
                processed_samples.append(sample)
                processed_hashes.add(get_sample_hash(sample))

    logger.info(f"✅ Loaded {len(processed_samples)} processed samples from checkpoint")
    return processed_samples, processed_hashes


def save_checkpoint(output_dir: Path, processed_samples: List[DataSample]):
    """Save checkpoint incrementally"""
    checkpoint_file = output_dir / "checkpoint_samples.jsonl"

    # Write all processed samples
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        for sample in processed_samples:
            sample_dict = sample.model_dump(mode="json")
            f.write(json.dumps(sample_dict, ensure_ascii=False, default=str) + "\n")

    logger.debug(f"💾 Checkpoint saved: {len(processed_samples)} samples")


def transform_samples(raw_samples, domains=None):
    """Transform raw samples to DataSample format"""
    if domains:
        samples = [
            DataSample(**sample)
            for sample in raw_samples
            if sample.get("metadata", {}).get("domain") in domains
        ]
    else:
        samples = [DataSample(**sample) for sample in raw_samples]

    # Set preference labels
    for sample in samples:
        for output in sample.output:
            output.answer.label["preference"] = (
                "chosen"
                if output.answer.label.get("is_preferred", False)
                else "rejected"
            )

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Run rubric generation on a preference dataset"
    )

    # Input/Output
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input preference dataset (JSONL format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rubric_generation_output",
        help="Output directory for generated rubrics",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-32b",
        help="LLM model name",
    )
    parser.add_argument(
        "--enable-thinking", type=bool, default=True, help="Enable LLM thinking mode"
    )
    # Generation settings
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        help="Filter by domains (e.g., 'general' 'math')",
    )
    parser.add_argument(
        "--generate-number",
        type=int,
        default=1,
        help="Number of rubrics to generate per sample",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Maximum iterative improvement epochs",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum concurrent threads",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for LLM calls",
    )

    parser.add_argument(
        "--sample-timeout",
        type=int,
        default=180,
        help="Maximum time (seconds) to process a single sample",
    )

    # Checkpoint and resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if available",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Process samples in batches (checkpoint saved after each batch, 0 to disable batching)",
    )
    parser.add_argument(
        "--disable-checkpoint",
        action="store_true",
        help="Disable checkpoint saving (process all at once)",
    )

    args = parser.parse_args()

    # Create output directory early
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    logger.info("=" * 80)
    logger.info("🚀 RUBRIC GENERATOR")
    logger.info("=" * 80)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Enable thinking: {args.enable_thinking}")
    logger.info(f"Max samples: {args.max_samples if args.max_samples > 0 else 'All'}")
    logger.info(f"Domains: {args.domains if args.domains else 'All'}")
    logger.info(f"Generate number: {args.generate_number}")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(
        f"Batch size: {args.batch_size if args.batch_size > 0 else 'Disabled (process all at once)'}"
    )
    logger.info(
        f"Checkpoint: {'Disabled' if args.disable_checkpoint else 'Enabled (save after each batch)'}"
    )
    logger.info(f"Resume: {args.resume}")
    logger.info("=" * 80)

    # Load data
    logger.info(f"\n📂 Loading data from {args.data_path}...")
    raw_samples = read_jsonl(args.data_path)
    logger.info(f"Loaded {len(raw_samples)} raw samples")

    # Transform samples
    all_samples = transform_samples(raw_samples, domains=args.domains)
    logger.info(f"Transformed {len(all_samples)} samples")

    # Limit samples if specified
    if args.max_samples > 0:
        all_samples = all_samples[: args.max_samples]
        logger.info(f"Limited to {len(all_samples)} samples")

    # Load checkpoint if resume is enabled
    processed_samples = []
    processed_hashes = set()

    if args.resume:
        processed_samples, processed_hashes = load_checkpoint(output_dir)

    # Filter out already processed samples
    samples_to_process = []
    for sample in all_samples:
        sample_hash = get_sample_hash(sample)
        if sample_hash not in processed_hashes:
            samples_to_process.append(sample)

    if args.resume and processed_samples:
        logger.info(f"📊 Already processed: {len(processed_samples)} samples")
        logger.info(f"🔄 Remaining to process: {len(samples_to_process)} samples")
    else:
        logger.info(f"🔄 Total samples to process: {len(samples_to_process)} samples")

    # Create LLM
    logger.info(f"\n🤖 Initializing LLM ({args.model})...")
    llm = OpenaiLLM(model=args.model, enable_thinking=args.enable_thinking)

    # Create generator
    logger.info("🔧 Creating rubric generator...")
    config = {
        "generate_number": args.generate_number,
        "max_retries": args.max_retries,
        "max_workers": args.max_workers,
        "max_epochs": args.max_epochs,
        "sample_timeout": args.sample_timeout,
    }
    generator = create_simple_generator(llm=llm, config=config)

    # Process in batches with checkpointing
    if len(samples_to_process) == 0:
        logger.info("✅ All samples already processed!")
    else:
        # Determine if batching is enabled
        enable_batching = args.batch_size > 0
        enable_checkpoint = not args.disable_checkpoint

        if enable_batching:
            logger.info(
                f"\n⚙️  Processing {len(samples_to_process)} samples in batches of {args.batch_size}..."
            )

            total_batches = (
                len(samples_to_process) + args.batch_size - 1
            ) // args.batch_size

            for batch_idx in range(0, len(samples_to_process), args.batch_size):
                batch_samples = samples_to_process[
                    batch_idx : batch_idx + args.batch_size
                ]
                batch_num = batch_idx // args.batch_size + 1

                logger.info(f"\n{'='*60}")
                logger.info(
                    f"📦 Batch {batch_num}/{total_batches}: Processing {len(batch_samples)} samples"
                )
                logger.info(f"{'='*60}")

                # Run generation for this batch
                _, batch_processed = generator.run_batch(
                    batch_samples, max_workers=args.max_workers
                )

                # Add to overall processed samples
                processed_samples.extend(batch_processed)

                # Save checkpoint after each batch (if enabled)
                if enable_checkpoint:
                    save_checkpoint(output_dir, processed_samples)
                    logger.info(
                        f"✅ Batch {batch_num} completed, checkpoint saved ({len(processed_samples)} total samples)"
                    )
                else:
                    logger.info(
                        f"✅ Batch {batch_num} completed ({len(processed_samples)} total samples)"
                    )
        else:
            # Process all samples at once (no batching)
            logger.info(
                f"\n⚙️  Processing all {len(samples_to_process)} samples at once..."
            )
            _, batch_processed = generator.run_batch(
                samples_to_process, max_workers=args.max_workers
            )
            processed_samples.extend(batch_processed)
            logger.info("✅ All samples processed")

    # Final checkpoint save
    if not args.disable_checkpoint:
        save_checkpoint(output_dir, processed_samples)
        logger.info("💾 Final checkpoint saved")

    # Collect statistics and rubrics
    successful_samples = [
        s
        for s in processed_samples
        if s.metadata.get("rubric_valid", "False") == "True"
    ]
    failed_samples = [
        s
        for s in processed_samples
        if s.metadata.get("rubric_valid", "False") == "False"
    ]

    # Collect all rubrics from successful samples
    rubrics = []
    for sample in successful_samples:
        sample_rubrics = sample.metadata.get("rubrics", [])
        rubrics.extend(sample_rubrics)

    # Save rubrics
    rubrics_file = output_dir / "rubrics.json"
    write_json(rubrics, str(rubrics_file))
    logger.info(f"\n💾 Saved {len(rubrics)} rubrics to {rubrics_file}")

    # Save statistics
    stats = {
        "total_samples": len(processed_samples),
        "successful_samples": len(successful_samples),
        "failed_samples": len(failed_samples),
        "success_rate": len(successful_samples) / len(processed_samples),
        "total_rubrics": len(rubrics),
        "avg_rubrics_per_sample": len(rubrics) / len(successful_samples)
        if successful_samples
        else 0,
        "epoch_distribution": {
            "total_samples": len(processed_samples),
            "successful_samples": len(successful_samples),
            "failed_samples": len(failed_samples),
        },
        "configuration": {
            "model": args.model,
            "enable_thinking": args.enable_thinking,
            "generate_number": args.generate_number,
            "max_epochs": args.max_epochs,
            "max_workers": args.max_workers,
            "max_retries": args.max_retries,
            "batch_size": args.batch_size,
            "checkpoint_enabled": not args.disable_checkpoint,
            "resumed": args.resume,
            "domains": args.domains,
        },
    }

    stats_file = output_dir / "statistics.json"
    write_json(stats, str(stats_file))
    logger.info(f"💾 Saved statistics to {stats_file}")

    # Save failed samples for analysis (if any)
    if failed_samples:
        failed_file = output_dir / "failed_samples.jsonl"
        with open(failed_file, "w", encoding="utf-8") as f:
            for sample in failed_samples:
                sample_dict = sample.model_dump(mode="json")
                f.write(json.dumps(sample_dict, ensure_ascii=False, default=str) + "\n")
        logger.info(f"⚠️  Saved {len(failed_samples)} failed samples to {failed_file}")

    logger.info("=" * 80)
    logger.info("✅ Generation completed!")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
