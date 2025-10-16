#!/usr/bin/env python3
"""
Rubric Structurer Runner Script

Transform rubrics into Theme-Tips format using LLM-based semantic analysis.
This script takes a list of rubrics and structures them into coherent themes
with supporting tips for better evaluation clarity.

Features:
- Load rubrics from JSON files (rubrics.json or results.json)
- LLM-based semantic analysis and grouping
- Theme-Tips format output
- Multiple output formats (detailed JSON, ready-to-use strings)

Usage:
    python run_rubric_structurer.py --input rubrics.json --themes 5
    python run_rubric_structurer.py --input results.json --output structured_results/ --model qwen3-32b
"""

import argparse
import traceback

from loguru import logger

from rm_gallery.core.reward.rubric.structurer import RubricStructurer


def main():
    """Main function for rubric structuring"""
    parser = argparse.ArgumentParser(
        description="Rubric Structurer - Transform rubrics into Theme-Tips format"
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file containing rubrics list (e.g., rubrics.json, results.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./rubric_structuring_results",
        help="Output directory for structured results",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-32b",
        help="LLM model name",
    )
    parser.add_argument(
        "--themes",
        type=int,
        default=5,
        help="Maximum number of themes to generate",
    )

    args = parser.parse_args()

    # Print configuration
    logger.info("=" * 80)
    logger.info("🎯 RUBRIC STRUCTURER")
    logger.info("=" * 80)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Target themes: {args.themes}")
    logger.info("=" * 80)

    try:
        # Initialize structurer
        logger.info("🔧 Initializing rubric structurer...")
        structurer = RubricStructurer(
            num_themes=args.themes,
            model_name=args.model,
            output_dir=args.output,
        )

        # Load rubrics from JSON file
        logger.info(f"📂 Loading rubrics from {args.input}...")
        rubrics = RubricStructurer.load_rubrics(args.input)
        logger.info(f"✅ Loaded {len(rubrics)} rubrics")

        if not rubrics:
            logger.error("❌ No rubrics found to structure")
            return

        # Run structuring
        logger.info(f"🤖 Starting LLM-based structuring into {args.themes} themes...")
        structured_rubrics, themes = structurer.structure_rubrics(rubrics)

        # Print results summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 STRUCTURING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📊 Input: {len(rubrics)} source rubrics")
        logger.info(f"📋 Output: {len(structured_rubrics)} Theme-Tips rubrics")
        logger.info(f"📁 Results saved to: {args.output}")
        logger.info("=" * 80)

        # Show theme previews
        if themes:
            logger.info("\n📝 Generated Themes Preview:")
            for i, (theme_id, theme_info) in enumerate(themes.items()):
                theme_text = theme_info.get("theme", "Unknown")
                tip_count = len(theme_info.get("tips", []))
                source_count = theme_info.get("rubric_count", 0)
                logger.info(
                    f"  {i+1}. {theme_text} ({tip_count} tips, {source_count} source rubrics)"
                )

    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {args.input}")
        logger.error("Please check the file path and try again.")
    except ValueError as e:
        logger.error(f"❌ Input file format error: {e}")
        logger.error("Please ensure the input file contains a valid rubrics list.")
    except Exception as e:
        logger.error(f"❌ Structuring failed: {e}")
        logger.debug("Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
