#!/usr/bin/env python3
"""
Rubric Structurer - Transform rubrics into Theme-Tips format

This module provides tools to analyze and structure a list of rubrics into
a coherent Theme-Tips format, where:
- Theme: A concise statement capturing the core evaluation focus
- Tips: Supporting bullet points that expand on the theme

The structuring process:
1. Analyzes underlying evaluation criteria from rubric examples
2. Groups similar criteria together
3. Synthesizes groups into distinct themes with supporting tips
4. Outputs structured, ready-to-use evaluation rubrics

Features:
- Compatible with rubric generator output (rubrics.json and results.json)
- Uses RubricStructuringTemplate for LLM-based semantic analysis
- Integrated with rubric generation pipeline
- Customizable number of theme categories
- Multiple output formats (detailed JSON, list, ready-to-use strings)

Usage:
    # As a library
    structurer = RubricStructurer(num_themes=5, model_name="gpt-4")
    structured_rubrics, themes_dict = structurer.structure_rubrics(rubrics)

    # Command line
    python structurer.py --input rubrics.json --output results/ --themes 5
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.rubric.base import RubricStructuringTemplate


def themes_to_rubric_strings(themes: Dict[int, Dict[str, Any]]) -> List[str]:
    """
    Convert themes dictionary to ready-to-use rubric strings.

    Args:
        themes: Dictionary of themes with theme and tips

    Returns:
        List of formatted rubric strings (Theme + Tips format)
    """
    rubric_strings = []
    for theme_id in sorted(themes.keys()):
        info = themes[theme_id]

        # Assemble into single string: Theme + Tips
        theme_str = f"Theme: {info['theme']}"
        tips_str = "\n".join(
            [f"- Tip {i+1}: {tip}" for i, tip in enumerate(info["tips"])]
        )

        # Combine into complete evaluation rubric string
        complete_rubric = f"{theme_str}\n{tips_str}"
        rubric_strings.append(complete_rubric)

    return rubric_strings


def save_structuring_results(
    themes: Dict[int, Dict[str, Any]], rubrics: List[str], output_dir: str
):
    """Save structuring results in multiple formats"""

    # Save detailed structured results
    detailed_results = {}
    for theme_id, info in themes.items():
        # Get actual rubric text for this theme
        theme_rubrics = [
            rubrics[idx] for idx in info["rubric_ids"] if idx < len(rubrics)
        ]

        detailed_results[f"theme_{theme_id+1}"] = {
            "theme": info["theme"],
            "tips": info["tips"],
            "rubric_count": info["rubric_count"],
            "rubric_ids": [
                idx + 1 for idx in info["rubric_ids"]
            ],  # Convert to 1-based for display
            "source_rubrics": theme_rubrics,
        }

    # Save detailed results
    with open(
        os.path.join(output_dir, "detailed_structured_results.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    # Convert to ready-to-use rubric strings and save
    ready_to_use_list = themes_to_rubric_strings(themes)
    with open(
        os.path.join(output_dir, "ready_to_use_rubrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(ready_to_use_list, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Structuring results saved to {output_dir}")


class RubricStructurer:
    """LLM-based Rubric structurer that transforms rubrics into Theme-Tips format

    This class takes a list of rubrics and uses LLM to:
    1. Analyze underlying evaluation criteria from rubric examples
    2. Group similar criteria together
    3. Synthesize groups into Theme-Tips structure (Theme + supporting Tips)
    4. Output structured, ready-to-use evaluation rubrics
    """

    def __init__(
        self,
        num_themes: int = 5,
        model_name: str = "qwen3-32b",
        output_dir: str = "rubric_structuring_results",
        enable_thinking: bool = True,
    ):
        """
        Initialize Rubric Structurer

        Args:
            num_themes: Maximum number of themes to generate
            model_name: LLM model name
            output_dir: Directory to save results
            enable_thinking: Whether to enable LLM thinking mode
        """
        self.num_themes = num_themes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize LLM
        self.llm = OpenaiLLM(model=model_name, enable_thinking=enable_thinking)

    @staticmethod
    def load_rubrics(file_path: str) -> List[str]:
        """
        Load rubrics from JSON file

        Args:
            file_path: Path to JSON file containing rubrics list

        Returns:
            List of rubric strings
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys
            for key in ["rubrics", "final_rubrics", "data", "items"]:
                if key in data and isinstance(data[key], list):
                    return data[key]

        raise ValueError(
            f"Cannot extract rubric list from {file_path}. "
            f"Expected a JSON list or dict with 'rubrics'/'final_rubrics' key."
        )

    def structure_rubrics(
        self, rubrics: List[str]
    ) -> tuple[List[str], Dict[int, Dict[str, Any]]]:
        """
        Main method: Structure rubrics into Theme-Tips format

        Args:
            rubrics: List of rubric strings to structure

        Returns:
            Tuple of (structured_rubrics_list, themes_dict)
        """

        logger.info(
            f"🎯 Starting rubric structuring for {len(rubrics)} rubrics, target themes: {self.num_themes}"
        )

        if len(rubrics) == 0:
            logger.error("❌ Input rubrics list is empty")
            return [], {}

        # Generate structuring prompt
        logger.info("🤖 Using LLM for rubric structuring...")
        prompt = RubricStructuringTemplate.format(
            rubrics=rubrics,
            num_categories=self.num_themes,
            enable_thinking=self.llm.enable_thinking
            if hasattr(self.llm, "enable_thinking")
            else False,
        )

        try:
            # Call LLM
            response = self.llm.simple_chat(query=prompt)

            logger.info("✅ LLM structuring completed, starting result parsing...")

            # Parse structuring results
            parsed_result = RubricStructuringTemplate.parse(response)

            if not parsed_result.rubrics:
                logger.error("❌ Failed to parse any rubric results")
                return [], {}

            # Convert rubrics list to themes dictionary
            themes = {}
            for i, rubric_data in enumerate(parsed_result.rubrics):
                # Get source_ids from LLM output (1-based) and convert to 0-based indices
                source_ids = rubric_data.get("source_ids", [])
                if source_ids:
                    # Convert 1-based to 0-based indices
                    rubric_ids = [
                        idx - 1
                        for idx in source_ids
                        if isinstance(idx, int) and 0 < idx <= len(rubrics)
                    ]
                else:
                    # Fallback: if no source_ids provided, assign empty list
                    rubric_ids = []
                    logger.warning(
                        f"Theme {i+1} ('{rubric_data.get('theme', 'Unknown')}'): "
                        f"No source_ids provided by LLM"
                    )

                themes[i] = {
                    "theme": rubric_data.get("theme", ""),
                    "tips": rubric_data.get("tips", []),
                    "rubric_ids": rubric_ids,
                    "rubric_count": len(rubric_ids),
                }

            # Save results
            save_structuring_results(themes, rubrics, str(self.output_dir))

            # Generate directly usable string list
            ready_to_use_list = themes_to_rubric_strings(themes)

            logger.info(
                f"🎉 Rubric structuring completed! Results saved in {self.output_dir}"
            )
            logger.info(
                f"📋 Generated {len(ready_to_use_list)} structured evaluation rubrics"
            )

            return ready_to_use_list, themes

        except Exception as e:
            logger.error(f"❌ LLM rubric structuring failed: {e}")
            return [], {}
