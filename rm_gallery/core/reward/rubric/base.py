from typing import Any, Dict, List

from pydantic import Field

from rm_gallery.core.reward.template import BasePromptTemplate


class BaseGeneratorTemplate(BasePromptTemplate):
    """Base template class for rubric generation tasks.

    Attributes:
        rubrics: Dictionary mapping rubric phrases to descriptions
    """

    # rubrics: List[str] = Field(
    #     default=...,
    #     description="""your rubrics without index""",
    # )


class RubricGenerateTemplate(BaseGeneratorTemplate):
    rubrics: List[str] = Field(
        default=...,
        description="""your rubrics without index""",
    )
    _schema_order: List[str] = ["think", "rubrics"]

    @classmethod
    def parse(cls, text: str):
        """Parse response text into structured rubrics dictionary.

        Args:
            text: Raw response text containing JSON-formatted rubrics

        Returns:
            cls instance with parsed rubrics
        """
        contents = cls._parse(text)
        rubrics = contents["rubrics"].strip().split("\n")
        rubrics = [p.strip() for p in rubrics if len(p.strip()) > 0]
        contents["rubrics"] = rubrics
        return cls(
            **contents,
        )

    @classmethod
    def format(
        cls,
        query: str,
        answers: List[str],
        preference: str | int,
        critics: List[str],
        number: int = 1,
        **kwargs,
    ) -> str:
        """Format prompt for rubric generation task.

        Args:
            query: Original query text
            answers: List of answer texts to compare
            preference: Index/ID of preferred answer
            number: Maximum number of rubrics to generate
            **kwargs: Additional template parameters

        Returns:
            Formatted prompt string
        """
        answer_str = ""
        for i, answer in enumerate(answers):
            answer_str += f"<answer_{i + 1}>\n{answer}\n</answer_{i + 1}>\n\n"
        critics_str = ""
        for i, critic in enumerate(critics):
            critics_str += f"<critic_{i + 1}>\n{critic}\n</critic_{i + 1}>\n\n"

        return f"""## Overview
You are an expert rubric writer for open-ended question. Your job is to
generate a self-contained set of evaluation criteria ("rubrics") for choosing a better answer from candidate answers to a given query. Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style, helpfulness, and common pitfalls. Each rubric item must be fully self-contained so that non-expert readers need not consult any external information.

I will give you:
1. the query(maybe contains history messages)
2. candidate answers
3. which answer is better than others
4. critics by the human experts, and you need to carefully read the critics provided by human experts and summarize the rubrics.

NOTE: The number of rubrics should be LESS THAN OR EQUAL TO {number}

## Query
{query}

## Candidate Answers
{answer_str}

## Better Answer
Answer {preference} is better than others.

## Critics
{critics_str}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class RubricReviseTemplate(BaseGeneratorTemplate):
    rubrics: List[str] = Field(
        default=...,
        description="""your improved rubrics without index""",
    )
    _schema_order: List[str] = ["think", "rubrics"]

    @classmethod
    def parse(cls, text: str):
        """Parse response text into structured rubrics dictionary.

        Args:
            text: Raw response text containing JSON-formatted rubrics

        Returns:
            cls instance with parsed rubrics
        """
        contents = cls._parse(text)
        rubrics = contents["rubrics"].strip().split("\n")
        rubrics = [p.strip() for p in rubrics if len(p.strip()) > 0]
        contents["rubrics"] = rubrics
        return cls(
            **contents,
        )

    @classmethod
    def format(
        cls,
        query: str,
        answers: List[str],
        preference: str | int,
        critics: List[str],
        number: int = 1,
        rubrics: List[str] | None = None,
        **kwargs,
    ) -> str:
        """Format prompt for rubric generation task.

        Args:
            query: Original query text
            answers: List of answer texts to compare
            preference: Index/ID of preferred answer
            number: Maximum number of rubrics to generate
            **kwargs: Additional template parameters

        Returns:
            Formatted prompt string
        """
        answer_str = ""
        for i, answer in enumerate(answers):
            answer_str += f"<answer_{i + 1}>\n{answer}\n</answer_{i + 1}>\n\n"
        critics_str = ""
        for i, critic in enumerate(critics):
            critics_str += f"<critic_{i + 1}>\n{critic}\n</critic_{i + 1}>\n\n"
        rubrics_str = ""
        for i, rubric in enumerate(rubrics):
            rubrics_str += f"<rubric_{i + 1}>\n{rubric}\n</rubric_{i + 1}>\n\n"

        return f"""## Overview
You are an expert rubric writer for open-ended question. A self-contained set of evaluation criteria ("rubrics") is needed for choosing a better answer from candidate answers to a given query.  Since the rubrics generated in the previous round failed to correctly select a better answer, you need to revise the rubrics. Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style,
helpfulness, and common pitfalls. Each rubric item must be fully self-contained so that non-expert readers need not consult
any external information.

I will give you:
1. the query(maybe contains history messages)
2. candidate answers
3. which answer is better than others
4. critics by the human experts, and you need to carefully read the critics provided by human experts and summarize the rubrics.
5. previous round rubrics that should to be improved

NOTE: The number of rubrics should be LESS THAN OR EQUAL TO {number}

## Query
{query}

## Candidate Answers
{answer_str}

## Better Answer
Answer {preference} is better than others.

## Critics
{critics_str}

## Previous Round Rubrics
{rubrics_str}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class RubricStructuringTemplate(BaseGeneratorTemplate):
    """Template for LLM semantic classification of rubrics"""

    rubrics: List[Dict[str, Any]] = Field(
        default=...,
        description="""A JSON list of rubrics, each containing:
        - theme: A concise statement capturing the core focus
        - tips: A list of specific guidance points (max 5)
        - source_ids: A list of input example numbers (1-based) that this rubric is derived from
        Each rubric must be independent and non-contradictory with others.
        Example format:
        ```json
        [
            {
                "theme": "Concise theme statement",
                "tips": ["Specific guidance point 1", "Specific guidance point 2"],
                "source_ids": [1, 3, 5, 8]
            }
        ]
        ```""",
    )
    num_categories: int = Field(
        default=5, description="Maximum number of rubrics to generate"
    )

    _schema_order: List[str] = ["think", "rubrics"]

    @classmethod
    def parse(cls, text: str) -> "RubricStructuringTemplate":
        """Parse response text into structured rubrics.

        Args:
            text: Response text containing XML-formatted rubrics

        Returns:
            RubricStructuringTemplate instance with parsed rubrics
        """
        contents = cls._parse(text)

        # Parse rubrics from JSON string
        try:
            import json

            rubrics = json.loads(contents.get("rubrics", "[]"))
        except (json.JSONDecodeError, ValueError):
            rubrics = []

        return cls(
            think=contents.get("think", ""),
            rubrics=rubrics,
            num_categories=contents.get("num_categories", 5),
        )

    @classmethod
    def format(cls, rubrics: List[str], num_categories: int = 5, **kwargs) -> str:
        """Format classification prompt"""
        rubrics_text = "\n".join(
            [f"{i+1}. {rubric}" for i, rubric in enumerate(rubrics)]
        )

        return f"""## Task Description
Your task is to generate a set of evaluation rubrics to identify the best answer, based on the suggestions for determining from the examples. I will give you some examples, and every example contains the query and suggestion which has been verified to help select the best answer.

## Input Examples (Suggestions for Evaluation)
{rubrics_text}

## Requirements
- Rubrics must be fully self-contained so that non-expert readers need not consult any external information.
- Each rubric should assess an independent dimension and be non-contradictory with others.
- Rubrics ensure that the overall judgment remains aligned and consistent for all examples.
- The number of rubrics should be LESS THAN OR EQUAL TO {num_categories}. The number of tips for each rubric should be LESS THAN OR EQUAL TO 5.
- Must strictly adhere to the Rubrics Format.

## Rubric Format
Each rubric consists of two parts:
- Theme: A concise and clear statement that captures the core focus of the rubric, and must be **necessary** for all queries with no assumption.
- Tips: Multiple bullet points that expand on or supplement the rubric and only focuses on some specific queries.

Here is an example of a rubric:
```
Theme: [Concise theme statement]
- Tip 1: [Specific guidance point]
- Tip 2: [Specific guidance point]
- Tip 3: [Specific guidance point]
- (Optional: More tips as needed)
```

## Expected Output Format
Please provide your response in the following structured format:

**Rubric 1:**
Theme: [Your theme statement]
- Tip 1: [Your tip]
- Tip 2: [Your tip]
- Tip 3: [Your tip]

**Rubric 2:**
Theme: [Your theme statement]
- Tip 1: [Your tip]
- Tip 2: [Your tip]

[Continue for all rubrics up to {num_categories}]

## Process
1. Based on the query and suggestions of each example, analyze the underlying evaluation criteria.
2. Group similar evaluation criteria together to form coherent rubrics.
3. Synthesize these groups into {num_categories} or fewer distinct rubrics, each with a clear theme and supporting tips.
4. For each generated rubric, record which input examples (by their numbers 1, 2, 3, ...) contributed to it in the "source_ids" field.
5. Ensure each rubric addresses different aspects of evaluation quality and maintains consistency across all examples.

NOTE: The number of rubrics should be LESS THAN OR EQUAL TO {num_categories}. The number of tips for each rubric should be LESS THAN OR EQUAL TO 5.
IMPORTANT: Each rubric MUST include a "source_ids" list indicating which input example numbers it was derived from.

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class RubricEvaluationTemplate(BasePromptTemplate):
    """Template for rubric-based pairwise evaluation"""

    preference: str = Field(
        default=...,
        description='Which response is better? Choose "A", "B", or "tie"',
    )

    @classmethod
    def parse(cls, text: str) -> "RubricEvaluationTemplate":
        """Parse evaluation response"""
        contents = cls._parse(text)
        return cls(
            think=contents.get("think", ""),
            preference=contents.get("preference", "tie").upper(),
        )

    @classmethod
    def format(
        cls,
        query: str,
        response_a: str,
        response_b: str,
        rubrics: str,
        **kwargs,
    ) -> str:
        """Format rubric evaluation prompt"""
        return f"""## Task Description
I will provide you with a set of rubrics, along with the current query and two responses. These rubrics are the primary basis for selecting the best answer. You must follow the steps specified in the Evaluation Process when conducting your evaluation process.

## Rubrics
{rubrics}

## Process
1. Confirm the task scenario of the current query and select the corresponding evaluation rubrics.
2. Identify the best response that meets the most selected rubrics.

## Query
{query}

## Response A
{response_a}

## Response B
{response_b}

## Output Requirements
{cls.schema(**kwargs)}
"""
