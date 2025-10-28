"""
Templates for Conflict Detector
"""

import re

from pydantic import BaseModel, Field


class PairComparisonTemplate(BaseModel):
    """Pairwise comparison template for conflict detection"""

    model_config = {"arbitrary_types_allowed": True}

    result_tag: str = Field(default="result")

    def format(self, query: str, answers: list[str]) -> str:
        """
        Generate pairwise comparison prompt

        Args:
            query: The user question
            answers: List of two answers to compare [answer_a, answer_b]

        Returns:
            Formatted prompt string
        """
        if len(answers) != 2:
            raise ValueError(
                f"Expected 2 answers for pairwise comparison, got {len(answers)}"
            )

        answer_a, answer_b = answers

        return f"""# AI Assistant Response Quality Evaluation

## User Query
{query}

## Response A
{answer_a}

## Response B
{answer_b}

## Instructions
Please evaluate which response is better in terms of helpfulness, accuracy, relevance, and overall quality.

Provide your evaluation result in the <{self.result_tag}> tags:
- "A": If Response A is clearly better
- "B": If Response B is clearly better
- "tie": If both responses are of similar quality

Before providing the result, you may explain your reasoning.

<{self.result_tag}>Your evaluation result (A/B/tie)</{self.result_tag}>
"""

    def parse(self, llm_output: str) -> "PairComparisonResult":
        """
        Parse the LLM output to extract comparison result

        Args:
            llm_output: Raw output from LLM

        Returns:
            PairComparisonResult object
        """
        # Extract reasoning (everything before the result tag)
        reasoning_match = re.search(
            rf"(.*?)<{self.result_tag}>", llm_output, re.DOTALL | re.IGNORECASE
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract result from XML tags
        pattern = rf"<{self.result_tag}>(.*?)</{self.result_tag}>"
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)

        if match:
            result = match.group(1).strip().lower()
            # Normalize result
            if "a" in result and "b" not in result:
                best_answer = "A"
            elif "b" in result:
                best_answer = "B"
            elif "tie" in result or "similar" in result or "equal" in result:
                best_answer = "tie"
            else:
                # Try to find clear preference
                if "response a" in result.lower() and "better" in result.lower():
                    best_answer = "A"
                elif "response b" in result.lower() and "better" in result.lower():
                    best_answer = "B"
                else:
                    best_answer = "tie"  # Default to tie if unclear
        else:
            # Fallback: search in entire output
            output_lower = llm_output.lower()
            if (
                "response a is better" in output_lower
                or "option a" in output_lower
                or "answer a" in output_lower
            ):
                best_answer = "A"
            elif (
                "response b is better" in output_lower
                or "option b" in output_lower
                or "answer b" in output_lower
            ):
                best_answer = "B"
            else:
                best_answer = "tie"

        return PairComparisonResult(best_answer=best_answer, reasoning=reasoning)


class PairComparisonResult(BaseModel):
    """Result of pairwise comparison"""

    best_answer: str = Field(description="Which answer is better: 'A', 'B', or 'tie'")
    reasoning: str = Field(default="", description="Reasoning for the decision")


class PointwiseTemplate(BaseModel):
    """Pointwise scoring template for conflict detection"""

    model_config = {"arbitrary_types_allowed": True}

    result_tag: str = Field(default="score")

    def format(self, query: str, response: str) -> str:
        """
        Generate pointwise scoring prompt

        Args:
            query: The user question
            response: The response to evaluate

        Returns:
            Formatted prompt string
        """
        return f"""# AI Assistant Response Quality Evaluation

## User Query
{query}

## Response
{response}

## Instructions
Please rate the quality of this response on a scale from 1 to 10, where:
- 1-3: Poor quality (incorrect, unhelpful, or irrelevant)
- 4-6: Acceptable quality (partially correct or helpful)
- 7-9: Good quality (correct, helpful, and relevant)
- 10: Excellent quality (comprehensive, accurate, and highly helpful)

Consider the following factors:
- Accuracy and correctness
- Helpfulness and relevance
- Clarity and completeness
- Overall quality

Provide your score in the <{self.result_tag}> tags as a single number between 1 and 10.

You may explain your reasoning before providing the score.

<{self.result_tag}>Your score (1-10)</{self.result_tag}>
"""

    def parse(self, llm_output: str) -> "PointwiseResult":
        """
        Parse the LLM output to extract score

        Args:
            llm_output: Raw output from LLM

        Returns:
            PointwiseResult object with score
        """
        # Extract reasoning (everything before the score tag)
        reasoning_match = re.search(
            rf"(.*?)<{self.result_tag}>", llm_output, re.DOTALL | re.IGNORECASE
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Try XML tag first
        pattern = rf"<{self.result_tag}>(.*?)</{self.result_tag}>"
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)

        score = 5.0  # Default medium score

        if match:
            score_str = match.group(1).strip()
            try:
                # Extract first number
                numbers = re.findall(r"\d+\.?\d*", score_str)
                if numbers:
                    score = float(numbers[0])
                    score = max(1.0, min(10.0, score))  # Clamp to [1, 10]
            except:
                pass
        else:
            # Fallback: search for score in entire output
            # Look for patterns like "Score: 8" or "8/10" or "8 out of 10"
            patterns = [
                r"score[:\s]+(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*/\s*10",
                r"rate[:\s]+(\d+(?:\.\d+)?)",
            ]

            for pattern in patterns:
                match = re.search(pattern, llm_output, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        score = max(1.0, min(10.0, score))
                        break
                    except:
                        continue

        return PointwiseResult(score=score, reasoning=reasoning)


class PointwiseResult(BaseModel):
    """Result of pointwise scoring"""

    score: float = Field(description="Score from 1 to 10")
    reasoning: str = Field(default="", description="Reasoning for the score")
