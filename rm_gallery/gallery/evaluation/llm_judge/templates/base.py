"""
VERL-Compatible Prompt Templates

Supports custom prompt functions and XML-based result parsing.
"""

import re
from typing import Callable, List, Optional

from pydantic import BaseModel, Field


class VERLPromptTemplate(BaseModel):
    """
    VERL-compatible prompt template base class

    Supports:
    - Custom prompt generation functions
    - XML-based result parsing
    - Flexible result tag names

    Parameters:
        prompt_function: Optional custom prompt generation function
        result_tag: XML tag name for result extraction (default: "result")
        parse_function: Optional custom result parsing function
    """

    model_config = {"arbitrary_types_allowed": True}

    prompt_function: Optional[Callable] = Field(default=None)
    result_tag: str = Field(default="result")
    parse_function: Optional[Callable] = Field(default=None)

    def generate_prompt(
        self,
        user_query: str,
        response_a: str,
        response_b: Optional[str] = None,
        reference: str = "",
        **kwargs,
    ) -> str:
        """
        Generate evaluation prompt

        If custom prompt_function is provided, use it;
        otherwise use default template
        """
        if self.prompt_function:
            return self.prompt_function(
                user_query=user_query,
                response_a=response_a,
                response_b=response_b,
                reference=reference,
                **kwargs,
            )
        else:
            return self._default_prompt(user_query, response_a, response_b, reference)

    def _default_prompt(
        self,
        user_query: str,
        response_a: str,
        response_b: Optional[str],
        reference: str,
    ) -> str:
        """Default pairwise comparison template"""
        reference_section = (
            f"\n## Reference Response\n{reference}\n" if reference else ""
        )

        return f"""# AI Assistant Response Quality Evaluation

## User Query
{user_query}

## Response A
{response_a}

## Response B
{response_b}
{reference_section}

## Instructions
Please evaluate which response is better. Provide your result in the <{self.result_tag}> tags:
- "A": If Response A is better
- "B": If Response B is better
- "tie": If both responses are of similar quality

<{self.result_tag}>Your evaluation result</{self.result_tag}>
"""

    def parse_result(self, llm_output: str) -> str:
        """
        Extract result from LLM output using XML tags

        Supports custom parse function
        """
        if self.parse_function:
            return self.parse_function(llm_output)

        # Default XML parsing
        pattern = f"<{self.result_tag}>(.*?)</{self.result_tag}>"
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)

        if match:
            result = match.group(1).strip().lower()
            return result

        # Fallback: try to match keywords directly
        output_lower = llm_output.lower()
        if (
            "response a" in output_lower or "option a" in output_lower
        ) and "better" in output_lower:
            return "a"
        elif (
            "response b" in output_lower or "option b" in output_lower
        ) and "better" in output_lower:
            return "b"
        elif (
            "tie" in output_lower
            or "similar" in output_lower
            or "equal" in output_lower
        ):
            return "tie"

        return "invalid"


class PointwisePromptTemplate(VERLPromptTemplate):
    """Pointwise scoring template (1-10 scale)"""

    result_tag: str = Field(default="score")

    def _default_prompt(
        self,
        user_query: str,
        response_a: str,
        response_b: Optional[str] = None,
        reference: str = "",
    ) -> str:
        """Generate pointwise scoring prompt"""
        reference_section = (
            f"\n## Reference Response\n{reference}\n" if reference else ""
        )

        return f"""# AI Assistant Response Quality Evaluation

## User Query
{user_query}

## Response
{response_a}
{reference_section}

## Instructions
Please rate the quality of the response on a scale from 1 to 10, where:
- 1-3: Poor quality
- 4-6: Acceptable quality
- 7-9: Good quality
- 10: Excellent quality

Provide your score in the <{self.result_tag}> tags.

<{self.result_tag}>Your score (1-10)</{self.result_tag}>
"""

    def parse_result(self, llm_output: str) -> float:
        """Parse score (1-10) from LLM output"""
        if self.parse_function:
            return self.parse_function(llm_output)

        # Try XML tag first
        pattern = f"<{self.result_tag}>(.*?)</{self.result_tag}>"
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)

        if match:
            score_str = match.group(1).strip()
            try:
                # Extract first number
                numbers = re.findall(r"\d+\.?\d*", score_str)
                if numbers:
                    score = float(numbers[0])
                    return max(1.0, min(10.0, score))  # Clamp to [1, 10]
            except:
                pass

        # Fallback: search for number in entire output
        numbers = re.findall(r"\b([1-9]|10)(?:\.\d+)?\b", llm_output)
        if numbers:
            try:
                score = float(numbers[0])
                return max(1.0, min(10.0, score))
            except:
                pass

        return 5.0  # Default medium score


class ListwisePromptTemplate(VERLPromptTemplate):
    """Listwise ranking template"""

    result_tag: str = Field(default="ranking")

    def generate_prompt(
        self,
        user_query: str = "",
        response_a: Optional[str] = None,
        response_b: Optional[str] = None,
        responses: List[str] = None,
        reference: str = "",
        **kwargs,
    ) -> str:
        """Generate listwise ranking prompt (multiple responses)"""
        # Support both single response and multiple responses for compatibility
        if responses is None:
            if response_a is not None:
                responses = [response_a]
                if response_b is not None:
                    responses.append(response_b)
            else:
                responses = []

        if self.prompt_function:
            return self.prompt_function(
                user_query=user_query,
                responses=responses,
                reference=reference,
                **kwargs,
            )

        return self._default_listwise_prompt(user_query, responses, reference)

    def _default_listwise_prompt(
        self, user_query: str, responses: List[str], reference: str
    ) -> str:
        """Default listwise ranking prompt"""
        reference_section = (
            f"\n## Reference Response\n{reference}\n" if reference else ""
        )

        responses_section = "\n\n".join(
            [f"## Response {i+1}\n{resp}" for i, resp in enumerate(responses)]
        )

        return f"""# AI Assistant Response Quality Evaluation

## User Query
{user_query}

{responses_section}
{reference_section}

## Instructions
Please rank all responses from best to worst.
Provide your ranking in the <{self.result_tag}> tags as a comma-separated list of response numbers.

Example: 3,1,2 (meaning Response 3 is best, then Response 1, then Response 2)

<{self.result_tag}>Your ranking</{self.result_tag}>
"""

    def parse_result(self, llm_output: str, n_responses: int) -> List[int]:
        """
        Parse ranking from LLM output

        Returns:
            List of response indices in ranked order (0-indexed)
        """
        if self.parse_function:
            return self.parse_function(llm_output, n_responses)

        # Try XML tag first
        pattern = f"<{self.result_tag}>(.*?)</{self.result_tag}>"
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)

        if match:
            ranking_str = match.group(1).strip()
        else:
            # Fallback: try to find comma-separated numbers
            ranking_str = llm_output

        # Extract numbers
        numbers = re.findall(r"\d+", ranking_str)

        if numbers:
            try:
                # Convert to 0-indexed and remove duplicates (preserve order)
                ranking = []
                seen = set()
                for n in numbers:
                    idx = int(n) - 1
                    if 1 <= int(n) <= n_responses and idx not in seen:
                        ranking.append(idx)
                        seen.add(idx)

                # Add missing indices
                all_indices = set(range(n_responses))
                missing = list(all_indices - seen)
                ranking.extend(missing)

                return ranking[:n_responses]
            except:
                pass

        # Default: natural order
        return list(range(n_responses))
