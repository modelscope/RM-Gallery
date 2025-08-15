"""
RewardBench2 Template with Principle Support
"""
import re
from typing import List
from pydantic import Field

from rm_gallery.core.reward.template import PrincipleListWiseTemplate


class RewardBench2PrincipleTemplate(PrincipleListWiseTemplate):
    """Template class for Reward-Bench-2 evaluation protocol with principle support.

    Generates structured prompts for list-wise comparison of multiple responses with 
    customizable evaluation principles.
    """

    # Override the required 'best' field to make it optional for Ties mode
    best: int = Field(default=1, description="which answer is the best? (1-4 for four-way comparison)")
    
    # Template response fields for both four-way and Ties modes
    reasoning: str = Field(default="", description="detailed reasoning for evaluation")
    best_answer: str = Field(default="", description="the best answer letter (A, B, C, or D)")
    rating: int = Field(default=-1, description="rating for individual response (1-10, used for Ties subset)")
    raw_judgment: str = Field(default="", description="raw LLM response")

    @classmethod
    def format(
        cls, 
        query: str, 
        answers: List[str], 
        principles: str = "", 
        enable_thinking: bool = False, 
        is_ties: bool = False,
        **kwargs
    ) -> str:
        """Generates evaluation prompt template for multi-response comparison.

        Args:
            query: User's original question
            answers: List of responses to compare
            principles: Custom evaluation principles
            enable_thinking: Whether to include thinking process in output
            is_ties: Whether this is Ties mode (single response rating)
            **kwargs: Additional template parameters

        Returns:
            Formatted prompt string with evaluation instructions and response template
        """
        if is_ties:
            return cls._format_ties_rating(query, answers[0], principles, enable_thinking, **kwargs)
        else:
            return cls._format_four_way(query, answers, principles, enable_thinking, **kwargs)

    @classmethod
    def _format_four_way(
        cls, 
        query: str, 
        answers: List[str], 
        principles: str = "", 
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """Format four-way comparison prompt with principles."""
        if len(answers) != 4:
            raise ValueError(f"Four-way comparison requires exactly 4 answers, got {len(answers)}")

        # Format answers with numbered tags
        answer_text = "\n".join([
            f"<answer_{i+1}>\n{answer}\n</answer_{i+1}>\n"
            for i, answer in enumerate(answers)
        ])

        # Format principles section
        if principles:
            principles_text = f"""You need to select and apply the appropriate evaluation principles based on the query scenario from the following options for evaluation:
{principles}"""
        else:
            principles_text = "Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity and level of detail."

        # Format output requirements
        output_schema = cls.schema(enable_thinking=enable_thinking)

        return f"""Please act as a fair judge to evaluate the response quality of four AI assistants to a user question.
Please begin your evaluation by comparing the four responses and providing a brief explanation. Avoid any position bias and ensure that the order of response presentation does not affect your decision. Do not let the length of responses influence your evaluation. Do not favor certain assistant names. Be as objective as possible.
{principles_text}

# Query
{query}

# Answers
{answer_text}

# Output Requirements
{output_schema}"""

    @classmethod
    def _format_ties_rating(
        cls, 
        query: str, 
        answer: str, 
        principles: str = "", 
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """Format Ties rating prompt for individual response evaluation."""
        # Format principles section for individual rating
        if principles:
            principles_text = f"""Please evaluate based on the following principles:
{principles}"""
        else:
            principles_text = "Your evaluation should consider factors such as helpfulness, relevance, and accuracy of the response."

        # Create a simplified schema for rating (just the rating field)
        rating_schema = f"""Please provide your evaluation in the following format:
<reasoning>
[Your detailed reasoning here]
</reasoning>

<rating>
[Your rating from 1 to 10]
</rating>"""

        return f"""### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
1- {principles_text}
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10. For your rating, only give a number between 1 and 10 (inclusive), do not use any markdown, and do not put any text after your final rating.

[Query]
{query}

[Response]
{answer}

[Your judgement]
{rating_schema if enable_thinking else ""}"""

    @classmethod
    def parse_four_way(cls, text: str):
        """Parse four-way comparison response."""
        # Try to parse using the schema format first (if using structured output)
        try:
            parsed = cls._parse(text)
            if "best" in parsed:
                # Convert numeric best to letter
                best_num = int(parsed["best"])
                best_letter = {1: "A", 2: "B", 3: "C", 4: "D"}.get(best_num, "A")
                return cls(
                    reasoning=parsed.get("reasoning", ""),
                    best_answer=best_letter,
                    best=best_num,
                    raw_judgment=text
                )
        except:
            pass

        # Fallback to RewardBench2 original format parsing
        if "[[A]]" in text:
            best_answer = "A"
            best_num = 1
        elif "[[B]]" in text:
            best_answer = "B"
            best_num = 2
        elif "[[C]]" in text:
            best_answer = "C"
            best_num = 3
        elif "[[D]]" in text:
            best_answer = "D"
            best_num = 4
        else:
            best_answer = "A"  # Default fallback
            best_num = 1

        return cls(
            reasoning=text.strip(),
            best_answer=best_answer,
            best=best_num,
            raw_judgment=text
        )

    @classmethod
    def parse_ties_rating(cls, text: str):
        """Parse Ties rating response to extract numerical rating."""
        # Try structured parsing first
        try:
            parsed = cls._parse(text)
            if "rating" in parsed:
                rating = int(parsed["rating"])
                if 1 <= rating <= 10:
                    return cls(
                        reasoning=parsed.get("reasoning", ""),
                        rating=rating,
                        best=1,  # Default value for Ties mode
                        raw_judgment=text
                    )
        except:
            pass

        # Fallback to original format parsing
        rating = -1
        match = re.search(r"\b([1-9]|10)\b\s*$", text.strip())
        if match:
            potential_rating = int(match.group(1))
            if 1 <= potential_rating <= 10:
                rating = potential_rating

        return cls(
            reasoning=text.strip(),
            rating=rating,
            best=1,  # Default value for Ties mode
            raw_judgment=text
        )

    @classmethod
    def parse(cls, text: str, is_ties: bool = False):
        """Main parse method that routes to appropriate sub-parser.

        Args:
            text: Raw LLM response text
            is_ties: Whether this is Ties subset or not

        Returns:
            RewardBench2PrincipleTemplate instance with parsed content
        """
        try:
            if is_ties:
                return cls.parse_ties_rating(text)
            else:
                return cls.parse_four_way(text)
        except Exception as e:
            # Fallback for any parsing errors
            return cls(
                reasoning=f"Parse error: {str(e)[:100]}",
                best_answer="A",
                best=1,
                rating=-1,
                raw_judgment=text
            )

    def get_system_prompt(self) -> str:
        """Get system prompt for four-way comparison."""
        return "You are a helpful assistant that evaluates AI responses according to given criteria."

    @property
    def best_index(self) -> int:
        """Convert letter answer to zero-based index."""
        letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        return letter_to_index.get(self.best_answer, 0)