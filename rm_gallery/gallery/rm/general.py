import re
from typing import List, Literal

from rm_gallery.core.grader import FunctionGrader, Grader, GraderMode, GraderScore
from rm_gallery.core.utils.tokenizer import get_tokenizer


@FunctionGrader.register("accuracy")
async def compute_accuracy(generated, reference) -> GraderScore:
    # Calculate accuracy (1.0 for exact match, 0.0 otherwise)
    accuracy = 1.0 if generated == reference else 0.0

    return GraderScore(
        score=accuracy,
        reason=f"Accuracy: {accuracy:.3f}",
    )


class F1ScoreGrader(Grader):
    """
    Calculate F1 score between generated content and reference answer at word level.

    This reward computes precision, recall and F1 score by comparing word overlap
    between generated and reference texts. Uses configurable tokenizer to support
    multilingual content including Chinese and English.
    """

    def __init__(
        self,
        name: str = "f1_score",
        tokenizer_type: Literal["tiktoken", "jieba", "simple"] = "tiktoken",
        encoding_name: str = "cl100k_base",
        chinese_only: bool = False,
    ):
        super().__init__(name=name, grader_mode=GraderMode.POINT_WISE, description="")
        # Initialize tokenizer
        self.tokenizer_type = tokenizer_type
        self.encoding_name = encoding_name
        self.chinese_only = chinese_only
        self._tokenizer = get_tokenizer(
            tokenizer_type=tokenizer_type,
            encoding_name=encoding_name,
            chinese_only=chinese_only,
        )

    async def evaluate(self, generated, reference) -> GraderScore:
        """
        Calculate F1 score.

        Args:
            sample: Data sample containing generated content and reference answer

        Returns:
            RewardResult: Reward result containing F1 score
        """

        # Tokenize using unified tokenizer
        generated_preprocessed = self._tokenizer.preprocess_text(
            generated, to_lower=True
        )
        reference_preprocessed = self._tokenizer.preprocess_text(
            reference, to_lower=True
        )

        generated_tokens = set(self._tokenizer.tokenize(generated_preprocessed))
        reference_tokens = set(self._tokenizer.tokenize(reference_preprocessed))

        # Calculate precision, recall and F1 score
        if not generated_tokens and not reference_tokens:
            precision = recall = f1 = 1.0
        elif not generated_tokens or not reference_tokens:
            precision = recall = f1 = 0.0
        else:
            intersection = generated_tokens.intersection(reference_tokens)
            precision = len(intersection) / len(generated_tokens)
            recall = len(intersection) / len(reference_tokens)
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        return GraderScore(
            score=f1,
            reason=f"F1 score: {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})",
            metadata={
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "generated_tokens": list(generated_tokens),
                "reference_tokens": list(reference_tokens),
                "tokenizer_type": self.tokenizer_type,
                "tokenizer_name": self._tokenizer.name,
            },
        )


@FunctionGrader.register(name="rouge")
async def compute_rouge(generated, reference) -> GraderScore:
    """
    Calculate ROUGE-L score between generated content and reference answer.
    """

    def _lcs_length(x: List[str], y: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    # Tokenization
    generated_tokens = generated.split()
    reference_tokens = reference.split()

    if not generated_tokens and not reference_tokens:
        rouge_l = 1.0
    elif not generated_tokens or not reference_tokens:
        rouge_l = 0.0
    else:
        # Calculate LCS length
        lcs_len = _lcs_length(generated_tokens, reference_tokens)

        # Calculate ROUGE-L
        if len(generated_tokens) == 0 or len(reference_tokens) == 0:
            rouge_l = 0.0
        else:
            precision = lcs_len / len(generated_tokens)
            recall = lcs_len / len(reference_tokens)
            rouge_l = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

    return GraderScore(
        score=rouge_l,
        reason=f"ROUGE-L score: {rouge_l:.3f}",
        metadata={
            "rouge_l": rouge_l,
            "generated_length": len(generated_tokens),
            "reference_length": len(reference_tokens),
            "lcs_length": _lcs_length(generated_tokens, reference_tokens)
            if generated_tokens and reference_tokens
            else 0,
        },
    )


class NumberAccuracyGrader(Grader):
    """
    Check numerical calculation accuracy by comparing numbers in generated vs reference content.

    This reward verifies if the numbers in the generated content match
    the numbers in the reference content within a specified tolerance.
    """

    def __init__(self, name: str = "number_accuracy", tolerance: float = 1e-6):
        super().__init__(name=name, grader_mode=GraderMode.POINTWISE, description="")
        self.tolerance = tolerance

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        # Match integers and floating point numbers
        number_pattern = r"-?\d+\.?\d*"
        numbers = re.findall(number_pattern, text)
        return [float(n) for n in numbers if n]

    async def evaluate(self, generated, reference) -> GraderScore:
        """
        Calculate number accuracy.
        """
        generated_numbers = self._extract_numbers(generated)
        reference_numbers = self._extract_numbers(reference)

        if not reference_numbers:
            return GraderScore(
                score=0.0,
                reason="No reference numbers to compare",
                metadata={
                    "generated_numbers": generated_numbers,
                    "reference_numbers": reference_numbers,
                },
            )
        if not generated_numbers:
            return GraderScore(
                score=0.0,
                reason="No numbers found in generated content",
                metadata={
                    "generated_numbers": generated_numbers,
                    "reference_numbers": reference_numbers,
                },
            )

        # Compare numbers (match in order)
        correct = 0
        total = min(len(generated_numbers), len(reference_numbers))

        for i in range(total):
            if abs(generated_numbers[i] - reference_numbers[i]) <= self.tolerance:
                correct += 1

        accuracy = correct / len(reference_numbers) if reference_numbers else 0.0

        return GraderScore(
            score=accuracy,
            reason=f"Number accuracy: {correct}/{len(reference_numbers)} numbers correct",
            metadata={
                "accuracy": accuracy,
                "correct_numbers": correct,
                "total_reference_numbers": len(reference_numbers),
                "generated_numbers": generated_numbers,
                "reference_numbers": reference_numbers,
            },
        )
