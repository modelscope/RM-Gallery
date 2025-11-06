from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from rm_gallery.core.grader import Grader, GraderMode, GraderScore


class MathVerifyGrader(Grader):
    """
    Verifies mathematical expressions using the math_verify library, supporting both LaTeX and plain expressions
    """

    def __init__(
        self,
        name: str = "math_verify",
        grader_mode: GraderMode = GraderMode.POINTWISE,
        timeout_score: float = 0.0,
    ):
        """
        Initialize the MathVerifyGrader

        Args:
            name: Name of the grader
            grader_mode: Grader mode
            timeout_score: Score to assign on timeout
        """
        super().__init__(name, grader_mode)
        self.timeout_score = timeout_score

    async def __call__(self, generated, reference) -> GraderScore:
        """
        Verify mathematical expressions

        Args:
            sample: Data sample containing mathematical content

        Returns:
            RewardResult: Reward result containing verification score
        """
        score = 0.0
        reason = "Verification failed or timed out"

        try:
            # Parse the reference (gold) answer
            # Use both LatexExtractionConfig and ExprExtractionConfig for maximum flexibility
            gold_parsed = parse(
                reference,
                extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            )

            # Parse the generated answer
            pred_parsed = parse(
                generated,
                extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            )

            # If both parsing succeeded and we have results
            if gold_parsed and pred_parsed:
                # Use the first parsed result from each
                gold_expr = gold_parsed[0]
                pred_expr = pred_parsed[0]

                # Verify if they match
                if verify(gold_expr, pred_expr):
                    score = 1.0
                    reason = f"({gold_parsed}, {pred_parsed})"
                else:
                    score = 0.0
                    reason = f"({gold_parsed}, {pred_parsed})"
            else:
                score = 0.0
                reason = f"Parsing failed - gold: {gold_parsed}, pred: {pred_parsed}"

        except Exception as e:
            score = self.timeout_score
            reason = f"Exception occurred: {str(e)}"

        return GraderScore(
            score=score,
            reason=reason,
            metadata={
                "generated": generated,
                "reference": reference,
                "score": score,
            },
        )
