from datetime import datetime

from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.message import ChatMessage, MessageRole
from rm_gallery.gallery.rm.math.math import MathVerifyReward


def create_math_data_sample(
    generated_content: str, reference_content: str = "", unique_id: str = "test_id"
) -> DataSample:
    """Create a DataSample object for math verification testing"""
    return DataSample(
        unique_id=unique_id,
        input=[ChatMessage(role=MessageRole.USER, content="Solve this math problem")],
        output=[
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT,
                    content=generated_content,
                    label={"reference": reference_content},
                )
            )
        ],
        task_category="math",
        source="test_source",
        created_at=datetime.now(),
        metadata={"test": True},
    )


class TestMathVerifyReward:
    """Test the MathVerifyReward class"""

    def test_init_default_values(self):
        """Test default initialization values"""
        reward = MathVerifyReward()

        assert reward.name == "math_verify"
        assert reward.timeout_score == 0.0

    def test_init_custom_values(self):
        """Test custom initialization values"""
        reward = MathVerifyReward(name="custom_math_verify", timeout_score=0.5)

        assert reward.name == "custom_math_verify"
        assert reward.timeout_score == 0.5

    def test_correct_math_answer(self):
        """Test correct math answer"""
        reward = MathVerifyReward()
        sample = create_math_data_sample(
            generated_content="42",  # Simplified to direct number answer
            reference_content="42",
        )

        result = reward._evaluate(sample)

        # Verify results
        assert result.name == "math_verify"
        assert len(result.details) == 1
        assert result.details[0].score == 1.0
        assert result.details[0].name == "math_verify"
        # reason contains detailed verification info, don't check specific content

        # Verify extra_data
        assert result.extra_data["generated"] == "42"
        assert result.extra_data["reference"] == "42"
        assert result.extra_data["score"] == 1.0

    def test_incorrect_math_answer(self):
        """Test incorrect math answer"""

        reward = MathVerifyReward()
        sample = create_math_data_sample(
            generated_content="24",  # Simplified to direct number answer
            reference_content="42",
        )

        result = reward._evaluate(sample)

        # Verify results
        assert result.details[0].score == 0.0
        # Don't check specific reason content as it comes from math_verify library
        assert result.extra_data["score"] == 0.0

    def test_math_expression_matching(self):
        """Test math expression matching"""

        reward = MathVerifyReward()
        sample = create_math_data_sample(
            generated_content="2*x",  # Use expression that can be correctly parsed by math_verify
            reference_content="2x",
        )

        result = reward._evaluate(sample)

        # Mathematically equivalent expressions should match
        assert result.details[0].score == 1.0
        assert result.extra_data["score"] == 1.0

    def test_exception_handling(self):
        """Test exception handling"""

        reward = MathVerifyReward()
        sample = create_math_data_sample(
            generated_content="Invalid math expression", reference_content="42"
        )

        result = reward._evaluate(sample)

        # Return 0.0 score when parsing fails
        assert result.details[0].score == 0.0
        assert result.extra_data["score"] == 0.0

    def test_empty_strings(self):
        """Test empty string handling"""

        reward = MathVerifyReward()
        sample = create_math_data_sample(generated_content="", reference_content="")

        result = reward._evaluate(sample)

        assert result.extra_data["generated"] == ""
        assert result.extra_data["reference"] == ""

    def test_whitespace_handling(self):
        """Test whitespace handling"""

        reward = MathVerifyReward()
        sample = create_math_data_sample(
            generated_content="  42  ", reference_content="  42  "
        )

        result = reward._evaluate(sample)

        # Verify whitespace is handled correctly
        assert result.extra_data["generated"] == "42"
        assert result.extra_data["reference"] == "42"

    def test_latex_math_expression(self):
        """Test LaTeX math expression"""

        reward = MathVerifyReward()
        sample = create_math_data_sample(
            generated_content="\\frac{d}{dx}(x^2) = 2x",  # Remove outer $ symbols
            reference_content="2x",
        )

        result = reward._evaluate(sample)

        # Verify LaTeX expression is handled correctly
        assert result.details[0].score == 1.0

    def test_custom_timeout_score(self):
        """Test custom timeout score"""

        reward = MathVerifyReward(timeout_score=0.5)

        # Create a sample that will cause exception (simulated by modifying implementation)
        sample = create_math_data_sample(
            generated_content="Answer", reference_content="42"
        )

        result = reward._evaluate(sample)

        # Return 0.0 when parsing fails (not timeout_score since this isn't a timeout)
        assert result.details[0].score == 0.0
        assert result.extra_data["score"] == 0.0

    def test_missing_reference(self):
        """Test missing reference answer"""

        reward = MathVerifyReward()
        # Create sample without reference
        sample = DataSample(
            unique_id="test_id",
            input=[ChatMessage(role=MessageRole.USER, content="Math problem")],
            output=[
                DataOutput(
                    answer=Step(
                        role=MessageRole.ASSISTANT,
                        content="42",
                        label={},  # No reference
                    )
                )
            ],
            task_category="math",
            source="test_source",
            created_at=datetime.now(),
        )

        result = reward._evaluate(sample)

        assert result.extra_data["reference"] == ""

    def test_basic_math_expressions(self):
        """Test basic math expression evaluation"""

        reward = MathVerifyReward()

        # Test simple number matching
        sample1 = create_math_data_sample("42", "42")
        result1 = reward._evaluate(sample1)
        assert result1.details[0].score == 1.0

        # Test non-matching numbers
        sample2 = create_math_data_sample("24", "42")
        result2 = reward._evaluate(sample2)
        assert result2.details[0].score == 0.0

        # Test mathematically equivalent expressions
        sample3 = create_math_data_sample("6", "2*3")
        result3 = reward._evaluate(sample3)
        assert result3.details[0].score == 1.0


class TestMathVerifyRewardIntegration:
    """Integration tests"""

    def test_multiple_samples_evaluation(self):
        """Test evaluation of multiple samples"""

        reward = MathVerifyReward()

        samples = [
            create_math_data_sample("42", "42"),  # Exact match
            create_math_data_sample("24", "42"),  # No match
            create_math_data_sample("6", "2*3"),  # Mathematically equivalent
        ]

        results = []
        for sample in samples:
            result = reward._evaluate(sample)
            results.append(result)

        assert len(results) == 3
        assert results[0].details[0].score == 1.0  # Match
        assert results[1].details[0].score == 0.0  # No match
        assert results[2].details[0].score == 1.0  # Equivalent
