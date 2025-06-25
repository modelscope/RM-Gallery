from datetime import datetime

from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.message import ChatMessage, MessageRole
from rm_gallery.gallery.rm.general import (
    AccuracyReward,
    F1ScoreReward,
    NumberAccuracyReward,
    RougeReward,
)


def create_data_sample(
    generated_content: str, reference_content: str = "", unique_id: str = "test_id"
) -> DataSample:
    """Create test DataSample object"""
    return DataSample(
        unique_id=unique_id,
        input=[ChatMessage(role=MessageRole.USER, content="Test question")],
        output=[
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT,
                    content=generated_content,
                    label={"reference": reference_content},
                )
            )
        ],
        task_category="test",
        source="test_source",
        created_at=datetime.now(),
        metadata={"test": True},
    )


class TestAccuracyReward:
    """Test AccuracyReward class"""

    def test_exact_match(self):
        """Test exact match"""
        reward = AccuracyReward()
        sample = create_data_sample("Hello world", "Hello world")

        result = reward._evaluate(sample)

        assert result.name == "accuracy"
        assert len(result.details) == 1
        assert result.details[0].score == 1.0
        assert result.details[0].name == "accuracy"
        assert "matches" in result.details[0].reason
        assert result.extra_data["accuracy"] == 1.0
        assert result.extra_data["generated"] == "Hello world"
        assert result.extra_data["reference"] == "Hello world"

    def test_no_match(self):
        """Test no match"""
        reward = AccuracyReward()
        sample = create_data_sample("Hello world", "Goodbye world")

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "does not match" in result.details[0].reason
        assert result.extra_data["accuracy"] == 0.0

    def test_whitespace_handling(self):
        """Test whitespace handling"""
        reward = AccuracyReward()
        sample = create_data_sample("  Hello world  ", "Hello world")

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0
        assert result.extra_data["generated"] == "Hello world"
        assert result.extra_data["reference"] == "Hello world"

    def test_empty_strings(self):
        """Test empty strings"""
        reward = AccuracyReward()
        sample = create_data_sample("", "")

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0

    def test_case_sensitive(self):
        """Test case sensitive"""
        reward = AccuracyReward()
        sample = create_data_sample("Hello World", "hello world")

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0


class TestF1ScoreReward:
    """Test F1ScoreReward class"""

    def test_perfect_match(self):
        """Test perfect match"""
        reward = F1ScoreReward()
        sample = create_data_sample("hello world test", "hello world test")

        result = reward._evaluate(sample)

        assert result.name == "f1_score"
        assert result.details[0].score == 1.0
        assert result.extra_data["f1_score"] == 1.0
        assert result.extra_data["precision"] == 1.0
        assert result.extra_data["recall"] == 1.0

    def test_partial_match(self):
        """Test partial match"""
        reward = F1ScoreReward()
        sample = create_data_sample("hello world", "hello test")

        result = reward._evaluate(sample)

        assert 0.0 < result.details[0].score < 1.0
        assert result.extra_data["f1_score"] > 0.0

    def test_no_match(self):
        """Test no match"""
        reward = F1ScoreReward()
        sample = create_data_sample("abc def", "xyz uvw")

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert result.extra_data["precision"] == 0.0
        assert result.extra_data["recall"] == 0.0

    def test_empty_strings(self):
        """Test empty strings"""
        reward = F1ScoreReward()
        sample = create_data_sample("", "")

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0

    def test_one_empty_string(self):
        """Test one empty string"""
        reward = F1ScoreReward()
        sample = create_data_sample("hello", "")

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0

    def test_different_tokenizers(self):
        """Test different tokenizers"""
        reward_jieba = F1ScoreReward(tokenizer_type="jieba")
        sample = create_data_sample("你好世界", "你好")

        result = reward_jieba._evaluate(sample)

        assert result.extra_data["tokenizer_type"] == "jieba"
        assert result.details[0].score > 0.0

        reward_simple = F1ScoreReward(tokenizer_type="simple")
        sample = create_data_sample("hello world", "hello")

        result = reward_simple._evaluate(sample)

        assert result.extra_data["tokenizer_type"] == "simple"
        assert result.details[0].score > 0.0


class TestRougeReward:
    """Test RougeReward class"""

    def test_identical_sequences(self):
        """Test identical sequences"""
        reward = RougeReward()
        sample = create_data_sample("hello world test", "hello world test")

        result = reward._evaluate(sample)

        assert result.name == "rouge"
        assert result.details[0].score == 1.0
        assert result.extra_data["rouge_l"] == 1.0

    def test_subsequence_match(self):
        """Test subsequence match"""
        reward = RougeReward()
        sample = create_data_sample("hello beautiful world", "hello world")

        result = reward._evaluate(sample)

        assert result.details[0].score > 0.0
        assert result.extra_data["lcs_length"] == 2

    def test_no_common_subsequence(self):
        """Test no common subsequence"""
        reward = RougeReward()
        sample = create_data_sample("abc def", "xyz uvw")

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert result.extra_data["lcs_length"] == 0

    def test_empty_strings(self):
        """Test empty strings"""
        reward = RougeReward()
        sample = create_data_sample("", "")

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0

    def test_case_insensitive(self):
        """Test case insensitive"""
        reward = RougeReward()
        sample = create_data_sample("Hello World", "hello world")

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0

    def test_different_lengths(self):
        """Test different lengths"""
        reward = RougeReward()
        sample = create_data_sample("a", "a b c d e")

        result = reward._evaluate(sample)

        assert result.details[0].score > 0.0
        assert result.extra_data["generated_length"] == 1
        assert result.extra_data["reference_length"] == 5
        assert result.extra_data["lcs_length"] == 1


class TestNumberAccuracyReward:
    """Test NumberAccuracyReward class"""

    def test_exact_number_match(self):
        """Test exact number match"""
        reward = NumberAccuracyReward()
        sample = create_data_sample("The answer is 42", "The answer is 42")

        result = reward._evaluate(sample)

        assert result.name == "number_accuracy"
        assert result.details[0].score == 1.0
        assert result.extra_data["accuracy"] == 1.0
        assert result.extra_data["correct_numbers"] == 1
        assert result.extra_data["total_reference_numbers"] == 1
        assert result.extra_data["generated_numbers"] == [42.0]
        assert result.extra_data["reference_numbers"] == [42.0]

    def test_multiple_numbers(self):
        """Test multiple numbers"""
        reward = NumberAccuracyReward()
        sample = create_data_sample("Numbers: 1, 2.5, -3", "Numbers: 1, 2.5, -3")

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0
        assert result.extra_data["correct_numbers"] == 3
        assert result.extra_data["generated_numbers"] == [1.0, 2.5, -3.0]
        assert result.extra_data["reference_numbers"] == [1.0, 2.5, -3.0]

    def test_partial_number_match(self):
        """Test partial number match"""
        reward = NumberAccuracyReward()
        sample = create_data_sample("Numbers: 1, 2, 3", "Numbers: 1, 2, 4")

        result = reward._evaluate(sample)

        assert result.details[0].score == 2.0 / 3.0  # 2 out of 3 correct
        assert result.extra_data["correct_numbers"] == 2

    def test_tolerance(self):
        """Test tolerance"""
        reward = NumberAccuracyReward(tolerance=0.1)
        sample = create_data_sample("Value: 3.14", "Value: 3.141")

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0

    def test_no_numbers_in_reference(self):
        """Test no numbers in reference"""
        reward = NumberAccuracyReward()
        sample = create_data_sample("The answer is 42", "No numbers here")

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "No reference numbers" in result.details[0].reason

    def test_no_numbers_in_generated(self):
        """Test no numbers in generated"""
        reward = NumberAccuracyReward()
        sample = create_data_sample("No numbers here", "The answer is 42")

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "No numbers found" in result.details[0].reason

    def test_floating_point_numbers(self):
        """Test floating point numbers"""
        reward = NumberAccuracyReward()
        sample = create_data_sample("Pi is 3.14159", "Pi is 3.14159")

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0
        assert result.extra_data["generated_numbers"] == [3.14159]

    def test_negative_numbers(self):
        """Test negative numbers"""
        reward = NumberAccuracyReward()
        sample = create_data_sample("Temperature: -5.2", "Temperature: -5.2")

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0
        assert result.extra_data["generated_numbers"] == [-5.2]

    def test_different_number_counts(self):
        """Test different number counts"""
        reward = NumberAccuracyReward()
        sample = create_data_sample("Numbers: 1, 2", "Numbers: 1, 2, 3, 4")

        result = reward._evaluate(sample)

        assert (
            result.details[0].score == 2.0 / 4.0
        )  # 2 matches out of 4 reference numbers
        assert result.extra_data["correct_numbers"] == 2


class TestRewardIntegration:
    """Test reward integration"""

    def test_all_rewards_with_same_sample(self):
        """Test all rewards with same sample"""
        sample = create_data_sample("The answer is 42", "The answer is 42")

        accuracy_reward = AccuracyReward()
        f1_reward = F1ScoreReward()
        rouge_reward = RougeReward()
        number_reward = NumberAccuracyReward()

        accuracy_result = accuracy_reward._evaluate(sample)
        f1_result = f1_reward._evaluate(sample)
        rouge_result = rouge_reward._evaluate(sample)
        number_result = number_reward._evaluate(sample)

        assert accuracy_result.details[0].score == 1.0
        assert f1_result.details[0].score == 1.0
        assert rouge_result.details[0].score == 1.0
        assert number_result.details[0].score == 1.0

    def test_reward_registry_integration(self):
        """Test reward registry integration"""
        from rm_gallery.core.reward.registry import RewardRegistry

        assert "accuracy" in RewardRegistry._registry
        assert "f1_score" in RewardRegistry._registry
        assert "rouge" in RewardRegistry._registry
        assert "number_accuracy" in RewardRegistry._registry

        accuracy_reward_class = RewardRegistry.get("accuracy")
        assert accuracy_reward_class is not None
        assert issubclass(accuracy_reward_class, AccuracyReward)

        f1_reward_class = RewardRegistry.get("f1_score")
        assert f1_reward_class is not None
        assert issubclass(f1_reward_class, F1ScoreReward)

        rouge_reward_class = RewardRegistry.get("rouge")
        assert rouge_reward_class is not None
        assert issubclass(rouge_reward_class, RougeReward)

        number_reward_class = RewardRegistry.get("number_accuracy")
        assert number_reward_class is not None
        assert issubclass(number_reward_class, NumberAccuracyReward)

        accuracy_reward = accuracy_reward_class()
        assert isinstance(accuracy_reward, AccuracyReward)

        f1_reward = f1_reward_class()
        assert isinstance(f1_reward, F1ScoreReward)

        rouge_reward = rouge_reward_class()
        assert isinstance(rouge_reward, RougeReward)

        number_reward = number_reward_class()
        assert isinstance(number_reward, NumberAccuracyReward)
