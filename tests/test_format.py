import json
from unittest.mock import patch

import pytest

from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.gallery.rm.format.format import (
    LengthPenaltyReward,
    NgramRepetitionPenaltyReward,
    PrivacyLeakageReward,
    ReasoningFormatReward,
    ReasoningToolCallFormatReward,
)


def create_sample(content: str) -> DataSample:
    """Create a DataSample for testing"""
    answer = Step(content=content, role="assistant")
    output = DataOutput(answer=answer)
    return DataSample(unique_id="test", output=[output])


class TestReasoningFormatReward:
    """Test ReasoningFormatReward class"""

    def test_valid_format(self):
        """Test valid format"""
        reward = ReasoningFormatReward()
        content = "<think>This is the thought process</think><answer>This is the answer</answer>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0
        assert "All format requirements met" in result.details[0].reason
        assert result.extra_data["has_think_tag"] is True
        assert result.extra_data["has_answer_tag"] is True

    def test_missing_think_tag(self):
        """Test missing think tag"""
        reward = ReasoningFormatReward()
        content = "<answer>This is the answer</answer>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "Missing <think></think> tags" in result.details[0].reason
        assert result.extra_data["has_think_tag"] is False
        assert result.extra_data["has_answer_tag"] is True

    def test_missing_answer_tag(self):
        """Test missing answer tag"""
        reward = ReasoningFormatReward()
        content = "<think>This is the thought process</think>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "Missing <answer></answer> tags" in result.details[0].reason
        assert result.extra_data["has_think_tag"] is True
        assert result.extra_data["has_answer_tag"] is False

    def test_missing_both_tags(self):
        """Test missing both tags"""
        reward = ReasoningFormatReward()
        content = "This is plain text"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        reason = result.details[0].reason
        assert "Missing <think></think> tags" in reason
        assert "Missing <answer></answer> tags" in reason

    def test_custom_tokens(self):
        """Test custom tokens"""
        reward = ReasoningFormatReward(think_token="thinking", answer_token="response")
        content = "<thinking>Thought</thinking><response>Answer</response>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0
        assert result.extra_data["think_token"] == "thinking"
        assert result.extra_data["answer_token"] == "response"


class TestReasoningToolCallFormatReward:
    """Test ReasoningToolCallFormatReward class"""

    def test_valid_think_answer_format(self):
        """Test valid think + answer format"""
        reward = ReasoningToolCallFormatReward()
        content = "<think>This is the thought process</think><answer>This is the answer</answer>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0
        assert (
            "Valid <think></think> + <answer></answer> format"
            in result.details[0].reason
        )

    def test_valid_think_tool_call_format(self):
        """Test valid think + tool call format"""
        reward = ReasoningToolCallFormatReward()
        tool_call_json = json.dumps(
            {"name": "calculate", "arguments": {"expression": "2+2"}}
        )
        content = (
            f"<think>I need to calculate</think><tool_call>{tool_call_json}</tool_call>"
        )
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0
        assert (
            "Valid <think></think> + <tool_call></tool_call> format with valid JSON"
            in result.details[0].reason
        )

    def test_multiple_tool_calls(self):
        """Test multiple tool calls"""
        reward = ReasoningToolCallFormatReward()
        tool_call1 = json.dumps({"name": "search", "arguments": {"query": "test"}})
        tool_call2 = json.dumps({"name": "calculate", "arguments": {"expr": "1+1"}})
        content = f"<think>Need to search and calculate</think><tool_call>{tool_call1}</tool_call><tool_call>{tool_call2}</tool_call>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 1.0
        assert result.extra_data["tool_call_count"] == 2

    def test_invalid_json_in_tool_call(self):
        """Test invalid JSON in tool call"""
        reward = ReasoningToolCallFormatReward()
        content = "<think>Thought</think><tool_call>{invalid json}</tool_call>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "Invalid JSON format in <tool_call> tags" in result.details[0].reason

    def test_missing_required_json_fields(self):
        """Test missing required JSON fields"""
        reward = ReasoningToolCallFormatReward()
        tool_call_json = json.dumps({"name": "test"})  # Missing arguments
        content = f"<think>Thought</think><tool_call>{tool_call_json}</tool_call>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0

    def test_invalid_combination_both_answer_and_tool_call(self):
        """Test invalid combination: both answer and tool call"""
        reward = ReasoningToolCallFormatReward()
        tool_call_json = json.dumps({"name": "test", "arguments": {}})
        content = f"<think>Thought</think><answer>Answer</answer><tool_call>{tool_call_json}</tool_call>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "Invalid combination" in result.details[0].reason

    def test_missing_think_tag(self):
        """Test missing think tag"""
        reward = ReasoningToolCallFormatReward()
        content = "<answer>Only answer</answer>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "Missing <think></think> tags" in result.details[0].reason


class TestLengthPenaltyReward:
    """Test LengthPenaltyReward class"""

    def test_acceptable_length(self):
        """Test acceptable length"""
        reward = LengthPenaltyReward(min_length=10, max_length=100)
        content = "This is a text with appropriate length, should not be penalized."
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "Length acceptable" in result.details[0].reason

    def test_too_short_penalty(self):
        """Test too short penalty"""
        reward = LengthPenaltyReward(min_length=50, max_length=100, penalty_rate=0.1)
        content = "Short text"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score < 0.0
        assert "Too short" in result.details[0].reason
        expected_penalty = -(50 - len(content)) * 0.1
        assert result.details[0].score == expected_penalty

    def test_too_long_penalty(self):
        """Test too long penalty"""
        reward = LengthPenaltyReward(min_length=10, max_length=20, penalty_rate=0.05)
        content = "This is a very long text content that should trigger the length penalty mechanism."
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score < 0.0
        assert "Too long" in result.details[0].reason
        expected_penalty = -(len(content) - 20) * 0.05
        assert result.details[0].score == expected_penalty


class TestNgramRepetitionPenaltyReward:
    """Test NgramRepetitionPenaltyReward class"""

    @patch("rm_gallery.core.utils.tokenizer.get_tokenizer")
    def test_no_repetition(self, mock_get_tokenizer):
        """Test no repetition"""
        reward = NgramRepetitionPenaltyReward(n=2, penalty_threshold=0.3)
        content = "This is a unique sentence."
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "repetition rate" in result.details[0].reason

    @patch("rm_gallery.core.utils.tokenizer.get_tokenizer")
    def test_high_repetition(self, mock_get_tokenizer):
        """Test high repetition rate"""
        reward = NgramRepetitionPenaltyReward(
            n=2, penalty_threshold=0.3, penalty_rate=1.0
        )
        content = "The the the the the"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score < 0.0

    @patch("rm_gallery.core.utils.tokenizer.get_tokenizer")
    def test_soft_penalty_mode(self, mock_get_tokenizer):
        """Test soft penalty mode"""
        reward = NgramRepetitionPenaltyReward(
            n=2, use_soft_penalty=True, max_penalty=-2.0, min_scaling=0.1
        )
        content = "Word word word word"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.extra_data["penalty_mode"] == "soft"

    @patch("rm_gallery.core.utils.tokenizer.get_tokenizer")
    def test_thought_scope_analysis(self, mock_get_tokenizer):
        """Test thought scope analysis"""
        reward = NgramRepetitionPenaltyReward(n=2, analyze_scope="thought")
        content = "<think>thinking process</think><answer>final answer</answer>"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.extra_data["analyze_scope"] == "thought"

    @patch("rm_gallery.core.utils.tokenizer.get_tokenizer")
    def test_text_too_short(self, mock_get_tokenizer):
        """Test text too short"""
        reward = NgramRepetitionPenaltyReward(n=3)
        content = "Hi"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "Text too short" in result.details[0].reason

    @patch("rm_gallery.core.utils.tokenizer.get_tokenizer")
    def test_no_thought_process_found(self, mock_get_tokenizer):
        """Test no thought process found"""
        reward = NgramRepetitionPenaltyReward(n=2, analyze_scope="thought")
        content = "No thought tag"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "No thought process found" in result.details[0].reason


class TestPrivacyLeakageReward:
    """Test PrivacyLeakageReward class"""

    def test_no_privacy_leaks(self):
        """Test no privacy leaks"""
        reward = PrivacyLeakageReward()
        content = "This is a normal text, no privacy information."
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert "No privacy leaks detected" in result.details[0].reason
        assert result.extra_data["total_leaks"] == 0

    def test_email_detection(self):
        """Test email detection"""
        reward = PrivacyLeakageReward(penalty_per_leak=-1.0)
        content = "Please contact me: test@example.com or user@gmail.com"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == -2.0
        assert "Privacy leaks detected" in result.details[0].reason
        assert result.extra_data["leak_types"]["email"] == 2

    def test_phone_detection(self):
        """Test phone detection"""
        reward = PrivacyLeakageReward(penalty_per_leak=-0.5)
        content = "My phone number is 123-456-7890"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == -0.5
        assert result.extra_data["leak_types"]["phone"] == 1

    def test_id_card_detection(self):
        """Test ID card detection"""
        reward = PrivacyLeakageReward()
        content = "ID card number: 110101199001011234"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score < 0.0
        assert result.extra_data["leak_types"]["id_card"] == 1

    def test_credit_card_detection(self):
        """Test credit card detection"""
        reward = PrivacyLeakageReward()
        content = "Credit card number: 1234 5678 9012 3456"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score < 0.0
        assert result.extra_data["leak_types"]["credit_card"] == 1

    def test_ip_address_detection(self):
        """Test IP address detection"""
        reward = PrivacyLeakageReward()
        content = "Server IP: 203.0.113.1"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score < 0.0
        assert result.extra_data["leak_types"]["ip_address"] == 1

    def test_multiple_leak_types(self):
        """Test multiple leak types"""
        reward = PrivacyLeakageReward(penalty_per_leak=-1.0)
        content = "Contact information: test@example.com, phone: 123-456-7890, IP: 203.0.113.1"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == -3.0
        leaks = result.extra_data["leaks"]
        leak_types = [leak["type"] for leak in leaks]
        assert "email" in leak_types
        assert "phone" in leak_types
        assert "ip_address" in leak_types

    def test_localhost_ip_exclusion(self):
        """Test localhost IP exclusion"""
        reward = PrivacyLeakageReward()
        content = "Local server: 127.0.0.1 and 192.168.1.1"
        sample = create_sample(content)

        result = reward._evaluate(sample)

        assert result.details[0].score == 0.0
        assert result.extra_data["total_leaks"] == 0


class TestIntegration:
    """Test integration"""

    def test_multiple_rewards_combination(self):
        """Test multiple rewards combination"""
        content = """
        <think>I need to think about this problem</think>
        <answer>Contact information: test@example.com, phone: 123-456-7890</answer>
        """
        sample = create_sample(content)

        format_reward = ReasoningFormatReward()
        format_result = format_reward._evaluate(sample)
        assert format_result.details[0].score == 1.0

        privacy_reward = PrivacyLeakageReward()
        privacy_result = privacy_reward._evaluate(sample)
        assert privacy_result.details[0].score < 0.0
        assert privacy_result.extra_data["total_leaks"] == 2

    @patch("rm_gallery.core.utils.tokenizer.get_tokenizer")
    def test_comprehensive_evaluation(self, mock_get_tokenizer):
        """Test comprehensive evaluation"""
        content = "<think>Deep thinking</think><answer>This is a perfect answer with correct format, appropriate length, no repetition, and no privacy leaks.</answer>"
        sample = create_sample(content)

        rewards = [
            ReasoningFormatReward(),
            LengthPenaltyReward(min_length=10, max_length=200),
            NgramRepetitionPenaltyReward(n=3),
            PrivacyLeakageReward(),
        ]

        results = []
        for reward in rewards:
            result = reward._evaluate(sample)
            results.append(result)

        for result in results:
            assert result.details[0].score >= 0.0


class TestHelperFunctions:
    """Test helper functions"""

    def test_create_sample_function(self):
        """Test create sample function"""
        content = "Test content"
        sample = create_sample(content)

        assert sample.unique_id == "test"
        assert len(sample.output) == 1
        assert sample.output[0].answer.content == content
        assert sample.output[0].answer.role == "assistant"

    def test_sample_structure_validation(self):
        """Test sample structure validation"""
        content = "<think>Thinking</think><answer>Answer</answer>"
        sample = create_sample(content)

        assert hasattr(sample, "unique_id")
        assert hasattr(sample, "output")
        assert hasattr(sample.output[0], "answer")
        assert hasattr(sample.output[0].answer, "content")
        assert hasattr(sample.output[0].answer, "role")

    def test_edge_case_empty_content(self):
        """Test edge case: empty content"""
        sample = create_sample("")

        assert sample.output[0].answer.content == ""

        rewards = [
            ReasoningFormatReward(),
            LengthPenaltyReward(),
            PrivacyLeakageReward(),
        ]

        for reward in rewards:
            result = reward._evaluate(sample)
            # should be able to handle, not throw exception
            assert hasattr(result, "details")
            assert len(result.details) > 0


if __name__ == "__main__":
    pytest.main([__file__])
