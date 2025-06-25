from unittest.mock import MagicMock, patch

import pytest

from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.core.reward.principle.generator import PrincipleGenerator


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=BaseLLM)
    llm.simple_chat.return_value = '<think>here is a reasoning trace</think><principles>```json{"test_key": "test_description"}```</principles>'
    llm.enable_thinking = True  # Add missing attribute
    return llm


@pytest.fixture
def sample_data():
    return DataSample(
        unique_id="test",
        input=[ChatMessage(role="user", content="Hello!")],
        output=[
            DataOutput(
                answer=Step(
                    role="assistant",
                    content="Hello! How can I assist you today?",
                    label={"preference": "chosen"},
                )
            ),
            DataOutput(
                answer=Step(
                    role="assistant", content="Hello!", label={"preference": "rejected"}
                )
            ),
        ],
    )


def test_generate(mock_llm: MagicMock, sample_data: DataSample):
    generator = PrincipleGenerator(
        llm=mock_llm, scenario="test", generate_number=1, cluster_number=1
    )

    result = generator.generate(sample_data)
    assert hasattr(result.input[-1], "additional_kwargs")
    assert "generate" in result.input[-1].additional_kwargs
    # Added verification of principle content
    assert result.input[-1].additional_kwargs["generate"]["principles"] == {
        "test_key": "test_description"
    }


def test_cluster(mock_llm: MagicMock, sample_data: DataSample):
    generator = PrincipleGenerator(
        llm=mock_llm, scenario="test", generate_number=1, cluster_number=1
    )

    # Modified to use real generate call
    generated_samples = [generator.generate(sample_data)]
    result = generator.cluster(generated_samples)

    assert isinstance(result, dict)
    # Changed to expect actual principle key from mock
    assert "test_key" in result
    # Added value verification
    assert result["test_key"] == "test_description"


@patch("rm_gallery.core.reward.principle.generator.ThreadPoolExecutor")
def test_run_batch(mock_executor, mock_llm: MagicMock, sample_data: DataSample):
    generator = PrincipleGenerator(
        llm=mock_llm, scenario="test", generate_number=1, cluster_number=1
    )

    # Fixed mock setup to return valid samples
    mock_executor.return_value.__enter__.return_value.submit.side_effect = [
        MagicMock(result=generator.generate(sample_data)),
        MagicMock(result=generator.generate(sample_data)),
    ]

    result = generator.run_batch([sample_data, sample_data], mock_executor.return_value)
    assert isinstance(result, dict)
    assert "test_key" in result
    assert result["test_key"] == "test_description"
