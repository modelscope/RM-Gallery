# -*- coding: utf-8 -*-
"""Unit tests for QwenVLModel."""

from unittest.mock import MagicMock, patch

import pytest

from openjudge.models.qwen_vl_model import QwenVLModel


def _fake_response(text: str = "OK"):
    response = MagicMock()
    response.status_code = 200
    response.message = ""
    response.output.choices = [MagicMock()]
    response.output.choices[0].message.content = [{"text": text}]
    return response


@pytest.mark.unit
class TestQwenVLModelTimeout:
    """Test cases for QwenVLModel's timeout handling."""

    @pytest.mark.parametrize(
        "init_kwargs, expect_request_timeout",
        [
            ({"timeout": 5.0}, 5.0),
            ({}, None),
            ({"timeout": None}, None),
        ],
        ids=["with_timeout", "defaults", "explicit_none"],
    )
    @patch("openjudge.models.qwen_vl_model.MultiModalConversation")
    def test_generate_forwards_request_timeout(
        self,
        mock_conversation,
        init_kwargs,
        expect_request_timeout,
    ):
        """timeout=N must reach DashScope as request_timeout=N, never as timeout=N."""
        mock_conversation.call.return_value = _fake_response()

        model = QwenVLModel(api_key="test-key", **init_kwargs)
        model.generate(text="hi")

        call_kwargs = mock_conversation.call.call_args[1]
        assert "timeout" not in call_kwargs

        if expect_request_timeout is None:
            assert "request_timeout" not in call_kwargs
        else:
            assert call_kwargs["request_timeout"] == expect_request_timeout
