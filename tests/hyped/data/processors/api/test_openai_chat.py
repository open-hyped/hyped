from unittest.mock import MagicMock, patch

import pytest
from datasets import Features, Sequence, Value
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)

from hyped.common.feature_key import Const, FeatureCollection
from hyped.data.processors.api.openai_chat import (
    OpenAIChatCompletion,
    OpenAIChatCompletionConfig,
)
from tests.hyped.data.processors.base import BaseTestDataProcessor


async def dummy_chat_completion_response(*args, **kwargs):
    return ChatCompletion(
        id="0",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    content="This is a completion message",
                    role="assistant",
                ),
            )
        ],
        created=0,
        model="dummy_model",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=10,
            prompt_tokens=15,
            total_tokens=25,
        ),
    )


class TestOpenAIChatCompletion(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "messages": Sequence(
                    {"role": Value("string"), "content": Value("string")}
                )
            }
        )

    @pytest.fixture
    def processor(self):
        # create a mock chat client to be used in the processor
        mock_chat_client = MagicMock()
        mock_chat_client.chat = MagicMock()
        mock_chat_client.chat.completions = MagicMock()
        mock_chat_client.chat.completions.create.side_effect = (
            dummy_chat_completion_response
        )

        # patch the async openai client with the mock client
        with patch(
            "hyped.data.processors.api.openai_chat.AsyncOpenAI",
            return_value=mock_chat_client,
        ):
            return OpenAIChatCompletion(
                OpenAIChatCompletionConfig(messages="messages")
            )

    @pytest.fixture
    def in_batch(self):
        return {
            "messages": [
                [
                    {"role": "system", "content": "This is a system message"},
                    {"role": "user", "content": "This is a user message"},
                ]
            ]
        }

    def test_warning_on_constant_prompt(self):
        with pytest.warns(UserWarning):
            OpenAIChatCompletion(
                OpenAIChatCompletionConfig(
                    # messages is a feature collection of only
                    # constant features
                    messages=FeatureCollection(
                        [
                            {
                                "role": Const("system"),
                                "content": Const("This is a system message"),
                            },
                            {
                                "role": Const("user"),
                                "content": Const("This is a user message"),
                            },
                        ]
                    )
                )
            )
