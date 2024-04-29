"""OpenAI LLM API Data Processor."""
import asyncio
from contextlib import nullcontext
from typing import Annotated, Any

from datasets import Features, Sequence, Value
from openai import AsyncOpenAI
from pydantic import Field

from hyped.common.feature_checks import raise_feature_is_sequence
from hyped.common.feature_key import FeatureCollection
from hyped.common.lazy import LazyInstance
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


class OpenAIChatCompletionConfig(BaseDataProcessorConfig):
    """OpenAI Chat Completion Data Processor Config.

    Attributes:
        messages (FeatureCollection):
            input chat messages, following the openai chat message format
        model (str):
            openai model to use
        max_concurrent_calls (None | int):
            the maximum number of concurrent calls to the api. When using
            multiple processes, each process can have up to a total of
            `max_concurrent_calls` calls to the api at a time.
    """

    messages: FeatureCollection

    model: str = "gpt-3.5-turbo-0125"
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    logit_bias: dict[int, Annotated[float, Field(ge=-100, le=100)]] = {}
    logprobs: bool = False
    top_logprobs: None | int = None
    temperature: float = Field(default=1, ge=0, le=2)
    top_p: float = Field(default=1, ge=0, le=1)
    max_tokens: None | int = None
    response_format: None | dict[str, str] = None
    seed: None | int = None
    stop: None | str = None

    max_concurrent_calls: None | int = None


class OpenAIChatCompletion(BaseDataProcessor[OpenAIChatCompletionConfig]):
    """OpenAI Chat Completion Data Processor.

    MacOS users might have to set the following environment variable when using
    this processor in a multiprocessing setting:

    .. code-block:: bash

        OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    """

    def __init__(self, config: OpenAIChatCompletionConfig) -> None:
        """Construct a new processor.

        Arguments:
            config (OpenAIChatCompletionConfig):
                configuration of the processor
        """
        super(OpenAIChatCompletion, self).__init__(config)
        # create semaphore object to control the maximum
        # number of concurrent calls to the api
        self.sem = (
            asyncio.Semaphore(value=self.config.max_concurrent_calls)
            if self.config.max_concurrent_calls is not None
            else nullcontext()
        )
        # create a lazy instance of the openai client
        self.client = LazyInstance(AsyncOpenAI)

        # check if there are any non-constant values in the messages
        if len(list(self.config.messages.feature_keys)) == 0:
            warnings.warn(
                "The specified messages do not contain any variables, i.e. "
                "FeatureKeys. The prompt will be constant for all examples.",
                UserWarning,
            )

    def map_features(self, features: Features) -> Features:
        """Map features."""
        # check the messages feature
        raise_feature_is_sequence(
            self.config.messages,
            self.config.messages.index_features(features),
            Features({"role": Value("string"), "content": Value("string")}),
        )

        return {
            "completion": {
                "run_id": Value("string"),
                "message": Value("string"),
                "tool_calls": Sequence(
                    {
                        "type": Value("string"),
                        "function": {
                            "name": Value("string"),
                            "arguments": Value("string"),
                        },
                    }
                ),
                "usage": {
                    "completion_tokens": Value("int32"),
                    "prompt_tokens": Value("int32"),
                    "total_tokens": Value("int32"),
                },
            }
        }

    async def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Process example."""
        with self.sem:
            resp = await self.client.chat.completions.create(
                messages=self.config.messages.index_example(example),
                model=self.config.model,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                logit_bias=self.config.logit_bias,
                logprobs=self.config.logprobs,
                top_logprobs=self.config.top_logprobs,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                response_format=self.config.response_format,
                seed=self.config.seed,
                stop=self.config.stop,
            )

        return {
            "run_id": resp.id,
            "completion": {
                "message": resp.choices[0].message.content,
                "function_call": resp.choices[0].message.function_call,
                "tool_calls": resp.choices[0].message.tool_calls,
            },
            "usage": {
                "completion_tokens": resp.usage.completion_tokens,
                "prompt_tokens": resp.usage.prompt_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
        }
