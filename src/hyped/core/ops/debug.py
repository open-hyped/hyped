"""This module defines a collection of debug nodes for inspecting data within processing pipelines.

These nodes provide functionalities such as printing samples, logging information, and asserting
conditions to aid in debugging and monitoring the flow and content of data as it moves through
the pipeline.
"""
import json

from ..nodes.base import RunContext
from ..nodes.debug import BaseDebugNode, BaseDebugNodeConfig
from ..typing import Feature

GREY = "\033[90m"
RESET = "\033[0m"

DEFAULT_FORMAT_STRING = f"{GREY}[Rank: {{rank}}, Index: {{index}}]{RESET} {{feature}}"


class PrintSampleConfig(BaseDebugNodeConfig):
    """Configuration for the :class:`PrintSample` node."""

    format_string: str = DEFAULT_FORMAT_STRING
    """The format of the print message.

    This string can include placeholders like '{rank}', '{index}', and '{feature}'
    which will be replaced with the corresponding values during processing.
    """

    indent: int | None = None
    """Number of spaces to use for indentation when printing the feature as JSON.
    If None, the feature will be printed without indentation.
    """


class PrintSample(BaseDebugNode[PrintSampleConfig]):
    """Debug node that prints the input feature with formatting.

    This node takes an input feature and prints it to the console.
    The output format can be customized using the `format_string` configuration.
    The feature can also be pretty-printed as JSON if an `indent` value is provided.
    """

    async def process(self, ctx: RunContext, feature: Feature) -> None:
        """Prints the input feature to the console with specified formatting.

        Args:
            ctx (RunContext): Context object containing runtime information,
                including the rank and index of the current sample.
            feature (Feature): The input feature to be printed.
        """
        if self.config.indent is not None:
            feature = json.dumps(feature, indent=self.config.indent)
        # format message
        msg = self.config.format_string.format(
            rank=ctx.rank,
            index=ctx.index,
            feature=feature,
        )
        # print message
        print(msg)  # noqa: T201


def print_(
    feature: Feature, format_string: str = DEFAULT_FORMAT_STRING, indent: int | None = None
) -> None:
    """Prints a given feature to the console.

    This function creates and calls a :class:`PrintSample` node to print the provided
    feature with optional formatting and indentation.

    Args:
        feature (Feature): The feature to be printed.
        format_string (str, optional): The format string for printing.
            Defaults to :const:`DEFAULT_FORMAT_STRING`.
        indent (int | None, optional): Number of spaces to indent the feature when
            serializing it to JSON. Defaults to None, in which case the raw feature
            is printed.
    """
    PrintSample(format_string=format_string, indent=indent).call(feature)
