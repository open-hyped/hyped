"""This module defines a collection of debug nodes for inspecting data within processing pipelines.

These nodes provide functionalities such as printing samples, logging information, and asserting
conditions to aid in debugging and monitoring the flow and content of data as it moves through
the pipeline.
"""
import json
from typing import overload

from ..nodes.base import RunContext
from ..nodes.debug import BaseDebugNode, BaseDebugNodeConfig
from ..typing import Feature

GREY = "\033[90m"
RESET = "\033[0m"


class PrintSampleConfig(BaseDebugNodeConfig):
    """Configuration for the :class:`PrintSample` node."""

    format_string: str = "{feature}"
    """The format string for printing.

    This string uses standard Python formatting syntax to incorporate positional
    and keyword arguments passed to the node's :func:`process` method.
    """

    indent: int | None = None
    """Number of spaces to use for indentation when printing feature arguments as JSON.

    If :code:`None`, the raw feature will be printed. If an integer is provided,
    feature arguments will be serialized to JSON with that level of indentation
    before being formatted according to :code:`format_string`.
    """


class PrintSample(BaseDebugNode[PrintSampleConfig]):
    """Debug node that prints the input feature with formatting.

    This node takes positional and keyword arguments representing features
    and prints them to the console according to the `format_string` configuration.
    Features can be optionally indented if the `indent` configuration is set.
    """

    # TODO: support args
    async def process(self, ctx: RunContext, **kwargs: Feature) -> None:
        """Prints the input features to the console with specified formatting.

        Args:
            ctx (RunContext): Context object containing runtime information,
                including the rank and index of the current sample.
            *args (Feature): Positional features to be printed. These will be
                formatted based on their order in the :code:`format_string`.
            **kwargs (Feature): Keyword features to be printed. These will be
                formatted based on their names in the :code:`format_string`.
        """
        args = []

        if self.config.indent is not None:
            args = (json.dumps(v, indent=self.config.indent) for v in args)
            kwargs = {k: json.dumps(v, indent=self.config.indent) for k, v in kwargs.items()}

        # format and print message
        msg = self.config.format_string.format(*args, **kwargs)
        print(f"{GREY}[Rank: {ctx.rank}, Index: {ctx.index}]{RESET} {msg}")  # noqa: T201


@overload
def print_(*args: Feature, indent: int | None = None, **named_features: Feature) -> None:
    ...


@overload
def print_(
    format: str, *feature: Feature, indent: int | None = None, **named_features: Feature
) -> None:
    ...


def print_(*args: str | Feature, indent: int | None = None, **kwargs: str | Feature) -> None:
    """Prints provided features to the console with optional formatting and indentation.

    This function creates and calls a :class:`PrintSample` node to print the provided
    features with optional formatting and indentation.

    There are two ways to call this function:

    1. :code:`print_(*features: Feature, **named_features: Feature)`:
       Prints one or more features using a default format string where each positional
       feature is represented by :code:`"{}"` and each keyword feature :code:`name=value`
       is represented by :code:`"{name}"`. An optional keyword-only :code:`indent` controls
       JSON pretty-printing of the features.

    2. :code:`print_(format_string: str, *features: Feature, **named_features: Feature)`:
       Prints one or more features using the provided :code:`format_string`. Positional
       features are passed as positional arguments to the format string, and keyword
       features are passed as keyword arguments. An optional keyword-only :code:`indent`
       controls JSON pretty-printing of the features.

    Args:
        *args: Positional arguments.
            - The first argument can optionally be a :code:`format_string`.
            - Subsequent arguments are positional features (:class:`Feature`) to be printed.
        indent: Keyword-only argument specifying the number of spaces to indent
            the output if the features are JSON serializable. Defaults to None (no indentation).
        **kwargs: Keyword arguments representing named features (:class:`Feature`) to be printed.
            These can be referenced by their names in the `format_string`.
    """
    args = list(args)
    # find the format string in the arguments
    format_string = (
        args.pop(0)
        if len(args) > 0 and isinstance(args[0], str)
        else kwargs.pop("format")
        if "format" in kwargs
        else None
    )

    # build the default format
    if format_string is None:
        format_string = " ".join(["{}"] * len(args) + [f"{k}={{{k}}}" for k in kwargs.keys()])

    if len(args) > 0:
        raise NotImplementedError()

    # create and call the debug node
    PrintSample(format_string=format_string, indent=indent).call(**kwargs)
