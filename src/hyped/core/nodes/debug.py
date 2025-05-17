"""Module defining the base classes for debug nodes in a data flow graph.

This module introduces the :class:`BaseDebugNodeConfig` for configuring debug
nodes and the abstract :class:`BaseDebugNode` class, which serves as the foundation
for creating custom debug processors. Debug nodes are specialized components
used for inspecting and monitoring data as it flows through the graph, without
altering the data itself or producing outputs for further processing.
"""
import asyncio
from abc import ABC, abstractmethod
from collections import deque
from typing import TypeVar, overload

import pyarrow as pa

from ..abc import AbstractDataFlow
from ..features.features import Feature as _Feature
from ..typing import Feature
from ..utils import map_recursive
from .base import ProcessMode, RunContext, _extract_builder_from_args
from .processor import BaseDataProcessor, BaseDataProcessorConfig


class BaseDebugNodeConfig(BaseDataProcessorConfig):
    """Base configuration class for debug nodes."""


D = TypeVar("D", bound=BaseDebugNodeConfig)


class BaseDebugNode(BaseDataProcessor[D], ABC):
    """Base class for debug nodes in a data flow graph.

    This abstract class defines the structure for debug nodes. Debug nodes are
    specialized data processors that are primarily used for introspection and
    monitoring of data flow. Unlike standard data processors, they do not produce
    output features that are integrated back into the main data flow.

    Subclasses must implement the :func:`process` method to define their specific
    debugging actions, such as logging or data inspection.
    """

    @overload
    async def process(self, ctx: RunContext, *args: Feature, **kwargs: Feature) -> None:
        ...

    @abstractmethod
    def process(self, ctx: RunContext, *args: Feature, **kwargs: Feature) -> None:
        """Process function for debugging.

        This abstract method defines the processing logic for the debug node.
        Subclasses must implement this method to perform specific debugging
        actions on the input features. This method can be either synchronous
        or asynchronous. Importantly, debug nodes do not return any features
        that are integrated into the data flow.

        Args:
            ctx (RunContext): The context for the current process call.
            *args (Feature): Positional input features to be inspected.
            **kwargs (Feature): Keyword input features to be inspected.
        """

    def call(
        self,
        *args: AbstractDataFlow | Feature,
        **kwargs: AbstractDataFlow | Feature,
    ) -> Feature:
        """Call the debug node, adding it to the underlying data flow.

        This method adds the debug node to the data flow graph. It validates the
        node's signature against the provided arguments, ensuring the correct
        input features are connected.

        Args:
            *args (AbstractDataFlow | Feature): Positional input arguments, which
                can be existing data flow nodes or features.
            **kwargs (AbstractDataFlow | Feature): Keyword input arguments, which
                can be existing data flow nodes or features.

        Returns:
            _Feature: A placeholder feature representing the debug operation in
            the graph. This feature is typically not used for further computation.

        Raises:
            RuntimeError: If the data flow graph cannot be inferred from the
                arguments.
        """
        # extract the graph builder and prepare the input arguments
        builder, args, kwargs = _extract_builder_from_args(args, kwargs)

        # bind arguments to signature and unpack dynamic keyword arguments
        bound_args = self.signature.bind(*args, **kwargs).arguments
        for param in self.signature.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                bound_args.update(bound_args.pop(param.name, {}))
                break

        # extract reference instances from all features in the inputs
        inputs = map_recursive(lambda _, x: x.ref if isinstance(x, _Feature) else x, bound_args)
        # add the compute node to the graph
        builder.compute(self, inputs)

    async def run(self, ctx: RunContext, arrays: dict[str, pa.Array]) -> None:
        """Execute the main processing logic for the debug node.

        This method executes the debugging actions defined in the :func:`process`
        method. It receives the input data as a dictionary of PyArrow arrays.
        Since debug nodes do not produce output features for the data flow,
        this method returns :code:`None`.

        Args:
            ctx (RunContext): The execution context for the current run.
            arrays (dict[str, pa.Array]): A dictionary mapping input names to
                :code:`PyArrow` arrays, representing the data to be inspected
                or used for debugging purposes.
        """
        # get the process mode
        mode = ProcessMode.from_decorated_fn(self.process)

        # prepare inputs and apply process function to all inputs
        inputs = mode.prepare(ctx, **arrays)
        outputs = (self.process(c, **kw) for c, kw in inputs)

        # gather all outputs in case the process function is a coroutine
        if self._is_process_async:
            await asyncio.gather(*outputs)
        else:
            deque(outputs, maxlen=0)
