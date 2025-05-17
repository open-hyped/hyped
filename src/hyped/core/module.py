"""Defines the base class for creating data processing modules.

This module provides the :class:`DataFlowModule` class, which serves as an
abstract base class for defining modular data processing workflows.  It
simplifies the creation of complex data flows by encapsulating processing
logic and automatically handling data flow construction.
"""
import inspect
from abc import ABC, abstractmethod
from functools import partialmethod, update_wrapper
from typing import Any, Callable, TypeVar

from jinja2 import Template
from matplotlib import pyplot as plt

from hyped.common._generic import solve_typevar
from hyped.common.utils import tmp_setattr

from .features.dtypes import DType, MappingType
from .features.features import MappingFeature, build_feature_from_annotation
from .flow import DEFAULT_NODE_FORMAT, DataFlow, ExecutableDataFlow, plot_data_flow
from .graph import DataFlowGraph
from .nodes.base import _extract_builder_from_args
from .typing import Feature


def _call_wrapper(self, *args: Any, __unwrapped_call: Callable, **kwargs: Any) -> Any:
    """Wraps the :code:`DataFlowModule.call` method to provide data flow context.

    This wrapper extracts the data flow builder from the arguments,
    creates a :class:`~.flow.DataFlow` instance, and makes it available
    via the :code:`_flow` attribute within the :code:`call` method.

    Args:
        self (DataFlowModule): The instance of the :class:`DataFlowModule`.
        *args:  Positional arguments passed to the :code:`call` method.
        __unwrapped_call (Callable): The original :code:`call` method.
        **kwargs: Keyword arguments passed to the :code:`call` method.

    Returns:
        Any: The result of the original :code:`call` method.
    """
    builder, _, _ = _extract_builder_from_args(args, kwargs)
    flow = DataFlow._from_builder(builder)

    with tmp_setattr(self, "_flow", flow):
        return __unwrapped_call(self, *args, **kwargs)


class DataFlowModule(ABC):
    """Base class for defining reusable data processing modules.

    A `DataFlowModule` encapsulates a data processing workflow. Subclasses
    must implement the :meth:`call` method, which defines the core logic
    of the module.
    """

    def __init_subclass__(cls) -> None:
        """Handles subclass initialization.

        This method wraps the :code:`call` method of the subclass with
        :func:`call_wrapper` to provide data flow context.
        """
        cls._unwrapped_call = cls.call
        cls.call = update_wrapper(
            wrapper=partialmethod(_call_wrapper, __unwrapped_call=cls._unwrapped_call),
            wrapped=cls._unwrapped_call,
        )

        return super().__init_subclass__()

    def __init__(self, debug: bool = True) -> None:
        """Initializes a DataFlowModule.

        Args:
            debug (bool, optional): Whether to enable debug mode when
                building the data flow. Defaults to :code:`True`.
        """
        self.debug = debug
        self._flow: None | DataFlow = None

    def _build_flow(self, debug: bool = True) -> ExecutableDataFlow:
        """Builds the executable data flow for this module.

        This method inspects the signature of the :meth:`call` method to
        determine the input and output features, constructs a :class:`~.flow.DataFlow`
        instance, and then builds an :class:`~.flow.ExecutableDataFlow`.

        Args:
            debug (bool, optional): Whether to build the flow in debug mode.
                Defaults to :code:`True`.

        Returns:
            ExecutableDataFlow: The built executable data flow.
        """
        typevar_mapping: dict[TypeVar, DType] = {}
        if hasattr(self, "__orig_class__") and hasattr(self, "__parameters__"):
            for v in self.__parameters__:
                annotation = solve_typevar(self.__orig_class__, v)
                typevar_mapping[v] = build_feature_from_annotation(annotation).dtype

        sig = inspect.signature(self._unwrapped_call)
        # build source features from signature
        src_dtype = MappingType.construct(
            {
                k: build_feature_from_annotation(
                    p.annotation, typevar_mapping=typevar_mapping
                ).dtype
                for k, p in sig.parameters.items()
            }
        )

        # create the data flow and call the module on the source features
        flow = DataFlow(src_dtype.hf_feature)
        output = self.call(**flow.source)

        # collect the output in a mapping
        if isinstance(output, dict):
            output = flow.collect(output)
        elif not isinstance(output, MappingFeature):
            output = flow.collect({"output": output})

        # build the data flow
        return flow.build(collect=output, debug=debug)

    @property
    def flow(self) -> DataFlow | ExecutableDataFlow:
        """The executable data flow for this module.

        This property lazily builds and returns the data flow instance.

        -   Within the :meth:`call` method, this returns a mutable
            :class:`~.flow.DataFlow` instance, giving access to functionality
            like `~.flow.DataFlow.collect`.

        -   Outside of :meth:`call`, this returns the build
            :class:`~.flow.ExecutableDataFlow`, which represents the
            fully constructed and optimized data flow, ready for execution.

        Returns:
            DataFlow | ExecutableDataFlow: The data flow.
        """
        if getattr(self, "_flow", None) is not None:
            return self._flow

        debug = getattr(self, "debug", True)
        return self._build_flow(debug=debug)

    def plot(
        self,
        node_format: str | Template = DEFAULT_NODE_FORMAT,
        with_edge_labels: bool = True,
        edge_font_size: int = 6,
        node_font_size: int = 6,
        node_size: int = 5_000,
        arrowsize: int = 25,
        color_map: dict[DataFlowGraph.NodeType, str] = {},
        legend: bool = True,
        legend_fontsize: int = 6,
        ax: None | plt.Axes = None,
    ) -> plt.Axes:
        """Plot a data flow graph.

        Args:
            flow (DataFlow): The data flow to plot.
            node_format (str | Template): The jinja template used to generate node labels.
            with_edge_labels (bool): Whether to include labels on the edges. Defaults to True.
            edge_font_size (int): The font size for edge labels. Defaults to 6.
            node_font_size (int): The font size for node labels. Defaults to 6.
            node_size (int): The size of the nodes. Defaults to 5_000.
            arrowsize (int): The size of the arrows on the edges. Defaults to 25.
            color_map (dict[None | type, str]): indicate custom color scheme based on the processor
                type. `None` refers to the source node.
            legend (bool): Whether to add a legend of the node types to the axes. Defaults to True.
            legend_fontsize (int): The font size for the legend. Defaults to 6.
            ax (Optional[plt.Axes]): Matplotlib axes object to draw the plot on. Defaults to None.

        Returns:
            plt.Axes: The Matplotlib axes object with the plot.
        """
        ax = plot_data_flow(
            flow=self.flow,
            node_format=node_format,
            with_edge_labels=with_edge_labels,
            edge_font_size=edge_font_size,
            node_font_size=node_font_size,
            node_size=node_size,
            arrowsize=arrowsize,
            color_map=color_map,
            legend=legend,
            legend_fontsize=legend_fontsize,
            ax=ax,
        )
        ax.set(title=type(self).__name__)

        return ax

    @abstractmethod
    def call(self, *args: Feature, **kwargs: Feature) -> Feature:
        """Defines the core logic of the data processing module.

        Subclasses must implement this method to specify how input features
        are processed to produce output features. The signature of this method
        determines the input features of the data flow, and the return type
        determines the output feature.

        Within this method, the :py:attr:`flow` property provides access to
        a :class:`~.flow.DataFlow` instance, giving access to functionality
        like `~.flow.DataFlow.collect`.

        Args:
            *args (Feature): Positional input features.
            **kwargs (Feature): Keyword input features.

        Returns:
            Feature: The output feature produced by the module.
        """
