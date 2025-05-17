"""Defines the base class for creating data processing modules.

This module provides the :class:`DataFlowModule` class, which serves as an
abstract base class for defining modular data processing workflows.  It
simplifies the creation of complex data flows by encapsulating processing
logic and automatically handling data flow construction.
"""
import inspect
from abc import ABC, abstractmethod
from typing import TypeVar

from jinja2 import Template
from matplotlib import pyplot as plt

from hyped.common._generic import solve_typevar

from .features.dtypes import DType, MappingType
from .features.features import MappingFeature, build_feature_from_annotation
from .flow import DEFAULT_NODE_FORMAT, DataFlow, ExecutableDataFlow, plot_data_flow
from .graph import DataFlowGraph
from .typing import Feature


class DataFlowModule(ABC):
    """Base class for defining reusable data processing modules.

    A `DataFlowModule` encapsulates a data processing workflow. Subclasses
    must implement the :meth:`call` method, which defines the core logic
    of the module.
    """

    def __init__(self, debug: bool = True) -> None:
        """Initializes a DataFlowModule.

        Args:
            debug (bool, optional): Whether to enable debug mode when
                building the data flow. Defaults to :code:`True`.
        """
        self.debug = debug

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

        sig = inspect.signature(self.call)
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
        if not isinstance(output, MappingFeature):
            output = flow.collect({"output": output})

        # build the data flow
        return flow.build(collect=output, debug=debug)

    @property
    def flow(self) -> ExecutableDataFlow:
        """The executable data flow for this module.

        This property lazily builds and returns the :class:`~.flow.ExecutableDataFlow`
        instance associated with this module.

        Returns:
            ExecutableDataFlow: The executable data flow.
        """
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

        Args:
            *args (Feature): Positional input features.
            **kwargs (Feature): Keyword input features.

        Returns:
            Feature: The output feature produced by the module.
        """
