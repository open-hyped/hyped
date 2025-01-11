"""Defines high-level interfaces for data processing workflows.

This module provides the :class:`DataFlow` class, which allows users to define and
execute complex data processing workflows. The workflows are represented as
directed acyclic graphs (DAGs) of data processors.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import pickle
import re
from dataclasses import replace
from itertools import groupby
from typing import Any, Generic, Literal, Mapping, TypeVar, get_args, overload

import datasets
import matplotlib.pyplot as plt
import nest_asyncio
import networkx as nx
import numpy as np
import pyarrow as pa
from matplotlib import colormaps

from hyped.common._pydantic import TypeAdapterWithArbitraryTypesAllowed
from hyped.common._worker import get_worker_info

from .abstract import AbstractDataFlow
from .executor import DataFlowExecutor, LazyDataFlowExecutor
from .features.dtypes import (
    DType,
    MappingType,
    build_dtype_from_hf_feature,
    build_dtype_from_python_object,
    cast_dtype,
    is_dtype_subset,
)
from .features.features import (
    BoolFeature,
    Feature,
    Float64Feature,
    Int64Feature,
    MappingFeature,
    SequenceFeature,
    StringFeature,
    build_feature_from_annotation,
    build_feature_from_reference,
)
from .features.reference import ConcreteReference, ForwardReference
from .features.session import ValidationSession
from .graph import DataFlowGraph
from .nodes.aggregator import DataAggregationManager
from .nodes.base import RunContext
from .optim import DataFlowGraphOptimizer
from .typing import IndexList, NodeId, Rank, cast
from .utils import NestedType, build_annotation_from_dtype, map_recursive, validate_hf_feature

logger = logging.getLogger(__name__)

# patch asyncio if running in an async environment, such as jupyter notebook
# this fixes #26
try:
    nest_asyncio._patch_asyncio()
    loop = asyncio.get_event_loop()
    nest_asyncio.apply(loop)
except ValueError:  # pragma: not covered
    # TODO: log warning
    pass


Dataset = TypeVar("Dataset", datasets.Dataset, datasets.DatasetDict)
ItDataset = TypeVar("ItDataset", datasets.IterableDataset, datasets.IterableDatasetDict)

T = TypeVar("T", bound=MappingFeature)


def plot_data_flow(
    flow: DataFlow,
    with_edge_labels: bool = True,
    edge_label_format: str = "{name}",
    src_node_label: str = "[ROOT]",
    edge_font_size: int = 6,
    node_font_size: int = 6,
    node_size: int = 5_000,
    arrowsize: int = 25,
    color_map: dict[
        Literal[
            DataFlowGraph.NodeType.SOURCE,
            DataFlowGraph.NodeType.DATA_PROCESSOR,
            DataFlowGraph.NodeType.DATA_AUGMENTOR,
            DataFlowGraph.NodeType.DATA_AGGREGATOR,
        ],
        str,
    ] = {},
    ax: None | plt.Axes = None,
) -> plt.Axes:
    """Plot a data flow graph.

    Args:
        flow (DataFlow): The data flow to plot.
        with_edge_labels (bool): Whether to include labels on the edges. Defaults to True.
        edge_label_format (str): Format string for edge labels. Defaults to "{name}".
        src_node_label (str): Label for the source node. Defaults to "[ROOT]".
        edge_font_size (int): The font size for edge labels. Defaults to 6.
        node_font_size (int): The font size for node labels. Defaults to 6.
        node_size (int): The size of the nodes. Defaults to 5_000.
        arrowsize (int): The size of the arrows on the edges. Defaults to 25.
        color_map (dict[None | type, str]): indicate custom color scheme based on the processor
            type. `None` refers to the source node.
        ax (Optional[plt.Axes]): Matplotlib axes object to draw the plot on. Defaults to None.

    Returns:
        plt.Axes: The Matplotlib axes object with the plot.
    """
    # create a plot axes
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(flow._graph.depth * 2, flow._graph.width * 2.5))

    # compute the node positions
    pos = nx.multipartite_layout(flow._graph, subset_key=DataFlowGraph.NodeAttribute.DEPTH)

    # build color map
    cmap = colormaps.get_cmap("Pastel1")
    default_color_map = {
        DataFlowGraph.NodeType.SOURCE: cmap.colors[0],
        DataFlowGraph.NodeType.CONST: cmap.colors[1],
        DataFlowGraph.NodeType.CAST: cmap.colors[5],
        DataFlowGraph.NodeType.COLLECT: cmap.colors[6],
        DataFlowGraph.NodeType.DATA_PROCESSOR: cmap.colors[2],
        DataFlowGraph.NodeType.DATA_AUGMENTOR: cmap.colors[3],
        DataFlowGraph.NodeType.DATA_AGGREGATOR: cmap.colors[4],
    }
    color_map = default_color_map | color_map

    # apply color map
    node_colors = [
        color_map[data[DataFlowGraph.NodeAttribute.NODE_TYPE]]
        for _, data in flow._graph.nodes(data=True)
    ]

    # plot the raw graph
    nx.draw(
        flow._graph,
        pos,
        with_labels=False,
        node_size=node_size,
        node_color=node_colors,
        arrowsize=arrowsize,
        ax=ax,
    )

    # limit the maximum number of character in a single line in nodes
    max_line_length = node_size // (node_font_size * 65)

    node_labels = {}
    # build node labels
    for node, data in flow._graph.nodes(data=True):
        if node == flow._graph.src_node_id:
            # add root node label
            node_labels[node] = src_node_label

        else:
            # build the node label of this node
            obj = data[DataFlowGraph.NodeAttribute.NODE_OBJ]
            if data[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.CAST:
                node_label = f"Cast[{str(data[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE])}]"
            else:
                node_label = str(obj)
            # split string into words
            words = re.split(r"(?<=[a-z])(?=[A-Z])", node_label)
            # group words such that each group has a limited number of
            # characers
            lengths = np.cumsum(list(map(len, words))) // max_line_length
            groups = groupby(range(len(words)), key=lengths.__getitem__)
            # join groups with newlines inbetween
            node_labels[node] = "\n".join(
                ["".join([words[i] for i in group]) for _, group in groups]
            )

    # add node labels
    nx.draw_networkx_labels(
        flow._graph,
        pos,
        labels=node_labels,
        font_size=node_font_size,
        ax=ax,
    )

    # add edge labels
    if with_edge_labels:
        # group multi-edges by their source and target nodes
        grouped_edges = groupby(
            sorted(flow._graph.edges(keys=True), key=lambda e: (e[0], e[1])),
            key=lambda e: (e[0], e[1]),
        )

        edge_labels = {}
        # build edge labels
        for edge, group in grouped_edges:
            edge_labels[edge] = "\n".join(
                [edge_label_format.format(name=key) for _, _, key in group]
            )

        # draw the edge labels
        nx.draw_networkx_edge_labels(
            flow._graph,
            pos,
            edge_labels=edge_labels,
            font_size=edge_font_size,
            ax=ax,
        )

    return ax


class DataFlow(AbstractDataFlow, Generic[T]):
    """Data Flow.

    The :class:`DataFlow` class allows users to create and manage directed acyclic
    graphs (DAGs) of data processors, facilitating complex data transformations and
    processing pipelines. Users can easily define source features, build sub-flows
    for specific outputs, and apply these workflows to batches of data or entire
    HuggingFace datasets.

    This class integrates various components such as the data flow graph and the
    executor to provide a seamless experience for processing data. It handles the
    internal state management, execution scheduling, and data flow dependencies to
    ensure efficient and accurate data processing.
    """

    def __init__(self, features: None | datasets.Features = None) -> None:
        """Initialize the DataFlow.

        Args:
            features (datasets.Features): The features of the source node.
        """
        self._graph: DataFlowGraph = DataFlowGraph()
        # save source features
        self._hf_source_features = features
        self._source_feature: None | T = None

    @property
    def _is_initialized(self) -> None:
        """Check if the data flow graph has been initialized.

        Returns:
            bool: True if the source node has been added to the graph, False otherwise.
        """
        return self._graph.src_node_id is not None

    @property
    def _source_annotation(self) -> Any:
        """Retrieves the source features annotation if present."""
        return None if not hasattr(self, "__orig_class__") else get_args(self.__orig_class__)[0]

    def _initialize(self) -> None:
        """Initialize the data flow graph by defining the source node.

        The method infers the source data type (:code:`src_dtype`) based on the
        provided HuggingFace features or the type annotation of the source
        node. It ensures that at least one of these is specified, and validates
        the consistency between the features and type annotation when both are provided.

        Raises:
            AssertionError: If the data flow graph is already initialized.
            RuntimeError: If neither HuggingFace features nor type annotation is provided.
        """
        # make sure the flow is not initialized yet
        assert not self._is_initialized, "DataFlow has already been initialized."

        src_type_annotation = self._source_annotation
        # infer the source data type from the hf
        # features and/or the source type annotation
        src_dtype: DType

        with ValidationSession() as session:
            if (src_type_annotation is not None) and (self._hf_source_features is not None):
                instance = validate_hf_feature(
                    self._hf_source_features, src_type_annotation, session=session
                )

            elif src_type_annotation is not None:
                # infer the source dtype from the type annotation
                instance = build_feature_from_annotation(src_type_annotation, session=session)

            elif self._hf_source_features is not None:
                # build the source dtype from the huggingface features
                src_dtype = build_dtype_from_hf_feature(self._hf_source_features)
                instance = MappingFeature(ForwardReference(dtype=src_dtype))

            else:
                # no input specified, at least argument or type hint is required
                raise RuntimeError(
                    "Initialization failed: At least one of 'features' or type annotation "
                    "must be provided to infer the source data type."
                )

            # add the source node to the graph with the node id
            src_ref = self._graph.add_source_node(instance.dtype)
            self._source_feature = replace(instance, ref=src_ref)

    @property
    def depth(self) -> int:
        """Computes the total depth of the data flow graph.

        The depth is defined as the maximum level of any node in the graph, where the root
        node has a depth of 0. This property calculates the depth by finding the maximum
        depth attribute among all nodes in the graph.

        Returns:
            int: The total depth of the graph.
        """
        return self._graph.depth  # pragma: not covered

    @property
    def width(self) -> int:
        """Computes the maximum width of the data flow graph.

        The width is defined as the maximum number of nodes present at any single depth level
        in the graph. This property calculates the width by grouping nodes by their depth and
        finding the largest group.

        Returns:
            int: The maximum width of the graph.
        """
        return self._graph.width  # pragma: not covered

    @property
    def source(self) -> T:
        """Get the source features.

        Returns:
            T: The reference to the source features.
        """
        if not self._is_initialized:
            self._initialize()

        return self._source_feature

    @overload
    def const(self, value: int) -> Int64Feature:
        ...

    @overload
    def const(self, value: float) -> Float64Feature:
        ...

    @overload
    def const(self, value: bool) -> BoolFeature:
        ...

    @overload
    def const(self, value: str) -> StringFeature:
        ...

    @overload
    def const(self, value: dict[str, Any]) -> MappingFeature:
        ...

    @overload
    def const(self, value: list[Any] | tuple[Any]) -> SequenceFeature:
        ...

    U = TypeVar("U", bound=Feature)

    @overload
    def const(self, value: Any, feature_type: type[U]) -> U:
        ...

    def const(self, value: Any, feature_type: None | type[Feature] = None) -> Feature:
        """Add a constant value as a node to the data flow.

        This method allows the user to add a constant value as a node in the data flow graph.
        The constant can either be associated with a specific feature type or have its
        data type inferred from the provided value.

        Args:
            value (Any): The constant value to add to the data flow graph.
            feature_type (None | type[Feature]): The explicit feature type for the value.
                If provided, the data type will be validated and inferred from this type.
                If not provided, the data type is inferred from the :code:`value`.

        Returns:
            Feature: The feature representation of the constant node added to the graph.

        Raises:
            TypeError: If the provided :code:`feature_type` is invalid or if the value cannot
                be validated against the :code:`feature_type`.
        """
        if feature_type is not None:
            # create a dummy feature to infer the data type
            # from the given feature type
            adapter = TypeAdapterWithArbitraryTypesAllowed(feature_type)
            dtype = adapter.validate_python(ForwardReference()).dtype
        else:
            # build the data type matching the object in case no data type was provided
            dtype = build_dtype_from_python_object(value)
        # add the constant node to the graph
        ref = self._graph.add_const_node(value, dtype)
        return build_feature_from_reference(ref)

    @overload
    def collect(self, collect: dict[NestedType[Any]]) -> MappingFeature:
        ...

    @overload
    def collect(self, collect: list[NestedType[Any]] | tuple[NestedType[Any]]) -> SequenceFeature:
        ...

    @overload
    def collect(self, collect: Any) -> Feature:
        ...

    def collect(self, collect: NestedType[Any]) -> MappingFeature | SequenceFeature | Feature:
        """Add a collect node to the data flow.

        The :func:`collect` method processes a nested structure (e.g., dicts, lists, tuples) of
        constants and features and adds a corresponding :code:`collect` node to the data flow
        graph. Constants are converted to constant nodes, while features are directly incorporated.

        Args:
            collect (NestedType[Any]): A nested structure of constants, `Feature` objects,
                and `Reference` instances. The structure may include dictionaries, lists,
                and tuples, where constants are automatically added as constant nodes.

        Returns:
            Mapping | Sequence | Feature: A feature or nested structure of features representing
                the collected data.

        Raises:
            NotImplementedError: If an empty sequence (list or tuple) is encountered.
            TypeError: If an unsupported type is encountered in the nested structure.
        """
        if isinstance(collect, Feature):
            # nothing to collect
            return collect

        # prepare the collect structure by extracting the references from the features
        # and add the collect node and all constants to the graph
        collect = map_recursive(lambda _, x: x.ref if isinstance(x, Feature) else x, collect)
        ref = self._graph.add_collect_node_with_constants(collect)
        # return the collect feature
        return build_feature_from_reference(ref)

    def build(
        self,
        collect: Feature | dict[str, Any],
        aggregate: None | Feature | dict[str, Any] = None,
        *,
        aggregation_manager: DataAggregationManager | None = None,
    ) -> ExecutableDataFlow:
        """Build an executable data flow for computing and collecting features.

        This method constructs a read-only, executable representation of the data flow
        for a specified :code:`collect` feature. Optionally, an :code:`aggregate` feature
        can also be included for dataset-wide computations. The resulting data flow is
        optimized for execution.

        Args:
            collect (Feature | dict[str, Any]): The feature to be computed and collected.
            aggregate (None | Feature): An optional feature for computing aggregated values
                across the dataset.
            *
            aggregation_manager (DataAggregationManager | None): The data aggregation manager
                instance.

        Returns:
            ExecutableDataFlow[T]: An executable data flow that encapsulates the graph,
            collect feature, and optional aggregate feature.

        Raises:
            RuntimeError: If the :code:`collect` feature does not belong to the
                current data flow graph.
            RuntimeError: If the :code:`aggregate` feature does not belong to the
                current data flow graph.
        """
        # collect the output features
        collect = self.collect(collect)
        aggregate = self.collect(aggregate) if aggregate is not None else None

        # make sure output features belong to this flow
        if collect.ref._graph is not self._graph:
            raise RuntimeError("The collect feature does not belong to this flow.")
        if (aggregate is not None) and (aggregate.ref._graph is not self._graph):
            raise RuntimeError("The aggregate feature does not belong to this flow.")

        # make sure flow is initialized
        if not self._is_initialized:
            self._initialize()

        # create a read-only view of the data flow graph
        graph = nx.restricted_view(self._graph, [], [])
        # build the reference instances to the graph view
        collect = replace(collect.ref, _graph=graph)
        aggregate = None if aggregate is None else replace(aggregate.ref, _graph=graph)
        # build executable data flow
        return ExecutableDataFlow(
            self._source_annotation, graph, collect, aggregate, aggregation_manager
        )

    @overload
    def apply(
        self,
        ds: Dataset,
        collect: Feature | dict[str, Any],
        *,
        batch_size: int = 1000,
        drop_last_batch: bool = False,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        writer_batch_size: int = 1000,
        num_proc: None | int = None,
        desc: None | str = None,
    ) -> Dataset:
        ...

    @overload
    def apply(
        self,
        ds: Dataset,
        collect: Feature | dict[str, Any],
        aggregate: Feature | dict[str, Any],
        *,
        batch_size: int = 1000,
        drop_last_batch: bool = False,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        writer_batch_size: int = 1000,
        num_proc: None | int = None,
        desc: None | str = None,
    ) -> tuple[Dataset, dict[str, Any]]:
        ...

    @overload
    def apply(
        self,
        ds: ItDataset,
        collect: Feature | dict[str, Any],
        *,
        batch_size: int = 1000,
        drop_last_batch: bool = False,
    ) -> ItDataset:
        ...

    @overload
    def apply(
        self,
        ds: ItDataset,
        collect: Feature | dict[str, Any],
        aggregate: Feature | dict[str, Any],
        *,
        batch_size: int = 1000,
        drop_last_batch: bool = False,
    ) -> tuple[ItDataset, dict[str, Any]]:
        ...

    def apply(
        self,
        ds: Dataset | ItDataset,
        collect: Feature,
        aggregate: None | Feature = None,
        **kwargs: Any,
    ) -> Dataset | ItDataset | tuple[Dataset, dict[str, Any]] | tuple[ItDataset, dict[str, Any]]:
        """Apply the data flow graph to a dataset.

        This method processes a given dataset or iterable dataset using the data flow graph,
        executing transformations based on the :code:`collect` and optional :code:`aggregate`
        features. The behavior of the processing depends on whether the dataset is an in-memory
        or streamed dataset (i.e. :class:`datasets.Dataset` or :class:`datasets.IterableDataset`)).

        Parameters:
            ds (Dataset | IterableDataset): The dataset to which the data flow graph will be
                applied.
            collect (Feature | dict[str, Any]): The collect feature, which defines the primary
                transformations applied to the dataset.
            aggregate (None | Feature | dict[str, Any]): An optional aggregate feature, which
                applies additional aggregate-level transformations. If not provided, aggregation
                is skipped.
            batch_size (int): The number of samples to process in a batch. Defaults to 1000.
            drop_last_batch (bool): Whether to drop the last batch if it is smaller than
                the specified batch size. Defaults to :code:`False`.
            keep_in_memory (bool): If :code:`True`, the resulting dataset is kept in memory.
                Defaults to :code:`False`. Only applies for :class:`datasets.Dataset`.
            load_from_cache_file (bool): Whether to load the resulting dataset from cache files
                when possible. Defaults to :code:`True`. Only applies for
                :class:`datasets.Dataset`.
            writer_batch_size (int): Batch size for writing results to cache files. Defaults to
                1000. Only applies for :class:`datasets.Dataset`.
            num_proc (None | int): The number of processes to use for parallel processing. Defaults
                to :code:`None`, which disables multiprocessing. Only applies for
                :class:`datasets.Dataset`.
            desc (Optional[str], optional, only for Dataset): A description for the progress bar
                displayed during processing. Defaults to :code:`None`. Only applies for
                :class:`datasets.Dataset`.

        Returns:
            Dataset | ItDataset | tuple[Dataset, dict[str, Any]] | tuple[ItDataset, dict[str, Any]]:
            The transformed dataset matching the input dataset type and a dictionary containing
            the aggregation results when aggregation is applied, i.e. the :code:`aggregate` input
            is specified.
        """
        # build and apply the data flow to the dataset
        flow = self.build(collect, aggregate)
        ds = flow.apply(ds, **kwargs)
        # return the transformed dataset and optionally the aggregates
        return ds if aggregate is None else (ds, flow.aggregates)

    @classmethod
    def deserialize(cls, data: str) -> ExecutableDataFlow:
        """Deserializes a JSON string into an :class:`ExecutableDataFlow` instance.

        This method parses a JSON string into a dictionary, validates its structure,
        and uses the data to reconstruct the executable data flow, including references
        to the collection and aggregation nodes.

        Args:
            data (str): The JSON string representing the serialized executable data flow.

        Returns:
            ExecutableDataFlow: The deserialized executable data flow.

        Raises:
            ValueError: If the input JSON string does not contain the required keys
                ("graph", "collect", and "aggregate").
        """
        return ExecutableDataFlow.deserialize(data)


class ExecutableDataFlow(AbstractDataFlow):
    """Executable data flow.

    This class validates and optimizes the input data flow graph and provides methods
    to execute the flow over a dataset. It manages source, collect, and aggregate
    features, optimizing and partitioning the graph into instance and aggregate graphs.

    This class is not meant to be instantiated directly. Use the :func:`DataFlow.build`
    function to construct an instance, ensuring proper setup and validation of the data
    flow.
    """

    def __init__(
        self,
        source_annotation: Any | None,
        graph: DataFlowGraph,
        collect: ConcreteReference,
        aggregate: ConcreteReference | None,
        aggregation_manager: DataAggregationManager | None,
    ) -> None:
        """Initialize a :class:`ExecutableDataFlow`.

        Args:
            source_annotation (Any): The source annotation.
            graph (DataFlowGraph): The data flow graph describing the pipeline.
            collect (ConcreteReference): A reference to the "collect" node in the graph.
            aggregate (ConcreteReference | None): A reference to the "aggregate" node,
                if applicable.
            aggregation_manager (DataAggregationManager | None): The data aggregation
                manager instance to use in case the flow contains aggregator nodes.

        Raises:
            RuntimeError: If the collect node is not part of the graph.
            RuntimeError: If the collect node is part of the :code:`AGGREGATED` partition.
            RuntimeError: If the collect feature is not a mapping.
            RuntimeError: If the aggregate node is not part of the graph.
            RuntimeError: If the aggregate node is not part of the :code:`AGGREGATED` partition.
            RuntimeError: If the aggregate feature is not a mapping.
        """
        self._source_annotation = source_annotation
        # make sure the collect feature belongs to the graph
        if (collect._graph is not graph) or (collect._node_id not in graph.nodes):
            raise RuntimeError(
                "The specified collect node does not belong to the provided data flow graph."
            )

        # make sure collect doesn't belong to the aggregate partition
        if graph.get_node_output_partition(collect._node_id) == DataFlowGraph.Partition.AGGREGATED:
            raise RuntimeError("The collect node cannot belong to the aggregated partition.")

        if aggregate is not None:
            if (aggregate._graph is not graph) or (aggregate._node_id not in graph.nodes):
                raise RuntimeError(
                    "The specified aggregate node does not belong to the provided data flow graph."
                )

            # make sure aggregate belongs to the aggregate partition
            if (
                graph.get_node_output_partition(aggregate._node_id)
                != DataFlowGraph.Partition.AGGREGATED
            ):
                raise RuntimeError("The aggregate node must belong to the aggregated partition.")

            # make sure the collect feature is a mapping
            if not isinstance(build_feature_from_reference(aggregate), MappingFeature):
                raise RuntimeError("The aggregate feature must be a mapping.")

        # optimize data flow
        graph = DataFlowGraphOptimizer().optimize(
            graph,
            leaf_nodes=(
                {collect._node_id} if aggregate is None else {collect._node_id, aggregate._node_id}
            ),
        )

        # create read only view on optimized graph
        self._graph: DataFlowGraph = nx.restricted_view(graph, [], [])
        # update the collect and aggregate references to the new graph
        collect = replace(collect, _graph=self._graph)
        aggregate = None if aggregate is None else replace(aggregate, _graph=self._graph)
        # get the source feature instance from the graph
        ref = ConcreteReference(_node_id=graph.src_node_id, _graph=self._graph)
        self._source_feature: MappingFeature = build_feature_from_reference(ref)
        self._collect_feature: MappingFeature = build_feature_from_reference(
            replace(collect, _graph=self._graph)
        )

        # make sure the source feature is a mapping
        if not isinstance(self._source_feature, MappingFeature):
            raise RuntimeError("The source feature must be a mapping.")

        # make sure the collect feature is a mapping
        if not isinstance(self._collect_feature, MappingFeature):
            raise RuntimeError("The collect feature must be a mapping.")

        # create read-only view on the instance partition of the graph
        self._instance_graph = graph.drop_partition(DataFlowGraph.Partition.AGGREGATED)
        self._aggregates_graph: None | DataFlowGraph = None

        self._aggregation_manager: None | DataAggregationManager = None
        self._aggregates_executor: None | DataFlowExecutor = None

        if aggregate is not None:
            # get all aggregator nodes in the instance graph
            nodes = self._instance_graph.nodes(data=DataFlowGraph.NodeAttribute.NODE_TYPE)
            nodes = [
                node
                for node, node_type in nodes
                if node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR
            ]
            # there must be at least one aggregator node in the graph
            assert len(nodes) > 0

            # build the data aggregation manager instance and the aggregates graph,
            # which implements the operations performed on aggregated values
            self._aggregation_manager = (
                self._build_aggregation_manager(nodes)
                if aggregation_manager is None
                else self._check_aggregation_manager(nodes, aggregation_manager)
            )
            self._aggregates_graph = self._build_aggregates_graph(aggregate, set(nodes))
            # initialize the aggregates executor from the graph and manager
            self._aggregates_executor = LazyDataFlowExecutor(
                self._aggregates_graph,
                replace(aggregate, _graph=self._aggregates_graph),
                self._aggregation_manager.values_proxy,
            )

        # create the executors
        self._instance_executor = DataFlowExecutor(
            self._instance_graph, collect, aggregation_manager=self._aggregation_manager
        )

    @property
    def depth(self) -> int:
        """Computes the total depth of the data flow graph.

        The depth is defined as the maximum level of any node in the graph, where the root
        node has a depth of 0. This property calculates the depth by finding the maximum
        depth attribute among all nodes in the graph.

        Returns:
            int: The total depth of the graph.
        """
        return self._graph.depth  # pragma: not covered

    @property
    def width(self) -> int:
        """Computes the maximum width of the data flow graph.

        The width is defined as the maximum number of nodes present at any single depth level
        in the graph. This property calculates the width by grouping nodes by their depth and
        finding the largest group.

        Returns:
            int: The maximum width of the graph.
        """
        return self._graph.width  # pragma: not covered

    @property
    def collect_feature(self) -> MappingFeature:
        """The collect feature of the executable data flow.

        Returns:
            MappingFeature: The output feature that is collected by the data flow when executed.
        """
        return self._collect_feature

    @property
    def aggregates_feature(self) -> None | MappingFeature:
        """The aggregates feature of the executable data flow.

        Returns:
            None | MappingFeature: The aggregates feature that is computed when the data flow is
                executed. :code:`None` in case no aggregates were specified when building the flow.
        """
        return (
            None
            if self._aggregates_executor is None
            else build_feature_from_reference(self._aggregates_executor.collect)
        )

    @property
    def aggregates(self) -> Mapping[str, Any]:  # pragma: not covered
        """Read-only view of the aggregated values computed during execution.

        Returns:
            MappingProxyType[str, Any]: A read-only mapping of aggregated values.
        """
        return self._aggregates_executor  # pragma: not covered

    def _build_aggregates_graph(
        self, aggregate: ConcreteReference, aggregator_nodes: set[NodeId]
    ) -> DataFlowGraph:
        """Build the aggregates graph.

        Constructs a subgraph modeling all operations performed on aggregates,
        up to the aggregator nodes. The resulting graph excludes the aggregator
        nodes themselves, as these are included separately in the instance graph.

        The source node provides all the aggregated features produced by the
        aggregator nodes. The values to these features are provided by the
        aggregation manager.

        Args:
            aggregate (ConcreteReference): The aggregate reference for which the dependency graph
                will be built.
            aggregator_nodes (set[NodeId]): A set of node IDs representing the aggregator nodes
                that act as stopping points in the dependency graph.

        Returns:
            DataFlowGraph: A new data flow graph representing the dependencies and operations
            leading to the aggregate, excluding the aggregator nodes.
        """
        # build the dependency graph of the aggregate up to the aggregator nodes
        # note that this also includes constants
        g = self._graph.dependency_graph({aggregate._node_id}, stop_nodes=aggregator_nodes)
        # remove the aggregator nodes themselves as these are included in the
        # instance graph
        g = nx.restricted_view(g, aggregator_nodes, [])

        h = DataFlowGraph()
        # add a source node
        source_type = MappingType.construct(
            {
                str(node): self._graph.nodes[node][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
                for node in aggregator_nodes
            }
        )
        source_ref = h.add_source_node(source_type)
        source = build_feature_from_reference(source_ref)

        # rebuild the aggregates graph
        for node_id in nx.topological_sort(g):
            # collect all the inputs to the node
            edges = self._graph.in_edges(node_id, keys=True)
            inputs = {
                key: source[u].ref if u in aggregator_nodes else ConcreteReference(u, h)
                for u, _, key in edges
            }
            # add the node to the graph
            data = self._graph.nodes[node_id]
            h.add_node(
                node_obj=data[DataFlowGraph.NodeAttribute.NODE_OBJ],
                node_type=data[DataFlowGraph.NodeAttribute.NODE_TYPE],
                inputs=inputs,
                output_type=data[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                node_id=node_id,
            )

        return h

    def _check_aggregation_manager(
        self, nodes: list[NodeId], manager: DataAggregationManager
    ) -> DataAggregationManager:
        """Checks the aggregation manager.

        Checks whether all aggregators are registered in the aggregation manager and
        the data types of the corresponding values match expectation.

        Args:
            nodes (list[NodeId]): A list of node IDs corresponding to the aggregator nodes
                for which the data aggregation manager is to be constructed.
            manager (DataAggregationManager): The aggregation manager to check.

        Returns:
            DataAggregationManager: The checked aggregation manager.
        """
        assert set(manager.values_proxy.keys()) == set(nodes)
        assert all(
            buf.type
            == self._graph.nodes[node_id][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE].arrow_type
            for node_id, buf in manager.values_proxy.items()
        )
        return manager

    def _build_aggregation_manager(self, nodes: list[NodeId]) -> DataAggregationManager:
        """Build the aggregation manager.

        Builds a data aggregation manager for the specified aggregator nodes.
        The manager contains the aggregator nodes and their respective runtime
        contexts, including input and output feature types.

        Args:
            nodes (list[NodeId]): A list of node IDs corresponding to the aggregator nodes
                for which the data aggregation manager is to be constructed.

        Returns:
            DataAggregationManager: An instance managing the aggregators and their
            associated runtime contexts, which includes node-specific metadata such as
            input/output feature types, rank, and index.
        """
        # build the run contexts for the aggregator nodes
        aggregators = [
            self._graph.nodes[node][DataFlowGraph.NodeAttribute.NODE_OBJ] for node in nodes
        ]
        contexts = [
            RunContext(
                node_id=node,
                index=[],
                rank=0,
                input_type=self._graph.nodes[node][DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE],
                output_type=self._graph.nodes[node][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
            )
            for node in nodes
        ]

        return DataAggregationManager(aggregators, contexts)

    @overload
    def apply(
        self,
        ds: Dataset,
        *,
        batch_size: int = 1000,
        drop_last_batch: bool = False,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        writer_batch_size: int = 1000,
        num_proc: None | int = None,
        desc: None | str = None,
    ) -> Dataset:
        ...

    @overload
    def apply(
        self, ds: ItDataset, *, batch_size: int = 1000, drop_last_batch: bool = False
    ) -> ItDataset:
        ...

    def apply(
        self,
        ds: Dataset | ItDataset,
        *,
        batch_size: int = 1000,
        drop_last_batch: bool = False,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        writer_batch_size: int = 1000,
        num_proc: None | int = None,
        desc: None | str = None,
    ) -> Dataset | ItDataset:
        """Apply the data flow graph to a dataset.

        This method processes a given dataset or iterable dataset using the data flow graph.
        The behavior of the processing depends on whether the dataset is an in-memory or
        streamed dataset (i.e. :class:`datasets.Dataset` or :class:`datasets.IterableDataset`)).

        If the dataset features do not fully align with the source features of the data flow
        graph, a fallback mechanism applies a casting operation. This ensures compatibility
        but may result in data loss or type adjustments. The casting adapts the dataset
        features to match the schema defined by the data flow graph.

        Parameters:
            ds (Dataset | IterableDataset): The dataset to which the data flow graph will be
                applied.
            batch_size (int): The number of samples to process in a batch. Defaults to 1000.
            drop_last_batch (bool): Whether to drop the last batch if it is smaller than
                the specified batch size. Defaults to :code:`False`.
            keep_in_memory (bool): If :code:`True`, the resulting dataset is kept in memory.
                Defaults to :code:`False`. Only applies for :class:`datasets.Dataset`.
            load_from_cache_file (bool): Whether to load the resulting dataset from cache files
                when possible. Defaults to :code:`True`. Only applies for
                :class:`datasets.Dataset`.
            writer_batch_size (int): Batch size for writing results to cache files. Defaults to
                1000. Only applies for :class:`datasets.Dataset`.
            num_proc (None | int): The number of processes to use for parallel processing. Defaults
                to :code:`None`, which disables multiprocessing. Only applies for
                :class:`datasets.Dataset`.
            desc (Optional[str], optional, only for Dataset): A description for the progress bar
                displayed during processing. Defaults to :code:`None`. Only applies for
                :class:`datasets.Dataset`.

        Returns:
            Dataset | ItDataset | tuple[Dataset: The transformed dataset matching the input
            dataset type.
        """
        kwargs = dict(
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            writer_batch_size=writer_batch_size,
            num_proc=num_proc,
            desc=desc,
        )

        # get the dataset features
        if isinstance(ds, (datasets.Dataset, datasets.IterableDataset)):
            features = ds.features
        elif isinstance(ds, (datasets.DatasetDict, datasets.IterableDatasetDict)):
            features = next(iter(ds.values())).features
        else:
            raise ValueError(
                "Expected one of `datasets.Dataset`, `datasets.DatasetDict`, "
                "`datasets.IterableDataset` or `datasets.IterableDatasetDict`,"
                "got %s" % type(ds)
            )

        if features is None:
            raise RuntimeError("Dataset features must not be None.")

        # make sure the dataset features are a subset of the source features of the data flow
        if not is_dtype_subset(self._source_feature.dtype, build_dtype_from_hf_feature(features)):
            if self._source_annotation is not None:
                try:
                    # check if the dataset features align with the data flow
                    # source annotation at all
                    validate_hf_feature(features, self._source_annotation)
                except Exception as e:
                    raise RuntimeError(
                        "Dataset features do not align with the expected data flow source "
                        "annotation. Ensure the dataset features matches the expected graph "
                        "definition."
                    ) from e

            logger.warning(
                "Dataset features do not fully align with the expected data flow features. "
                "A cast operation will be applied to adapt the dataset features to match the "
                "expected schema defined in the graph. This may result in potential data loss "
                "or type adjustments during casting."
            )

            try:
                # create a new data flow instance with the dataset features
                flow = (
                    DataFlow(features)
                    if self._source_annotation is None
                    else DataFlow[self._source_annotation](features)
                )
                # cast the source dataset features to the expected feature type this takes
                # the used dataset features (i.e. flow.source.dtype) and casts them to the
                # expected source features (i.e. self._source_features.dtype)
                # note that cast_dtype supports the target dtype to specify only a subset
                # of the source dtype
                dtype = cast_dtype(flow.source.dtype, self._source_feature.dtype)
                annotation = build_annotation_from_dtype(dtype)
                casted_source = cast(annotation, flow.source)  # type: ignore
                # rebuild the data flow graph by attaching it to the casted source
                # node of the new flow and build it, use the same aggregation
                # manager to make sure the aggregation is not reset
                collect, aggregate = self.attach(casted_source)
                exec_flow = flow.build(
                    collect, aggregate, aggregation_manager=self._aggregation_manager
                )

            except Exception as e:
                # unable to cast dataset features to flow features and applying the flow
                raise RuntimeError(
                    "Failed to cast dataset features to match the data flow graph. Ensure "
                    "the dataset features are compatible with the graph schema."
                ) from e

            # apply the new data flow to the dataset
            return exec_flow._internal_apply(ds, **kwargs)

        return self._internal_apply(ds, **kwargs)

    def attach(self, node: MappingFeature) -> tuple[MappingFeature, MappingFeature | None]:
        """Attach the data flow graph to a given node.

        This method attaches a data flow graph (`self`) to a specified node of an existing graph.
        Attaching the data flow graph means that the source node of the current graph is connected
        to the given node, effectively embedding the data flow graph into the target graph at
        the specified location.

        Parameters:
            node (MappingFeature): The target node in the existing graph to which the
                data flow graph will be attached.

        Returns:
            tuple[MappingFeature, MappingFeature | None]: A tuple containing:
                - The updated collect feature, which now references the corresponding node
                  in the attached graph.
                - The updated aggregate feature, or :code:`None` if the data flow graph does not
                  have an aggregate feature.
        """
        ref: ConcreteReference = node.ref
        # attach the graph to the referenced node
        g: DataFlowGraph = ref._graph
        g.attach_graph_to_node(ref._node_id, self._graph)

        # get the collect node id, use the node id of the node to attach to
        # in case the source features are collected by the original graph
        collect_id = self.collect_feature.ref._node_id
        collect_id = collect_id if collect_id != self._graph.src_node_id else node.ref._node_id
        # do the same for the aggregate id
        aggregate_id: None | NodeId = None
        if self.aggregates_feature is not None:
            aggregate_id = self.aggregates_feature.ref._node_id
            aggregate_id = (
                aggregate_id if aggregate_id != self._graph.src_node_id else node.ref._node_id
            )

        # create the references to the collect and aggregate references
        return (
            build_feature_from_reference(ConcreteReference(collect_id, g)),
            build_feature_from_reference(ConcreteReference(aggregate_id, g))
            if aggregate_id is not None
            else None,
        )

    def arrow_process(
        self, batch: pa.Table, index: IndexList, rank: None | Rank = None
    ) -> pa.Table:
        """Process a batch of data in the form of a pyarrow table.

        Args:
            batch (pa.Table): The batch of data to process.
            index (IndexList): The index of the batch.
            rank (None | Rank): The rank of the process in a distributed setting.

        Returns:
            pa.Table: The processed batch of data as a PyArrow Table.
        """
        if rank is None:
            # try to get multiprocessing rank from worker info
            worker_info = get_worker_info()
            rank = 0 if worker_info is None else worker_info.rank

        # create a new event loop to execute the flow in
        loop = asyncio.new_event_loop()
        # schedule the execution for the current batch
        future = self._instance_executor.execute(batch.to_struct_array(), index, rank)
        out = loop.run_until_complete(future)
        # close the event loop
        loop.close()
        return pa.table(out, schema=self._collect_feature.dtype.arrow_schema)

    def _internal_apply(
        self,
        ds: Dataset | ItDataset,
        batch_size: int,
        drop_last_batch: bool,
        keep_in_memory: bool,
        load_from_cache_file: bool,
        writer_batch_size: int,
        num_proc: None | int,
        desc: None | str,
    ) -> Dataset | ItDataset:
        """(Internal) Apply the data flow to a dataset.

        Args:
            ds (D): The dataset to process.
            batch_size (int): Batch Size.
            drop_last_batch (bool): Whether to drop the last batch.
            keep_in_memory (bool): If :code:`True`, the resulting dataset is kept in memory.
                Only applies for :class:`datasets.Dataset`.
            load_from_cache_file (bool): Load the resulting dataset from cache files
                when possible. Only applies for :class:`datasets.Dataset`.
            writer_batch_size (int): Batch size for writing results to cache files.
                Only applies for :class:`datasets.Dataset`.
            num_proc (None | int): Number of processes to use for parallel processing.
                Only applies for :class:`datasets.Dataset`.
            desc (Optional[str], optional, only for Dataset): A description for the progress bar
                displayed during processing. Only applies for :class:`datasets.Dataset`.

        Returns:
            Dataset | ItDataset: The processed dataset.
        """
        if isinstance(ds, datasets.Dataset):
            # use arrow formatter and only the required input columns
            prepared_ds = ds.with_format(type="arrow", columns=list(self._source_feature.keys()))

            # compute the new fingerprint
            fingerprint = datasets.fingerprint.generate_fingerprint(prepared_ds)
            fingerprint = datasets.fingerprint.update_fingerprint(
                fingerprint, transform=self.serialize(), transform_args={}
            )
            # use pyarrow table as output format for in-memory
            # datasets that support caching
            transformed_ds = prepared_ds.map(
                self.arrow_process,
                with_indices=True,
                with_rank=True,
                batched=True,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                keep_in_memory=keep_in_memory,
                load_from_cache_file=load_from_cache_file,
                writer_batch_size=writer_batch_size,
                num_proc=num_proc,
                desc=desc,
                new_fingerprint=fingerprint,
            )
            # get the format of the input dataset
            input_format = (
                ds.format if isinstance(ds, datasets.Dataset) else next(iter(ds.values())).format
            )
            transformed_ds.set_format(
                type=input_format["type"],
                format_kwargs=input_format["format_kwargs"],
                output_all_columns=True,
            )
            return transformed_ds

        elif isinstance(ds, datasets.DatasetDict):
            # apply to each dataset in the dataset dict
            return datasets.DatasetDict(
                {
                    key: self._internal_apply(
                        ds=val,
                        batch_size=batch_size,
                        drop_last_batch=drop_last_batch,
                        keep_in_memory=keep_in_memory,
                        load_from_cache_file=load_from_cache_file,
                        writer_batch_size=writer_batch_size,
                        num_proc=num_proc,
                        desc=desc or key,
                    )
                    for key, val in ds.items()
                }
            )

        elif isinstance(ds, datasets.IterableDataset):
            # use arrow formatter and only the required input columns
            prepared_ds = ds.with_format(type="arrow")
            # iterable dataset class doesn't support pyarrow
            # outputs in map function, but it also doesn't cache
            # and thus doesn't need the features while processing
            transformed_ds = prepared_ds.map(
                self.arrow_process,
                with_indices=True,
                batched=True,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=(
                    set(self._source_feature.keys()) - set(self._collect_feature.keys())
                ),
                features=self._collect_feature.dtype.hf_feature,
            )
            # unset the dataset format
            return transformed_ds.with_format(type=None)

        elif isinstance(ds, datasets.IterableDatasetDict):
            # use arrow formatter and only the required input columns
            prepared_ds = ds.with_format(type="arrow")
            # iterable dataset class doesn't support pyarrow
            # outputs in map function, but it also doesn't cache
            # and thus doesn't need the features while processing
            transformed_ds = prepared_ds.map(
                self.arrow_process,
                with_indices=True,
                batched=True,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=(
                    set(self._source_feature.keys()) - set(self._collect_feature.keys())
                ),
            )
            # iterable dataset dict doesn't support features argument to map function
            for split in ds.values():
                split.info.features = self._collect_feature.dtype.hf_feature

            # unset the dataset format
            return transformed_ds.with_format(type=None)

    def serialize(self, indent: None | int = None) -> str:
        """Serializes the executable data flow into a JSON string.

        This method creates a dictionary representation of the entire data flow,
        including the graph and the references for collection and aggregation nodes.
        The dictionary is then serialized into a JSON string.

        Args:
            indent (None | int, optional): If provided, formats the output with the
                given indentation level. If :code:`None`, the JSON is serialized without
                indentation.

        Returns:
            str: The JSON string representing the serialized executable data flow.
        """
        source_annotation: None | Any = None
        if self._source_annotation is not None:
            try:
                # try to dump the source annotation
                source_annotation = pickle.dumps(self._source_annotation)
                source_annotation = base64.b64encode(source_annotation).decode("utf-8")
            except AttributeError as e:
                logger.warning(
                    logger.warning(
                        f"Failed to serialize the source annotation due to Exception: {str(e)}. "
                        "The source annotation will be set to None."
                    )
                )

        # build data dictionary representing the full graph
        data = {}
        data["source_annotation"] = source_annotation
        data["collect"] = self._instance_executor.collect._node_id
        data["aggregate"] = (
            None
            if self._aggregates_executor is None
            else self._aggregates_executor.collect._node_id
        )
        data["graph"] = self._graph.to_dict()
        # serialize the dictionary
        return json.dumps(data, indent=indent, sort_keys=True)

    @classmethod
    def deserialize(cls, data: str) -> ExecutableDataFlow:
        """Deserializes a JSON string into an :class:`ExecutableDataFlow` instance.

        This method parses a JSON string into a dictionary, validates its structure,
        and uses the data to reconstruct the executable data flow, including references
        to the collection and aggregation nodes.

        Args:
            data (str): The JSON string representing the serialized executable data flow.

        Returns:
            ExecutableDataFlow: The deserialized executable data flow.

        Raises:
            ValueError: If the input JSON string does not contain the required keys
                ("graph", "collect", and "aggregate").
        """
        # load json string
        data = json.loads(data)

        # check data dict
        if set(data.keys()) != {"source_annotation", "graph", "collect", "aggregate"}:
            raise ValueError(
                "Missing required keys in the serialized data, expected "
                "['source_annotation', 'graph', 'collect', 'aggregate'], "
                f"got {list(data.keys())}."
            )

        # deserialize source annotation
        source_annotation = data["source_annotation"]
        if source_annotation is not None:
            source_annotation = base64.b64decode(source_annotation)
            source_annotation = pickle.loads(source_annotation)

        # deserialize data flow graph
        graph = DataFlowGraph.from_dict(data["graph"])

        # build collect and aggregate references
        collect = ConcreteReference(_node_id=data["collect"], _graph=graph)
        aggregate = (
            None
            if data["aggregate"] is None
            else ConcreteReference(_node_id=data["aggregate"], _graph=graph)
        )

        # construct executable data flow
        return ExecutableDataFlow(source_annotation, graph, collect, aggregate, None)
