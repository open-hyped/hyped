"""Defines high-level interfaces for data processing workflows.

This module provides the :class:`DataFlow` class, which allows users to define and
execute complex data processing workflows. The workflows are represented as
directed acyclic graphs (DAGs) of data processors.
"""

from __future__ import annotations

import asyncio
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
import pydantic
from matplotlib import colormaps

from hyped.common._worker import get_worker_info

from .abstract import AbstractDataFlow
from .executor import DataFlowExecutor, LazyDataFlowExecutor
from .features.features import (
    BoolFeature,
    Feature,
    Float64Feature,
    Int64Feature,
    MappingFeature,
    SequenceFeature,
    StringFeature,
    build_feature_from_annotation,
    build_feature_from_dtype,
)
from .features.reference import FeatureKey, Reference
from .features.types import MappingType, Type
from .graph import DataFlowGraph
from .nodes.aggregator import DataAggregationManager
from .nodes.base import RunContext
from .optim import DataFlowGraphOptimizer
from .typing import IndexList, NodeId, Rank
from .utils import (
    NestedType,
    build_dtype_from_hf_feature,
    build_dtype_from_python_object,
    is_dtype_subset,
    map_recursive,
)

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
    edge_label_format: str = "{name}={key}",
    src_node_label: str = "[ROOT]",
    edge_font_size: int = 6,
    node_font_size: int = 6,
    node_size: int = 5_000,
    arrowsize: int = 25,
    color_map: dict[
        Literal[
            DataFlowGraph.NodeType.SOURCE,
            DataFlowGraph.NodeType.DATA_PROCESSOR,
            DataFlowGraph.NodeType.DATA_AUGMENTER,
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
        edge_label_format (str): Format string for edge labels. Defaults to "{name}={key}".
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
        DataFlowGraph.NodeType.COLLECT: cmap.colors[5],
        DataFlowGraph.NodeType.DATA_PROCESSOR: cmap.colors[2],
        DataFlowGraph.NodeType.DATA_AUGMENTER: cmap.colors[3],
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
            # get the processor type name of the current node
            proc = data[DataFlowGraph.NodeAttribute.NODE_OBJ]
            node_label = type(proc).__name__
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
            sorted(flow._graph.edges(data=True), key=lambda e: (e[0], e[1])),
            key=lambda e: (e[0], e[1]),
        )

        edge_labels = {}
        # build edge labels
        for edge, group in grouped_edges:
            edge_labels[edge] = "\n".join(
                [
                    edge_label_format.format(
                        name=data[DataFlowGraph.EdgeAttribute.NAME],
                        key=str(data[DataFlowGraph.EdgeAttribute.KEY]),
                    )
                    for _, _, data in group
                ]
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

        # try to get the source type from the type variable
        src_type_annotation: None | type[T] = (
            None if not hasattr(self, "__orig_class__") else get_args(self.__orig_class__)[0]
        )

        # infer the source data type from the hf
        # features and/or the source type annotation
        src_dtype: Type

        if (src_type_annotation is not None) and (self._hf_source_features is not None):
            ref = Reference(_graph=self._graph)
            # create a dummy feature instance according to the huggingface features
            hf_dtype = build_dtype_from_hf_feature(self._hf_source_features)
            instance = build_feature_from_dtype(ref, hf_dtype)
            # validate the feature instance with respect to the type annotation
            try:
                adapter = pydantic.TypeAdapter(src_type_annotation)
                instance = adapter.validate_python(instance, context={"strict": True})
            except pydantic.ValidationError as e:
                raise RuntimeError(
                    f"The provided HuggingFace features '{self._hf_source_features}' are "
                    f"incompatible with the type annotation {src_type_annotation}."
                ) from e
            # use the data type of the validated instance as the source data type
            src_dtype = instance.dtype

        elif src_type_annotation is not None:
            # infer the source dtype from the type annotation
            src_dtype = build_feature_from_annotation(
                Reference(_graph=self._graph),
                src_type_annotation,
            ).dtype

        elif self._hf_source_features is not None:
            # build the source dtype from the huggingface features
            src_dtype = build_dtype_from_hf_feature(self._hf_source_features)

        else:
            # no input specified, at least argument or type hint is required
            raise RuntimeError(
                "Initialization failed: At least one of 'features' or type annotation must be "
                "provided to infer the source data type."
            )

        # add the source node to the graph with the node id
        src_ref = self._graph.add_source_node(src_dtype)
        self._source_feature = build_feature_from_dtype(src_ref, src_dtype)

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
            adapter = pydantic.TypeAdapter(feature_type)
            dtype = adapter.validate_python(Reference()).dtype
        else:
            # build the data type matching the object in case no data type was provided
            dtype = build_dtype_from_python_object(value)
        # add the constant node to the graph
        ref = self._graph.add_const_node(value, dtype)
        return self._graph.get_feature_from_reference(ref)

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
        return self._graph.get_feature_from_reference(ref)

    def build(
        self,
        collect: Feature | dict[str, Any],
        aggregate: None | Feature | dict[str, Any] = None,
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
        collect = Reference(collect.ref._key, collect.ref._node_id, graph)
        aggregate = (
            None
            if aggregate is None
            else Reference(aggregate.ref._key, aggregate.ref._node_id, graph)
        )
        # build executable data flow
        return ExecutableDataFlow(graph, collect, aggregate)

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
        self, graph: DataFlowGraph, collect: Reference, aggregate: Reference | None
    ) -> None:
        """Initialize a :class:`ExecutableDataFlow`.

        Args:
            graph (DataFlowGraph): The data flow graph describing the pipeline.
            collect (Reference): A reference to the "collect" node in the graph.
            aggregate (Reference | None): A reference to the "aggregate" node, if applicable.

        Raises:
            RuntimeError: If the collect node is not part of the graph.
            RuntimeError: If the collect node is part of the :code:`AGGREGATED` partition.
            RuntimeError: If the collect feature is not a mapping.
            RuntimeError: If the aggregate node is not part of the graph.
            RuntimeError: If the aggregate node is not part of the :code:`AGGREGATED` partition.
            RuntimeError: If the aggregate feature is not a mapping.
        """
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
            if not isinstance(graph.get_feature_from_reference(aggregate), MappingFeature):
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
        ref = Reference(_node_id=graph.src_node_id, _graph=self._graph)
        self._source_feature: MappingFeature = self._graph.get_feature_from_reference(ref)
        self._collect_feature: MappingFeature = self._graph.get_feature_from_reference(
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
            self._aggregation_manager = self._build_aggregation_manager(nodes)
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
    def aggregates(self) -> Mapping[str, Any]:  # pragma: not covered
        """Read-only view of the aggregated values computed during execution.

        Returns:
            MappingProxyType[str, Any]: A read-only mapping of aggregated values.
        """
        return self._aggregates_executor  # pragma: not covered

    def _build_aggregates_graph(
        self, aggregate: Reference, aggregator_nodes: set[NodeId]
    ) -> DataFlowGraph:
        """Build the aggregates graph.

        Constructs a subgraph modeling all operations performed on aggregates,
        up to the aggregator nodes. The resulting graph excludes the aggregator
        nodes themselves, as these are included separately in the instance graph.

        The source node provides all the aggregated features produced by the
        aggregator nodes. The values to these features are provided by the
        aggregation manager.

        Args:
            aggregate (Reference): The aggregate reference for which the dependency graph
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
        source_type = MappingType.from_dict(
            {
                str(node): self._graph.nodes[node][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
                for node in aggregator_nodes
            }
        )
        source_ref = h.add_source_node(source_type)

        # rebuild the aggregates graph
        for node_id in nx.topological_sort(g):
            # collect all the inputs to the node
            edges = self._graph.in_edges(node_id, data=DataFlowGraph.EdgeAttribute.KEY, keys=True)
            inputs = {
                name: (
                    replace(source_ref, _key=FeatureKey(str(u), *key))
                    if u in aggregator_nodes
                    else Reference(key, u, h)
                )
                for u, _, name, key in edges
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

        # build the arrow type from the dataset features
        ds_dtype: MappingType = build_dtype_from_hf_feature(features)

        # make sure the necessary features are contained in the dataset
        if not is_dtype_subset(self._source_feature.dtype, ds_dtype):
            raise RuntimeError(
                f"Expected input schema doesn't match dataset:\n"
                f"Expected feature type: {self._source_feature.dtype}\n"
                f"But received type: {ds_dtype}"
            )

        return self._internal_apply(
            ds,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            writer_batch_size=writer_batch_size,
            num_proc=num_proc,
            desc=desc,
        )

    def pyarrow_process(
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
        if isinstance(ds, (datasets.Dataset, datasets.DatasetDict)):
            # use arrow formatter and only the required input columns
            prepared_ds = ds.with_format(type="arrow", columns=list(self._source_feature.keys()))
            # use pyarrow table as output format for in-memory
            # datasets that support caching
            transformed_ds = prepared_ds.map(
                self.pyarrow_process,
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

        elif isinstance(ds, datasets.IterableDataset):
            # use arrow formatter and only the required input columns
            prepared_ds = ds.with_format(type="arrow")
            # iterable dataset class doesn't support pyarrow
            # outputs in map function, but it also doesn't cache
            # and thus doesn't need the features while processing
            transformed_ds = prepared_ds.map(
                self.pyarrow_process,
                with_indices=True,
                batched=True,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=(
                    set(self._source_feature.keys()) - set(self._collect_feature.keys())
                ),
                features=datasets.Features.from_arrow_schema(
                    self._collect_feature.dtype.arrow_schema
                ),
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
                self.pyarrow_process,
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
                split.info.features = datasets.Features.from_arrow_schema(
                    self._collect_feature.dtype.arrow_schema
                )

            # unset the dataset format
            return transformed_ds.with_format(type=None)
