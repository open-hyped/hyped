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
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar, get_args, overload

import datasets
import matplotlib.pyplot as plt
import nest_asyncio
import networkx as nx
import numpy as np
import pyarrow as pa
import pydantic
from matplotlib import colormaps

from hyped.common._worker import get_worker_info
from hyped.common.typing import IndexList, NodeId, Rank
from hyped.common.utils import tmp_setattr

from .abstract import AbstractDataFlow
from .executor import DataFlowExecutor, LazyDataFlowExecutor
from .features.features import (
    _Bool,
    _Feature,
    _Float64,
    _Int64,
    _String,
    build_feature_from_annotation,
    build_feature_from_dtype,
)
from .features.reference import FeatureKey, Reference
from .features.types import MappingType, Type
from .graph import DataFlowGraph
from .nodes.aggregator import DataAggregationManager
from .nodes.base import RunContext
from .optim import DataFlowGraphOptimizer
from .typing import Mapping, Sequence
from .utils import build_dtype_from_hf_feature, build_dtype_from_object, is_dtype_subset

D = TypeVar(
    "D",
    datasets.Dataset,
    datasets.DatasetDict,
    datasets.IterableDataset,
    datasets.IterableDatasetDict,
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


T = TypeVar("T", bound=Mapping)


class DataFlow(AbstractDataFlow, Generic[T]):
    """High-level interface for defining and executing data processing workflows.

    The DataFlow class allows users to create and manage directed acyclic graphs
    (DAGs) of data processors, facilitating complex data transformations and
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
        return self._graph.src_node_id is not None

    def _initialize(self) -> None:
        # make sure the flow is not initialized yet
        assert not self._is_initialized

        # try to get the source type from the type variable
        src_type_annotation: None | type[T] = (
            None if not hasattr(self, "__orig_class__") else get_args(self.__orig_class__)[0]
        )

        # infer the source data type from the hf
        # features and/or the source type annotation
        src_dtype: Type

        if (src_type_annotation is not None) and (self._hf_source_features is not None):
            ref = Reference(FeatureKey(), "DummyNode", self._graph)
            # create a dummy feature instance according to the huggingface features
            hf_dtype = build_dtype_from_hf_feature(self._hf_source_features)
            instance = build_feature_from_dtype(ref, hf_dtype)
            # validate the feature instance with respect to the type annotation
            adapter = pydantic.TypeAdapter(src_type_annotation)
            instance = adapter.validate_python(instance, context={"strict": True})
            # use the data type of the validated instance as the source data type
            src_dtype = instance.dtype

        elif src_type_annotation is not None:
            # infer the source dtype from the type annotation
            src_dtype = build_feature_from_annotation(
                Reference(FeatureKey(), "DummyNode", self._graph),
                src_type_annotation,
            ).dtype

        elif self._hf_source_features is not None:
            # build the source dtype from the huggingface features
            src_dtype = build_dtype_from_hf_feature(self._hf_source_features)

        else:
            # no input specified, at least argument or type hint is required
            raise RuntimeError()

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
        return self._graph.depth

    @property
    def width(self) -> int:
        """Computes the maximum width of the data flow graph.

        The width is defined as the maximum number of nodes present at any single depth level
        in the graph. This property calculates the width by grouping nodes by their depth and
        finding the largest group.

        Returns:
            int: The maximum width of the graph.
        """
        return self._graph.width

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
    def const(self, value: int) -> _Int64:
        ...

    @overload
    def const(self, value: float) -> _Float64:
        ...

    @overload
    def const(self, value: bool) -> _Bool:
        ...

    @overload
    def const(self, value: str) -> _String:
        ...

    @overload
    def const(self, value: dict) -> Mapping:
        ...

    @overload
    def const(self, value: list | tuple) -> Sequence:
        ...

    U = TypeVar("U", bound=_Feature)

    @overload
    def const(self, value: Any, feature_type: type[U]) -> U:
        ...

    def const(self, value: Any, feature_type: None | type = None) -> _Feature:
        if feature_type is not None:
            # create a dummy feature to infer the data type
            # from the given feature type
            adapter = pydantic.TypeAdapter(feature_type)
            dtype = adapter.validate_python(Reference()).dtype
        else:
            # build the data type matching the object in case no data type was provided
            dtype = build_dtype_from_object(value)
        # add the constant node to the graph
        ref = self._graph.add_const_node(value, dtype)
        return self._graph.get_feature_from_reference(ref)

    @overload
    def collect(self, collect: dict) -> Mapping:
        ...

    @overload
    def collect(self, collect: list | tuple) -> Sequence:
        ...

    @overload
    def collect(self, collect: Any) -> _Feature:
        ...

    def collect(self, collect: dict | list) -> Mapping | Sequence | _Feature:
        ref = self._graph.add_collect_node(collect)
        return self._graph.get_feature_from_reference(ref)

    def build(
        self,
        collect: _Feature,
        aggregate: None | _Feature = None,
    ) -> ExecutableDataFlow[T]:
        if collect.ref._graph is not self._graph:
            raise RuntimeError("The collect feature does not belong to this flow.")

        # make sure flow graph is initialized
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

    def apply(
        self,
        ds: D,
        collect: Reference,
        aggregate: None | Reference = None,
        **kwargs,
    ) -> tuple[D, None | MappingProxyType[str, Any]]:
        return self.build(collect, aggregate).apply(ds, **kwargs)

    def plot(
        self,
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
        """Plot the data flow graph.

        Args:
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
            _, ax = plt.subplots(1, 1, figsize=(self.depth * 2, self.width * 2.5))

        # compute the node positions
        pos = nx.multipartite_layout(self._graph, subset_key=DataFlowGraph.NodeAttribute.DEPTH)

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
            for _, data in self._graph.nodes(data=True)
        ]

        # plot the raw graph
        nx.draw(
            self._graph,
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
        for node, data in self._graph.nodes(data=True):
            if node == self._graph.src_node_id:
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
            self._graph,
            pos,
            labels=node_labels,
            font_size=node_font_size,
            ax=ax,
        )

        # add edge labels
        if with_edge_labels:
            # group multi-edges by their source and target nodes
            grouped_edges = groupby(
                sorted(self._graph.edges(data=True), key=lambda e: (e[0], e[1])),
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
                self._graph,
                pos,
                edge_labels=edge_labels,
                font_size=edge_font_size,
                ax=ax,
            )

        return ax


class ExecutableDataFlow(DataFlow[T]):
    def __init__(
        self, graph: DataFlowGraph, collect: Reference, aggregate: Reference | None
    ) -> None:
        super(ExecutableDataFlow, self).__init__(self)

        # make sure the collect feature belongs to the graph
        if (collect._graph is not graph) or (collect._node_id not in graph.nodes):
            raise RuntimeError("The collect node does not belong to this flow.")

        # make sure collect doesn't belong to the aggregate partition
        if graph.get_node_output_partition(collect._node_id) == DataFlowGraph.Partition.AGGREGATED:
            raise RuntimeError()

        if aggregate is not None:
            if (aggregate._graph is not graph) or (aggregate._node_id not in graph.nodes):
                raise RuntimeError("The aggregate node does not belong to this flow.")

            # make sure aggregate belongs to the aggregate partition
            if (
                graph.get_node_output_partition(aggregate._node_id)
                != DataFlowGraph.Partition.AGGREGATED
            ):
                raise RuntimeError()

            # make sure the collect feature is a mapping
            if not isinstance(graph.get_feature_from_reference(aggregate), Mapping):
                raise RuntimeError("aggregate must be mapping")

        # get the source feature instance from the graph
        ref = Reference(_node_id=graph.src_node_id, _graph=graph)
        self._source_feature = graph.get_feature_from_reference(ref)
        self._collect_feature = graph.get_feature_from_reference(collect)

        # make sure the source feature is a mapping
        if not isinstance(self._source_feature, Mapping):
            raise RuntimeError("Source must be mapping")

        # make sure the collect feature is a mapping
        if not isinstance(self._collect_feature, Mapping):
            raise RuntimeError("Collect must be mapping")

        # optimize data flow
        graph = DataFlowGraphOptimizer().optimize(
            graph,
            leaf_nodes=(
                {collect._node_id} if aggregate is None else {collect._node_id, aggregate._node_id}
            ),
        )

        # create read only view on optimized graph
        self._graph = nx.restricted_view(graph, [], [])
        # create read-only view on the instance partition of the graph
        self._instance_graph = graph.drop_partition(DataFlowGraph.Partition.AGGREGATED)
        self._aggregates_graph: DataFlowGraph = None

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
            self._aggregates_executor = self._build_aggregates_executor(aggregate, set(nodes))

        # create the executors
        self._instance_executor = DataFlowExecutor(
            self._instance_graph, collect, aggregation_manager=self._aggregation_manager
        )

    @property
    def aggregates(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(self._aggregates_executor)

    def _build_aggregates_executor(
        self, aggregate: Reference, aggregator_nodes: set[NodeId]
    ) -> DataFlowExecutor:
        # build the dependency graph of the aggregate up to the aggregator nodes
        # note that this also includes constants
        G = self._graph.dependency_graph({aggregate._node_id}, stop_nodes=aggregator_nodes)
        # remove the aggregator nodes themselves as these are executed in the
        # instance graph
        G = nx.restricted_view(G, aggregator_nodes, [])

        H = DataFlowGraph()
        # add a source node
        source_type = MappingType.from_dict(
            {
                str(node): self._graph.nodes[node][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
                for node in aggregator_nodes
            }
        )
        source_ref = H.add_source_node(source_type)

        # rebuild the aggregates graph
        for node_id in nx.topological_sort(G):
            # collect all the inputs to the node
            edges = self._graph.in_edges(node_id, data=DataFlowGraph.EdgeAttribute.KEY, keys=True)
            inputs = {
                name: (
                    replace(source_ref, _key=FeatureKey(str(u), *key))
                    if u in aggregator_nodes
                    else Reference(key, u, H)
                )
                for u, _, name, key in edges
            }
            # add the node to the graph
            data = self._graph.nodes[node_id]
            H.add_node(
                node_obj=data[DataFlowGraph.NodeAttribute.NODE_OBJ],
                node_type=data[DataFlowGraph.NodeAttribute.NODE_TYPE],
                inputs=inputs,
                output_type=data[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                node_id=node_id,
            )

        # build the executor
        return LazyDataFlowExecutor(
            H, replace(aggregate, _graph=H), self._aggregation_manager.values_proxy
        )

    def _build_aggregation_manager(self, nodes: list[NodeId]) -> DataAggregationManager:
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

    def _initialize(self) -> None:
        raise EnvironmentError()

    def apply(self, ds: D, **kwargs) -> D:
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
        ds_dtype = build_dtype_from_hf_feature(features)

        # make sure the necessary features are contained in the dataset
        if not is_dtype_subset(self._source_feature.dtype, ds_dtype):
            raise RuntimeError(
                f"Expected input schema doesn't match dataset:\n"
                f"Expected feature type: {self._source_feature.dtype}\n"
                f"But received type: {ds_dtype}"
            )

        # set the source feature to match the dataset while executing the flow
        ds_source_feature = replace(self._source_feature, dtype=ds_dtype)
        with tmp_setattr(self, "_source_feature", ds_source_feature):
            ds = self._internal_apply(ds, **kwargs)

        if isinstance(ds, (datasets.IterableDataset, datasets.IterableDatasetDict)):
            # set output features for lazy datasets manually
            # TODO: implement a arrow_type -> hf feature convert function
            raise NotImplementedError()

        # return the processed dataset and a snapshot of the aggregated values
        return ds

    def batch_process(
        self, batch: dict[str, list[Any]], index: IndexList, rank: None | Rank = None
    ) -> pa.Table:
        """Process a batch of data.

        Args:
            batch (dict[str, list[Any]]): The batch of data to process.
            index (IndexList): The index of the batch.
            rank (None | Rank): The rank of the process in a distributed setting.

        Returns:
            pa.Table: The processed batch of data as a PyArrow Table.

        Raises:
            AssertionError: If the flow has not been build yet.
        """
        if isinstance(batch, datasets.formatting.formatting.LazyBatch):
            batch = dict(batch)

        if rank is None:
            # try to get multiprocessing rank from worker info
            worker_info = get_worker_info()
            rank = 0 if worker_info is None else worker_info.rank

        # create a new event loop to execute the flow in
        loop = asyncio.new_event_loop()
        # schedule the execution for the current batch
        batch = pa.table(batch, schema=self._source_feature.dtype.arrow_schema)
        future = self._instance_executor.execute(batch.to_struct_array(), index, rank)
        out = loop.run_until_complete(future)
        # close the event loop
        loop.close()

        return pa.table(out, schema=self._collect_feature.dtype.arrow_schema)

    def _batch_process_to_pydict(
        self,
        batch: dict[str, list[Any]],
        index: IndexList,
        rank: None | Rank = None,
    ) -> dict[str, list[Any]]:
        """Process a batch of data and convert to a python dictionary.

        Args:
            batch (dict[str, list[Any]]): The batch of data to process.
            index (IndexList): The index of the batch.
            rank (None | Rank): The rank of the process in a
                multiprocessing setting.

        Returns:
            dict: The processed data as a python dictionary
        """
        return self.batch_process(batch, index, rank).to_pydict()

    def _internal_apply(self, ds: D, **kwargs) -> D:
        """(Internal) Apply the data flow to a dataset.

        Args:
            ds (D): The dataset to process.
            **kwargs: Additional arguments for dataset mapping. For more
                information please refer to the HuggingFace documentation
                of the Datasets.map function for the respective dataset type.

        Returns:
            D: The processed dataset.
        """
        # required settings
        kwargs["batched"] = True
        kwargs["with_indices"] = True
        # for non-iterable datasets the map function should provide the rank
        if isinstance(ds, (datasets.Dataset, datasets.DatasetDict)):
            kwargs["with_rank"] = True

        if isinstance(ds, (datasets.Dataset, datasets.DatasetDict)):
            # use pyarrow table as output format for in-memory
            # datasets that support caching since it includes
            # the output feature information
            return ds.map(self.batch_process, **kwargs)

        elif isinstance(ds, (datasets.IterableDataset, datasets.IterableDatasetDict)):
            # iterable dataset class doesn't support pyarrow
            # outputs in map function, but it also doesn't cache
            # and thus doesn't need the features while processing
            return ds.map(
                self._batch_process_to_pydict,
                remove_columns=(
                    set(self.src_features.feature_.keys())
                    - set(self._executor.collect.feature_.keys())
                ),
                **kwargs,
            )
