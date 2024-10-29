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
from typing import Any, Generic, Literal, TypeVar, get_args

import datasets
import matplotlib.pyplot as plt
import nest_asyncio
import networkx as nx
import numpy as np
import pyarrow as pa
import pydantic
from matplotlib import colormaps

from hyped.common._worker import get_worker_info
from hyped.common.typing import Aggregate, IndexList, Rank
from hyped.common.utils import tmp_setattr

from .abstract import AbstractDataFlow
from .executor import DataFlowExecutor
from .features.features import _Feature, build_feature_from_annotation, build_feature_from_dtype
from .features.reference import FeatureKey, Reference
from .features.types import Type
from .graph import DataFlowGraph
from .typing import Mapping
from .utils import build_dtype_from_hf_feature, is_dtype_subset

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

    U = TypeVar("U")

    def const(self, value: Any, feature_type: type[T]) -> T:
        # create a dummy feature to infer the data type
        # from the given feature type
        ref = Reference(FeatureKey(), "DummyNode", self._graph)
        dtype = pydantic.TypeAdapter(feature_type).validate_python(ref).dtype
        # add the constant node to the graph
        ref = self._graph.add_const_node(value, dtype)
        return build_feature_from_dtype(ref, dtype)

    def build(
        self,
        collect: _Feature,
        aggregate: None | _Feature = None,
    ) -> ExecutableDataFlow[T]:
        if collect.ref._graph is not self._graph:
            raise RuntimeError("The collect feature does not belong to this flow.")

        # create a copy of the graph and update the collect reference accordingly
        graph = self._graph.copy()
        collect = Reference(collect.ref._key, collect.ref._node_id, graph)
        # build executable data flow
        return ExecutableDataFlow(graph, collect)

    def apply(
        self,
        ds: D,
        collect: Reference,
        aggregate: None | Reference = None,
        **kwargs,
    ) -> tuple[D, None | Aggregate | MappingProxyType[str, Any]]:
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
    def __init__(self, graph: DataFlowGraph, collect: Reference) -> None:
        super(ExecutableDataFlow, self).__init__(self)

        # make sure the collect feature belongs to the graph
        if collect._graph is not graph:
            raise RuntimeError("The collect feature does not belong to this flow.")

        # TODO: optimize data flow

        # create read-only view on graph
        self._graph: DataFlowGraph = nx.restricted_view(graph, [], [])
        # get the source feature instance from the graph
        ref = Reference(FeatureKey(), graph.src_node_id, self._graph)
        self._source_feature = self._graph.get_feature_from_reference(ref)

        # make sure the source feature is a mapping type
        if not isinstance(self._source_feature, Mapping):
            raise RuntimeError("Source must be mapping")

        # create the executor instance
        self._executor = DataFlowExecutor(self._graph, collect, aggregation_manager=None)

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
        future = self._executor.execute(batch.to_struct_array(), index, rank)
        out = loop.run_until_complete(future)
        # close the event loop
        loop.close()

        return pa.table(out if isinstance(out, pa.StructArray) else {"value": out})

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
