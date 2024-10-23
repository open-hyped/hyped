"""Defines high-level interfaces for data processing workflows.

This module provides the :class:`DataFlow` class, which allows users to define and
execute complex data processing workflows. The workflows are represented as
directed acyclic graphs (DAGs) of data processors.
"""

from __future__ import annotations

import asyncio
import re
from itertools import groupby
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar, get_args

import datasets
import matplotlib.pyplot as plt
import nest_asyncio
import networkx as nx
import numpy as np
import pyarrow as pa
from datasets.features.features import FeatureType
from matplotlib import colormaps

from hyped.common._worker import get_worker_info
from hyped.common.typing import Aggregate, Batch, IndexList, Rank
from hyped.common.utils import tmp_setattr
from hyped.core.features.feature_key import FeatureKey

from .abstract import AbstractDataFlow
from .executor import DataFlowExecutor
from .features.factories import BaseFeatureFactory, DefaultFeatureFactory
from .features.features import _Feature
from .features.reference import Reference
from .graph import DataFlowGraph
from .refs.ref import FeatureRef
from .typing import Mapping
from .utils import FeatureFactoryFromHuggingFace, _is_type_subset

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


T = TypeVar("T", bound=_Feature)


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

        self._graph = DataFlowGraph()
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

        # build the source type factory for the
        src_type_factory: BaseFeatureFactory

        if self._hf_source_features is not None:
            # create the source type from the given source features
            src_type_factory = FeatureFactoryFromHuggingFace[src_type_annotation](
                self._hf_source_features
            )

        elif src_type_annotation is not None:
            # infer type from type annotation using default type resolvers
            src_type_factory = DefaultFeatureFactory[src_type_annotation]()

        else:
            # no input specified, at least argument or type hint is required
            raise RuntimeError()

        # add the source node to the graph with the node id
        node_id = self._graph.add_source_node(src_type_factory._pa_type)
        self._source_feature = src_type_factory(FeatureKey(), node_id, self._graph)

        if not isinstance(self._source_feature, Mapping):
            raise RuntimeError("Must be mapping")

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

    @property
    def out_features(self) -> FeatureRef:
        raise NotImplementedError()

    @property
    def aggregates(self) -> None | MappingProxyType[str, Any]:
        raise NotImplementedError()

    def const(self, value: Any, feature: None | FeatureType = None) -> FeatureRef:
        raise NotImplementedError()

    def build(
        self,
        collect: Reference,
        aggregate: None | Reference = None,
    ) -> ExecutableDataFlow[T]:
        if collect._graph is not self._graph:
            raise RuntimeError("The collect feature does not belong to this flow.")

        # create a copy of the graph
        graph = self._graph.copy()
        # update the references to the new graph
        collect = Reference(collect._key, collect._node_id, graph)

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

        # create source schema from source feature
        self._source_schema = pa.schema(self._source_feature._pa_type)

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
        src_schema = features.arrow_schema

        # make sure the necessary features are contained in the dataset
        if not _is_type_subset(self._source_feature._pa_type, pa.struct(src_schema)):
            raise RuntimeError(
                f"Expected input schema doesn't match dataset:\n"
                f"Expected feature type: {self._source_feature._pa_type}\n"
                f"But received type: {pa.struct(src_schema)}"
            )

        # set the source feature to the dataset arrow type
        # while executing the flow
        with tmp_setattr(self, "_source_schema", src_schema):
            ds = self._internal_apply(ds, **kwargs)

        if isinstance(ds, (datasets.IterableDataset, datasets.IterableDatasetDict)):
            # set output features for lazy datasets manually
            # TODO: implement a arrow_type -> hf feature convert function
            raise NotImplementedError()

        # return the processed dataset and a snapshot of the aggregated values
        return ds

    def batch_process(
        self, batch: dict[str, list[Any]], index: IndexList, rank: None | Rank = None
    ) -> Batch:
        """Process a batch of data.

        Args:
            batch (dict[str, list[Any]]): The batch of data to process.
            index (IndexList): The index of the batch.
            rank (None | Rank): The rank of the process in a distributed setting.

        Returns:
            Table: The processed batch of data as a PyArrow Table.

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
        batch = pa.table(batch, schema=self._source_schema)
        future = self._executor.execute(batch, index, rank)
        out = loop.run_until_complete(future)
        # close the event loop
        loop.close()

        return out

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
