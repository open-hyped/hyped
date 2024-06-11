"""Defines the structure of data flow graphs and their components.

This module provides the :class:`DataFlowGraph` class and related components, which represent
the structure of a data processing workflow. The graph consists of nodes (data processors)
and edges (data flow between processors).
"""


from __future__ import annotations

from enum import Enum
from itertools import groupby

import datasets
import networkx as nx

from hyped.common.feature_key import FeatureKey
from hyped.data.flow.core.nodes.aggregator import BaseDataAggregator
from hyped.data.flow.core.nodes.processor import BaseDataProcessor
from hyped.data.flow.core.refs.inputs import InputRefs
from hyped.data.flow.core.refs.outputs import OutputRefs
from hyped.data.flow.core.refs.ref import AggregationRef, FeatureRef

SRC_NODE_ID = 0


class DataFlowGraph(nx.MultiDiGraph):
    """A multi-directed graph representing a data flow of data processors.

    This class is used internally to define a directed acyclic graph (DAG)
    where nodes represent data processors of `BaseDataProcessor` type, and
    edges define the data flow between these processors.
    """

    class NodeType(Enum):
        """Enum representing types of nodes in the data flow graph."""

        SOURCE = "SOURCE_NODE"
        """
        Represents a source node in the data flow graph.

        This type of node acts as the starting point of the data flow graph,
        typically representing raw input data sources.
        """

        DATA_PROCESSOR = "DATA_PROCESSOR_NODE"
        """
        Represents a data processor node in the data flow graph.

        This type of node represents a data processing component within the
        data flow graph. Data processors perform specific transformations
        on input data and produce output data based on defined processing logic.
        """

        DATA_AGGREGATOR = "DATA_AGGREGATOR_NODE"
        """
        Represents a data aggregator node in the data flow graph.

        This type of node is responsible for aggregating data from multiple
        sources or processing stages within the data flow graph. Aggregator
        nodes typically perform dataset-wide computations or combine data
        from different sources into a unified representation.
        """

    class NodeProperty(str, Enum):
        """Enum representing properties of a node in the data flow graph."""

        PROCESSOR = "processor"
        """
        Represents the data processor associated with the node.

        Type: :code:`None`| :class:`BaseDataProcessor` | :class:`BaseDataAugmentor`

        This property holds a reference to the data processor instance that the node
        represents within the data flow graph. Set to :code:`None` for the source node.
        """

        # TODO: rename this property to NODE_TYPE
        PROCESSOR_TYPE = "processor_type"
        """
        Represents the type of the data processor associated with the node.

        Type: :class:`NodeType`

        This property indicates the type of data processor. It helps in categorizing
        and identifying the nature of the processor in the data flow graph.
        """

        IN_FEATURES = "in_features"
        """
        Represents the input features associated with the node.

        Type: :class:`datasets.Features`

        This property contains the input features required by the data processor,
        encapsulated in a HuggingFace `datasets.Features` instance. It defines the
        structure and types of data that is expected by this node.
        """

        OUT_FEATURES = "out_features"
        """
        Represents the output features associated with the node.

        Type: :class:`datasets.Features`

        This property contains the output features produced by the data processor,
        encapsulated in a HuggingFace `datasets.Features` instance. It defines the
        structure and types of data that are output by this node.
        """

        DEPTH = "depth"
        """
        Represents the depth of the node within the data flow graph.

        Type: :class:`int`

        This property indicates the level of the node in the graph, with the root
        node having a depth of 0. It is used to understand the hierarchical position
        of the node relative to other nodes in the data flow.
        """

    class EdgeProperty(str, Enum):
        """Enum representing properties of an edge in the data flow graph."""

        NAME = "name"
        """
        Represents the name of the edge.

        Type: :class:`str`

        This property corresponds to the keyword of the argument used as an input
        to the processor, linking the edge to a specific input parameter.

        The name is also used by NetworkX as an identifier to distinguish multiedges
        between a pair of nodes. It serves as a unique identifier for the edge.
        """

        KEY = "feature_key"
        """
        Represents the key of the feature associated with the edge.

        Type: :class:`FeatureKey`

        This property specifies which subfeature of the output of the source node
        is flowing through the edge. It defines the particular feature that is being
        transmitted from one node to another in the data flow graph.
        """

    @property
    def depth(self) -> int:
        """Computes the total depth of the data flow graph.

        The depth is defined as the maximum level of any node in the graph, where the root
        node has a depth of 0. This property calculates the depth by finding the maximum
        depth attribute among all nodes in the graph.

        Returns:
            int: The total depth of the graph.
        """
        return (
            max(
                nx.get_node_attributes(
                    self, DataFlowGraph.NodeProperty.DEPTH
                ).values()
            )
            + 1
        )

    @property
    def width(self) -> int:
        """Computes the width of the data flow graph.

        The width is defined as the maximum number of nodes present at any single depth level
        in the graph. This property calculates the width by grouping nodes by their depth and
        finding the largest group.

        Returns:
            int: The maximum width of the graph.
        """
        # group nodes by their layer
        depths = nx.get_node_attributes(self, DataFlowGraph.NodeProperty.DEPTH)
        layers = groupby(sorted(self, key=depths.get), key=depths.get)
        # find larges layer in graph
        return max(len(list(layer)) for _, layer in layers)

    def add_source_node(self, features: datasets.Features) -> int:
        """Add a the source node to the graph.

        This method adds a source node to the graph, which acts as the initial
        data provider for the data flow.

        Args:
            features (datasets.Features): The features of the source node.

        Returns:
            FeatureRef: A reference to the input features.

        Raises:
            AssertionError: If the graph already contains a source node
        """
        assert SRC_NODE_ID not in self, "Graph already contains a source node"
        # add src node to graph
        self.add_node(
            SRC_NODE_ID,
            **{
                DataFlowGraph.NodeProperty.PROCESSOR: None,
                DataFlowGraph.NodeProperty.PROCESSOR_TYPE: DataFlowGraph.NodeType.SOURCE,
                DataFlowGraph.NodeProperty.IN_FEATURES: features,
                DataFlowGraph.NodeProperty.OUT_FEATURES: features,
                DataFlowGraph.NodeProperty.DEPTH: 0,
            },
        )
        return SRC_NODE_ID

    def add_processor_node(
        self,
        processor: BaseDataProcessor | BaseDataAggregator,
        inputs: InputRefs,
        output_features: None | datasets.Features,
    ) -> int:
        """Add a processor node to the graph.

        This method adds a processor node to the graph and creates the
        necessary edges to define the data flow from input nodes to this
        processor.

        Args:
            processor (BaseDataProcessor | BaseDataAggregator): The processor or aggregator to add.
            inputs (InputRefs): The input references for the processor.
            output_features (None | datasets.Features):
                The output features generated by the processor. Must be None for aggregator nodes.

        Returns:
            int: The node id of the processor within the graph.

        Raises:
            RuntimeError: If input references are not from this data flow.
            AssertionError: If the graph does not contain a source node.
            AssertionError: If input references are not of the expected type
            AssertionError: If input features do not match the output features
                of the referred node.
        """
        # make sure the input refs match the processor
        assert SRC_NODE_ID in self, "No source node in graph."
        assert isinstance(inputs, processor._in_refs_type), (
            f"Expected input references of type {processor._in_refs_type}, "
            f"but got {type(inputs)}"
        )
        # aggregators have no output features
        if isinstance(processor, BaseDataAggregator):
            assert output_features is None

        # get processor type
        processor_type = (
            DataFlowGraph.NodeType.DATA_PROCESSOR
            if isinstance(processor, BaseDataProcessor)
            else DataFlowGraph.NodeType.DATA_AGGREGATOR
            if isinstance(processor, BaseDataAggregator)
            else None
        )
        assert (
            processor_type is not None
        ), f"Invalid processor type {type(processor)}."

        # add processor to graph
        depth = -1
        node_id = self.number_of_nodes()
        self.add_node(
            node_id,
            **{
                DataFlowGraph.NodeProperty.PROCESSOR: processor,
                DataFlowGraph.NodeProperty.PROCESSOR_TYPE: processor_type,
                DataFlowGraph.NodeProperty.IN_FEATURES: inputs.features_,
                DataFlowGraph.NodeProperty.OUT_FEATURES: output_features,
                DataFlowGraph.NodeProperty.DEPTH: -1,  # placeholder
            },
        )
        # add dependency edges to graph
        for name, ref in inputs.named_refs.items():
            # make sure the inputs come from this flow
            if ref.flow_ != self:
                raise RuntimeError(
                    "Input reference does not belong to this data flow."
                )
            # make sure the input is a valid output of the referred node
            assert ref.node_id_ in self
            assert (
                ref.key_.index_features(
                    self.nodes[ref.node_id_][
                        DataFlowGraph.NodeProperty.OUT_FEATURES
                    ]
                )
                is not None
            )
            # add edge to other nodes
            x = self.add_edge(
                ref.node_id_,
                node_id,
                key=name,
                **{
                    DataFlowGraph.EdgeProperty.NAME: name,
                    DataFlowGraph.EdgeProperty.KEY: ref.key_,
                },
            )
            # update the depth of the node based on the depth of the source node
            depth = max(
                depth,
                self.nodes[ref.node_id_][DataFlowGraph.NodeProperty.DEPTH] + 1,
            )

        # update the depth of the node
        self.nodes[node_id][DataFlowGraph.NodeProperty.DEPTH] = depth
        # make sure the graph is a DAG
        assert nx.is_directed_acyclic_graph(self)

        return node_id

    def get_node_output_ref(
        self, node_id: int
    ) -> FeatureRef | OutputRefs | AggregationRef:
        """Retrieves the output reference for a given node in the data flow graph.

        This method returns an appropriate output reference based on the type of the node specified by the
        given node ID. The method constructs the appropriate reference object based on the node type:

            - For source nodes, this method builds a feature reference using the output features of the node.
            - For data processor nodes, it retrieves the processor's output references type and constructs the full output reference.
            - For data aggregator nodes, it builds a data aggregation reference using the node's value type.

        Args:
            node_id (int): The ID of the node for which to retrieve the output reference.

        Returns:
            FeatureRef | OutputRefs | AggregationRef: The output reference associated with the specified node.

        Raises:
            KeyError: If the node ID does not exist in the data flow graph.
            TypeError: If the node type is not recognized.
        """
        if node_id not in self:
            raise KeyError(
                f"Node ID {node_id} does not exist in the data flow graph."
            )

        # get node
        node = self.nodes[node_id]
        node_type = node[DataFlowGraph.NodeProperty.PROCESSOR_TYPE]

        if node_type == DataFlowGraph.NodeType.SOURCE:
            features = node[DataFlowGraph.NodeProperty.OUT_FEATURES]
            # build feature reference
            return FeatureRef(
                key_=FeatureKey(),
                node_id_=node_id,
                flow_=self,
                feature_=features,
            )

        elif node_type == DataFlowGraph.NodeType.DATA_PROCESSOR:
            # get processor and output features
            proc = node[DataFlowGraph.NodeProperty.PROCESSOR]
            features = node[DataFlowGraph.NodeProperty.OUT_FEATURES]
            # build the full output reference
            return proc._out_refs_type(self, node_id, features)

        elif node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR:
            # get aggregator and build reference
            proc = node[DataFlowGraph.NodeProperty.PROCESSOR]
            return AggregationRef(
                node_id_=node_id, flow_=self, type_=proc._value_type
            )

        else:
            raise TypeError(
                f"Unrecognized node type {node_type} for node ID {node_id}."
            )

    def dependency_graph(self, nodes: set[int]) -> DataFlowGraph:
        """Generate the dependency subgraph for a given node.

        This method generates a subgraph containing all nodes that the given
        set of nodes depend on directly or indirectly.

        Args:
            nodes (set[int]): The node IDs for which to generate the dependency graph.

        Returns:
            DataFlowGraph: A subgraph representing the dependencies.
        """
        visited = set()
        nodes = nodes.copy()
        # search through dependency graph
        while len(nodes) > 0:
            node = nodes.pop()
            visited.add(node)
            nodes.update(self.predecessors(node))

        return self.subgraph(visited)
