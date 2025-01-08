"""Defines the structure of data flow graphs and their components.

This module provides the :class:`DataFlowGraph` class and related components, which represent
the structure of a data processing workflow. The graph consists of nodes (data processors)
and edges (data flow between processors).
"""


from __future__ import annotations

import operator
from enum import Enum
from functools import wraps
from itertools import groupby
from typing import Any, Hashable
from uuid import UUID

import networkx as nx
import numpy as np
import pyarrow as pa

from .abstract import AbstractDataFlowGraph
from .features.dtypes import (
    MappingType,
    SequenceType,
    Type,
    build_dtype_from_python_object,
    build_type_from_dict,
    cast_dtype,
)
from .features.engine import FeatureEngine
from .features.features import build_feature_from_reference
from .features.reference import ConcreteReference
from .nodes.aggregator import BaseDataAggregator
from .nodes.augmentor import BaseDataAugmentor
from .nodes.base import BaseNode, RunContext
from .nodes.collect import CollectNode
from .nodes.const import ConstNode
from .nodes.processor import BaseDataProcessor
from .registry.config import AutoConfigurable
from .typing import NodeId, PartitionId
from .utils import NestedType, map_recursive

rng = np.random.Generator(np.random.PCG64(42))


def random_uuid() -> UUID:
    """Generates a random UUID using a numpy random number generator.

    This function uses the random number generator to produce 16 random bytes,
    which are then used to create a UUID (version 4). The resulting UUID is
    unique and reproducible based on the underlying random byte generation.

    Returns:
        UUID: A randomly generated UUID created from 16 random bytes.
    """
    return UUID(bytes=rng.bytes(16))


def _compute_node_depth(g: nx.DiGraph) -> dict[Hashable, int]:
    """Compute the depth of each node in the graph.

    This function calculates the depth for each node in the provided networkx
    graph, where depth is defined as the shortest path length from a designated
    root node to each node in the graph. If the graph is a directed acyclic graph
    (DAG), the root is typically a node with no incoming edges.

    Args:
        g (nx.Graph): The networkx graph for which to compute node depths.

    Returns:
        dict: A dictionary mapping each node to its depth in the graph.

    Raises:
        ValueError: If the graph contains cycles.
        ValueError: If a root node cannot be determined.
    """
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Graph contains a cycle.")

    node_depths = {}
    # trafers graph in topological order and compute the node depth
    for partition_node in nx.topological_sort(g):
        node_depths[partition_node] = max(
            (node_depths[parent] + 1 for parent in g.predecessors(partition_node)),
            default=0,
        )

    return node_depths


def _build_dependency_graph(
    g: nx.DiGraph, nodes: set[Hashable], stop_nodes: set[Hashable] = set()
) -> nx.DiGraph:
    """Build a subgraph containing all dependencies for a given set of nodes.

    This function constructs a subgraph from a directed graph :code:`G` by including
    all nodes that are dependencies (predecessors) of the specified :code:`nodes`,
    except those that reach any node in :code:`stop_nodes`. The resulting subgraph
    consists of the nodes in :code:`nodes` and all their upstream dependencies, but
    traversal stops at nodes in :code:`stop_nodes`.

    Args:
        g (nx.DiGraph): The original directed graph.
        nodes (set[Hashable]): A set of nodes for which to build the dependency subgraph.
        stop_nodes (set[Hashable]): A set of nodes that serve as cut-off points in
            the traversal. Dependencies beyond these nodes will not be included
            in the subgraph.

    Returns:
        nx.DiGraph: A subgraph of :code:`G` containing the specified nodes and their dependencies,
        with paths beyond :code:`stop_nodes` excluded.

    Raises:
        AssertionError: If any node in :code:`nodes` is not present in :code:`G`.
    """
    assert all(node in g for node in nodes), "All nodes must be present in the graph 'G'."

    visited = set()
    to_visit = nodes.copy()

    # Traverse dependencies
    while to_visit:
        node = to_visit.pop()
        if node not in visited:
            visited.add(node)
            # Only add predecessors if the current node is not a stop node
            if node not in stop_nodes:
                to_visit.update(pred for pred in g.predecessors(node) if pred not in visited)

    return g.subgraph(visited)


class DataFlowGraph(nx.MultiDiGraph, AbstractDataFlowGraph):
    """A multi-directed graph representing a data flow of data processors.

    This class is used internally to define a directed acyclic graph (DAG)
    where nodes represent data processors of `BaseDataProcessor` type, and
    edges define the data flow between these processors.
    """

    class GraphAttribute(str, Enum):
        """Enum representing properties of the data flow graph."""

        SRC_NODE_ID = "src_node_id"
        """
        Property representing the source node ID.

        This property identifies the source node ID of an edge in the graph.
        """

    class Partition(str, Enum):
        """Enum representing predefined partitions in the data flow graph."""

        CONST = "CONSTANT"
        """
        Represents the partition containing all constant nodes.

        This partition includes nodes that hold constant values used in the
        data processing flow. This convers the actual constant nodes introducing
        constant values to the flow, as well as computations on only constant
        values.
        """

        DEFAULT = "DEFAULT"
        """
        Represents the default partition for nodes.

        This partition is assigned to the source node and is inherited by
        its sub-graph.
        """

        AGGREGATED = "AGGREGATED"
        """
        Represents the partition containing aggregated values.

        This partition includes all aggregated values in a data flow, i.e.
        nodes that process the output of aggregator nodes.
        """

    class NodeType(str, Enum):
        """Enum representing types of nodes in the data flow graph."""

        SOURCE = "SOURCE_NODE"
        """
        Represents a source node in the data flow graph.

        This type of node acts as the starting point of the data flow graph,
        typically representing raw input data sources.
        """

        CONST = "CONST_NODE"
        """
        Represents a constant node in the data flow graph.

        This type of node introduces constant values into the data flow, serving
        as fixed inputs to the subsequent processing stages.
        """

        COLLECT = "COLLECT_NODE"
        """
        Represents a collect node in the data flow graph.

        This type of node collects features from multiple upstream nodes into a single
        (nested) feature. This enables the combination of outputs from various sources
        or transformations into a unified structure, which can be further processed
        downstream.
        """

        CAST = "CAST_NODE"
        """
        Represents a cast node in the data flow graph.

        This type of node is responsible for type conversion within the data flow.
        It transforms data from one type to another, ensuring compatibility between
        different nodes or preparing the data for specific processing requirements.
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

        This type of node is responsible for aggregating samples. Aggregator
        nodes typically perform dataset-wide computations.
        """

        DATA_AUGMENTOR = "DATA_AUGMENTOR_NODE"
        """
        Represents a data augmentor node in the data flow graph.

        This type of node is responsible for modifying the dataset by generating
        new samples from existing ones or filtering out certain samples. Data
        augmentor nodes are used to expand or contract the dataset.
        """

    class NodeAttribute(str, Enum):
        """Enum representing properties of a node in the data flow graph."""

        NODE_OBJ = "node_object"
        """
        The object associated with the node.

        The value of this property is dependent on the type of node. For
        nodes of type :class:`NodeType.DATA_PROCESSOR`, this property
        refers to the processor instance of the node.
        """

        NODE_TYPE = "node_type"
        """
        Indicates the type of node.

        Type: :class:`NodeType`

        This property indicates the type of data processor. It helps in categorizing
        and identifying the nature of the processor in the data flow graph.
        """

        IN_FEATURE_TYPE = "in_feature_type"
        """
        Represents the input features associated with the node.

        Type: :class:`MappingType`

        This property contains the input features required by the data processor,
        in the form of a mapping type instance. It defines the structure and types
        of the inputs expected by the node.
        """

        OUT_FEATURE_TYPE = "out_feature_type"
        """
        Represents the output features associated with the node.

        Type: :class:`Type`

        This property contains the output feature produced by the data processor,
        as a :code:`Type` instance. It defines the structure and types of data that
        is generated by this node.
        """

        PARTITION = "partition"
        """
        Represents the partition to which the node belongs.

        Type: :class:`PartitionId`

        This property indicates the specific partition of the data flow graph that
        the node is part of, which can be used to group nodes by different semantics.
        """

        DEPTH = "depth"
        """
        Represents the depth of the node within the data flow graph.

        Type: :class:`int`

        This property indicates the level of the node in the graph, with the root
        node having a depth of 0. It is used to understand the hierarchical position
        of the node relative to other nodes in the data flow.
        """

    @wraps(nx.MultiDiGraph)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DataFlowGraph.

        Args:
            *args: Positional arguments forwarded to the init of the :class:`MultiDiGraph`.
            **kwargs: Keyword arguments forwarded to the init of the :class:`MultiDiGraph`.
        """
        super(DataFlowGraph, self).__init__(*args, **kwargs)

        # set default source node id
        if DataFlowGraph.GraphAttribute.SRC_NODE_ID not in self.graph:
            self.graph[DataFlowGraph.GraphAttribute.SRC_NODE_ID] = None

        # reset source node id for subgraphs
        if (self.src_node_id is not None) and (self.src_node_id not in self):
            self.graph[DataFlowGraph.GraphAttribute.SRC_NODE_ID] = None

    @property
    def src_node_id(self) -> None | NodeId:
        """Get the source node ID of the data flow graph.

        This property returns the source node ID associated with the data flow graph.
        The source node is the entrypoint for inputs to the data flow.

        Returns:
            None | NodeId: The uuid of the source node. :code:`None` if graph has no source node.
        """
        return self.graph[DataFlowGraph.GraphAttribute.SRC_NODE_ID]

    @property
    def src_dtype(self) -> MappingType:
        """Get the source data type.

        This property retrieves the data type of the source node in the data flow graph.
        The source data type defines the structure and types of features expected by the
        source node, which serve as the initial inputs to the data flow graph.

        Returns:
            MappingType: The data type of the source node.
        """
        return self.nodes[self.src_node_id][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]

    @property
    def depth(self) -> int:
        """Computes the total depth of the data flow graph.

        The depth is defined as the maximum level of any node in the graph, where the root
        node has a depth of 0. This property calculates the depth by finding the maximum
        depth attribute among all nodes in the graph.

        Returns:
            int: The total depth of the graph.
        """
        return max(nx.get_node_attributes(self, DataFlowGraph.NodeAttribute.DEPTH).values()) + 1

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
        depths = nx.get_node_attributes(self, DataFlowGraph.NodeAttribute.DEPTH)
        layers = groupby(sorted(self, key=depths.get), key=depths.get)
        # find larges layer in graph
        return max(len(list(layer)) for _, layer in layers)

    def build_partition_graph(self) -> nx.DiGraph:
        """Construct a partition graph from the data flow graph.

        This method builds a directed graph where each node represents a partition
        within the data flow graph, and edges represent the flow of data between
        these partitions. The resulting partition graph is required to have a tree
        structure, where each partition (except the root) has a single parent partition.

        Note that while the data flow graph itself is asyclic, the partition graph
        doesn't need to be.

        Note that while the partition graph includes the constant partition, it does not
        model the flow of constants to other partitions, i.e. the constant partition is
        not isolated from the remaining partition graph. Reason for this design choice is
        that constants are special in that they can be used in any partition.

        Returns:
            nx.DiGraph: A directed graph representing the partitioned data flow.

        Raises:
            AssertionError: If the resulting partition graph is not a tree (i.e., any node
                has more than one incoming edge).
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(
            [
                DataFlowGraph.Partition.CONST.value,
                DataFlowGraph.Partition.DEFAULT.value,
            ]
        )

        # TODO: the aggregated partition is a special case since multiple
        #       independent partitions can point into it, this is not captured
        #       in the partition graph yet and would also break the tree structure
        #       asserted below
        for node_id, attrs in self.nodes(data=True):
            src_partition = attrs[DataFlowGraph.NodeAttribute.PARTITION]
            tgt_partition = self.get_node_output_partition(node_id)

            # make sure the source partition is contained in the partition graph
            assert (
                src_partition in graph
            ), f"The source partition '{src_partition}' is not present in the partition graph."

            # don't include edges from the constant partition in partition graph
            if src_partition == DataFlowGraph.Partition.CONST:
                continue

            # add the target partition to the partition graph
            if tgt_partition not in graph:
                graph.add_node(tgt_partition)

            # only add a connection if there is no path from the source to the
            # target partition yet
            if (src_partition != tgt_partition) and not nx.has_path(
                graph, src_partition, tgt_partition
            ):
                graph.add_edge(src_partition, tgt_partition)

        return graph

    def add_node(
        self,
        node_obj: Any,
        node_type: DataFlowGraph.NodeType,
        inputs: dict[str, ConcreteReference],
        output_type: Type,
        node_id: None | NodeId = None,
    ) -> ConcreteReference:
        """Adds a new node to the data flow graph.

        This function creates a node in the graph with specified attributes such as
        type, inputs, and output data type. It computes the node's partition, depth,
        and input type dynamically based on its inputs and ensures the graph remains
        a Directed Acyclic Graph (DAG) after adding the node. The function also
        establishes dependency edges between the new node and its input nodes.

        Args:
            node_obj (Any): The object associated with the node.
            node_type (DataFlowGraph.NodeType): The type of the node.
            inputs (dict[str, ConcreteReference]): A dictionary mapping input names to
                :class:`ConcreteReference` instances that represent dependencies of this
                node on other nodes in the graph.
            output_type (Type): The output data type produced by this node.
            node_id (None | NodeId): The unique identifier for the node. If :code:`None`,
                a random UUID will be generated. Defaults to :code:`None`.

        Returns:
            ConcreteReference: A reference to the newly created node.

        """
        # infer the output partition of the node from the node type and input references
        partition = self.infer_node_partition(node_type, list(inputs.values()))
        # aggregated partition currently only supports processor type nodes
        if (partition == DataFlowGraph.Partition.AGGREGATED) and (
            node_type not in {DataFlowGraph.NodeType.COLLECT, DataFlowGraph.NodeType.DATA_PROCESSOR}
        ):
            raise NotImplementedError(
                f"Aggregator outputs may only be processed by data processors "
                f"or collect operations, got {node_type}."
            )

        # compute the depth of the node in the graph based on it's input references
        depth = (
            0
            if inputs is None
            else max(
                (
                    self.nodes[ref._node_id][DataFlowGraph.NodeAttribute.DEPTH] + 1
                    for ref in inputs.values()
                ),
                default=0,
            )
        )

        # build the input data type from the input references
        input_type = MappingType.construct(
            {key: self.get_dtype_from_reference(ref) for key, ref in inputs.items()}
        )

        # create a random node id if no was given
        node_id = node_id if node_id is not None else str(random_uuid())
        # add the node to the graph
        super(DataFlowGraph, self).add_node(
            node_id,
            **{
                DataFlowGraph.NodeAttribute.NODE_OBJ.value: node_obj,
                DataFlowGraph.NodeAttribute.NODE_TYPE.value: node_type,
                DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE.value: input_type,
                DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE.value: output_type,
                DataFlowGraph.NodeAttribute.PARTITION.value: partition,
                DataFlowGraph.NodeAttribute.DEPTH.value: depth,
            },
        )

        # add dependency edges to graph
        for name, ref in inputs.items():
            # add the edge
            self.add_edge(
                ref._node_id,
                node_id,
                key=name,
            )

        # make sure the graph is a DAG
        assert nx.is_directed_acyclic_graph(self)

        return ConcreteReference(node_id, self)

    def add_source_node(self, data_type: Type, node_id: None | NodeId = None) -> ConcreteReference:
        """Add a the source node to the graph.

        This method adds a source node to the graph, which acts as the initial
        data provider for the data flow.

        Args:
            data_type (Type): The data type representing the source features.
            node_id (None | NodeId): The id of the node, defaults to a random uuid.

        Returns:
            ConcreteReference: A reference object to the source node.

        Raises:
            AssertionError: If the graph already contains a source node
        """
        # make sure the graph has no source node yet
        if self.src_node_id is not None:
            raise RuntimeError("Graph already contains a source node.")

        # add the node to the graph
        ref = self.add_node(
            node_obj=None,
            node_type=DataFlowGraph.NodeType.SOURCE,
            inputs={},
            output_type=data_type,
            node_id=node_id,
        )
        # save source node id as graph attribute
        self.graph[DataFlowGraph.GraphAttribute.SRC_NODE_ID] = ref._node_id
        # return the reference to the source node
        return ref

    def add_const_node(
        self, value: Any, dtype: Type, node_id: None | NodeId = None
    ) -> ConcreteReference:
        """Adds a constant node to the data flow graph.

        This function creates a node representing a constant value in the graph. The
        value is wrapped in a :code:`PyArrow` array of the specified data type.

        Args:
            value (Any): The constant value to be added to the graph.
            dtype (Type): The data type of the constant value.
            node_id (None | NodeId): A unique identifier for the node. If :code:`None`,
                a random UUID is generated. Defaults to :code:`None`.

        Returns:
            ConcreteReference: A reference to the newly created constant node.
        """
        # make sure the data type matches the value
        array = pa.array([value], type=dtype.arrow_type)
        # create a random node id if not provided
        node_id = node_id if node_id is not None else str(random_uuid())

        # add the node to the graph
        return self.add_node(
            node_obj=ConstNode(value=array),
            node_type=DataFlowGraph.NodeType.CONST,
            inputs={},
            output_type=dtype,
            node_id=node_id,
        )

    def add_collect_node(
        self, collect: NestedType[ConcreteReference], node_id: None | NodeId = None
    ) -> ConcreteReference:
        """Adds a collect node to the data flow graph.

        A collect node combines features from other nodes, organizing them into a nested
        structure as specified by the :code:`collect` argument. It enables combining outputs
        from multiple nodes into a single, structured output.

        Args:
            collect (NestedType[ConcreteReference]): A nested structure (e.g., lists, dictionaries)
                containing :class:`ConcreteReference` objects that specify the features to collect.
            node_id (None | NodeId): A unique identifier for the node.
                If :code:`None`, a random UUID is generated. Defaults to :code:`None`.

        Returns:
            ConcreteReference: A reference to the newly created collect node.
        """
        inputs: dict[str, ConcreteReference] = {}
        # extract flat inputs to collect node from structure
        map_recursive(
            lambda p, r: (
                None
                if not isinstance(r, ConcreteReference)
                else operator.setitem(inputs, ".".join(map(str, p)), r)
            ),
            collect,
        )

        # build the nested value lookup structure
        lookup: NestedType[str] = map_recursive(
            lambda p, v: ".".join(map(str, p)) if isinstance(v, ConcreteReference) else v, collect
        )

        # create the collect node object
        obj = CollectNode(lookup=lookup)

        # build the output type of the collect operation
        output_type, required_casts = obj.build_output_type(self, inputs)
        # add required cast nodes to graph
        for key, dtype in required_casts.items():
            inputs[key] = self.add_cast_node(inputs[key], dtype)

        # add the node object
        return self.add_node(
            node_obj=obj,
            node_type=DataFlowGraph.NodeType.COLLECT,
            inputs=inputs,
            output_type=output_type,
            node_id=node_id,
        )

    def add_collect_node_with_constants(
        self,
        collect: NestedType[ConcreteReference | Any],
        dtype: None | Type = None,
        node_id: None | NodeId = None,
    ) -> ConcreteReference:
        """Add a collect node and all contained constants to a data flow graph.

        This function processes a nested structure of references and constants,
        adding constants to the graph as nodes if necessary. It then combines the
        resolved references and constants into a single :code:`collect` node, returning
        a reference to this node.

        Args:
            graph (DataFlowGraph): The data flow graph to which the nodes will be added.
            collect (NestedType[ConcreteReference | Any]): A nested structure containing constants
                (e.g., integers, floats, or strings) and/or references to existing nodes in
                the graph.
            dtype (None | Type): The expected data type of the :code:`collect` structure. If
                :code:`None`, the function attempts to infer the type where possible.
            node_id (None | NodeId): A unique identifier for the node. If :code:`None`, a random
                UUID is generated. Defaults to :code:`None`.

        Returns:
            ConcreteReference: A reference to the :code:`collect` node added to the graph.

        Raises:
            AssertionError: If the provided :code:`dtype` does not match the structure
                of :code:`collect`.
            TypeError: If an unsupported type is found in :code:`collect`.
        """

        def add_constants(
            val: NestedType[ConcreteReference | str | int | float], dtype: None | Type = None
        ) -> NestedType[ConcreteReference]:
            if isinstance(val, dict):
                assert (dtype is None) or isinstance(dtype, MappingType)

                return {
                    key: add_constants(item, dtype[key] if dtype is not None else None)
                    for key, item in val.items()
                }

            elif isinstance(val, (list, tuple)):
                assert (dtype is None) or isinstance(dtype, SequenceType)

                if dtype is None:
                    if len(val) == 0:
                        raise NotImplementedError()
                    # try to infer the dtype from the reference instances in the sequence
                    if any(isinstance(r, ConcreteReference) for r in val):
                        ref = next(r for r in val if isinstance(r, ConcreteReference))
                        dtype = self.get_dtype_from_reference(ref)

                else:
                    # otherwise use the value type from the given dtype
                    dtype = dtype.value_type

                # recurse on all items in the sequence
                return type(val)([add_constants(item, dtype) for item in val])

            elif not isinstance(val, ConcreteReference):
                # add the constant node
                return self.add_const_node(
                    val, dtype=dtype if dtype is not None else build_dtype_from_python_object(val)
                )

            elif isinstance(val, ConcreteReference):
                return val

            else:  # pragma: not covered
                raise TypeError(f"Unsupported type encountered in 'collect': {val}.")

        # add all constants to the graph
        collect = add_constants(collect, dtype)

        # add the collect node to the graph and return the reference to it
        return self.add_collect_node(collect, node_id)

    def add_cast_node(
        self, ref: ConcreteReference, dtype: Type, node_id: None | NodeId = None
    ) -> ConcreteReference:
        """Add a cast node to the data flow graph.

        This method adds a cast node, which transforms data from one type to another,
        ensuring compatibility between nodes or preparing the data for specific processing
        requirements. The method validates type compatibility and computes the resulting
        type if the cast is feasible.

        Args:
            ref (ConcreteReference): A reference to the input data to be cast.
            dtype (Type): The target type to which the input data should be cast.
            node_id (None | NodeId): An optional unique identifier for the node. If not
                provided, a random UUID is generated. Defaults to :code:`None`.

        Returns:
            ConcreteReference: A reference to the added cast node, allowing access to its output.

        Raises:
            RuntimeError: If the source and target types are incompatible or if other
                casting constraints are violated.
        """
        try:
            dtype = cast_dtype(self.get_dtype_from_reference(ref), dtype)
        except RuntimeError as e:
            raise RuntimeError(
                f"Error while casting from {self.get_dtype_from_reference(ref)} to {dtype}."
            ) from e

        # create a random node id if not provided
        node_id = node_id if node_id is not None else str(random_uuid())
        # add the node to the graph
        return self.add_node(
            node_obj=None,
            node_type=DataFlowGraph.NodeType.CAST,
            inputs={"value": ref},
            output_type=dtype,
            node_id=node_id,
        )

    def add_compute_node(
        self,
        obj: BaseNode,
        inputs: dict[str, ConcreteReference],
        node_id: None | NodeId = None,
    ) -> ConcreteReference:
        """Add a compute node to the data flow graph.

        This method adds a compute node, which performs a transformation or computation
        on input features, to the data flow graph. The node type is dynamically determined
        based on the class of the provided node object, and appropriate edges are created
        to define the data flow.

        Args:
            obj (BaseNode): The compute node object, which defines the transformation logic.
                Supported subclasses include :class:`BaseDataProcessor`,
                :class:`BaseDataAggregator`, and :class:`BaseDataAugmentor`.
            inputs (dict[str, ConcreteReference]): A mapping of input names to references for input
                features consumed by this compute node.
            node_id (None | NodeId): An optional unique identifier for the node. If not
                provided, a random UUID is generated.

        Returns:
            ConcreteReference: A reference instance to the added compute node, allowing access
            to its output features.

        Raises:
            AssertionError: If the node object is invalid or its type cannot be determined.
            RuntimeError: If any input reference do not belong to this data flow.
        """
        # get processor type
        node_type = (
            DataFlowGraph.NodeType.SOURCE
            if obj is None
            else DataFlowGraph.NodeType.DATA_PROCESSOR
            if isinstance(obj, BaseDataProcessor)
            else DataFlowGraph.NodeType.DATA_AGGREGATOR
            if isinstance(obj, BaseDataAggregator)
            else DataFlowGraph.NodeType.DATA_AUGMENTOR
            if isinstance(obj, BaseDataAugmentor)
            else None
        )
        # make sure the object is valid
        assert node_type is not None, f"Invalid node type {type(obj)}."

        # build the input features
        input_features = {key: build_feature_from_reference(ref) for key, ref in inputs.items()}
        # create a type validation engine instance
        name = f"DataFlowGraph.add_node({type(obj).__qualname__})"
        with FeatureEngine(name, obj.config, obj.signature) as engine:
            # validate the input features to the node
            engine.validate_signature()
            engine.validate_arguments(**input_features)
            # get the output feature type of the node for the given inputs
            feature_type = engine.build_return_feature(input_features).dtype

        return self.add_node(
            node_obj=obj,
            node_type=node_type,
            inputs=inputs,
            output_type=feature_type,
            node_id=node_id,
        )

    def get_node_output_partition(self, node_id: NodeId) -> PartitionId:
        """Determine the output partition for a given node in the data flow graph.

        This method determines the partition that the output of a specified node
        belongs to, based on the node's type. Different types of nodes may direct
        their output to different partitions, reflecting their role in the data flow:

        - :code:`DATA_AGGREGATOR`: Outputs always point to the :code:`AGGREGATED` partition,
          even though the aggregator node itself is not part of this partition.

        - :code:`DATA_AUGMENTOR`: Outputs always point to their own partition, using the
          node's ID as the partition name. Like aggregators, augmentors are not part of
          the partition they point to.

        - Other node types: Outputs remain within the partition specified by the
          node's :code:`PARTITION` attribute.

        Args:
            node_id (NodeId): The ID of the node for which to determine the output partition.

        Returns:
            PartitionId: The partition that the node's output will be directed to.
        """
        input_node = self.nodes[node_id]
        input_node_type = input_node[DataFlowGraph.NodeAttribute.NODE_TYPE]

        if input_node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR:
            # data aggregator outputs always point into
            # the aggregated partition while the aggregator
            # node itself is not part of the aggregated partition
            return DataFlowGraph.Partition.AGGREGATED.value

        elif input_node_type == DataFlowGraph.NodeType.DATA_AUGMENTOR:
            # build a mock run context
            ctx = RunContext(
                node_id=node_id,
                index=[],
                rank=0,
                input_type=input_node[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE],
                output_type=input_node[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
            )
            # infer the output partition of the aggregation operation
            node_obj = input_node[DataFlowGraph.NodeAttribute.NODE_OBJ]
            partition = input_node[DataFlowGraph.NodeAttribute.PARTITION]
            return node_obj.infer_output_partition(ctx, partition)

        else:
            # other node types don't transition between partitions
            return input_node[DataFlowGraph.NodeAttribute.PARTITION]

    def infer_node_partition(
        self, node_type: DataFlowGraph.NodeType, refs: list[ConcreteReference]
    ) -> PartitionId:
        """Infer the appropriate partition for a given node.

        This method determines the partition to which a node belongs, based on its
        type and the partitions of its input references.

        The method handles different node types as follows:
        - :code:`SOURCE`: Assigned to the default partition.
        - :code:`CONST`: Assigned to the constant partition.
        - Other node types: The partition is inferred based on the partitions of
        the input references.

        The partition is inferred from the inputs as follows:
        - If the input references are from constant partitions, the node is also assigned
        to the constant partition.
        - If any input comes from an aggregated partition, the node is assigned to the
        aggregated partition.
        - If multiple candidate partitions are found, the method selects the partition
        that is deepest in the partition graph.

        If the resulting partitions include a mix of independent partitions, an error is raised.
        Independent partitions are those that are not connected in the partition graph, meaning
        the resulting partition graph would not form a valid tree structure. This ensures that
        the partition graph remains a tree, with each partition having a single parent and no
        cycles or disjoint components.

        Args:
            node_type (DataFlowGraph.NodeType): The type of the node for which to infer
                the partition.
            refs (list[ConcreteReference]): A list of input references associated with the node.

        Returns:
            PartitionId: The inferred partition for the node.

        Raises:
            AssertionError: If no input references are provided.
            RuntimeError: If conflicting partitions are mixed.
        """
        if node_type == DataFlowGraph.NodeType.SOURCE:
            # source node is added to the default partition
            return DataFlowGraph.Partition.DEFAULT.value

        if node_type == DataFlowGraph.NodeType.CONST:
            # contants are added to the constant partition
            return DataFlowGraph.Partition.CONST.value

        # partition could not be inferred
        assert len(refs) > 0, "Partition cannot be inferred for nodes without any input references."

        # get the input partitions
        candidate_partitions = {self.get_node_output_partition(ref._node_id) for ref in refs}

        if candidate_partitions == {DataFlowGraph.Partition.CONST}:
            # if all inputs come from the constant partition, then this node
            # is also part of the constant partition
            return DataFlowGraph.Partition.CONST.value

        if DataFlowGraph.Partition.AGGREGATED in candidate_partitions:
            # if the inputs come directly from an aggregator or from the
            # aggregated partition, then stay in the aggregated partition

            if (
                len(
                    candidate_partitions
                    - {
                        DataFlowGraph.Partition.AGGREGATED,
                        DataFlowGraph.Partition.CONST,
                    }
                )
                != 0
            ):
                raise RuntimeError("Cannot mix aggregated and non-aggregated features.")

            return DataFlowGraph.Partition.AGGREGATED.value

        # remove the constant partition from the set of candidates
        # if there is any other partition to select from
        candidate_partitions -= {DataFlowGraph.Partition.CONST}

        if len(candidate_partitions) == 1:
            return next(iter(candidate_partitions))

        # build the partition graph and compute the depth of each node
        p_graph = self.build_partition_graph()
        p_depths = _compute_node_depth(p_graph)

        # select the candidate partition that is deepest in the partition graph
        # as it is the only one that can consume all candidate partitions
        candidate = max(candidate_partitions, key=p_depths.get)

        p_dep_graph = _build_dependency_graph(p_graph, {candidate})
        # build the set of allowed partitions
        # note that the constant partition is always allowed
        allowed_partitions = set(p_dep_graph.nodes())
        allowed_partitions.add(DataFlowGraph.Partition.CONST.value)
        # make sure only allowed partitions are used
        if not candidate_partitions.issubset(allowed_partitions):
            raise RuntimeError("Cannot mix independent partitions.")

        return candidate

    def get_dtype_from_reference(self, ref: ConcreteReference) -> Type:
        """Helper function to get the data type of a referenced feature.

        Args:
            ref (ConcreteReference): The reference for which to get the type.

        Returns:
            Type: The data type of the referenced feature.

        Raises:
            RuntimeError: If the reference does not refer to this graph.
            RuntimeError: If the reference's node ID is not contained in the graph.
        """
        if ref._graph is not self:
            raise RuntimeError("ConcreteReference does not belong to this graph.")

        if ref._node_id not in self.nodes:
            raise RuntimeError(f"Node with ID '{ref._node_id}' is not contained in the graph.")

        # get the output type of the referenced node
        return self.nodes[ref._node_id][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]

    def dependency_graph(
        self, nodes: set[NodeId], stop_nodes: set[NodeId] = set()
    ) -> DataFlowGraph:
        """Generate the dependency subgraph for a given set of nodes.

        This method generates a subgraph that includes all nodes that the specified
        :code:`nodes` depend on, either directly or indirectly, up to any defined cut-off
        points.

        Args:
            nodes (set[NodeId]): The node IDs for which to generate the dependency graph.
            stop_nodes (set[NodeId]): A set of node IDs that serve as cut-off points in
                the traversal. When a dependency chain reaches any node in :node:`stop_nodes`,
                it stops there, excluding that node's dependencies from the subgraph.
                This allows limiting the scope of the dependency graph by excluding
                deeper dependencies beyond these nodes.

        Returns:
            DataFlowGraph: A subgraph representing the dependencies of the specified
            :code:`nodes`, excluding paths beyond any nodes in :code:`stop_nodes`.
        """
        return _build_dependency_graph(self, nodes, stop_nodes)

    def get_partition(self, partition: PartitionId) -> DataFlowGraph:
        """Extract a subgraph containing only nodes from a specific partition.

        This method creates a subgraph from the current graph by selecting
        nodes that belong to a specified partition.

        Args:
            partition (PartitionId): The partition identifier.

        Returns:
            DataFlowGraph: The subgraph containing nodes of the specified partition.
        """
        # get all nodes to the provided partition
        partition = [
            i
            for i, data in self.nodes(data=True)
            if data[DataFlowGraph.NodeAttribute.PARTITION] == partition
        ]
        # build the sub-graph of only the provided partition
        return self.subgraph(partition)

    def drop_partition(self, partition: PartitionId) -> DataFlowGraph:
        """Drop a specified partition from the graph.

        This method creates a subgraph from the current graph by excluding
        nodes that belong to a specified partition.

        Args:
            partition (PartitionId): The partition identifier.

        Returns:
            DataFlowGraph: The subgraph excluding nodes of the specified partition.
        """
        # get all nodes to the provided partition
        remainder = [
            i
            for i, data in self.nodes(data=True)
            if data[DataFlowGraph.NodeAttribute.PARTITION] != partition
        ]
        # build the sub-graph of only the provided partition
        return self.subgraph(remainder)

    def subgraph_in_edges(self, subgraph: DataFlowGraph) -> list[tuple[int, int, str]]:
        """Get incoming edges to a subgraph from nodes outside the subgraph.

        This method returns a list of edges that point to nodes within the
        specified subgraph from nodes outside the subgraph.

        Args:
            subgraph (DataFlowGraph): The subgraph of interest.

        Returns:
            list[tuple[int, int, str]]: The incoming edges to the subgraph.
        """
        return [e for e in self.in_edges(subgraph, keys=True) if e[0] not in subgraph]

    def subgraph_out_edges(self, subgraph: DataFlowGraph) -> list[tuple[int, int, str]]:
        """Get outgoing edges from a subgraph to nodes outside the subgraph.

        This method returns a list of edges that point from nodes within the
        specified subgraph to nodes outside the subgraph.

        Args:
            subgraph (DataFlowGraph): The subgraph of interest.

        Returns:
            list[tuple[int, int, str]]: The outgoing edges from the subgraph.
        """
        return [e for e in self.out_edges(subgraph, keys=True) if e[1] not in subgraph]

    def recompute_depths(self) -> None:
        """Recompute the depth of all nodes in the data flow graph.

        This method recalculates the depth of each node based on the topological
        order of the graph. The depth of a node is defined as the length of the
        longest path from the source node to the node.
        """
        node_depths = _compute_node_depth(self)
        nx.set_node_attributes(self, node_depths, DataFlowGraph.NodeAttribute.DEPTH)

    def attach_graph_to_node(self, node_id: NodeId, graph: DataFlowGraph) -> DataFlowGraph:
        """Attach a data flow graph to a specific node within the current graph.

        This method embeds a separate data flow graph (:code:`graph`) into the current graph by
        attaching the source node of the :code:`graph` to the node identified by :code:`node_id`
        in the current graph. During this process, the nodes and edges of the `graph` are
        integrated into the current graph, preserving their relationships, configurations and
        node ids.

        Parameters:
            node_id (NodeId): The ID of the target node in the current graph to which
                the source node of the :code:`graph` will be attached.
            graph (DataFlowGraph): The data flow graph to be attached to the current graph.

        Returns:
            DataFlowGraph: The updated graph after the attachment.

        Raises:
            RuntimeError: If :code:`node_id` is not present in the current graph.
            NotImplementedError: If the method encounters an unsupported node type in the `graph`.
        """
        # TODO: implement tests

        if node_id not in self.nodes:
            raise RuntimeError(f"Node ID {node_id} does not exist in the current graph.")

        # mapping node ids of the 'graph' to references in the new graph 'h'
        node_id_mapping: dict[NodeId, ConcreteReference] = {
            graph.src_node_id: ConcreteReference(node_id, _graph=self)
        }

        for node_id in nx.topological_sort(graph):
            node_attrs = graph.nodes[node_id]
            node_obj = node_attrs[DataFlowGraph.NodeAttribute.NODE_OBJ]
            node_type = node_attrs[DataFlowGraph.NodeAttribute.NODE_TYPE]
            # collect all input references to the current node
            inputs = {
                key: node_id_mapping[in_node_id]
                for in_node_id, _, key in graph.in_edges(node_id, keys=True)
            }

            # handle different node types
            if node_type == DataFlowGraph.NodeType.SOURCE:
                pass

            elif node_type == DataFlowGraph.NodeType.COLLECT:

                def resolve(_, k):
                    return inputs[k] if isinstance(k, str) else k

                collect = map_recursive(resolve, node_obj.config.lookup)
                node_id_mapping[node_id] = self.add_collect_node(collect, node_id=node_id)

            elif node_type == DataFlowGraph.NodeType.CONST:
                node_id_mapping[node_id] = self.add_const_node(
                    node_obj.config.value.to_pylist()[0],
                    dtype=node_attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                    node_id=node_id,
                )

            elif node_type == DataFlowGraph.NodeType.CAST:
                node_id_mapping[node_id] = self.add_cast_node(
                    inputs["value"],
                    dtype=node_attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                    node_id=node_id,
                )

            elif node_type in {
                DataFlowGraph.NodeType.DATA_PROCESSOR,
                DataFlowGraph.NodeType.DATA_AUGMENTOR,
                DataFlowGraph.NodeType.DATA_AGGREGATOR,
            }:
                node_id_mapping[node_id] = self.add_compute_node(node_obj, inputs, node_id=node_id)

            else:
                raise NotImplementedError(f"Unexpected node type: {node_type}")

        return self

    def to_dict(self) -> dict:
        """Serializes the data flow graph into a dictionary representation.

        This method generates a dictionary that represents the entire data flow graph,
        including the nodes and edges. Each node is serialized based on its type,
        with special handling for constants, casts, and various data processing nodes.
        Additionally, the feature types for each node are serialized.

        Returns:
            dict: A dictionary representation of the data flow graph, including nodes
                with their serialized objects and feature types, and links between them.
        """
        # get dictionary representation of data flow graph
        data = nx.node_link_data(self, edges="edges")

        for node in data["nodes"]:
            # serialize node object
            obj = node[DataFlowGraph.NodeAttribute.NODE_OBJ]
            node[DataFlowGraph.NodeAttribute.NODE_OBJ] = (
                None if obj is None else obj.config.to_dict()
            )
            # serialize feature types
            node[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] = node[
                DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE
            ].to_dict()
            node[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] = node[
                DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE
            ].to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict) -> DataFlowGraph:
        """Deserializes a dictionary into a data flow graph.

        This method converts a dictionary (typically one generated by :code:`to_dict`)
        back into a :class:`DataFlowGraph` instance. It handles the deserialization of
        node types, partitions, feature types, and node objects. For each node, the
        method reconstructs its associated object based on the node's type. Additionally,
        edge data is deserialized and used to re-establish links between the nodes.

        Args:
            data (dict): A dictionary representation of a data flow graph, including
                node data and links between them.

        Returns:
            DataFlowGraph: The deserialized data flow graph.
        """
        for node in data["nodes"]:
            # deserialize node types
            node[DataFlowGraph.NodeAttribute.NODE_TYPE] = DataFlowGraph.NodeType(
                node[DataFlowGraph.NodeAttribute.NODE_TYPE]
            )
            # deserialize partition
            if node[DataFlowGraph.NodeAttribute.PARTITION] in set(DataFlowGraph.Partition):
                node[DataFlowGraph.NodeAttribute.PARTITION] = DataFlowGraph.Partition(
                    node[DataFlowGraph.NodeAttribute.PARTITION]
                )
            # deserialize feature types
            node[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] = build_type_from_dict(
                node[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE]
            )
            node[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] = build_type_from_dict(
                node[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
            )
            # deserialize node objects
            obj = node[DataFlowGraph.NodeAttribute.NODE_OBJ]
            node[DataFlowGraph.NodeAttribute.NODE_OBJ] = (
                None if obj is None else AutoConfigurable.from_config_dict(obj)
            )

        return DataFlowGraph(nx.node_link_graph(data, edges="edges"))
