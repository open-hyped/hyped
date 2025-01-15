"""This module implements the Data Flow Graph Builder."""
import operator
from collections import OrderedDict
from typing import Any

import networkx as nx
import pyarrow as pa

from .abc import AbstractDataFlowGraphBuilder
from .features.dtypes import (
    DType,
    MappingType,
    SequenceType,
    build_dtype_from_python_object,
    cast_dtype,
)
from .features.engine import FeatureEngine
from .features.features import Feature, build_feature_from_reference
from .features.reference import ConcreteReference
from .graph import DataFlowGraph, _build_dependency_graph, _compute_node_depth
from .nodes.aggregator import BaseDataAggregator
from .nodes.augmentor import BaseDataAugmentor
from .nodes.base import BaseNode, RunContext
from .nodes.collect import CollectNode
from .nodes.const import ConstNode
from .nodes.processor import BaseDataProcessor
from .nodes.trace import TraceNode
from .typing import NodeId, PartitionId
from .utils import NestedType, map_recursive, random_uuid


class DataFlowGraphBuilder(AbstractDataFlowGraphBuilder):
    """The Data Flow Graph Builder.

    The builder encapsulates higher-level logic for constructing cohesive data flow graphs. It
    abstracts away the low-level details of input preparation, type conversions, and intermediate
    data representation.

    For instance, when adding a compute node, the builder may also model any necessary input
    preparation steps, such as ensuring that constant inputs are represented as constant nodes
    in the graph.
    """

    def __init__(self, graph: None | DataFlowGraph = None) -> None:
        """Initialize the builder instance.

        Args:
            graph (None | DataFlowGraph): The initial data flow graph.
                Defaults to :code:`None`
        """
        self._graph = graph if graph is not None else DataFlowGraph()

    @property
    def graph(self) -> DataFlowGraph:
        """The data flow graph."""
        return self._graph

    def _add_node_to_graph(
        self,
        node_obj: Any,
        node_type: DataFlowGraph.NodeType,
        inputs: dict[str, ConcreteReference],
        output_dtype: DType,
        node_id: None | NodeId = None,
    ) -> ConcreteReference:
        """Add a single node to the data flow graph.

        Args:
            node_obj (Any): The object associated with the node.
            node_type (DataFlowGraph.NodeType): The type of the node.
            inputs (dict[str, ConcreteReference]): A dictionary mapping input names to
                :class:`ConcreteReference` instances that represent dependencies of this
                node on other nodes in the graph.
            output_dtype (DType): The output data type produced by this node.
            node_id (None | NodeId): The unique identifier for the node. If :code:`None`,
                a random UUID will be generated. Defaults to :code:`None`.

        Returns:
            ConcreteReference: A reference to the newly added node.
        """
        # create fixed ordering over inputs
        # this is to ensure that whenever the input type is inferred from the dictionary
        # the ordering of the fields in the input mapping is consistent
        inputs = OrderedDict[str, ConcreteReference](inputs)
        input_ids = OrderedDict({key: ref._node_id for key, ref in inputs.items()})

        # create a random node id if no was given
        node_id = node_id if node_id is not None else str(random_uuid())

        # infer the node partition
        partition = self._infer_node_partition(node_type, list(input_ids.values()))
        out_partition = self._infer_node_output_partition(
            node_id, node_obj, node_type, partition, input_ids, output_dtype
        )

        # aggregated partition currently only supports processor type nodes
        if (partition == DataFlowGraph.Partition.AGGREGATED) and (
            node_type
            not in {
                DataFlowGraph.NodeType.CAST,
                DataFlowGraph.NodeType.COLLECT,
                DataFlowGraph.NodeType.DATA_PROCESSOR,
            }
        ):
            raise NotImplementedError(
                f"Aggregator outputs may only be processed by data processors "
                f"or collect operations, got {node_type}."
            )

        # trace all inputs to the node partition
        traced_input_ids = OrderedDict(
            (key, self.trace(ref, partition)._node_id) for key, ref in inputs.items()
        )

        # add the node to the graph
        node_id = self._graph.add_node(
            node_obj=node_obj,
            node_type=node_type,
            inputs=traced_input_ids,
            output_dtype=output_dtype,
            partition=partition,
            out_partition=out_partition,
            node_id=node_id,
        )

        return ConcreteReference(node_id, self._graph, self)

    def _infer_node_partition(
        self, node_type: DataFlowGraph.NodeType, in_nodes: list[NodeId]
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
            in_nodes (list[NodeId]): A list of the input node ids of the node.

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
        assert (
            len(in_nodes) > 0
        ), "Partition cannot be inferred for nodes without any input references."

        # get the input partitions
        candidate_partitions = {
            self._graph.nodes[node_id][DataFlowGraph.NodeAttribute.OUT_PARTITION]
            for node_id in in_nodes
        }

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
        p_graph = self._build_partition_graph()
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

    def _infer_node_output_partition(
        self,
        node_id: NodeId,
        node_obj: Any,
        node_type: DataFlowGraph.NodeType,
        partition: PartitionId,
        inputs: dict[str, NodeId],
        output_dtype: DType,
    ) -> PartitionId:
        """Determine the output partition for a given node in the data flow graph.

        This method determines the partition that the output of a specified node
        belongs to, based on the node's type. Different types of nodes may direct
        their output to different partitions, reflecting their role in the data flow:

        - :code:`DATA_AGGREGATOR`: Outputs always point to the :code:`AGGREGATED` partition,
          even though the aggregator node itself is not part of this partition.

        - :code:`DATA_AUGMENTOR`: Outputs point to partition specified by the aggregator
          itself. Like aggregators, augmentors are not part of the partition they point to.

        - Other node types: Outputs remain within the partition specified by the
          node's :code:`PARTITION` attribute.

        Args:
            node_id (NodeId): The ID of the node for which to determine the output partition.
            node_obj (Any): The node object.
            node_type (DataFlowGraph.NodeType): The node type.
            partition (PartitionId): The partition of the node.
            inputs (dict[str, NodeId]): The inputs to the node.
            output_dtype (DType): The output data type of the node.

        Returns:
            PartitionId: The partition that the node's output will be directed to.
        """
        if node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR:
            # data aggregator outputs always point into
            # the aggregated partition while the aggregator
            # node itself is not part of the aggregated partition
            return DataFlowGraph.Partition.AGGREGATED.value

        elif node_type == DataFlowGraph.NodeType.DATA_AUGMENTOR:
            # build the input data type from the inputs
            input_dtype = MappingType(
                tuple(
                    (key, self._graph.get_output_dtype(node_id)) for key, node_id in inputs.items()
                )
            )
            # build a mock run context
            ctx = RunContext(
                node_id=node_id,
                index=[],
                rank=0,
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                session=None,
            )
            # infer the output partition of the node
            return node_obj.infer_output_partition(ctx, partition)

        elif node_type == DataFlowGraph.NodeType.TRACE:
            # trace nodes always point into their target partition
            return node_obj.config.path[-1]

        else:
            # other node types don't transition between partitions
            return partition

    def _build_partition_graph(self) -> nx.DiGraph:
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
        g = nx.DiGraph()
        g.add_nodes_from(
            [
                DataFlowGraph.Partition.CONST.value,
                DataFlowGraph.Partition.DEFAULT.value,
            ]
        )

        for _, attrs in self._graph.nodes(data=True):
            src_partition = attrs[DataFlowGraph.NodeAttribute.PARTITION]
            out_partition = attrs[DataFlowGraph.NodeAttribute.OUT_PARTITION]

            # make sure the source partition is contained in the partition graph
            assert (
                src_partition in g
            ), f"The source partition '{src_partition}' is not present in the partition graph."

            # don't include edges from the constant partition in partition graph
            if src_partition == DataFlowGraph.Partition.CONST:
                continue

            # add the target partition to the partition graph
            if out_partition not in g:
                g.add_node(out_partition)

            # only add a connection if there is no path from the source to the
            # target partition yet
            if (src_partition != out_partition) and not nx.has_path(
                g, src_partition, out_partition
            ):
                g.add_edge(src_partition, out_partition)

        return g

    def _add_consts_from_nested(
        self, val: NestedType[ConcreteReference | Any], dtype: None | DType = None
    ) -> NestedType[ConcreteReference]:
        if isinstance(val, dict):
            assert (dtype is None) or isinstance(dtype, MappingType)

            return {
                key: self._add_consts_from_nested(item, dtype[key] if dtype is not None else None)
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
                    dtype = self._graph.get_output_dtype(ref._node_id)

            else:
                # otherwise use the value type from the given dtype
                dtype = dtype.value_type

            # recurse on all items in the sequence
            return type(val)([self._add_consts_from_nested(item, dtype) for item in val])

        elif not isinstance(val, ConcreteReference):
            # add the constant node
            return self.const(
                val, dtype=dtype if dtype is not None else build_dtype_from_python_object(val)
            )

        elif isinstance(val, ConcreteReference):
            return val

        else:  # pragma: not covered
            raise TypeError(f"Unsupported type encountered in 'collect': {val}.")

    def source(self, dtype: DType, node_id: NodeId | None = None) -> ConcreteReference:
        """Add a the source node to the graph.

        Args:
            dtype (DType): The data type representing the source features.
            node_id (None | NodeId): The id of the node, defaults to a random uuid.

        Returns:
            ConcreteReference: A reference object to the source node.
        """
        # create a random node id if no was given
        node_id = node_id if node_id is not None else str(random_uuid())
        # add the source node to the graph
        node_id = self._graph.add_source_node(dtype, node_id)
        # build the concrete reference to the source node
        return ConcreteReference(node_id, self._graph, self)

    def const(self, value: Any, dtype: DType, node_id: None | NodeId = None) -> ConcreteReference:
        """Adds a constant node to the data flow graph.

        This function creates a node representing a constant value in the graph. The
        value is wrapped in a :code:`PyArrow` array of the specified data type.

        Args:
            value (Any): The constant value to be added to the graph.
            dtype (DType): The data type of the constant value.
            node_id (None | NodeId): A unique identifier for the node. If :code:`None`,
                a random UUID is generated. Defaults to :code:`None`.

        Returns:
            ConcreteReference: A reference to the newly created constant node.
        """
        # make sure the data type matches the value
        array = pa.array([value], type=dtype.arrow_type)
        # add the node to the graph
        return self._add_node_to_graph(
            node_obj=ConstNode(value=array),
            node_type=DataFlowGraph.NodeType.CONST,
            inputs={},
            output_dtype=dtype,
            node_id=node_id,
        )

    def cast(
        self, ref: ConcreteReference, dtype: DType, node_id: None | NodeId = None
    ) -> ConcreteReference:
        """Add a cast node to the data flow graph.

        This method adds a cast node, which transforms data from one type to another,
        ensuring compatibility between nodes or preparing the data for specific processing
        requirements. The method validates type compatibility and computes the resulting
        type if the cast is feasible.

        Args:
            ref (ConcreteReference): A reference to the input data to be cast.
            dtype (DType): The target type to which the input data should be cast.
            node_id (None | NodeId): An optional unique identifier for the node. If not
                provided, a random UUID is generated. Defaults to :code:`None`.

        Returns:
            ConcreteReference: A reference to the added cast node, allowing access to its output.

        Raises:
            RuntimeError: If the source and target types are incompatible or if other
                casting constraints are violated.
        """
        try:
            dtype = cast_dtype(self._graph.get_output_dtype(ref._node_id), dtype)
        except RuntimeError as e:
            raise RuntimeError(
                f"Error while casting from {self._graph.get_output_dtype(ref._node_id)} to {dtype}."
            ) from e

        # add the node to the graph
        return self._add_node_to_graph(
            node_obj=None,
            node_type=DataFlowGraph.NodeType.CAST,
            inputs={"value": ref},
            output_dtype=dtype,
            node_id=node_id,
        )

    def collect(
        self,
        collect: NestedType[ConcreteReference | Any],
        dtype: None | DType = None,
        node_id: None | NodeId = None,
    ) -> ConcreteReference:
        """Adds a collect node to the data flow graph.

        A collect node combines features from other nodes, organizing them into a nested structure
        as specified by the :code:`collect` argument. It enables combining outputs from multiple
        nodes into a single, structured output. Additionally, the function processes the nested
        structure to check for constants to add to the graph before collecting.

        Args:
            collect (NestedType[ConcreteReference]): A nested structure (e.g., lists, dictionaries)
                containing :class:`ConcreteReference` objects that specify the features to collect.
            dtype (None | DType): The expected data type of the :code:`collect` structure. If
                :code:`None`, the function attempts to infer the type where possible.
            node_id (None | NodeId): A unique identifier for the node.
                If :code:`None`, a random UUID is generated. Defaults to :code:`None`.

        Returns:
            ConcreteReference: A reference to the newly created collect node.
        """
        # add all constant nodes contained in the collect structure to the graph
        collect = self._add_consts_from_nested(collect, dtype)

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
        input_dtypes = {k: self._graph.get_output_dtype(r._node_id) for k, r in inputs.items()}
        output_type, required_casts = obj.build_output_type(input_dtypes)
        # add required cast nodes to graph
        # TODO: autocast
        for key, dtype in required_casts.items():
            inputs[key] = self.cast(inputs[key], dtype)

        return self._add_node_to_graph(
            node_obj=obj,
            node_type=DataFlowGraph.NodeType.COLLECT,
            inputs=inputs,
            output_dtype=output_type,
            node_id=node_id,
        )

    def trace(
        self, ref: ConcreteReference, tgt: PartitionId, node_id: None | NodeId = None
    ) -> ConcreteReference:
        """Adds a trace node to the data flow graph.

        The trace node follows the shortest path through the partition graph from the source
        partition of the provided reference to the specified target partition. It transforms
        the values associated with the reference according to the trace indices of the path.

        If the source and target partitions are the same, no trace node is added, and the
        reference is returned as-is.

        Args:
            ref (ConcreteReference): A reference to the value or node whose partition needs to
                be traced.
            tgt (PartitionId): The identifier of the target partition in the partition graph.
            node_id (None | NodeId): A unique identifier for the trace node. If :code:`None`, a
                random UUID is generated. Defaults to :code:`None`.

        Returns:
            ConcreteReference: A reference to the newly created trace node, or the original
                reference if no trace is required.
        """
        # get the partition of the reference
        src = self._graph.nodes[ref._node_id][DataFlowGraph.NodeAttribute.OUT_PARTITION]

        # check if the value needs to be traced
        if src == tgt:
            return ref

        if src == DataFlowGraph.Partition.CONST:
            # constant partition can be directly traced to any partition
            path = (src, tgt)
        else:
            # compute the shortest path through the partition graph
            partition_graph = self._build_partition_graph()
            path = nx.shortest_path(partition_graph, src, tgt)

        # get the output data type of the trace node
        out_dtype = self.graph.nodes[ref._node_id][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
        # add the trace node to the graph
        return self._add_node_to_graph(
            node_obj=TraceNode(path=path),
            node_type=DataFlowGraph.NodeType.TRACE,
            inputs={"value": ref},
            output_dtype=out_dtype,
            node_id=node_id,
        )

    def compute(
        self,
        obj: BaseNode,
        inputs: dict[str, ConcreteReference | Any],
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

        def convert_to_feature(_, x):
            return build_feature_from_reference(x) if isinstance(x, ConcreteReference) else x

        def convert_to_ref(_, x):
            return x.ref if isinstance(x, Feature) else x

        # convert input references to feature instances for input validation
        inputs = map_recursive(convert_to_feature, inputs)

        # create the type engine from the node signature
        name = f"{type(obj).__qualname__}"
        with FeatureEngine(name, obj.config, obj.signature) as engine:
            # validate the node signature and input arguments
            engine.validate_signature()
            engine.validate_arguments(**inputs)
            # split the input features from the input constants
            features, objects, object_dtypes = engine.get_features_and_objects(**inputs)
            # collect all objects
            for key, val in objects.items():
                # convert all features in the collect object to references for the collect
                # node to catch them
                ref = self.collect(map_recursive(convert_to_ref, val), object_dtypes[key])
                features[key] = build_feature_from_reference(ref)
            # build the output feature of the node
            output_dtype = engine.build_return_feature(features).dtype

        input_refs = {key: feature.ref for key, feature in features.items()}
        # add the node to the graph
        return self._add_node_to_graph(
            node_obj=obj,
            node_type=node_type,
            inputs=input_refs,
            output_dtype=output_dtype,
            node_id=node_id,
        )

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

        if node_id not in self._graph.nodes:
            raise RuntimeError(f"Node ID {node_id} does not exist in the current graph.")

        # mapping node ids of the 'graph' to references in the new graph 'h'
        node_id_mapping: dict[NodeId, ConcreteReference] = {
            graph.src_node_id: ConcreteReference(node_id, self._graph, self)
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
                node_id_mapping[node_id] = self.collect(collect, node_id=node_id)

            elif node_type == DataFlowGraph.NodeType.CONST:
                node_id_mapping[node_id] = self.const(
                    node_obj.config.value.to_pylist()[0],
                    dtype=node_attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                    node_id=node_id,
                )

            elif node_type == DataFlowGraph.NodeType.CAST:
                node_id_mapping[node_id] = self.cast(
                    inputs["value"],
                    dtype=node_attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                    node_id=node_id,
                )

            elif node_type in {
                DataFlowGraph.NodeType.DATA_PROCESSOR,
                DataFlowGraph.NodeType.DATA_AUGMENTOR,
                DataFlowGraph.NodeType.DATA_AGGREGATOR,
            }:
                node_id_mapping[node_id] = self.compute(node_obj, inputs, node_id)

            else:
                raise NotImplementedError(f"Unexpected node type: {node_type}")

        return self
