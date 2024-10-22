"""Manages the execution of data flow graphs.

This module provides the :code:`DataFlowExecutor` class, which is responsible for executing
a data flow graph. It manages the execution state, ensures that data processors
are executed in the correct order, and collects the results.

Classes:
    - :class:`ExecutionState`: Tracks the state during the execution of a data flow graph.
    - :class:`DataFlowExecutor`: Executes a data flow graph, managing the execution of each node
      and collecting results.
"""
import asyncio
from collections import defaultdict

import datasets
import networkx as nx
import numpy as np
import pyarrow as pa

from hyped.common.feature_key import FeatureKey
from hyped.common.typing import Batch, IndexList, NodeId, Rank, TraceIndexList

from .graph import DataFlowGraph
from .nodes.aggregator import DataAggregationManager
from .nodes.base import IOContext
from .refs.ref import FeatureRef


class ExecutionState(object):
    """Tracks the state during the execution of a data flow graph.

    This class is used internally to manage the state of data processing as the
    data flow graph is executed, keeping track of outputs, indexes, and node readiness.
    """

    def __init__(
        self,
        graph: DataFlowGraph,
        p_graph: nx.DiGraph,
        batch: Batch,
        index: IndexList,
        rank: Rank,
    ):
        """Initialize the execution state.

        Args:
            graph (DataFlowGraph): The data flow graph being executed.
            p_graph (nx.DiGraph): The partition graph to the data flow graph.
            batch (Batch): The initial batch of data.
            index (IndexList): The index of the batch.
            rank (Rank): The rank of the process in a distributed setting.
        """
        self.rank = rank
        self.graph = graph
        self.p_graph = p_graph

        # partition graph attributes
        self.index = {
            DataFlowGraph.PredefinedPartition.DEFAULT: index,
            DataFlowGraph.PredefinedPartition.CONST: [0],
        }
        self.traces: dict[tuple[str, str], np.ndarray] = {}

        # execution graph attributes
        self.outputs = {graph.src_node_id: batch}
        self.ready = {
            node_id: asyncio.Event()
            for node_id, attrs in graph.nodes(data=True)
            if (
                (node_id != graph.src_node_id)
                and (
                    attrs[DataFlowGraph.NodeAttribute.NODE_TYPE]
                    != DataFlowGraph.NodeType.DATA_AGGREGATOR
                )
            )
        }

    async def wait_for(self, node_id: str) -> None:
        """Wait until the specified node is ready.

        Args:
            node_id (str): The ID of the node to wait for.
        """
        if node_id in self.ready:
            await self.ready[node_id].wait()

    def trace_through_partition_path(
        self, values: list[pa.Array], src: str, tgt: str
    ) -> list[pa.Array]:
        """Apply trace indices through the partition path to transform the provided values.

        This method traces the path in the partition graph from the source partition to the
        target partition. It applies the trace indices of each partition transition to transform
        the provided values. This transformation might involve reordering, filtering, or otherwise
        modifying the values according to the computed trace indices.

        Args:
            values (list[list[Any]]): A list of lists where each inner list contains values
                that are transformed based on the computed trace indices.
            src (str): The identifier of the source partition in the partition graph.
            tgt (str): The identifier of the target partition in the partition graph.

        Returns:
            list[list[Any]]: A list of lists where each inner list has been transformed
                according to the computed trace indices, reflecting the path from the source
                to the target partition.

        Raises:
            AssertionError: If either the source or target partitions are not present in
                the partition graph, or if the lengths of the inner lists in `values` do not
                match the length of the index associated with the source partition.
        """
        # make sure the partitions are valid nodes in the partition graph
        assert src in self.p_graph, f"Source partition '{src}' not included in partition graph."
        assert tgt in self.p_graph, f"Target partition '{tgt}' not included in partition graph."
        # get the index to the source partition
        index = self.index[src]
        assert all(len(vals) == len(index) for vals in values)

        trace_index = np.arange(len(index))
        # follow the path from partition u to partition v and apply the trace
        # of each partition transition to build the final trace index
        while src != tgt:
            edge = next(iter(self.p_graph.out_edges(src)))
            trace_index = trace_index[self.traces[edge]]
            _, src = edge

        # apply the final trace index to the given values
        return [vals.take(trace_index) for vals in values]

    def register_partition_trace(
        self, node_id: NodeId, trace_index: TraceIndexList, index: IndexList
    ) -> None:
        """Register trace and index mappings for the transition between partitions.

        This method registers the trace indices and index mappings for a node's output partition
        transition in the partition graph. It ensures that there is a valid edge between the source
        and target partitions and then stores the trace and index information accordingly.

        Args:
            node_id (NodeId): The identifier of the node for which to register the partition trace.
            trace_index (TraceIndexList): The trace indices representing how to transition between
                the source and target partitions.
            index (IndexList): The index values to be used for the output partition, which will be
                transformed according to the computed trace indices.

        Raises:
            AssertionError: If there is no edge between the source and target partitions in the
                partition graph.
        """
        u = self.graph.nodes[node_id][DataFlowGraph.NodeAttribute.PARTITION]
        v = self.graph.get_node_output_partition(node_id)
        # must be an edge in the partition graph
        assert self.p_graph.has_edge(
            u, v
        ), f"No edge between partitions '{u}' and '{v}' in the partition graph."
        # register trace and index
        self.traces[(u, v)] = np.asarray(trace_index)
        self.index[v] = self.trace_through_partition_path([pa.array(index)], src=u, tgt=v)[0]

    def collect_value(self, ref: FeatureRef) -> Batch:
        """Collect the values requested by the feature reference.

        Args:
            ref (FeatureRef): The feature reference indicating which
                values to collect.

        Returns:
            Batch: The collected batch of data.

        Raises:
            AssertionError: If the feature reference does not contain
                expected feature types.
        """
        assert isinstance(ref.feature_, (datasets.Features, dict)), (
            f"Expected features of type datasets.Features or dict, " f"but got {type(ref.feature_)}"
        )
        batch = ref.key_.index_batch(self.outputs[ref.node_id_])
        return pa.table(batch)

    def collect_inputs(self, node_id: NodeId) -> tuple[Batch, IndexList]:
        """Collect inputs for a given node.

        Args:
            node_id (NodeId): The ID of the node for which to collect inputs.

        Returns:
            tuple[Batch, IndexList]: The collected inputs to the processor
                and the corresponding index

        Raises:
            AssertionError: If inputs are collected from a node that is not ready.
            AssertionError: If the collected values are not of the expected type.
        """
        inputs = dict()
        src_partitions = defaultdict(list)
        # TODO: first group edges by reference to the same feature
        #       then collect the feature only once
        for u, _, name, data in self.graph.in_edges(node_id, keys=True, data=True):
            assert (u == self.graph.src_node_id) or self.ready[
                u
            ].is_set(), f"Node {u} is not ready."
            # get the values requested from the batch
            key: FeatureKey = data[DataFlowGraph.EdgeAttribute.KEY]
            values = key.index_batch(self.outputs[u])

            partition = self.graph.get_node_output_partition(u)
            # store the values in inputs and keep track of the source partition
            inputs[name] = values
            src_partitions[partition].append(name)

        # get the node partition and the partition info
        tgt_partition = self.graph.nodes[node_id][DataFlowGraph.NodeAttribute.PARTITION]
        index = self.index[tgt_partition]

        # we dont need to trace the values of the target partition
        src_partitions.pop(tgt_partition, None)
        # handle the constant partition as an edge case
        for name in src_partitions.pop(DataFlowGraph.PredefinedPartition.CONST, []):
            inputs[name] = pa.chunked_array([inputs[name]] * len(index))

        for src, names in src_partitions.items():
            # trace values from their origin partition to the target partition
            values = [inputs[name] for name in names]
            values = self.trace_through_partition_path(values, src=src, tgt=tgt_partition)
            # update the values in the inputs
            inputs.update(dict(zip(names, values)))

        input_batch = pa.table(list(inputs.values()), names=list(inputs.keys()))

        return input_batch, index

    def capture_output(self, node_id: NodeId, output: Batch) -> None:
        """Capture the output of a node.

        Args:
            node_id (NodeId): The ID of the node producing the output.
            output (Batch): The output batch of data.

        Raises:
            AssertionError: If the node is already set
        """
        assert not self.ready[node_id].is_set(), f"Node {node_id} is already set."

        self.outputs[node_id] = output
        self.ready[node_id].set()


class DataFlowExecutor(object):
    """Executes a data flow graph.

    This class provides the low-level functionality for executing a data flow
    graph, managing the execution of each node and collecting results.
    """

    def __init__(
        self,
        graph: DataFlowGraph,
        collect: FeatureRef,
        aggregation_manager: None | DataAggregationManager,
    ) -> None:
        """Initialize the executor.

        Args:
            graph (DataFlowGraph): The data flow graph to execute.
            collect (FeatureRef): The feature reference to collect results.
            aggregation_manager (None | DataAggregationManager):
                The manager responsible for handling data aggregation. Can be
                None if the graph has no aggregator nodes.


        Raises:
            TypeError: If the collect feature is not of type datasets.Features.
        """
        if not isinstance(collect.feature_, (datasets.Features, dict)):
            raise TypeError(
                f"Expected collect feature of type datasets.Features or dict, "
                f"but got {type(collect.feature_)}"
            )

        self.graph = graph
        self.p_graph = graph.build_partition_graph()
        self.collect = collect
        self.aggregation_manager = aggregation_manager

    async def execute_node(self, node_id: NodeId, state: ExecutionState) -> None:
        """Execute a single node in the data flow graph.

        Args:
            node_id (NodeId): The ID of the node to execute.
            state (ExecutionState): The current execution state.

        Raises:
            AssertionError: If the output batch size doesn't match
                the input batch size.
        """
        if self.graph.in_degree(node_id) > 0:
            # wait for all dependencies of the current node
            deps = self.graph.predecessors(node_id)
            futures = map(state.wait_for, deps)
            await asyncio.gather(*futures)

        # collect inputs for processor execution
        inputs, index = state.collect_inputs(node_id)
        node_attrs = self.graph.nodes[node_id]
        node_obj = node_attrs[DataFlowGraph.NodeAttribute.NODE_OBJ]
        node_type = node_attrs[DataFlowGraph.NodeAttribute.NODE_TYPE]

        if node_type == DataFlowGraph.NodeType.CONST:
            # get constants from node object
            consts = node_obj.get_const_batch(batch_size=1)
            state.capture_output(node_id, consts)
            # done
            return

        # build the io context
        io = IOContext(
            node_id=node_id,
            inputs=node_attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPEs],
            outputs=node_attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
        )

        if node_type == DataFlowGraph.NodeType.DATA_PROCESSOR:
            # run processor and check the output batch size
            out = await node_obj.batch_process(inputs, index, state.rank, io)
            assert out.num_rows == len(index), "Output values length does not match index length."
            # capture output in execution state
            state.capture_output(node_id, out)

        elif node_type == DataFlowGraph.NodeType.DATA_AUGMENTER:
            # run processor and check the output batch size
            out, trace_index = await node_obj.batch_process(inputs, index, state.rank, io)
            # register output partition and capture output in execution state
            state.register_partition_trace(node_id, trace_index, index)
            state.capture_output(node_id, out)

        elif node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR:
            # run aggregator
            await self.aggregation_manager.aggregate(node_obj, inputs, index, state.rank, io)

    async def execute(self, batch: Batch, index: IndexList, rank: Rank) -> Batch:
        """Execute the entire data flow graph.

        Args:
            batch (Batch): The initial batch of data.
            index (IndexList): The index of the batch.
            rank (Rank): The rank of the process in a multiprocessing setting.

        Returns:
            Batch: The final collected batch of data.
        """
        # create an execution state
        state = ExecutionState(self.graph, self.p_graph, batch, index, rank)
        # execute all processors in the flow
        await asyncio.gather(
            *[
                self.execute_node(node_id, state)
                for node_id in self.graph.nodes()
                if node_id != self.graph.src_node_id
            ]
        )
        # collect output values
        return state.collect_value(self.collect)
