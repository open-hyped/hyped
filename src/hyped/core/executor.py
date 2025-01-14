"""Manages the execution of data flow graphs.

This module provides the :code:`DataFlowGraphExecutor` class, which is responsible for executing
a data flow graph. It manages the execution state, ensures that data processors
are executed in the correct order, and collects the results.
"""
import asyncio
import logging
import typing
from contextlib import nullcontext
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from hyped.common._worker import get_worker_info

from .features.dtypes import MappingType
from .features.reference import ConcreteReference
from .graph import DataFlowGraph
from .nodes.aggregator import DataAggregationManager
from .nodes.base import BaseNode, RunContext, RunSession
from .nodes.const import ConstNode
from .typing import IndexList, NodeId, Rank, TraceIndexList

logger = logging.getLogger(__name__)


class ExecutionState(object):
    """Tracks the state during the execution of a data flow graph.

    This class is used internally to manage the state of data processing as the
    data flow graph is executed, keeping track of outputs, indexes, and node readiness.
    """

    def __init__(
        self,
        graph: DataFlowGraph,
        batch: pa.Array,
        index: IndexList,
        rank: Rank,
    ):
        """Initialize the execution state.

        Args:
            graph (DataFlowGraph): The data flow graph being executed.
            batch (pa.Array): The initial batch of data.
            index (IndexList): The index of the batch.
            rank (Rank): The rank of the process in a distributed setting.
        """
        self.rank = rank
        self.graph = graph

        # partition graph attributes
        self.index = {
            DataFlowGraph.Partition.DEFAULT: index,
            DataFlowGraph.Partition.CONST: [0],
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

    def register_partition_trace(self, node_id: NodeId, trace_index: TraceIndexList) -> None:
        """Register trace and index mappings for the transition between partitions.

        This method registers the trace indices and index mappings for a node's output partition
        transition in the partition graph. It ensures that there is a valid edge between the source
        and target partitions and then stores the trace and index information accordingly.

        Args:
            node_id (NodeId): The identifier of the node for which to register the partition trace.
            trace_index (TraceIndexList): The trace indices representing how to transition between
                the source and target partitions.

        Raises:
            AssertionError: If there is no edge between the source and target partitions in the
                partition graph.
            AssertionError: If the source partition is not registered yet.
        """
        u = self.graph.nodes[node_id][DataFlowGraph.NodeAttribute.PARTITION]
        v = self.graph.nodes[node_id][DataFlowGraph.NodeAttribute.OUT_PARTITION]

        if u != v:
            # register the trace
            self.traces[(u, v)] = np.asarray(trace_index)
            # get the index of the source partition and transform it
            assert u in self.index, f"Partition {u} not registered yet!"
            self.index[v] = np.asarray(self.index[u])[trace_index].tolist()

    def collect_value(self, ref: ConcreteReference) -> pa.Array:
        """Collect the values requested by the feature reference.

        Args:
            ref (ConcreteReference): The feature reference indicating which
                values to collect.

        Returns:
            pa.Array: The collected pyarrow array of data.

        Raises:
            AssertionError: If the feature reference does not contain
                expected feature types.
        """
        return self.outputs[ref._node_id]

    def collect_inputs(self, node_id: NodeId) -> tuple[dict[str, pa.Array], IndexList]:
        """Collect inputs for a given node.

        Args:
            node_id (NodeId): The ID of the node for which to collect inputs.

        Returns:
            tuple[dict[str, pa.Array], IndexList]: The collected inputs to the processor
                and the corresponding index
        """
        # collect all inputs from the incoming edges
        inputs = {key: self.outputs[u] for u, _, key in self.graph.in_edges(node_id, keys=True)}
        # get the index
        partition = self.graph.nodes[node_id][DataFlowGraph.NodeAttribute.PARTITION]
        index = self.index[partition]

        return inputs, index

    def capture_output(self, node_id: NodeId, output: pa.Array) -> None:
        """Capture the output of a node.

        Args:
            node_id (NodeId): The ID of the node producing the output.
            output (pa.Array): The output batch of data.

        Raises:
            AssertionError: If the node is already set
        """
        assert not self.ready[node_id].is_set(), f"Node {node_id} is already set."
        # apply indexing wrapper to array, required for feature key
        # indexing of arrow arrays
        self.outputs[node_id] = output
        self.ready[node_id].set()


class DataFlowGraphExecutor(object):
    """Executes a data flow graph.

    This class provides the low-level functionality for executing a data flow
    graph, managing the execution of each node and collecting results.
    """

    def __init__(
        self,
        graph: DataFlowGraph,
        collect: ConcreteReference,
        aggregation_manager: None | DataAggregationManager,
    ) -> None:
        """Initialize the executor.

        Args:
            graph (DataFlowGraph): The data flow graph to execute.
            collect (ConcreteReference): The feature reference to collect results.
            aggregation_manager (None | DataAggregationManager):
                The manager responsible for handling data aggregation. Can be
                None if the graph has no aggregator nodes.

        Raises:
            TypeError: If the collect feature is not of type datasets.Features.
        """
        self.graph = graph
        self.collect = collect
        self.aggregation_manager = aggregation_manager

        self.session: None | RunSession = None

        if (aggregation_manager is None) and any(
            node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR
            for _, node_type in graph.nodes(data=DataFlowGraph.NodeAttribute.NODE_TYPE)
        ):
            raise RuntimeError(
                "The data flow graph includes one or more aggregator nodes, but no aggregation "
                "manager was provided."
            )

    def build_run_context(self, node_id: NodeId, index: IndexList, rank: Rank) -> RunContext:
        """Build the run context for a specified node.

        Args:
            node_id (NodeId): The id of the node.
            index (IndexList): The index list of the current batch.
            rank (Rank): The multiprocess rank.

        Returns:
            RunContext: The context for the node execution.
        """
        node_attrs = self.graph.nodes[node_id]
        # build the run context
        return RunContext(
            node_id=node_id,
            index=index,
            rank=rank,
            input_dtype=node_attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE],
            output_dtype=node_attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
            session=self.session,
        )

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

        # build the run context
        ctx = self.build_run_context(node_id, index, state.rank)

        with (
            node_obj.with_state(self.session.get_context(node_id))
            if isinstance(node_obj, BaseNode)
            else nullcontext()
        ):
            if node_type == DataFlowGraph.NodeType.CONST:
                assert isinstance(node_obj, ConstNode)
                # for constant nodes the node object is a pyarrow array
                # of a single entry holding the value
                state.capture_output(node_id, node_obj.config.value)

            elif node_type == DataFlowGraph.NodeType.CAST:
                # cast the value to the expected data type
                cast_value = pc.cast(inputs["value"], ctx.output_dtype.arrow_type)
                state.capture_output(node_id, cast_value)

            elif node_type == DataFlowGraph.NodeType.COLLECT:
                # collect values and capture values
                values = node_obj.collect(ctx, inputs)
                state.capture_output(node_id, values)

            elif node_type == DataFlowGraph.NodeType.TRACE:
                # trace the value through to the target partition
                values = node_obj.trace_values_through_partition_path(
                    ctx, inputs["value"], state.traces
                )
                state.capture_output(node_id, values)

            elif node_type == DataFlowGraph.NodeType.DATA_PROCESSOR:
                # run processor and check the output batch size
                out = await node_obj.run(ctx, inputs)
                assert out.type == ctx.output_dtype.arrow_type, "Unexpected output type"
                assert len(out) == len(index), "Output values length does not match index length."
                # capture output in execution state
                state.capture_output(node_id, out)

            elif node_type == DataFlowGraph.NodeType.DATA_AUGMENTOR:
                # run processor and check the output batch size
                out, trace_index = await node_obj.run(ctx, inputs)
                assert out.type == ctx.output_dtype.arrow_type, "Unexpected output type"
                # register output partition and capture output in execution state
                state.register_partition_trace(node_id, trace_index)
                state.capture_output(node_id, out)

            elif node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR:
                # run aggregator
                assert self.aggregation_manager is not None
                await self.aggregation_manager.aggregate(node_obj, ctx, inputs)

            else:  # pragma: not covered
                raise TypeError(f"Unsupported node type: {node_type}")

    async def execute(self, batch: pa.Array, index: IndexList, rank: Rank) -> pa.Array:
        """Execute the entire data flow graph.

        Args:
            batch (pa.Array): The initial batch of data.
            index (IndexList): The index of the batch.
            rank (Rank): The rank of the process in a multiprocessing setting.

        Returns:
            pa.Array: The final collected batch of data.
        """
        # create an execution state
        state = ExecutionState(self.graph, batch, index, rank)
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

    def run(self, batch: pa.Table, index: IndexList, rank: Rank | None = None) -> pa.Table:
        """Run the executor for a given batch.

        Args:
            batch (pa.Table): The initial batch of data.
            index (IndexList): The index of the batch.
            rank (Rank | None): The rank of the process in a multiprocessing setting.

        Returns:
            pa.Table: The final collected batch of data.
        """
        if rank is None:
            # try to get multiprocessing rank from worker info
            worker_info = get_worker_info()
            rank = 0 if worker_info is None else worker_info.rank

        if self.session is None:
            self.init_run_session(rank)

        # execute
        future = self.execute(batch.to_struct_array(), index=index, rank=rank)
        out = self.session._loop.run_until_complete(future)
        # convert output to pyarrow table
        return pa.table(out, schema=self.collect.get_dtype().arrow_schema)

    def init_run_session(self, rank: Rank) -> None:
        """Initialize the :class:`RunSession`.

        Args:
            rank (Rank): The rank of the process in a multiprocessing setting.
        """
        assert self.session is None
        # set the run session for the executor
        self.session = RunSession()
        # initialize all nodes
        for node_id in self.graph.nodes():
            node_obj = self.graph.nodes[node_id][DataFlowGraph.NodeAttribute.NODE_OBJ]
            if isinstance(node_obj, BaseNode):
                # get the node state and create a shallow copy
                state = node_obj.get_state()
                state = {
                    "__dict__": state["__dict__"].copy(),
                    "__slots__": tuple(state["__slots__"]),
                }
                # initialize the node and capture the state
                with node_obj.with_state(state):
                    ctx = self.build_run_context(node_id, [], rank)
                    node_obj.initialize(ctx)
                # capture the initialized state in the session
                self.session.set_context(node_id, state)

        logger.info(f"Set up run session instance {self.session.session_id}.")

    def reset_run_session(self) -> None:
        """Reset the run session."""
        if self.session is not None:
            # close the event loop and reset the session
            self.session._loop.close()
            self.session = None


class LazyDataFlowGraphExecutor(typing.Mapping, DataFlowGraphExecutor):
    """A lazy executor for a data flow graph.

    This class extends the :class:`DataFlowGraphExecutor` to compute outputs only
    when requested and when the inputs have changed. It implements a mapping
    interface to provide read-only access to the output values of the data flow.

    The execution is triggered lazily upon accessing an output feature, ensuring
    that the computation is performed only when necessary. The inputs are cached
    and compared to avoid redundant computations.
    """

    def __init__(
        self,
        graph: DataFlowGraph,
        collect: ConcreteReference,
        input_proxy: MappingProxyType[str, pa.Scalar],
    ) -> None:
        """Initialize the executor.

        Args:
            graph (DataFlowGraph): The data flow graph to execute.
            collect (ConcreteReference): The feature reference to collect results.
            input_proxy (MappingProxyType[str, Any]): A read-only proxy for the input data.
        """
        DataFlowGraphExecutor.__init__(self, graph, collect, None)

        self._proxy = input_proxy
        self._proxy_snapshot: None | dict[str, pa.Array] = None
        self._value_snapshot: None | dict[str, pa.Scalar] = None

        # create the run session
        self.init_run_session(0)

    def keys(self) -> typing.Iterable[str]:
        """Get the keys of the output features.

        Returns:
            Iterable[Hashable]: An iterable of the output feature keys.
        """
        # get the data type of the collect feature
        dtype = self.graph.get_output_dtype(self.collect._node_id)
        assert isinstance(dtype, MappingType)

        return dtype.keys()

    def _get_values(self) -> MappingProxyType[str, typing.Any]:
        """Compute and cache the output values if the inputs have changed.

        Returns:
            MappingProxyType[Hashable, Any]: A read-only proxy to the computed output values.
        """
        proxy_snapshot = dict(self._proxy)

        if (self._proxy_snapshot is None) or (proxy_snapshot != self._proxy_snapshot):
            # convert pyarrow scalars to arrays for execution
            array = pa.table(proxy_snapshot, schema=self.graph.src_dtype.arrow_schema)
            # run the executor
            future = self.execute(array.to_struct_array(), index=[0], rank=0)
            output = self.session._loop.run_until_complete(future)
            # parse the outputs and store them as the snapshot
            self._proxy_snapshot = proxy_snapshot
            self._out_snapshot = output[0].as_py()

        return MappingProxyType(self._out_snapshot)

    def __getitem__(self, key: str) -> typing.Any:
        """Get the value associated with the specified key.

        Args:
            key (Hashable): The key of the desired output value.

        Returns:
            Any: The value associated with the specified key.

        Raises:
            KeyError: If the key is not in the output features.
        """
        if key not in self.keys():
            raise KeyError(key)

        return self._get_values()[key]

    def __iter__(self):
        """Get an iterator over the keys of the output features.

        Returns:
            Iterator[Hashable]: An iterator over the output feature keys.
        """
        return iter(self.keys())

    def __len__(self):
        """Get the number of output features.

        Returns:
            int: The number of output features.
        """
        return len(self.keys())

    def __repr__(self):
        """Get the string representation of the LazyFlowOutput.

        Returns:
            str: The string representation of the LazyFlowOutput.
        """
        return "LazyFlowOutput(input_proxy=%s, executor=%s)" % (
            self._proxy,
            self,
        )

    def __str__(self):
        """Get the string representation of the LazyFlowOutput.

        Returns:
            str: The string representation of the LazyFlowOutput.
        """
        return str(dict(self))
