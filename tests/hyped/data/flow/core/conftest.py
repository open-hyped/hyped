import pytest
from datasets import Features, Value

from hyped.data.flow.core.executor import ExecutionState
from hyped.data.flow.core.flow import DataFlow
from hyped.data.flow.core.graph import DataFlowGraph

from .mock import MockAggregator, MockInputRefs, MockProcessor


@pytest.fixture(autouse=True)
def reset_mocks():
    MockProcessor.process.reset_mock()
    MockAggregator.initialize.reset_mock()
    MockAggregator.extract.reset_mock()
    MockAggregator.update.reset_mock()


@pytest.fixture
def setup_graph():
    # create graph
    graph = DataFlowGraph()
    # add source node
    src_features = Features({"x": Value("int64")})
    src_node_id = graph.add_source_node(src_features)
    # create processor
    p = MockProcessor()
    a = MockAggregator()

    # create input refs from source features
    i = MockInputRefs(
        a=graph.get_node_output_ref(src_node_id).x,
        b=graph.get_node_output_ref(src_node_id).x,
    )
    o = p._out_refs_type.build_features(p.config, i)

    # add nodes
    proc_node = graph.add_processor_node(p, i, o)
    agg_node = graph.add_processor_node(a, i, None)

    return graph, proc_node, agg_node


@pytest.fixture
def setup_state(setup_graph):
    graph, proc_node, agg_node = setup_graph
    # create state
    batch, index, rank = {"x": [1, 2, 3]}, [0, 1, 2], 0
    state = ExecutionState(graph, batch, index, rank)
    # return setup
    return state, graph, proc_node, agg_node


@pytest.fixture
def setup_flow(setup_graph):
    graph, proc_node, agg_node = setup_graph
    # create data flow
    flow = DataFlow(Features({"x": Value("int64")}))
    flow._graph = graph
    # return setup
    return flow, graph, proc_node, agg_node
