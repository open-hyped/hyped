import networkx as nx
import pytest
from datasets import Features, Value

from hyped.data.flow.core.graph import DataFlowGraph
from hyped.data.flow.core.nodes.const import Const
from hyped.data.flow.core.optim import DataFlowGraphOptimizer
from hyped.data.flow.core.refs.ref import FeatureRef
from hyped.data.flow.processors.ops.collect import (
    CollectFeatures,
    NestedContainer,
)

from .mock import MockInputRefs, MockProcessor


def new_graph():
    # create graph
    graph = DataFlowGraph()
    # add source node
    src_features = Features({"x": Value("int64")})
    src_node_id = graph.add_source_node(src_features)
    # return graph and source node id
    return graph, src_node_id


def add_processor(graph, node_A, node_B, **kwargs):
    # create processor
    p = MockProcessor(**kwargs)
    i = MockInputRefs(
        a=graph.get_node_output_ref(node_A),
        b=graph.get_node_output_ref(node_B),
    )
    o = p._out_refs_type.build_features(p.config, i)
    # add new processor to graph
    return graph.add_processor_node(p, i, o)


def cse_test_cases():
    test_cases = []

    # create simple graph
    graph, src_node_id = new_graph()
    add_processor(graph, src_node_id, src_node_id)
    add_processor(graph, src_node_id, src_node_id)
    # create target to simple graph
    target, src_node_id = new_graph()
    add_processor(target, src_node_id, src_node_id)
    # add test case
    test_cases.append((graph, target))

    # create slightly more complex graph
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id)
    node_id_2 = add_processor(graph, src_node_id, src_node_id)
    add_processor(graph, node_id_1, node_id_2)
    # create target to graph
    target, src_node_id = new_graph()
    node_id_1 = add_processor(target, src_node_id, src_node_id)
    add_processor(target, node_id_1, node_id_1)
    # add test case
    test_cases.append((graph, target))

    # create graph with no redundant nodes
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id, i=0)
    node_id_2 = add_processor(graph, src_node_id, src_node_id, i=1)
    # create target graph
    target, src_node_id = new_graph()
    node_id_1 = add_processor(target, src_node_id, src_node_id, i=0)
    node_id_2 = add_processor(target, src_node_id, src_node_id, i=1)
    # add test case
    test_cases.append((graph, target))

    # create graph with no two branches including redundant nodes
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id, i=0)
    node_id_2 = add_processor(graph, src_node_id, node_id_1)
    node_id_2 = add_processor(graph, src_node_id, node_id_1)
    node_id_3 = add_processor(graph, src_node_id, src_node_id, i=1)
    # create target graph
    target, src_node_id = new_graph()
    node_id_1 = add_processor(target, src_node_id, src_node_id, i=0)
    node_id_2 = add_processor(target, src_node_id, node_id_1)
    node_id_3 = add_processor(target, src_node_id, src_node_id, i=1)
    # add test case
    test_cases.append((graph, target))

    # test constant nodes
    graph, _ = new_graph()
    Const(value=0).to(graph)
    Const(value=0).to(graph)
    Const(value=1).to(graph)
    # create target graph
    target, _ = new_graph()
    Const(value=0).to(target)
    Const(value=1).to(target)
    # add test case
    test_cases.append((graph, target))

    # test collect nodes
    graph, _ = new_graph()
    ref_1 = Const(value=0).to(graph)
    ref_2 = Const(value=1).to(graph)
    CollectFeatures().call(
        collection=NestedContainer[FeatureRef](data={"a": ref_1, "b": ref_2})
    )
    CollectFeatures().call(
        collection=NestedContainer[FeatureRef](data={"a": ref_1, "b": ref_2})
    )
    # create target graph
    target, _ = new_graph()
    ref_1 = Const(value=0).to(target)
    ref_2 = Const(value=1).to(target)
    CollectFeatures().call(
        collection=NestedContainer[FeatureRef](data={"a": ref_1, "b": ref_2})
    )
    # add test case
    test_cases.append((graph, target))

    return test_cases


def optimize_test_cases():
    test_cases = []

    # create simple graph
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id)
    node_id_2 = add_processor(graph, src_node_id, src_node_id)
    # create target to simple graph
    target, src_node_id = new_graph()
    node_id_3 = add_processor(target, src_node_id, src_node_id)
    # add test case
    test_cases.append((graph, target, node_id_1))
    test_cases.append((graph, target, node_id_2))

    # create simple graph
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id)
    node_id_2 = add_processor(graph, src_node_id, src_node_id)
    node_id_3 = add_processor(graph, src_node_id, src_node_id)
    node_id_4 = add_processor(graph, node_id_1, node_id_2)
    # create target to simple graph
    target, src_node_id = new_graph()
    node_id_1 = add_processor(target, src_node_id, src_node_id)
    node_id_2 = add_processor(target, node_id_1, node_id_1)
    # add test case
    test_cases.append((graph, target, node_id_4))

    return test_cases


@pytest.mark.parametrize("graph, target", cse_test_cases())
def test_optimizer_cse(graph, target):
    # apply cse
    optim = DataFlowGraphOptimizer()
    cse_graph = optim.cse(graph)
    # check topology of cse graph
    assert nx.is_isomorphic(cse_graph, target)


@pytest.mark.parametrize("graph, target, leaf_node", optimize_test_cases())
def test_optimizer(graph, target, leaf_node):
    # apply cse
    optim = DataFlowGraphOptimizer()
    optim_graph = optim.optimize(graph, {leaf_node})
    # check topology of cse graph
    assert nx.is_isomorphic(optim_graph, target)
