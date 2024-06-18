import pytest
from datasets import Features, Value

from hyped.data.flow.core.graph import DataFlowGraph
from hyped.data.flow.core.nodes.const import Const
from hyped.data.flow.core.refs.ref import AggregationRef, FeatureRef

from .mock import MockAggregator, MockInputRefs, MockOutputRefs, MockProcessor


class TestDataFlowGraph:
    def test_add_source_node(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # check source node was added
        assert src_node_id in graph
        # check node properties
        node = graph.nodes[src_node_id]
        assert node[DataFlowGraph.NodeAttribute.DEPTH] == 0
        assert node[DataFlowGraph.NodeAttribute.NODE_OBJ] is None
        assert (
            node[DataFlowGraph.NodeAttribute.NODE_TYPE]
            == DataFlowGraph.NodeType.SOURCE
        )
        assert node[DataFlowGraph.NodeAttribute.IN_FEATURES] is None
        assert node[DataFlowGraph.NodeAttribute.OUT_FEATURES] == src_features

        # try to add another source node
        with pytest.raises(RuntimeError):
            graph.add_source_node(src_features)

    def test_add_processor_node(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create processor
        p = MockProcessor()
        i = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add processor to graph
        node_id = graph.add_processor_node(p, i, o)

        # check processor node was added
        assert node_id in graph
        # check node properties
        node = graph.nodes[node_id]
        assert node[DataFlowGraph.NodeAttribute.DEPTH] == 1
        assert node[DataFlowGraph.NodeAttribute.NODE_OBJ] == p
        assert (
            node[DataFlowGraph.NodeAttribute.NODE_TYPE]
            == DataFlowGraph.NodeType.DATA_PROCESSOR
        )
        assert node[DataFlowGraph.NodeAttribute.IN_FEATURES] == i.features_
        assert node[DataFlowGraph.NodeAttribute.OUT_FEATURES] == o
        # check edges
        assert graph.has_edge(src_node_id, node_id)
        for n, r in i.named_refs.items():
            assert n in graph[src_node_id][node_id]
            assert (
                graph[src_node_id][node_id][n][DataFlowGraph.EdgeAttribute.KEY]
                == r.key_
            )

    def test_add_aggregator_node(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        a = MockAggregator()
        i = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        # add aggregator node
        node_id = graph.add_processor_node(a, i, None)

        # check processor node was added
        assert node_id in graph
        # check node properties
        node = graph.nodes[node_id]
        assert node[DataFlowGraph.NodeAttribute.DEPTH] == 1
        assert node[DataFlowGraph.NodeAttribute.NODE_OBJ] == a
        assert (
            node[DataFlowGraph.NodeAttribute.NODE_TYPE]
            == DataFlowGraph.NodeType.DATA_AGGREGATOR
        )
        assert node[DataFlowGraph.NodeAttribute.IN_FEATURES] == i.features_
        assert node[DataFlowGraph.NodeAttribute.OUT_FEATURES] is None
        # check edges
        assert graph.has_edge(src_node_id, node_id)
        for n, r in i.named_refs.items():
            assert n in graph[src_node_id][node_id]
            assert (
                graph[src_node_id][node_id][n][DataFlowGraph.EdgeAttribute.KEY]
                == r.key_
            )

    def test_partition(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # source node is part of the default partition
        # this is by definition of the default partition: all child nodes
        # of the source node are part of the the default partition
        assert (
            graph.nodes[src_node_id][DataFlowGraph.NodeAttribute.PARTITION]
            == DataFlowGraph.PredefinedPartition.DEFAULT
        )

        # create processor
        p = MockProcessor()
        i = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add processor to graph
        node_id = graph.add_processor_node(p, i, o)

        # the processor is a child of the source node and thus
        # should be part of the default partition
        assert (
            graph.nodes[node_id][DataFlowGraph.NodeAttribute.PARTITION]
            == DataFlowGraph.PredefinedPartition.DEFAULT
        )

        # add constant node
        c = Const(value=0)
        co = c._out_refs_type.build_features(c.config, None)
        const_node = graph.add_processor_node(c, None, co)

        # create processor
        p = MockProcessor()
        i = MockInputRefs(
            a=graph.get_node_output_ref(const_node).value,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add processor to graph
        node_id = graph.add_processor_node(p, i, o)

        # the processor inherits from a constant and the source node
        # so it should still be part of the default partition
        assert (
            graph.nodes[node_id][DataFlowGraph.NodeAttribute.PARTITION]
            == DataFlowGraph.PredefinedPartition.DEFAULT
        )

        # create processor
        p = MockProcessor()
        i = MockInputRefs(
            a=graph.get_node_output_ref(const_node).value,
            b=graph.get_node_output_ref(const_node).value,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add processor to graph
        node_id = graph.add_processor_node(p, i, o)

        # the processor inherits only from constant nodes and
        # thus it's output is also considered to be constant
        assert (
            graph.nodes[node_id][DataFlowGraph.NodeAttribute.PARTITION]
            == DataFlowGraph.PredefinedPartition.CONST
        )

    def test_depth_and_width(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)
        # check depth of source node
        assert graph.nodes[src_node_id][DataFlowGraph.NodeAttribute.DEPTH] == 0
        # check graph properties
        assert graph.depth == 1
        assert graph.width == 1

        # create processor
        p = MockProcessor()

        # create input refs from source features
        i1 = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i1)
        # add first level processor
        node_id_1 = graph.add_processor_node(p, i1, o)
        assert graph.nodes[node_id_1][DataFlowGraph.NodeAttribute.DEPTH] == 1
        # check graph properties
        assert graph.depth == 2
        assert graph.width == 1

        # create input refs from first-level outputs
        i2 = MockInputRefs(
            a=graph.get_node_output_ref(node_id_1).y,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i2)
        # add second level processor
        node_id_2 = graph.add_processor_node(p, i2, o)
        assert graph.nodes[node_id_2][DataFlowGraph.NodeAttribute.DEPTH] == 2
        # check graph properties
        assert graph.depth == 3
        assert graph.width == 1

        # create in put refs from source and first level nodes
        i3 = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i3)
        # add third level processor
        node_id_3 = graph.add_processor_node(p, i3, o)
        assert graph.nodes[node_id_3][DataFlowGraph.NodeAttribute.DEPTH] == 2
        # check graph properties
        assert graph.depth == 3
        assert graph.width == 2

        # create in put refs from source and second level nodes
        i4 = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(node_id_2).y,
        )
        o = p._out_refs_type.build_features(p.config, i4)
        # add third level processor
        node_id_4 = graph.add_processor_node(p, i4, o)
        assert graph.nodes[node_id_4][DataFlowGraph.NodeAttribute.DEPTH] == 3
        # check graph properties
        assert graph.depth == 4
        assert graph.width == 2

    def test_add_processor_invalid_input(self):
        g1 = DataFlowGraph()
        g2 = DataFlowGraph()
        # mock features
        src_features = Features({"x": Value("int64")})
        out_features = Features({"y": Value("int64")})
        # add source nodes
        g1_src_node_id = g1.add_source_node(src_features)
        g2_src_node_id = g2.add_source_node(src_features)
        # create processor instance
        p = MockProcessor()
        # add valid nodes
        g1.add_processor_node(
            p,
            MockInputRefs(
                a=g1.get_node_output_ref(g1_src_node_id).x,
                b=g1.get_node_output_ref(g1_src_node_id).x,
            ),
            out_features,
        )
        g2.add_processor_node(
            p,
            MockInputRefs(
                a=g2.get_node_output_ref(g2_src_node_id).x,
                b=g2.get_node_output_ref(g2_src_node_id).x,
            ),
            out_features,
        )
        # try add invalid node
        with pytest.raises(RuntimeError):
            g1.add_processor_node(
                p,
                MockInputRefs(
                    a=g2.get_node_output_ref(g2_src_node_id).x,
                    b=g1.get_node_output_ref(g1_src_node_id).x,
                ),
                out_features,
            )

    def test_get_node_output_ref(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create mock processor
        p = MockProcessor()
        # create input refs from source features
        i = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add processor to the graph
        node_id = graph.add_processor_node(p, i, o)

        # test feature reference to source features
        ref = graph.get_node_output_ref(src_node_id)
        assert (
            ref.model_dump()
            == FeatureRef(
                node_id_=src_node_id,
                key_=tuple(),
                flow_=graph,
                feature_=src_features,
            ).model_dump()
        )
        # test feature reference to processor output
        ref = graph.get_node_output_ref(node_id)
        assert isinstance(ref, MockOutputRefs)
        assert (
            ref.model_dump() == MockOutputRefs(graph, node_id, o).model_dump()
        )

        # test invalid node id
        with pytest.raises(KeyError):
            graph.get_node_output_ref(-1)

    def test_dependency_graph(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create processor
        p = MockProcessor()
        # create input refs from source features
        i = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add first level processor
        node_id_1 = graph.add_processor_node(p, i, o)

        # create input refs from first-level outputs
        i = MockInputRefs(
            a=graph.get_node_output_ref(node_id_1).y,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add second level processor
        node_id_2 = graph.add_processor_node(p, i, o)

        # create input refs from first-level outputs
        i = MockInputRefs(
            a=graph.get_node_output_ref(node_id_1).y,
            b=graph.get_node_output_ref(node_id_2).y,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add third level processor
        node_id_3 = graph.add_processor_node(p, i, o)

        subgraph = graph.dependency_graph({src_node_id})
        assert set(subgraph.nodes) == {src_node_id}

        subgraph = graph.dependency_graph({node_id_1})
        assert set(subgraph.nodes) == {src_node_id, node_id_1}

        subgraph = graph.dependency_graph({node_id_2})
        assert set(subgraph.nodes) == {src_node_id, node_id_1, node_id_2}

        subgraph = graph.dependency_graph({node_id_3})
        assert set(subgraph.nodes) == {
            src_node_id,
            node_id_1,
            node_id_2,
            node_id_3,
        }
