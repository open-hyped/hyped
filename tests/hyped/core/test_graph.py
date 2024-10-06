from unittest.mock import MagicMock, call, patch

import networkx as nx
import pytest
from datasets import Features, Value

from hyped.core.graph import DataFlowGraph, _build_dependency_graph, _compute_node_depth
from hyped.core.nodes.base import BaseNodeConfig
from hyped.core.nodes.const import Const
from hyped.core.refs.ref import FeatureRef

from .mock import MockAggregator, MockAugmenter, MockOutputRefs, MockProcessor


class TestComputeNodeDepth:
    def test_empty_graph(self):
        G = nx.DiGraph()
        expected = {}
        assert _compute_node_depth(G) == expected

    def test_single_node(self):
        G = nx.DiGraph()
        G.add_node("A")
        expected = {"A": 0}
        assert _compute_node_depth(G) == expected

    def test_linear_chain(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        expected = {"A": 0, "B": 1, "C": 2, "D": 3}
        assert _compute_node_depth(G) == expected

    def test_branching_graph(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "D"),
                ("D", "E"),
            ]
        )
        expected = {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3}
        assert _compute_node_depth(G) == expected

    def test_multiple_roots(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("A", "C"),
                ("B", "C"),
                ("C", "D"),
            ]
        )
        G.add_node("E")  # Disconnected root node
        expected = {"A": 0, "B": 0, "C": 1, "D": 2, "E": 0}
        assert _compute_node_depth(G) == expected

    def test_disconnected_graph(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
            ]
        )
        G.add_edges_from(
            [
                ("D", "E"),
                ("E", "F"),
            ]
        )
        expected = {"A": 0, "B": 1, "C": 2, "D": 0, "E": 1, "F": 2}
        assert _compute_node_depth(G) == expected

    def test_cycle_graph(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "A"),
            ]
        )
        with pytest.raises(nx.NetworkXUnfeasible):
            _compute_node_depth(G)

    def test_graph_with_self_loop(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("A", "A"),
            ]
        )
        with pytest.raises(nx.NetworkXUnfeasible):
            _compute_node_depth(G)

    def test_complex_dag(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "D"),
                ("C", "E"),
                ("D", "F"),
                ("E", "F"),
                ("F", "G"),
            ]
        )
        expected = {
            "A": 0,
            "B": 1,
            "C": 1,
            "D": 2,
            "E": 2,
            "F": 3,
            "G": 4,
        }
        assert _compute_node_depth(G) == expected


class TestBuildDependencyGraph:
    def test_empty_graph(self):
        G = nx.DiGraph()
        nodes = set()
        subgraph = _build_dependency_graph(G, nodes)
        expected_subgraph = nx.DiGraph()
        assert nx.is_isomorphic(subgraph, expected_subgraph)

    def test_single_node_no_dependencies(self):
        G = nx.DiGraph()
        G.add_node("A")
        nodes = {"A"}
        subgraph = _build_dependency_graph(G, nodes)
        expected_subgraph = nx.DiGraph()
        expected_subgraph.add_node("A")
        assert nx.is_isomorphic(subgraph, expected_subgraph)

    def test_single_node_with_dependencies(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        nodes = {"D"}
        subgraph = _build_dependency_graph(G, nodes)
        expected_subgraph = nx.DiGraph()
        expected_subgraph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        assert nx.is_isomorphic(subgraph, expected_subgraph)

    def test_multiple_nodes_with_shared_dependencies(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("A", "E"),
                ("E", "F"),
                ("F", "D"),
            ]
        )
        nodes = {"D", "F"}
        subgraph = _build_dependency_graph(G, nodes)
        expected_subgraph = nx.DiGraph()
        expected_subgraph.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("A", "E"),
                ("E", "F"),
                ("F", "D"),
            ]
        )
        assert nx.is_isomorphic(subgraph, expected_subgraph)

    def test_node_not_in_graph(self):
        G = nx.DiGraph()
        G.add_node("A")
        nodes = {"B"}
        with pytest.raises(AssertionError, match="All nodes must be present in the graph 'G'."):
            _build_dependency_graph(G, nodes)

    def test_disconnected_graph(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
            ]
        )
        G.add_edges_from(
            [
                ("D", "E"),
                ("E", "F"),
            ]
        )
        nodes = {"C", "F"}
        subgraph = _build_dependency_graph(G, nodes)
        expected_subgraph = nx.DiGraph()
        expected_subgraph.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("D", "E"),
                ("E", "F"),
            ]
        )
        assert nx.is_isomorphic(subgraph, expected_subgraph)

    def test_no_dependencies(self):
        G = nx.DiGraph()
        G.add_nodes_from(["A", "B", "C"])
        nodes = {"A", "B"}
        subgraph = _build_dependency_graph(G, nodes)
        expected_subgraph = nx.DiGraph()
        expected_subgraph.add_nodes_from(["A", "B"])
        assert nx.is_isomorphic(subgraph, expected_subgraph)

    def test_complex_graph(self):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("E", "F"),
                ("F", "G"),
                ("C", "G"),
                ("G", "H"),
            ]
        )
        nodes = {"D", "H"}
        subgraph = _build_dependency_graph(G, nodes)
        expected_subgraph = nx.DiGraph()
        expected_subgraph.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("E", "F"),
                ("C", "G"),
                ("F", "G"),
                ("G", "H"),
            ]
        )
        assert nx.is_isomorphic(subgraph, expected_subgraph)


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
        assert node[DataFlowGraph.NodeAttribute.NODE_OBJ] is None
        assert node[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.SOURCE
        assert node[DataFlowGraph.NodeAttribute.IN_FEATURES] is None
        assert node[DataFlowGraph.NodeAttribute.OUT_FEATURES] == src_features

        # try to add another source node
        with pytest.raises(RuntimeError):
            graph.add_source_node(src_features)

    @pytest.mark.parametrize(
        "node_type, node_class",
        [
            (DataFlowGraph.NodeType.DATA_PROCESSOR, MockProcessor),
            (DataFlowGraph.NodeType.DATA_AGGREGATOR, MockAggregator),
            (DataFlowGraph.NodeType.DATA_AUGMENTER, MockAugmenter),
        ],
    )
    def test_add_node(self, node_type, node_class):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create processor
        p = node_class()
        i = p._in_refs_validator.validate(
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
        assert node[DataFlowGraph.NodeAttribute.NODE_OBJ] == p
        assert node[DataFlowGraph.NodeAttribute.NODE_TYPE] == node_type
        assert node[DataFlowGraph.NodeAttribute.IN_FEATURES] == i.features_
        assert node[DataFlowGraph.NodeAttribute.OUT_FEATURES] == o
        # check edges
        assert graph.has_edge(src_node_id, node_id)
        for n, r in i.named_refs.items():
            assert n in graph[src_node_id][node_id]
            assert graph[src_node_id][node_id][n][DataFlowGraph.EdgeAttribute.KEY] == r.key_

    def test_depth_and_width(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)
        # check graph properties
        assert graph.depth == 1
        assert graph.width == 1

        # create processor
        p = MockProcessor()

        # create input refs from source features
        i1 = p._in_refs_validator.validate(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i1)
        # add first level processor
        node_id_1 = graph.add_processor_node(p, i1, o)
        # check graph properties
        assert graph.depth == 2
        assert graph.width == 1

        # create input refs from first-level outputs
        i2 = p._in_refs_validator.validate(
            a=graph.get_node_output_ref(node_id_1).y,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i2)
        # add second level processor
        node_id_2 = graph.add_processor_node(p, i2, o)
        # check graph properties
        assert graph.depth == 3
        assert graph.width == 1

        # create in put refs from source and first level nodes
        i3 = p._in_refs_validator.validate(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i3)
        # add third level processor
        graph.add_processor_node(p, i3, o)
        # check graph properties
        assert graph.depth == 3
        assert graph.width == 2

        # create in put refs from source and second level nodes
        i4 = p._in_refs_validator.validate(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(node_id_2).y,
        )
        o = p._out_refs_type.build_features(p.config, i4)
        # add third level processor
        graph.add_processor_node(p, i4, o)
        # check graph properties
        assert graph.depth == 4
        assert graph.width == 2

        # get all depths
        depths = nx.get_node_attributes(graph, DataFlowGraph.NodeAttribute.DEPTH)
        # manually set all depths to -1
        nx.set_node_attributes(graph, -1, DataFlowGraph.NodeAttribute.DEPTH)
        assert nx.get_node_attributes(graph, DataFlowGraph.NodeAttribute.DEPTH) != depths
        # recompute the depth values
        graph.recompute_depths()

        # check if depths are recomputed correctly
        assert nx.get_node_attributes(graph, DataFlowGraph.NodeAttribute.DEPTH) == depths

    def test_error_on_mixing_flows(self):
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
            p._in_refs_validator.validate(
                a=g1.get_node_output_ref(g1_src_node_id).x,
                b=g1.get_node_output_ref(g1_src_node_id).x,
            ),
            out_features,
        )
        g2.add_processor_node(
            p,
            p._in_refs_validator.validate(
                a=g2.get_node_output_ref(g2_src_node_id).x,
                b=g2.get_node_output_ref(g2_src_node_id).x,
            ),
            out_features,
        )
        # try add invalid node
        with pytest.raises(RuntimeError):
            g1.add_processor_node(
                p,
                p._in_refs_validator.validate(
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
        i = p._in_refs_validator.validate(
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
        assert ref.model_dump() == MockOutputRefs(graph, node_id, o).model_dump()

        # test invalid node id
        with pytest.raises(KeyError):
            graph.get_node_output_ref(-1)

    def test_get_partition(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create processor
        p = MockProcessor()
        # create input refs from source features
        i = p._in_refs_validator.validate(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add first level processor
        proc_node_id = graph.add_processor_node(p, i, o)

        # add constant to graph
        c = Const(value=0)
        co = c._out_refs_type.build_features(c.config, None)
        const_node_id = graph.add_processor_node(c, None, co)

        # get constant partition
        const_graph = graph.get_partition(DataFlowGraph.PredefinedPartition.CONST)
        # check nodes in constant partition
        assert const_node_id in const_graph
        assert proc_node_id not in const_graph
        assert src_node_id not in const_graph

        # get default partition
        default_graph = graph.get_partition(DataFlowGraph.PredefinedPartition.DEFAULT)
        # check nodes in default partition
        assert const_node_id not in default_graph
        assert proc_node_id in default_graph
        assert src_node_id in default_graph

    def test_drop_partition(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create processor
        p = MockProcessor()
        # create input refs from source features
        i = p._in_refs_validator.validate(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add first level processor
        proc_node_id = graph.add_processor_node(p, i, o)

        # add constant to graph
        c = Const(value=0)
        co = c._out_refs_type.build_features(c.config, None)
        const_node_id = graph.add_processor_node(c, None, co)

        # drop constant partition
        non_const_graph = graph.drop_partition(DataFlowGraph.PredefinedPartition.CONST)
        # check nodes in default partition
        assert const_node_id not in non_const_graph
        assert proc_node_id in non_const_graph
        assert src_node_id in non_const_graph

        # drop default partition
        non_default_graph = graph.drop_partition(DataFlowGraph.PredefinedPartition.DEFAULT)
        # check nodes in constant partition
        assert const_node_id in non_default_graph
        assert proc_node_id not in non_default_graph
        assert src_node_id not in non_default_graph

    def test_subgraph_in_edges(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create processor
        p = MockProcessor()
        # create input refs from source features
        i = p._in_refs_validator.validate(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add first level processor
        node_id_1 = graph.add_processor_node(p, i, o)

        # create input refs from first-level outputs
        i = p._in_refs_validator.validate(
            a=graph.get_node_output_ref(node_id_1).y,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add second level processor
        node_id_2 = graph.add_processor_node(p, i, o)

        # check trivial cases
        edges = graph.subgraph_in_edges(graph.subgraph([src_node_id]))
        assert edges == []
        edges = graph.subgraph_in_edges(graph.subgraph([src_node_id, node_id_1]))
        assert edges == []
        edges = graph.subgraph_in_edges(graph.subgraph([src_node_id, node_id_1, node_id_2]))
        assert edges == []

        # check non-trivial cases
        edges = graph.subgraph_in_edges(graph.subgraph([node_id_1, node_id_2]))
        assert edges == [
            (src_node_id, node_id_1, "a"),
            (src_node_id, node_id_1, "b"),
        ]
        edges = graph.subgraph_in_edges(graph.subgraph([node_id_2]))
        assert edges == [
            (node_id_1, node_id_2, "a"),
            (node_id_1, node_id_2, "b"),
        ]

    def test_subgraph_out_edges(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create processor
        p = MockProcessor()
        # create input refs from source features
        i = p._in_refs_validator.validate(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add first level processor
        node_id_1 = graph.add_processor_node(p, i, o)

        # create input refs from first-level outputs
        i = p._in_refs_validator.validate(
            a=graph.get_node_output_ref(node_id_1).y,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add second level processor
        node_id_2 = graph.add_processor_node(p, i, o)

        # check trivial cases
        edges = graph.subgraph_out_edges(graph.subgraph([src_node_id, node_id_1, node_id_2]))
        assert edges == []
        edges = graph.subgraph_out_edges(graph.subgraph([node_id_1, node_id_2]))
        assert edges == []
        edges = graph.subgraph_out_edges(graph.subgraph([node_id_2]))
        assert edges == []

        # check non-trivial cases
        edges = graph.subgraph_out_edges(graph.subgraph([src_node_id]))
        assert edges == [
            (src_node_id, node_id_1, "a"),
            (src_node_id, node_id_1, "b"),
        ]
        edges = graph.subgraph_out_edges(graph.subgraph([src_node_id, node_id_1]))
        assert edges == [
            (node_id_1, node_id_2, "a"),
            (node_id_1, node_id_2, "b"),
        ]

    def test_get_node_output_partition(self):
        # Setup a mock DataFlowGraph object
        graph = DataFlowGraph()
        graph.nodes = {
            "aggregator_node": {
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                DataFlowGraph.NodeAttribute.PARTITION: "partition_1",
            },
            "augmenter_node": {
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.DATA_AUGMENTER,
                DataFlowGraph.NodeAttribute.PARTITION: "partition_2",
            },
            "processor_node": {
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.DATA_PROCESSOR,
                DataFlowGraph.NodeAttribute.PARTITION: "partition_3",
            },
            "source_node": {
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.SOURCE,
                DataFlowGraph.NodeAttribute.PARTITION: "partition_4",
            },
            "const_node": {
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.CONST,
                DataFlowGraph.NodeAttribute.PARTITION: "partition_5",
            },
        }

        # Test that an aggregator node returns the 'AGGREGATED' partition.
        assert (
            graph.get_node_output_partition("aggregator_node")
            == DataFlowGraph.PredefinedPartition.AGGREGATED.value
        )

        # Test that an augmenter node returns its own node_id as the partition.
        node_id = "augmenter_node"
        assert graph.get_node_output_partition("augmenter_node") == "augmenter_node"

        # Test that a non-aggregator/augmenter node returns its own partition.
        for node_id, expected_partition in [
            ("processor_node", "partition_3"),
            ("source_node", "partition_4"),
            ("const_node", "partition_5"),
        ]:
            assert graph.get_node_output_partition(node_id) == expected_partition

        # Test that an invalid node raises a KeyError.
        with pytest.raises(KeyError):
            graph.get_node_output_partition("invalid_node")

    def test_partition_graph_tree_structure(self):
        """Test that the partition graph is a valid tree structure."""
        # Initialize DataFlowGraph instance
        dfg = DataFlowGraph()

        # shorthand to the predefined partitions
        const = DataFlowGraph.PredefinedPartition.CONST.value
        default = DataFlowGraph.PredefinedPartition.DEFAULT.value

        # Define partitions
        dfg.add_node("A", partition=default)
        dfg.add_node("B", partition=const)
        dfg.add_node("C", partition="Partition2")
        dfg.add_node("D", partition="Partition2")
        dfg.add_node("E", partition="Partition3")

        # Define output partitions
        dfg.get_node_output_partition = lambda node_id: {
            "A": "Partition2",
            "B": "Partition2",
            "C": "Partition3",
            "D": "Partition3",
            "E": "Partition3",
        }[node_id]

        # Build the partition graph
        partition_graph = dfg.build_partition_graph()

        # Expected graph structure
        expected_graph = nx.DiGraph()
        expected_graph.add_node(const)
        expected_graph.add_edges_from(
            [
                (default, "Partition2"),
                ("Partition2", "Partition3"),
            ]
        )

        # Validate the structure
        assert nx.is_isomorphic(
            partition_graph, expected_graph
        ), "Partition graph structure does not match expected tree structure."

    def test_partition_graph_with_single_partition(self):
        """Test that the partition graph handles a single partition correctly."""
        dfg = DataFlowGraph()

        # shorthand to the predefined partitions
        const = DataFlowGraph.PredefinedPartition.CONST.value
        default = DataFlowGraph.PredefinedPartition.DEFAULT.value

        # Add nodes all in the same partition
        dfg.add_node("A", partition=default)
        dfg.add_node("B", partition=default)
        dfg.add_node("C", partition=default)

        # Mock the output partition method
        dfg.get_node_output_partition = lambda node_id: default

        # Build the partition graph
        partition_graph = dfg.build_partition_graph()

        # Expected graph should just contain the single partition node
        expected_graph = nx.DiGraph()
        expected_graph.add_nodes_from([default, const])

        # Validate the structure
        assert nx.is_isomorphic(
            partition_graph, expected_graph
        ), "Partition graph structure for single partition is incorrect."

    def test_partition_graph_with_multiple_root_partitions(self):
        dfg = DataFlowGraph()

        # shorthand to the predefined partitions
        default = DataFlowGraph.PredefinedPartition.DEFAULT.value

        # Define partitions
        dfg.add_node("A", partition=default)
        dfg.add_node("B", partition=default)
        dfg.add_node("C", partition="Partition2")
        dfg.add_node("D", partition="Partition3")

        # Define output partitions that would violate the tree structure
        dfg.get_node_output_partition = lambda node_id: {
            "A": "Partition2",
            "B": "Partition3",
            "C": "Partition4",
            "D": "Partition4",
        }[node_id]

        # Expecting an assertion error because 'Partition4' would have multiple incoming edges
        with pytest.raises(
            AssertionError,
            match="The partition graph must be a tree structure",
        ):
            dfg.build_partition_graph()

    def test_infer_node_partition_basic_cases(self):
        graph = DataFlowGraph()

        # source nodes should always be part of the default partition
        assert (
            graph.infer_node_partition(DataFlowGraph.NodeType.SOURCE, None, [])
            == DataFlowGraph.PredefinedPartition.DEFAULT.value
        )

        # constant nodes should always be part of the constant partition
        assert (
            graph.infer_node_partition(DataFlowGraph.NodeType.CONST, None, [])
            == DataFlowGraph.PredefinedPartition.CONST.value
        )

        # cannot infer node partition without valid inputs
        with pytest.raises(AssertionError):
            graph.infer_node_partition(DataFlowGraph.NodeType.DATA_PROCESSOR, None, [])

    @pytest.mark.parametrize(
        "input_partitions,config,expected_partition",
        [
            (
                [
                    DataFlowGraph.PredefinedPartition.CONST.value,
                    DataFlowGraph.PredefinedPartition.CONST.value,
                ],
                BaseNodeConfig(),
                DataFlowGraph.PredefinedPartition.CONST.value,
            ),
            (
                [
                    DataFlowGraph.PredefinedPartition.CONST.value,
                    DataFlowGraph.PredefinedPartition.CONST.value,
                ],
                BaseNodeConfig(is_deterministic=False),
                DataFlowGraph.PredefinedPartition.DEFAULT.value,
            ),
            (
                [
                    DataFlowGraph.PredefinedPartition.AGGREGATED.value,
                    DataFlowGraph.PredefinedPartition.AGGREGATED.value,
                ],
                BaseNodeConfig(),
                DataFlowGraph.PredefinedPartition.AGGREGATED.value,
            ),
            (
                [
                    DataFlowGraph.PredefinedPartition.CONST.value,
                    DataFlowGraph.PredefinedPartition.AGGREGATED.value,
                ],
                BaseNodeConfig(),
                DataFlowGraph.PredefinedPartition.AGGREGATED.value,
            ),
            (
                [
                    DataFlowGraph.PredefinedPartition.DEFAULT.value,
                    DataFlowGraph.PredefinedPartition.DEFAULT.value,
                ],
                BaseNodeConfig(),
                DataFlowGraph.PredefinedPartition.DEFAULT.value,
            ),
            (
                [
                    DataFlowGraph.PredefinedPartition.CONST.value,
                    DataFlowGraph.PredefinedPartition.DEFAULT.value,
                ],
                BaseNodeConfig(),
                DataFlowGraph.PredefinedPartition.DEFAULT.value,
            ),
        ],
    )
    def test_infer_node_partition_simple(self, input_partitions, config, expected_partition):
        mock_input_refs = [MagicMock() for _ in range(len(input_partitions))]
        mock_input_refs_partitions = {
            mock.node_id_: part for mock, part in zip(mock_input_refs, input_partitions)
        }

        mock_node_output_partition = MagicMock()
        mock_node_output_partition.side_effect = mock_input_refs_partitions.get

        with patch(
            "hyped.core.graph.DataFlowGraph.get_node_output_partition",
            mock_node_output_partition,
        ):
            partition = DataFlowGraph().infer_node_partition(
                DataFlowGraph.NodeType.DATA_PROCESSOR, config, mock_input_refs
            )
            assert partition == expected_partition
            mock_node_output_partition.assert_has_calls(
                [call(ref.node_id_) for ref in mock_input_refs], any_order=True
            )

    def test_infer_node_partition_with_partition_graph(self):
        # create a mock partition graph consisting of a simple
        # path and one additional node that is not connected
        mock_p_graph = nx.path_graph(
            [DataFlowGraph.PredefinedPartition.DEFAULT, "A", "B"],
            create_using=nx.DiGraph,
        )
        mock_p_graph.add_node("C")

        # create the data flow graph object and mock the partition graph constructor
        graph = DataFlowGraph()
        graph.build_partition_graph = MagicMock(return_value=mock_p_graph)

        # create mock input references with partitions in the path
        mock_input_refs = [MagicMock(), MagicMock()]
        mock_input_refs_partitions = {
            mock_input_refs[0].node_id_: "A",
            mock_input_refs[1].node_id_: "B",
        }

        with patch(
            "hyped.core.graph.DataFlowGraph.get_node_output_partition",
            mock_input_refs_partitions.get,
        ):
            # infer the partition from the inputs
            partition = graph.infer_node_partition(
                DataFlowGraph.NodeType.DATA_PROCESSOR, BaseNodeConfig(), mock_input_refs
            )
            # make sure the selected partition is the input partition
            # that is deepest in the path
            assert partition == "B"

        # create mock inputs that connect independent partitions
        mock_input_refs_partitions = {
            mock_input_refs[0].node_id_: "A",
            mock_input_refs[1].node_id_: "C",
        }

        with patch(
            "hyped.core.graph.DataFlowGraph.get_node_output_partition",
            mock_input_refs_partitions.get,
        ):
            with pytest.raises(RuntimeError):
                graph.infer_node_partition(
                    DataFlowGraph.NodeType.DATA_PROCESSOR, BaseNodeConfig(), mock_input_refs
                )
