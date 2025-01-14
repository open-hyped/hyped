from typing import Any, Hashable
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest

from hyped.core.builder import DataFlowGraphBuilder
from hyped.core.features.dtypes import BoolType as MockType
from hyped.core.features.dtypes import DType, MappingType
from hyped.core.features.reference import ConcreteReference
from hyped.core.graph import DataFlowGraph
from hyped.core.ops.mapping import MappingGetItem
from hyped.core.optim import DataFlowGraphOptimizer

from .utils import build_graph


def node_match(
    n1: dict[DataFlowGraph.NodeAttribute, Any], n2: dict[DataFlowGraph.NodeAttribute, Any]
) -> bool:
    node_type_1 = n1[DataFlowGraph.NodeAttribute.NODE_TYPE]
    node_type_2 = n2[DataFlowGraph.NodeAttribute.NODE_TYPE]

    # for compute nodes, make sure the objects match
    if node_type_1 in {
        DataFlowGraph.NodeType.DATA_PROCESSOR,
        DataFlowGraph.NodeType.DATA_AUGMENTOR,
        DataFlowGraph.NodeType.DATA_AGGREGATOR,
    }:
        return (node_type_1 == node_type_2) and (
            n1[DataFlowGraph.NodeAttribute.NODE_OBJ] == n2[DataFlowGraph.NodeAttribute.NODE_OBJ]
        )
    # for other nodes just make sure the node types match
    return node_type_1 == node_type_2


class TestDataFlowGraphOptimizer:
    @pytest.mark.parametrize(
        "graph, target_graph",
        [
            # simple graph without common sub-expressions
            (
                build_graph(
                    [(0, 1)],
                    {0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
                    {0: "SOURCE", 1: "PROC"},
                ),
                build_graph(
                    [(0, 1)],
                    {0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
                    {0: "SOURCE", 1: "PROC"},
                ),
            ),
            # complex graph without common sub-expression
            (
                build_graph(
                    [(0, 1, "x"), (0, 2, "y")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "A", 2: "B"},
                ),
                build_graph(
                    [(0, 1, "x"), (0, 2, "y")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "A", 2: "B"},
                ),
            ),
            # graph with simple common subexpression
            (
                build_graph(
                    [(0, 1, "x"), (0, 2, "x"), (1, 3, "x"), (2, 3, "y")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "A", 2: "A", 3: "C"},
                ),
                build_graph(
                    [(0, 1, "x"), (1, 2, "x"), (1, 2, "y")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "A", 2: "C"},
                ),
            ),
            # graph with common sub-expression chain
            (
                build_graph(
                    [
                        (0, 1, "x"),
                        (1, 2, "x"),
                        (2, 3, "x"),
                        (0, 4, "x"),
                        (4, 5, "x"),
                        (5, 6, "x"),
                        (3, 7, "a"),
                        (6, 7, "b"),
                    ],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        4: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        5: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        6: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        7: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "A", 2: "B", 3: "C", 4: "A", 5: "B", 6: "C", 7: "OUT"},
                ),
                build_graph(
                    [(0, 1, "x"), (1, 2, "x"), (2, 3, "x"), (3, 4, "a"), (3, 4, "b")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        4: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "A", 2: "B", 3: "C", 4: "OUT"},
                ),
            ),
        ],
    )
    def test_cse(self, graph: DataFlowGraph, target_graph: DataFlowGraph) -> None:
        def node_match(
            n1: dict[DataFlowGraph.NodeAttribute, Any], n2: dict[DataFlowGraph.NodeAttribute, Any]
        ) -> bool:
            # make sure the node type and objects match
            return (
                n1[DataFlowGraph.NodeAttribute.NODE_TYPE]
                == n2[DataFlowGraph.NodeAttribute.NODE_TYPE]
            ) and (
                n1[DataFlowGraph.NodeAttribute.NODE_OBJ] == n2[DataFlowGraph.NodeAttribute.NODE_OBJ]
            )

        # apply common subexpression evaluation to the graph
        # and compare to the target graph
        optim_graph, _ = DataFlowGraphOptimizer().cse(graph)
        assert nx.is_isomorphic(optim_graph, target_graph, node_match=node_match)
        assert optim_graph.src_node_id == graph.src_node_id

    @pytest.mark.parametrize(
        "graph, target_graph, leaf_nodes",
        [
            # simple graph without constants
            (
                build_graph(
                    [(0, 1)],
                    {0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
                    {0: "SOURCE", 1: "PROC"},
                ),
                build_graph(
                    [(0, 1)],
                    {0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
                    {0: "SOURCE", 1: "PROC"},
                ),
                {1},
            ),
            # simple graph without constant expressions to evaluate
            (
                build_graph(
                    [(0, 1, "x"), (2, 1, "y"), (3, 1, "z")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.CONST,
                        3: DataFlowGraph.NodeType.CONST,
                    },
                    {0: "SOURCE", 1: "PROC", 2: "A", 3: "B"},
                ),
                build_graph(
                    [(0, 1, "x"), (2, 1, "y"), (3, 1, "z")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.CONST,
                        3: DataFlowGraph.NodeType.CONST,
                    },
                    {0: "SOURCE", 1: "PROC", 2: "A", 3: "B"},
                ),
                {1},
            ),
            # simple graph with constant expression
            (
                build_graph(
                    [(0, 1, "x"), (2, 4, "y"), (3, 4, "z"), (4, 1, "y")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.CONST,
                        3: DataFlowGraph.NodeType.CONST,
                        4: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "PROC", 2: "A", 3: "B", 4: "PROC"},
                ),
                build_graph(
                    [(0, 1, "x"), (2, 1, "y")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.CONST,
                    },
                    {0: "SOURCE", 1: "PROC", 2: "PROC"},
                ),
                {1},
            ),
            # graph with constant expression as leaf node
            (
                build_graph(
                    [(0, 1, "x"), (2, 4, "y"), (3, 4, "z")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.CONST,
                        3: DataFlowGraph.NodeType.CONST,
                        4: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "PROC", 2: "A", 3: "B", 4: "PROC"},
                ),
                build_graph(
                    [(0, 1, "x")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.CONST,
                    },
                    {0: "SOURCE", 1: "PROC", 2: "PROC"},
                ),
                {4},
            ),
        ],
    )
    @patch("hyped.core.builder.pa", MagicMock())
    @patch("hyped.core.builder.ConstNode", MagicMock())
    @patch("hyped.core.optim.DataFlowGraphExecutor", MagicMock())
    def test_constant_evaluation(
        self, graph: DataFlowGraph, target_graph: DataFlowGraph, leaf_nodes: set[Hashable]
    ) -> None:
        # this test only checks the morphology but not the actual values of the evaluated constants
        # however this is done by the data flow executor which is tested in itself
        with patch(
            "hyped.core.optim.DataFlowGraphExecutor.execute", AsyncMock(return_value=MagicMock())
        ):
            # apply constant evaluation to the graph and compare to the target graph
            optim_graph = DataFlowGraphOptimizer().constant_evaluation(graph, leaf_nodes)
            assert nx.is_isomorphic(optim_graph, target_graph, node_match=node_match)

    @pytest.mark.parametrize(
        "edges_with_keys, src_dtype, accessed_src_dtype",
        [
            (
                [(0, 1, "x"), (0, 2, "y")],
                MappingType.construct({"x": MockType, "y": MockType}),
                MappingType.construct({"x": MockType, "y": MockType}),
            ),
            (
                [(0, 1, "x"), (0, 2, "x")],
                MappingType.construct({"x": MockType, "y": MockType}),
                MappingType.construct({"x": MockType}),
            ),
            (
                [(0, 1, "y"), (0, 2, "x"), (2, 3, "a")],
                MappingType.construct(
                    {"x": MappingType.construct({"a": MockType, "b": MockType}), "y": MockType}
                ),
                MappingType.construct({"x": MappingType.construct({"a": MockType}), "y": MockType}),
            ),
        ],
    )
    def test_accessed_src_dtype_property(
        self,
        edges_with_keys: list[tuple[Hashable, Hashable, str]],
        src_dtype: DType,
        accessed_src_dtype: DType,
    ) -> None:
        builder = DataFlowGraphBuilder()
        builder.source(src_dtype, 0)
        # add all edges assuming that all non-source nodes
        # are get-item nodes
        for u, v, k in edges_with_keys:
            builder.compute(
                MappingGetItem(key=k), {"mapping": ConcreteReference(u, builder.graph, builder)}, v
            )
        # apply accessed fields
        DataFlowGraphOptimizer().apply_accessed_fields(builder.graph)
        # check the accessed source data type
        assert builder.graph.src_dtype == accessed_src_dtype

    @pytest.mark.parametrize(
        "graph, target_graph, leaf_nodes",
        [
            # simple graph with nothing to optimize
            (
                build_graph(
                    [(0, 1)],
                    {0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
                    {0: "SOURCE", 1: "PROC"},
                ),
                build_graph(
                    [(0, 1)],
                    {0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
                    {0: "SOURCE", 1: "PROC"},
                ),
                {1},
            ),
            # graph with simple common subexpression
            (
                build_graph(
                    [(0, 1, "x"), (0, 2, "x"), (1, 3, "x"), (2, 3, "y")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "A", 2: "A", 3: "C"},
                ),
                build_graph(
                    [(0, 1, "x"), (1, 2, "x"), (1, 2, "y")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "A", 2: "C"},
                ),
                {3},
            ),
            # simple graph with constant expression
            (
                build_graph(
                    [(0, 1, "x"), (2, 4, "y"), (3, 4, "z"), (4, 1, "y")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.CONST,
                        3: DataFlowGraph.NodeType.CONST,
                        4: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    {0: "SOURCE", 1: "PROC", 2: "A", 3: "B", 4: "PROC"},
                ),
                build_graph(
                    [(0, 1, "x"), (2, 1, "y")],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.CONST,
                    },
                    {0: "SOURCE", 1: "PROC", 2: "PROC"},
                ),
                {1},
            ),
        ],
    )
    @patch("hyped.core.builder.pa", MagicMock())
    @patch("hyped.core.builder.ConstNode", MagicMock())
    @patch("hyped.core.optim.DataFlowGraphExecutor", MagicMock())
    def test_optimize(
        self, graph: DataFlowGraph, target_graph: DataFlowGraph, leaf_nodes: set[Hashable]
    ) -> None:
        with patch(
            "hyped.core.optim.DataFlowGraphExecutor.execute", AsyncMock(return_value=MagicMock())
        ):
            # apply constant evaluation to the graph and compare to the target graph
            optim_graph = DataFlowGraphOptimizer().optimize(graph, leaf_nodes)
            assert nx.is_isomorphic(optim_graph, target_graph, node_match=node_match)
            assert optim_graph.src_node_id == graph.src_node_id
