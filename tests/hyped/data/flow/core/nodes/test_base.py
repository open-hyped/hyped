from unittest.mock import MagicMock, patch

import pytest
from datasets import Features, Value

from hyped.data.flow.core.nodes.base import (
    BaseNode,
    BaseNodeConfig,
    InputRefs,
    OutputRefs,
)
from hyped.data.flow.core.refs.inputs import InputRefsContainer

from ..mock import MockInputRefs, MockOutputRefs


class MockConfig(BaseNodeConfig):
    ...


def test_basics():
    class MockNode(BaseNode[MockConfig, MockInputRefs, MockOutputRefs]):
        ...

    assert MockNode.Config == MockConfig

    with patch(
        "hyped.data.flow.core.nodes.base.InputRefsValidator"
    ) as mock_validator:
        inst = MockNode()
        # check members
        assert inst._in_refs_type == MockInputRefs
        assert inst._out_refs_type == MockOutputRefs

        mock_validator.assert_called_once_with(inst.config, MockInputRefs)
        assert inst.required_input_keys == mock_validator().required_keys

    class MockNode(BaseNode[MockConfig, None, MockOutputRefs]):
        ...

    with patch(
        "hyped.data.flow.core.nodes.base.InputRefsValidator"
    ) as mock_validator:
        inst = MockNode()
        # check members
        assert inst._in_refs_type is type(None)
        assert inst._out_refs_type == MockOutputRefs
        assert inst._in_refs_validator is None

        assert not mock_validator.called
        assert inst.required_input_keys == set()


def test_call():
    # output features matching the MockOutputRefs
    out_features = Features({"y": Value("int64")})

    class MockNode(BaseNode[MockConfig, MockInputRefs, MockOutputRefs]):
        ...

    # create mock flow
    mock_flow = MagicMock()
    mock_flow.add_processor_node = MagicMock(return_value="node_id")
    # create mock input container
    mock_inputs = MagicMock()
    # create mock validator
    mock_validator = MagicMock()
    mock_validator().validate = MagicMock(return_value=mock_inputs)

    # patch input references validator
    with patch(
        "hyped.data.flow.core.nodes.base.InputRefsValidator", mock_validator
    ):
        # create mock inputs
        x = MagicMock()
        x.flow_ = mock_flow
        # create instance of the mock node and call it
        inst = MockNode()
        inst.call(a=x, b=x)

        # make sure the validator was called correctly
        mock_validator().validate.asser_called_once_with(flow=None, a=x, b=x)
        # make sure the node was added to the flow
        mock_flow.add_processor_node.assert_called_once_with(
            inst, mock_inputs, out_features
        )


def test_call_infer_flow_from_inputs():
    # output features matching the MockOutputRefs
    out_features = Features({"y": Value("int64")})

    class MockNode(BaseNode[MockConfig, MockInputRefs, MockOutputRefs]):
        ...

    # create mock flow
    mock_flow = MagicMock()
    mock_flow.add_processor_node = MagicMock(return_value="node_id")
    # create mock input container
    mock_inputs = MagicMock()
    # create mock validator
    mock_validator = MagicMock()
    mock_validator().validate = MagicMock(return_value=mock_inputs)

    # patch input references validator
    with patch(
        "hyped.data.flow.core.nodes.base.InputRefsValidator", mock_validator
    ):
        # create mock inputs
        a, b = MagicMock(), MagicMock()
        # create instance of the mock node and call it
        inst = MockNode()
        inst.call(flow=mock_flow, a=a, b=b)

        # make sure the validator was called correctly
        mock_validator().validate.asser_called_once_with(flow=None, a=a, b=b)
        # make sure the node was added to the flow
        mock_flow.add_processor_node.assert_called_once_with(
            inst, mock_inputs, out_features
        )


def test_call_no_inputs():
    # output features matching the MockOutputRefs
    out_features = Features({"y": Value("int64")})

    class MockNode(BaseNode[MockConfig, None, MockOutputRefs]):
        ...

    # create mock flow
    mock_flow = MagicMock()
    mock_flow.add_processor_node = MagicMock(return_value="node_id")
    # create mock input container
    mock_inputs = MagicMock()
    # create mock validator
    mock_validator = MagicMock()
    mock_validator().validate = MagicMock(return_value=mock_inputs)

    # patch input references validator
    with patch(
        "hyped.data.flow.core.nodes.base.InputRefsValidator", mock_validator
    ):
        # create instance of the mock node and call it
        inst = MockNode()
        inst.call(flow=mock_flow)
        # no validator for nodes without inputs
        assert not mock_validator().validate.called
        # make sure the node was added to the flow
        mock_flow.add_processor_node.assert_called_once_with(
            inst, None, out_features
        )

    with pytest.raises(RuntimeError):
        # create mock instance and call it without specifying the flow
        inst = MockNode()
        inst.call()
