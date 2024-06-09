from datasets import Features, Value

from hyped.data.flow.processors.ops.binary import (
    Add,
    AddConfig,
    ClosedOp,
    ClosedOpConfig,
    Comparator,
    ComparatorConfig,
    Div,
    DivConfig,
    Equals,
    EqualsConfig,
    FloorDiv,
    FloorDivConfig,
    GreaterThan,
    GreaterThanConfig,
    GreaterThanOrEqual,
    GreaterThanOrEqualConfig,
    LessThan,
    LessThanConfig,
    LessThanOrEqual,
    LessThanOrEqualConfig,
    LogicalAnd,
    LogicalAndConfig,
    LogicalOp,
    LogicalOpConfig,
    LogicalOr,
    LogicalOrConfig,
    LogicalXOr,
    LogicalXOrConfig,
    Mod,
    ModConfig,
    Mul,
    MulConfig,
    NotEquals,
    NotEqualsConfig,
    Pow,
    PowConfig,
    Sub,
    SubConfig,
)
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestComparator(BaseDataProcessorTest):
    processor_type = Comparator
    processor_config = ComparatorConfig(op=lambda a, b: False)

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, False]}


class TestEquals(BaseDataProcessorTest):
    processor_type = Equals
    processor_config = EqualsConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, False, False]}


class TestNotEquals(BaseDataProcessorTest):
    processor_type = NotEquals
    processor_config = NotEqualsConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, True]}


class TestLessThan(BaseDataProcessorTest):
    processor_type = LessThan
    processor_config = LessThanConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, False]}


class TestLessThanOrEqual(BaseDataProcessorTest):
    processor_type = LessThanOrEqual
    processor_config = LessThanOrEqualConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, True, False]}


class TestGreaterThan(BaseDataProcessorTest):
    processor_type = GreaterThan
    processor_config = GreaterThanConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, True]}


class TestGreaterThanOrEqual(BaseDataProcessorTest):
    processor_type = GreaterThanOrEqual
    processor_config = GreaterThanOrEqualConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, False, True]}


class TestLogicalOp(BaseDataProcessorTest):
    processor_type = LogicalOp
    processor_config = LogicalOpConfig(op=lambda a, b: False)

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, False]}


class TestLogicalAnd(BaseDataProcessorTest):
    processor_type = LogicalAnd
    processor_config = LogicalAndConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, True]}


class TestLogicalOr(BaseDataProcessorTest):
    processor_type = LogicalOr
    processor_config = LogicalOrConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, True]}


class TestLogicalXOr(BaseDataProcessorTest):
    processor_type = LogicalXOr
    processor_config = LogicalXOrConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, False]}


class TestClosedOp_Int32_Int32(BaseDataProcessorTest):
    processor_type = ClosedOp
    processor_config = ClosedOpConfig(op=lambda a, b: 0)

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 0], "b": [0, 0, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 0, 0]}


class TestClosedOp_Int16_Int32(BaseDataProcessorTest):
    processor_type = ClosedOp
    processor_config = ClosedOpConfig(op=lambda a, b: 0)

    input_features = Features({"a": Value("int16"), "b": Value("int32")})
    input_data = {"a": [0, 0, 0], "b": [0, 0, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 0, 0]}


class TestClosedOp_Float32_Float32(BaseDataProcessorTest):
    processor_type = ClosedOp
    processor_config = ClosedOpConfig(op=lambda a, b: 0.0)

    input_features = Features({"a": Value("float32"), "b": Value("float32")})
    input_data = {"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("float32")})
    expected_output_data = {"result": [0.0, 0.0, 0.0]}


class TestClosedOp_Int32_Float32(BaseDataProcessorTest):
    processor_type = ClosedOp
    processor_config = ClosedOpConfig(op=lambda a, b: 0.0)

    input_features = Features({"a": Value("int32"), "b": Value("float32")})
    input_data = {"a": [0, 0, 0], "b": [0.0, 0.0, 0.0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("float32")})
    expected_output_data = {"result": [0.0, 0.0, 0.0]}


class TestAdd(BaseDataProcessorTest):
    processor_type = Add
    processor_config = AddConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [2, 4, 6]}


class TestSub(BaseDataProcessorTest):
    processor_type = Sub
    processor_config = SubConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 0, 0]}


class TestMul(BaseDataProcessorTest):
    processor_type = Mul
    processor_config = MulConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 4, 9]}


class TestPow(BaseDataProcessorTest):
    processor_type = Pow
    processor_config = PowConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 4, 27]}


class TestMod(BaseDataProcessorTest):
    processor_type = Mod
    processor_config = ModConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 5, 6], "b": [2, 2, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 1, 2]}


class TestFloorDiv(BaseDataProcessorTest):
    processor_type = FloorDiv
    processor_config = FloorDivConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 6, 9], "b": [2, 4, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [2, 1, 2]}


class TestDiv(BaseDataProcessorTest):
    processor_type = Div
    processor_config = DivConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 6, 9], "b": [2, 4, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("float32")})
    expected_output_data = {"result": [2.0, 1.5, 2.25]}
