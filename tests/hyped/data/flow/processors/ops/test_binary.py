from datasets import Features, Sequence, Value

from hyped.data.flow.processors.ops import binary
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestEquals(BaseDataProcessorTest):
    processor_type = binary.Equals
    processor_config = binary.EqualsConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, False, False]}


class TestNotEquals(BaseDataProcessorTest):
    processor_type = binary.NotEquals
    processor_config = binary.NotEqualsConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, True]}


class TestLessThan(BaseDataProcessorTest):
    processor_type = binary.LessThan
    processor_config = binary.LessThanConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, False]}


class TestLessThanOrEqual(BaseDataProcessorTest):
    processor_type = binary.LessThanOrEqual
    processor_config = binary.LessThanOrEqualConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, True, False]}


class TestGreaterThan(BaseDataProcessorTest):
    processor_type = binary.GreaterThan
    processor_config = binary.GreaterThanConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, True]}


class TestGreaterThanOrEqual(BaseDataProcessorTest):
    processor_type = binary.GreaterThanOrEqual
    processor_config = binary.GreaterThanOrEqualConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, False, True]}


class TestLogicalAnd(BaseDataProcessorTest):
    processor_type = binary.LogicalAnd
    processor_config = binary.LogicalAndConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, True]}


class TestLogicalOr(BaseDataProcessorTest):
    processor_type = binary.LogicalOr
    processor_config = binary.LogicalOrConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, True]}


class TestLogicalXOr(BaseDataProcessorTest):
    processor_type = binary.LogicalXOr
    processor_config = binary.LogicalXOrConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, False]}


class TestClosedOp_Add_Int32_Int32(BaseDataProcessorTest):
    processor_type = binary.Add
    processor_config = binary.AddConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    expected_output_features = Features({"result": Value("int32")})


class TestClosedOp_Add_Int16_Int32(BaseDataProcessorTest):
    processor_type = binary.Add
    processor_config = binary.AddConfig()

    input_features = Features({"a": Value("int16"), "b": Value("int32")})
    expected_output_features = Features({"result": Value("int32")})


class TestClosedOp_Add_Float32_Float32(BaseDataProcessorTest):
    processor_type = binary.Add
    processor_config = binary.AddConfig()

    input_features = Features({"a": Value("float32"), "b": Value("float32")})
    expected_output_features = Features({"result": Value("float32")})


class TestClosedOp_Add_Int32_Float32(BaseDataProcessorTest):
    processor_type = binary.Add
    processor_config = binary.AddConfig()

    input_features = Features({"a": Value("int32"), "b": Value("float32")})
    expected_output_features = Features({"result": Value("float32")})


class TestAdd(BaseDataProcessorTest):
    processor_type = binary.Add
    processor_config = binary.AddConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [2, 4, 6]}


class TestSub(BaseDataProcessorTest):
    processor_type = binary.Sub
    processor_config = binary.SubConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 0, 0]}


class TestMul(BaseDataProcessorTest):
    processor_type = binary.Mul
    processor_config = binary.MulConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 4, 9]}


class TestPow(BaseDataProcessorTest):
    processor_type = binary.Pow
    processor_config = binary.PowConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 4, 27]}


class TestMod(BaseDataProcessorTest):
    processor_type = binary.Mod
    processor_config = binary.ModConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 5, 6], "b": [2, 2, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 1, 2]}


class TestFloorDiv(BaseDataProcessorTest):
    processor_type = binary.FloorDiv
    processor_config = binary.FloorDivConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 6, 9], "b": [2, 4, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int64")})
    expected_output_data = {"result": [2, 1, 2]}


class TestDiv(BaseDataProcessorTest):
    processor_type = binary.TrueDiv
    processor_config = binary.TrueDivConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 6, 9], "b": [2, 4, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("float64")})
    expected_output_data = {"result": [2.0, 1.5, 2.25]}


class TestElementWiseEquals(BaseDataProcessorTest):
    processor_type = binary.Equals
    processor_config = binary.EqualsConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [[0, 0, 1], [0, 1, 0]], "b": [[1, 0, 1], [1, 1, 1]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {
        "result": [[False, True, True], [False, True, False]]
    }


class TestElementWiseNotEquals(BaseDataProcessorTest):
    processor_type = binary.NotEquals
    processor_config = binary.NotEqualsConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [[0, 0, 1], [0, 1, 0]], "b": [[1, 0, 1], [1, 1, 1]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {
        "result": [[True, False, False], [True, False, True]]
    }


class TestElementWiseLessThan(BaseDataProcessorTest):
    processor_type = binary.LessThan
    processor_config = binary.LessThanConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [[1, 2, 3], [3, 5, 6]], "b": [[0, 1, 4], [4, 5, 5]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {
        "result": [[False, False, True], [True, False, False]]
    }


class TestElementWiseLessThanOrEqual(BaseDataProcessorTest):
    processor_type = binary.LessThanOrEqual
    processor_config = binary.LessThanOrEqualConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [[1, 2, 3], [4, 5, 6]], "b": [[0, 1, 3], [4, 5, 5]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {
        "result": [[False, False, True], [True, True, False]]
    }


class TestElementWiseGreaterThan(BaseDataProcessorTest):
    processor_type = binary.GreaterThan
    processor_config = binary.GreaterThanConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [[0, 1, 4], [4, 5, 5]], "b": [[1, 2, 3], [3, 5, 6]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {
        "result": [[False, False, True], [True, False, False]]
    }


class TestElementWiseGreaterThanOrEqual(BaseDataProcessorTest):
    processor_type = binary.GreaterThanOrEqual
    processor_config = binary.GreaterThanOrEqualConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [[0, 1, 3], [4, 5, 5]], "b": [[1, 2, 3], [4, 5, 6]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {
        "result": [[False, False, True], [True, True, False]]
    }


class TestElementWiseLogicalAnd(BaseDataProcessorTest):
    processor_type = binary.LogicalAnd
    processor_config = binary.LogicalAndConfig()

    input_features = Features(
        {"a": Sequence(Value("bool")), "b": Sequence(Value("bool"))}
    )
    input_data = {
        "a": [[True, True, False], [True, True]],
        "b": [[True, False, True], [False, True]],
    }
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[True, False, False], [False, True]]}


class TestElementWiseLogicalOr(BaseDataProcessorTest):
    processor_type = binary.LogicalOr
    processor_config = binary.LogicalOrConfig()

    input_features = Features(
        {"a": Sequence(Value("bool")), "b": Sequence(Value("bool"))}
    )
    input_data = {
        "a": [[True, True, False], [True, False]],
        "b": [[True, False, True], [False, False]],
    }
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[True, True, True], [True, False]]}


class TestElementWiseLogicalXOr(BaseDataProcessorTest):
    processor_type = binary.LogicalXOr
    processor_config = binary.LogicalXOrConfig()

    input_features = Features(
        {"a": Sequence(Value("bool")), "b": Sequence(Value("bool"))}
    )
    input_data = {
        "a": [[True, True, False], [True, False]],
        "b": [[True, False, True], [False, False]],
    }
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[False, True, True], [True, False]]}


class TestElementWiseAdd(BaseDataProcessorTest):
    processor_type = binary.Add
    processor_config = binary.AddConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("float64"))}
    )
    input_data = {"a": [[1, 2, 3], [4, 5]], "b": [[0, 1, -1], [10, -10]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[1, 3, 2], [14, -5]]}


class TestElementWiseSub(BaseDataProcessorTest):
    processor_type = binary.Sub
    processor_config = binary.SubConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("float64"))}
    )
    input_data = {"a": [[1, 2, 3], [4, 5]], "b": [[0, 1, -1], [10, -10]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[1, 1, 4], [-6, 15]]}


class TestElementWiseMul(BaseDataProcessorTest):
    processor_type = binary.Mul
    processor_config = binary.MulConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("float64"))}
    )
    input_data = {
        "a": [[1, 2, 3], [4, 5]],
        "b": [[0.5, 2.0, -1.0], [10.0, -10.0]],
    }
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[0.5, 4.0, -3.0], [40.0, -50.0]]}


class TestElementWisePow(BaseDataProcessorTest):
    processor_type = binary.Pow
    processor_config = binary.PowConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [[2, 3, 4], [1, 5]], "b": [[3, 2, 1], [2, 3]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[8, 9, 4], [1, 125]]}


class TestElementWiseMod(BaseDataProcessorTest):
    processor_type = binary.Mod
    processor_config = binary.ModConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [[10, 15, 20], [8, 12]], "b": [[3, 4, 5], [3, 5]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[1, 3, 0], [2, 2]]}


class TestElementWiseFloorDiv(BaseDataProcessorTest):
    processor_type = binary.FloorDiv
    processor_config = binary.FloorDivConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [[10, 15, 20], [8, 12]], "b": [[3, 4, 5], [3, 5]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("int64"))})
    expected_output_data = {"result": [[3, 3, 4], [2, 2]]}


class TestElementWiseTrueDiv(BaseDataProcessorTest):
    processor_type = binary.TrueDiv
    processor_config = binary.TrueDivConfig()

    input_features = Features(
        {"a": Sequence(Value("float32")), "b": Sequence(Value("float32"))}
    )
    input_data = {
        "a": [[10.0, 15.5, 20.0], [8.0, 12.5]],
        "b": [[2.0, 4, 5.0], [2, 5.0]],
    }
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[5.0, 3.875, 4.0], [4, 2.5]]}


class TestElementWiseValueBroadcastFirstArg(BaseDataProcessorTest):
    processor_type = binary.Add
    processor_config = binary.AddConfig()

    input_features = Features(
        {"a": Value("float64"), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [5.8, 9.0], "b": [[1, 2, 3], [4, 5]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[6.8, 7.8, 8.8], [13, 14]]}


class TestElementWiseValueBroadcastSecondArg(BaseDataProcessorTest):
    processor_type = binary.Add
    processor_config = binary.AddConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Value("float64")}
    )
    input_data = {"a": [[1, 2, 3], [4, 5]], "b": [5.8, 9.0]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[6.8, 7.8, 8.8], [13, 14]]}
