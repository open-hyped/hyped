from datasets import Features, Value

from hyped.data.flow.processors.ops.value import numeric
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestNeg(BaseDataProcessorTest):
    processor_type = numeric.Neg
    processor_config = numeric.NegConfig()

    input_features = Features({"a": Value("int32")})
    input_data = {"a": [1, -2, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [-1, 2, 0]}


class TestInvert(BaseDataProcessorTest):
    processor_type = numeric.Invert
    processor_config = numeric.InvertConfig()

    input_features = Features({"a": Value("int32")})
    input_data = {"a": [1, -2, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [-2, 1, -1]}


class TestAbs(BaseDataProcessorTest):
    processor_type = numeric.Abs
    processor_config = numeric.AbsConfig()

    input_features = Features({"a": Value("int32")})
    input_data = {"a": [1, -2, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 2, 0]}


class TestLessThan(BaseDataProcessorTest):
    processor_type = numeric.LessThan
    processor_config = numeric.LessThanConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, False]}


class TestLessThanOrEqual(BaseDataProcessorTest):
    processor_type = numeric.LessThanOrEqual
    processor_config = numeric.LessThanOrEqualConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, True, False]}


class TestGreaterThan(BaseDataProcessorTest):
    processor_type = numeric.GreaterThan
    processor_config = numeric.GreaterThanConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, True]}


class TestGreaterThanOrEqual(BaseDataProcessorTest):
    processor_type = numeric.GreaterThanOrEqual
    processor_config = numeric.GreaterThanOrEqualConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, False, True]}


class TestClosedOp_Add_Int32_Int32(BaseDataProcessorTest):
    processor_type = numeric.Add
    processor_config = numeric.AddConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    expected_output_features = Features({"result": Value("int32")})


class TestClosedOp_Add_Int16_Int32(BaseDataProcessorTest):
    processor_type = numeric.Add
    processor_config = numeric.AddConfig()

    input_features = Features({"a": Value("int16"), "b": Value("int32")})
    expected_output_features = Features({"result": Value("int32")})


class TestClosedOp_Add_Float32_Float32(BaseDataProcessorTest):
    processor_type = numeric.Add
    processor_config = numeric.AddConfig()

    input_features = Features({"a": Value("float32"), "b": Value("float32")})
    expected_output_features = Features({"result": Value("float32")})


class TestClosedOp_Add_Int32_Float32(BaseDataProcessorTest):
    processor_type = numeric.Add
    processor_config = numeric.AddConfig()

    input_features = Features({"a": Value("int32"), "b": Value("float32")})
    expected_output_features = Features({"result": Value("float32")})


class TestAdd(BaseDataProcessorTest):
    processor_type = numeric.Add
    processor_config = numeric.AddConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [2, 4, 6]}


class TestSub(BaseDataProcessorTest):
    processor_type = numeric.Sub
    processor_config = numeric.SubConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 0, 0]}


class TestMul(BaseDataProcessorTest):
    processor_type = numeric.Mul
    processor_config = numeric.MulConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 4, 9]}


class TestPow(BaseDataProcessorTest):
    processor_type = numeric.Pow
    processor_config = numeric.PowConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 4, 27]}


class TestMod(BaseDataProcessorTest):
    processor_type = numeric.Mod
    processor_config = numeric.ModConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 5, 6], "b": [2, 2, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 1, 2]}


class TestFloorDiv(BaseDataProcessorTest):
    processor_type = numeric.FloorDiv
    processor_config = numeric.FloorDivConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 6, 9], "b": [2, 4, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [2, 1, 2]}


class TestDiv(BaseDataProcessorTest):
    processor_type = numeric.TrueDiv
    processor_config = numeric.TrueDivConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 6, 9], "b": [2, 4, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("float32")})
    expected_output_data = {"result": [2.0, 1.5, 2.25]}
