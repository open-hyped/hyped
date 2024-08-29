from datasets import Features, Sequence, Value

from hyped.data.flow.processors.ops.sequence.element_wise import unary
from hyped.data.flow.processors.ops.sequence.element_wise import binary
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestElementWiseNeg(BaseDataProcessorTest):
    processor_type = unary.ElementWiseNeg
    processor_config = unary.ElementWiseNegConfig()

    input_features = Features({"a": Sequence(Value("int32"))})
    input_data = {"a": [[1, 2, 3], [1, 2]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[-1, -2, -3], [-1, -2]]}


class TestElementWiseAbs(BaseDataProcessorTest):
    processor_type = unary.ElementWiseAbs
    processor_config = unary.ElementWiseAbsConfig()

    input_features = Features({"a": Sequence(Value("float32"))})
    input_data = {"a": [[1, -2, 3], [-1, -2]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float32"))})
    expected_output_data = {"result": [[1, 2, 3], [1, 2]]}


class TestElementWiseInvert(BaseDataProcessorTest):
    processor_type = unary.ElementWiseInvert
    processor_config = unary.ElementWiseInvertConfig()

    input_features = Features({"a": Sequence(Value("int64"))})
    input_data = {"a": [[1, 2, 3], [-1, -2, -3]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("int64"))})
    expected_output_data = {"result": [[-2, -3, -4], [0, 1, 2]]}


class TestElementWiseBooleanInvert(BaseDataProcessorTest):
    processor_type = unary.ElementWiseBooleanInvert
    processor_config = unary.ElementWiseBooleanInvertConfig()

    input_features = Features({"a": Sequence(Value("bool"))})
    input_data = {"a": [[True, False, False], [False, True, False]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[False, True, True], [True, False, True]]}


class TestElementWiseEquals(BaseDataProcessorTest):
    processor_type = binary.ElementWiseEquals
    processor_config = binary.ElementWiseEqualsConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))})
    input_data = {"a": [[0, 0, 1], [0, 1, 0]], "b": [[1, 0, 1], [1, 1, 1]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[False, True, True], [False, True, False]]}


class TestElementWiseNotEquals(BaseDataProcessorTest):
    processor_type = binary.ElementWiseNotEquals
    processor_config = binary.ElementWiseNotEqualsConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))})
    input_data = {"a": [[0, 0, 1], [0, 1, 0]], "b": [[1, 0, 1], [1, 1, 1]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[True, False, False], [True, False, True]]}


class TestElementWiseLessThan(BaseDataProcessorTest):
    processor_type = binary.ElementWiseLessThan
    processor_config = binary.ElementWiseLessThanConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))})
    input_data = {"a": [[1, 2, 3], [3, 5, 6]], "b": [[0, 1, 4], [4, 5, 5]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[False, False, True], [True, False, False]]}


class TestElementWiseLessThanOrEqual(BaseDataProcessorTest):
    processor_type = binary.ElementWiseLessThanOrEqual
    processor_config = binary.ElementWiseLessThanOrEqualConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))})
    input_data = {"a": [[1, 2, 3], [4, 5, 6]], "b": [[0, 1, 3], [4, 5, 5]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[False, False, True], [True, True, False]]}


class TestElementWiseGreaterThan(BaseDataProcessorTest):
    processor_type = binary.ElementWiseGreaterThan
    processor_config = binary.ElementWiseGreaterThanConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))})
    input_data = {"a": [[0, 1, 4], [4, 5, 5]], "b": [[1, 2, 3], [3, 5, 6]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[False, False, True], [True, False, False]]}


class TestElementWiseGreaterThanOrEqual(BaseDataProcessorTest):
    processor_type = binary.ElementWiseGreaterThanOrEqual
    processor_config = binary.ElementWiseGreaterThanOrEqualConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))})
    input_data = {"a": [[0, 1, 3], [4, 5, 5]], "b": [[1, 2, 3], [4, 5, 6]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[False, False, True], [True, True, False]]}


class TestElementWiseLogicalAnd(BaseDataProcessorTest):
    processor_type = binary.ElementWiseLogicalAnd
    processor_config = binary.ElementWiseLogicalAndConfig()

    input_features = Features({"a": Sequence(Value("bool")), "b": Sequence(Value("bool"))})
    input_data = {"a": [[True, True, False], [True, True]], "b": [[True, False, True], [False, True]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[True, False, False], [False, True]]}


class TestElementWiseLogicalOr(BaseDataProcessorTest):
    processor_type = binary.ElementWiseLogicalOr
    processor_config = binary.ElementWiseLogicalOrConfig()

    input_features = Features({"a": Sequence(Value("bool")), "b": Sequence(Value("bool"))})
    input_data = {"a": [[True, True, False], [True, False]], "b": [[True, False, True], [False, False]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[True, True, True], [True, False]]}


class TestElementWiseLogicalXOr(BaseDataProcessorTest):
    processor_type = binary.ElementWiseLogicalXOr
    processor_config = binary.ElementWiseLogicalXOrConfig()

    input_features = Features({"a": Sequence(Value("bool")), "b": Sequence(Value("bool"))})
    input_data = {"a": [[True, True, False], [True, False]], "b": [[True, False, True], [False, False]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {"result": [[False, True, True], [True, False]]}


class TestElementWiseAdd(BaseDataProcessorTest):
    processor_type = binary.ElementWiseAdd
    processor_config = binary.ElementWiseAddConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("float64"))})
    input_data = {"a": [[1, 2, 3], [4, 5]], "b": [[0, 1, -1], [10, -10]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[1, 3, 2], [14, -5]]}


class TestElementWiseSub(BaseDataProcessorTest):
    processor_type = binary.ElementWiseSub
    processor_config = binary.ElementWiseSubConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("float64"))})
    input_data = {"a": [[1, 2, 3], [4, 5]], "b": [[0, 1, -1], [10, -10]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[1, 1, 4], [-6, 15]]}


class TestElementWiseMul(BaseDataProcessorTest):
    processor_type = binary.ElementWiseMul
    processor_config = binary.ElementWiseMulConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("float64"))})
    input_data = {"a": [[1, 2, 3], [4, 5]], "b": [[0.5, 2.0, -1.0], [10.0, -10.0]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[0.5, 4.0, -3.0], [40.0, -50.0]]}


class TestElementWisePow(BaseDataProcessorTest):
    processor_type = binary.ElementWisePow
    processor_config = binary.ElementWisePowConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))})
    input_data = {"a": [[2, 3, 4], [1, 5]], "b": [[3, 2, 1], [2, 3]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[8, 9, 4], [1, 125]]}


class TestElementWiseMod(BaseDataProcessorTest):
    processor_type = binary.ElementWiseMod
    processor_config = binary.ElementWiseModConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))})
    input_data = {"a": [[10, 15, 20], [8, 12]], "b": [[3, 4, 5], [3, 5]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[1, 3, 0], [2, 2]]}


class TestElementWiseFloorDiv(BaseDataProcessorTest):
    processor_type = binary.ElementWiseFloorDiv
    processor_config = binary.ElementWiseFloorDivConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))})
    input_data = {"a": [[10, 15, 20], [8, 12]], "b": [[3, 4, 5], [3, 5]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[3, 3, 4], [2, 2]]}


class TestElementWiseTrueDiv(BaseDataProcessorTest):
    processor_type = binary.ElementWiseTrueDiv
    processor_config = binary.ElementWiseTrueDivConfig()

    input_features = Features({"a": Sequence(Value("float32")), "b": Sequence(Value("float32"))})
    input_data = {"a": [[10.0, 15.5, 20.0], [8.0, 12.5]], "b": [[2.0, 4, 5.0], [2, 5.0]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float32"))})
    expected_output_data = {"result": [[5.0, 3.875, 4.0], [4, 2.5]]}


class TestElementWiseValueBroadcastFirstArg(BaseDataProcessorTest):
    processor_type = binary.ElementWiseAdd
    processor_config = binary.ElementWiseAddConfig()

    input_features = Features({"a": Value("float64"), "b": Sequence(Value("int32"))})
    input_data = {"a": [5.8, 9.0], "b": [[1, 2, 3], [4, 5]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[6.8, 7.8, 8.8], [13, 14]]}


class TestElementWiseValueBroadcastSecondArg(BaseDataProcessorTest):
    processor_type = binary.ElementWiseAdd
    processor_config = binary.ElementWiseAddConfig()

    input_features = Features({"a": Sequence(Value("int32")), "b": Value("float64")})
    input_data = {"a": [[1, 2, 3], [4, 5]], "b": [5.8, 9.0]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float64"))})
    expected_output_data = {"result": [[6.8, 7.8, 8.8], [13, 14]]}
