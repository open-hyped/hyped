from datasets import Features, Value

from hyped.nodes._ops.aggregate import (
    MeanAggregator,
    MeanAggregatorConfig,
    SumAggregator,
    SumAggregatorConfig,
)
from tests.hyped.nodes.base import BaseDataAggregatorTest


class TestSum(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = SumAggregator
    aggregator_config = SumAggregatorConfig()
    # input
    input_features = Features({"x": Value("int32")})
    input_data = {"x": list(range(100))}
    input_index = list(range(100))

    expected_value_feature = Features({"value": Value("float64")})
    # expected initial value
    expected_initial_value = {"value": 0}
    expected_initial_state = None
    # expected output
    expected_output_value = {"value": sum(range(100))}
    expected_output_state = None


class TestSumWithOffset(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = SumAggregator
    aggregator_config = SumAggregatorConfig(start=-10)
    # input
    input_features = Features({"x": Value("int32")})
    input_data = {"x": list(range(100))}
    input_index = list(range(100))

    expected_value_feature = Features({"value": Value("float64")})
    # expected initial value
    expected_initial_value = {"value": -10}
    expected_initial_state = None
    # expected output
    expected_output_value = {"value": sum(range(100)) - 10}
    expected_output_state = None


class TestMean(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = MeanAggregator
    aggregator_config = MeanAggregatorConfig()
    # input
    input_features = Features({"x": Value("int32")})
    input_data = {"x": list(range(100))}
    input_index = list(range(100))

    expected_value_feature = Features({"value": Value("float64")})
    # expected initial value
    expected_initial_value = {"value": 0}
    expected_initial_state = 0
    # expected output
    expected_output_value = {"value": sum(range(100)) / 100.0}
    expected_output_state = 100


class TestMeanWithOffset(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = MeanAggregator
    aggregator_config = MeanAggregatorConfig(start=-10, start_count=5)
    # input
    input_features = Features({"x": Value("int32")})
    input_data = {"x": list(range(100))}
    input_index = list(range(100))

    expected_value_feature = Features({"value": Value("float64")})
    # expected initial value
    expected_initial_value = {"value": -10}
    expected_initial_state = 5
    # expected output
    expected_output_value = {"value": (sum(range(100)) - 10 * 5) / (100 + 5)}
    expected_output_state = 105
