"""Module for testing data aggregator nodes in a data processing pipeline.

This module defines the :class:`BaseDataAggregatorTest` class, which extends
the :class:`BaseNodeTest` class to provide specialized functionality for
testing :class:`BaseDataAggregator` nodes.
"""
from typing import Any, ClassVar

from hyped.core.nodes.aggregator import BaseDataAggregator
from hyped.typing import Feature

from .base import BaseNodeTest


class BaseDataAggregatorTest(BaseNodeTest):
    """Base class for testing data aggregators.

    This class extends :class:`BaseNodeTest` and provides additional functionality
    for testing data aggregator nodes specifically.

    Subclasses of :class:`BaseDataAggregatorTest` must define the :code:`aggregator` class
    variable, which represents the specific data aggregator (node) to be tested. The
    :func:`execute_test` method is implemented to execute the aggregator, validate its
    behavior, and check that the resulting output matches expectations.

    This class provides a testing framework for verifying data aggregators by checking
    both the correctness of the aggregated output and handling potential errors or mismatches.
    """

    aggregator: ClassVar[BaseDataAggregator]
    """The data aggregator (node) to be tested.

    This class variable must be set by subclasses to specify the aggregator node
    being tested. It should be an instance of :class:`BaseDataAggregator` or a subclass
    thereof that defines the specific data aggregation logic to be validated in the test.
    """

    expected_output_data: ClassVar[None | Any] = None
    """The expected output data for comparison with the actual output.

    This should contain the expected aggregated output value. If :code:`None`, no
    specific output data is provided by default, and the test will only verify the
    aggregator's general behavior.
    """

    def execute_test(self) -> tuple[Feature, dict[str, Any]]:
        """Executes the test for the data aggregator node.

        This method performs the following steps to test the behavior of the aggregator node:

        1. Calls the aggregator node using the :func:`call_node` method to generate a data flow
           and obtain the output from the node.
        2. Builds the data flow, collecting the source data and defining an aggregation step
           using :func:`flow.build`.
        3. Executes the flow by passing it to the :func:`execute_flow` method, which processes
           the data and generates the aggregated output.
        4. Compares the resulting aggregated output data against the expected output using
           the :func:`check_output_data_matches_expectation` method.

        This method ensures that the aggregator node behaves as expected and that the
        generated aggregated output matches the predefined expectations, including handling any
        specified errors or mismatches.

        Returns:
            output_feature, output_data: tuple[Feature, list[dict[str, Any]]]: The output feature
                that the aggregator call returns, as well as the aggregated data.
        """
        # call node and build flow
        flow, output_feature = self.call_node(type(self).aggregator)
        flow = flow.build(collect=flow.source, aggregate={"output": output_feature})
        # execute the flow and check the output data
        _ = self.execute_flow(flow)
        aggregate_data = flow.aggregates["output"]
        type(self).check_output_data_matches_expectation(aggregate_data)

        return output_feature, aggregate_data
