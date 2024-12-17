"""Module for testing data processor nodes in a data processing pipeline.

This module defines the :class:`BaseDataProcessorTest` class, which extends
the :class:`BaseNodeTest` class to provide specialized functionality for
testing data :class:`BaseDataProcessor` nodes.
"""
from typing import ClassVar

from hyped.core.nodes.processor import BaseDataProcessor

from .base import BaseNodeTest


class BaseDataProcessorTest(BaseNodeTest):
    """Base class for testing data processors.

    This class extends :class:`BaseNodeTest` and provides additional functionality
    for testing data processor nodes specifically.

    Subclasses of :class:`BaseDataProcessorTest` must define the :code:`processor` class variable,
    which represents the specific data processor (node) to be tested. The :func:`execute_test`
    method is implemented to execute the processor, validate its behavior, and check that
    the resulting output matches expectations.

    This class provides a testing framework for verifying data processors by checking
    both the correctness of the output and handling potential errors or mismatches.
    """

    processor: ClassVar[BaseDataProcessor]
    """The data processor (node) to be tested.

    This class variable must be set by subclasses to specify the processor node
    being tested. It should be an instance of `BaseDataProcessor` or a subclass
    thereof that defines the specific data transformation or processing logic to be
    validated in the test.
    """

    def execute_test(self) -> None:
        """Executes the test for the data processor node.

        This method performs the following steps to test the behavior of the processor node:

        1. Calls the processor node using the :func:`call_node` method to generate a data flow
        and obtain the output from the node.
        2. Builds the data flow and collects the output into the flow using :func:`flow.build`.
        3. Executes the flow by passing it to the :func:`execute_flow` method, which processes the
           data and generates output.
        4. Compares the resulting output data against the expected output using
           the :func:`check_output_data_matches_expectation` method.

        This method ensures that the processor node behaves as expected and that the
        generated output matches the predefined expectations, including handling any
        specified errors or mismatches.
        """
        # call node and build flow
        flow, output = self.call_node(type(self).processor)
        flow = flow.build(collect={"output": output})
        # execute the flow and check the output data
        output_data = self.execute_flow(flow)
        type(self).check_output_data_matches_expectation(output_data["output"])
