"""Module providing the base test for nodes in a data processing pipeline.

This module provides a base class (:class:`BaseNodeTest`) designed to facilitate the
testing of nodes within a data processing pipeline. It defines common functionality
to set up and validate test configurations, handle input data, and check the execution
of nodes.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, ClassVar

import pyarrow as pa
import pydantic
import pytest

from hyped.common._pydantic import TypeAdapterWithArbitraryTypesAllowed
from hyped.core.features.dtypes import MappingType
from hyped.core.features.features import MappingFeature, build_feature_from_annotation
from hyped.core.features.session import ValidationSession
from hyped.core.flow import DataFlow, ExecutableDataFlow
from hyped.core.nodes.base import BaseNode
from hyped.core.typing import Feature, IndexList, Rank


class InvalidTestSetupError(Exception):
    """Exception raised for errors in the test setup or configuration."""

    def __init__(self, message: str):
        """Initializes the :class:`InvalidTestSetup` exception."""
        detailed_message = (
            f"Test Configuration Error: {message}\n"
            "Note: This issue originates from the test configuration and not the "
            "code being tested."
        )
        super().__init__(detailed_message)


class TestSuccessful(Exception):  # noqa: N818
    """Exception raised to interrupt a test case with successful completion.

    By raising this exception, tests can be terminated early without raising
    any errors, providing a convenient way to signal that the test has reached
    its intended conclusion without issues.
    """


class BaseNodeTest(ABC):
    """Base class for testing nodes in a data processing pipeline.

    This abstract base class provides methods to set up test configurations,
    validate input data, build expected outputs, and check the execution of nodes.

    Test subclasses must define the :func:`execute_test` method, which contains the
    specific test logic for a node.
    """

    input_features: ClassVar[dict[str, Feature]]
    """A dictionary defining the input features for the node being tested.

    The keys must match the expected input fields for the node. The values are
    :class:`Feature` annotations that describe the structure and data types of
    the expected inputs.
    """

    input_data: ClassVar[None | list[dict[str, Any]]] = None
    """A batch of input data to be used during the test.

    This should contain dictionaries where each dictionary corresponds to a set
    of input values for the node. If set to :code:`None`, the node execution is
    not tested.
    """

    input_index: ClassVar[None | IndexList] = None
    """A list of indices corresponding to the input data.

    If :code:`None`, the indices will be generated automatically based on the length
    of the input data.
    """

    input_rank: ClassVar[Rank] = 0
    """The rank of the input data for the node."""

    expected_output_feature: ClassVar[None | Feature] = None
    """The expected output feature of the node.

    This defines the structure of the output that the node is expected to produce.
    If :code:`None`, it indicates that no specific output feature is expected by
    default, meaning no test for the output feature is conducted.
    """

    expected_output_data: ClassVar[None | list[Any]] = None
    """A list of expected output data for comparison with the actual output.

    This should contain the expected values for the node's output. If :code:`None`, no
    specific output data is provided by default.
    """

    expected_verification_error: ClassVar[None | Exception | type[Exception]] = None
    """The exception expected during the input verification phase of the node.

    If the node call is expected to fail verification, this class variable should
    be set to the corresponding exception type. If :code:`None`, no verification
    error is expected.
    """

    expected_execution_error: ClassVar[None | Exception | type[Exception]] = None
    """The exception expected during the execution phase of the test.

    If the node is expected to raise an error during execution, this class variable
    should be set to the corresponding exception type. If :code:`None`, no execution
    error is expected, and the test should pass without such an error.
    """

    @pytest.fixture(autouse=True, scope="class")
    def check_test_setup(self) -> None:
        """Fixture to check the validity of the test setup.

        Ensures that the necessary data and configuration are present before
        the test runs.

        Raises:
            InvalidTestSetup: If input data is not provided when expected output data
            or execution errors are defined.
        """
        if type(self).input_features is None:
            raise InvalidTestSetupError("No input features specified.")

        if type(self).input_data is None:
            if type(self).expected_output_data is not None:
                raise InvalidTestSetupError(
                    "Input data must be provided when expected output data is defined."
                )

            if type(self).expected_execution_error is not None:
                raise InvalidTestSetupError(
                    "Input data must be provided when expected execution errors are defined."
                )

    @classmethod
    def build_input_dtype(cls) -> MappingType:
        """Builds the data input type based on :code:`input_features`.

        Returns:
            MappingType: A mapping representing the input feature types.

        Raises:
            InvalidTestSetup: If :code:`input_features` is not properly configured.
        """
        return MappingType.construct(
            {
                key: build_feature_from_annotation(annotation).dtype
                for key, annotation in cls.input_features.items()
            }
        )

    @classmethod
    def build_input_arrow_table(cls) -> pa.Array:
        """Builds a PyArrow Table from the :code:`input_data`.

        Returns:
            pa.Array: A PyArrow array representing the input data.

        Raises:
            InvalidTestSetup: If input data is provided but :code:`input_features` is not set
            correctly.
        """
        if cls.input_data is None:
            return None
        try:
            return pa.Table.from_pylist(cls.input_data, schema=cls.build_input_dtype().arrow_schema)
        except pa.ArrowTypeError as e:
            raise InvalidTestSetupError(
                f"Failed to convert input data to PyArrow Table: {str(e)}. "
                "Please ensure that the input data matches the expected types "
                "defined in 'input_features'."
            ) from e

    @classmethod
    def build_input_index(cls) -> IndexList:
        """Builds an index list for the input data.

        If :code:`input_index` is not provided, a default index based on the
        length of :code:`input_data` is generated.

        Returns:
            IndexList: A list of indices for the input data.

        Raises:
            InvalidTestSetup: If :code:`input_data` is not provided and
                :code:`input_index` is not set.
        """
        if cls.input_index is not None:
            return cls.input_index
        if cls.input_data is None:
            raise InvalidTestSetupError(
                "Input data must be specified to build an input index. "
                "Ensure that 'input_data' is provided or set 'input_index' explicitly."
            )
        return list(range(len(cls.input_data)))

    def setup_data_flow(self, flow: DataFlow) -> MappingFeature:
        """Hook to setup of the data flow graph before adding the node.

        This method can be overridden to implement custom logic for configuring
        the data flow. It is called before the node to be tested is added, allowing
        the setup of additional processing steps, transformations, or configurations
        within the data flow.

        By default, this method returns the source features of the data flow,
        which serve as inputs to the node being tested. Override this method to
        modify the inputs or structure of the data flow as needed.

        The return value of this method is used as the input to the :code:`call`
        method of the node being tested. By default, this method returns the source
        features of the data flow (:code:`flow.source`). Override this method to modify
        the inputs or structure of the data flow as needed for your testing scenario.

        Args:
            flow (DataFlow): The data flow graph to be set up.

        Returns:
            MappingFeature: The inputs to the node, by default derived from
            :code:`flow.source`.
        """
        return flow.source

    def call_node(self, node: BaseNode) -> tuple[DataFlow, Feature]:
        """Calls the node in a new data flow graph.

        This method creates a fresh data flow graph using the :code:`input_features`
        as the source features. It calls the node with the built data flow and checks
        if the output matches the expected type. It also checks for any expected
        verification errors.

        Args:
            node (BaseNode): The node to be tested.

        Returns:
            tuple[DataFlow, Feature]: A tuple containing the data flow and
            the output feature.
        """
        cls = type(self)
        flow = DataFlow(cls.build_input_dtype().hf_feature)
        source = self.setup_data_flow(flow)

        with (
            pytest.raises(cls.expected_verification_error)
            if cls.expected_verification_error is not None
            else nullcontext()
        ):
            # call the node while catching potential verification error
            output = node.call(**source)

        if cls.expected_verification_error is not None:
            # successfully catched verification error
            raise TestSuccessful()

        if cls.expected_output_feature is not None:
            try:
                with ValidationSession() as session:
                    adapter = TypeAdapterWithArbitraryTypesAllowed(cls.expected_output_feature)
                    adapter.validate_python(
                        output,
                        context={
                            "config": node.config,
                            "inputs": {},
                            "typevars": {},
                            "session": session,
                        },
                    )
            except pydantic.ValidationError as e:
                raise AssertionError(
                    f"Output feature mismatch: Expected output feature to conform to "
                    f"{cls.expected_output_feature}, but got {output} of type {type(output)}."
                ) from e

        return flow, output

    def execute_flow(self, flow: ExecutableDataFlow) -> None | pa.Table:
        """Executes the given data flow.

        This method executes an executable data flow and checks if an expected
        execution error occurs. If no error is expected, it validates that the
        execution completes successfully.

        Args:
            flow (ExecutableDataFlow): The data flow to be executed.

        Returns:
            None

        Raises:
            TestSuccessful: If the expected execution error is successfully caught.
        """
        cls = type(self)
        if cls.input_data is not None:
            input_array = cls.build_input_arrow_table()
            input_index = cls.build_input_index()

            with (
                pytest.raises(cls.expected_execution_error)
                if cls.expected_execution_error is not None
                else nullcontext()
            ):
                # execute data flow while cathcing potential execution error
                output_data = flow.arrow_process(input_array, input_index, cls.input_rank)

            if cls.expected_execution_error is not None:
                # successfully catched execution error
                raise TestSuccessful()

            return output_data

    @classmethod
    def check_output_data_matches_expectation(cls, actual_data: Any) -> None:
        """Checks if the output data matches the expected output.

        Args:
            actual_data (pa.Array): The actual output data to be validated.

        Raises:
            AssertionError: If the output data does not match the expected output.
        """
        if cls.expected_output_data is not None:
            assert cls.expected_output_data == actual_data, (
                f"Output data mismatch:\n"
                f"Expected: {cls.expected_output_data}\n"
                f"Actual: {actual_data}"
            )

    def test_case(self) -> None:
        """Main entrypoint.

        Runs the test case by invoking the :func:`execute_test` method. It wraps the
        test execution in a try-except statement to catch :class:`TestSuccessful` exceptions.
        This allows to easily interrupt tests from anywhere without them failing.
        """
        try:
            self.execute_test()
        except TestSuccessful:
            return

    @abstractmethod
    def execute_test(self) -> Any:
        """Test execution.

        Abstract method that must be implemented in subclasses. Contains the specific
        test logic for the node being tested.
        """
        ...
