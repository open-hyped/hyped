import io
import sys
from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

from hyped.core.ops.debug import PrintSample, print_
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.typing import String


@contextmanager
def capture_stdout() -> Generator[io.StringIO, None, None]:
    with io.StringIO() as captured_output:
        sys.stdout = captured_output
        yield captured_output

    # Restore stdout to its original state (the console) - important!
    sys.stdout = sys.__stdout__


class TestPrintSample(BaseDataProcessorTest):
    processor = PrintSample(format_string="{feature}")
    input_features = {"feature": String}
    input_data = [{"feature": "Hello World"}]
    expected_output_feature = None
    expected_output_data = None

    def execute_test(self) -> None:
        with capture_stdout() as stdout:
            # build and execute flow
            flow, _ = self.call_node(type(self).processor)
            flow = flow.build(collect=flow.source, debug=True)
            self.execute_flow(flow)
            # check if sample was printed to stdout
            assert "Hello World" in stdout.getvalue()

    @patch("hyped.core.ops.debug.PrintSample")
    def test_print_operator(self, print_sample_mock: MagicMock) -> None:
        x = MagicMock()
        y = MagicMock()

        # test print without specific template
        print_(x=x, y=y, indent=2)
        print_sample_mock.assert_called_once_with(format_string="x={x} y={y}", indent=2)
        print_sample_mock.return_value.call.assert_called_once_with(x=x, y=y)

        print_sample_mock.reset_mock()
        # test print with custom template
        print_("My Template {x} {y}", x=x, y=y, indent=2)
        print_sample_mock.assert_called_once_with(format_string="My Template {x} {y}", indent=2)
        print_sample_mock.return_value.call.assert_called_once_with(x=x, y=y)
