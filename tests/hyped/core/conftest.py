import pytest

from .mock import MockAggregator, MockProcessor


@pytest.fixture(autouse=True)
def reset_mocks():
    MockProcessor.process.reset_mock()
    MockAggregator.initialize.reset_mock()
    MockAggregator.extract.reset_mock()
    MockAggregator.update.reset_mock()
