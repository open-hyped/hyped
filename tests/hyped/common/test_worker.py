from unittest.mock import patch

import pytest

import hyped.common._worker
from hyped.common._worker import get_worker_info, set_worker_info


@pytest.fixture(autouse=True)
def reset_worker_info():
    hyped.common._worker._worker_info = None


@patch("hyped.common._worker.is_package_installed", return_value=False)
def test_set_worker_info(mock_is_package_installed):
    # Set worker info
    worker_info = set_worker_info(rank=1, num_workers=4, seed=123, extra_data="test")

    # Verify the worker info object
    assert worker_info.rank == 1
    assert worker_info.num_workers == 4
    assert worker_info.seed == 123
    assert worker_info.ctx.extra_data == "test"


@patch("hyped.common._worker.is_package_installed", return_value=False)
def test_get_worker_info_after_set(mock_is_package_installed):
    # Set worker info and retrieve it
    set_worker_info(rank=2, num_workers=8, seed=456, extra_data="test")
    retrieved_info = get_worker_info()

    # Verify that the retrieved info matches the set info
    assert retrieved_info.rank == 2
    assert retrieved_info.num_workers == 8
    assert retrieved_info.seed == 456
    assert retrieved_info.ctx.extra_data == "test"

    # update ctx
    retrieved_info.ctx.new_extra_data = "test-2"
    assert get_worker_info().ctx.new_extra_data == "test-2"


@patch("hyped.common._worker.is_package_installed", return_value=False)
def test_set_worker_info_already_set(mock_is_package_installed):
    # Set worker info
    set_worker_info(rank=3, num_workers=10, seed=789)

    # Ensure setting worker info again raises an AssertionError
    with pytest.raises(AssertionError, match="Worker info already set"):
        set_worker_info(rank=4, num_workers=12, seed=999)


@patch("hyped.common._worker.is_package_installed", return_value=True)
def test_get_worker_info_with_torch_no_info(mock_is_installed):
    # Mock that torch worker info is None
    worker_info = get_worker_info()

    # Ensure that the function returns None if torch worker info is not available
    assert worker_info is None


@patch("hyped.common._worker.is_package_installed", return_value=True)
def test_set_worker_info_with_torch(mock_is_installed):
    # Mock PyTorch's WorkerInfo class

    # Set worker info
    worker_info = set_worker_info(rank=7, num_workers=3, seed=999, dataset="my_dataset")

    # Verify that the returned worker info is correct
    assert worker_info.rank == 7
    assert worker_info.num_workers == 3
    assert worker_info.seed == 999
    assert worker_info.ctx.dataset == "my_dataset"

    # update ctx
    worker_info.ctx.new_extra_data = "test-2"
    # get worker
    worker_info = get_worker_info()

    # Verify that the returned worker info is correct
    assert worker_info.rank == 7
    assert worker_info.num_workers == 3
    assert worker_info.seed == 999
    assert worker_info.ctx.dataset == "my_dataset"
    assert worker_info.ctx.new_extra_data == "test-2"
