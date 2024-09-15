"""Module for managing worker information in a multiprocessing context.

This module provides functionality to set and retrieve worker information,
such as rank, number of workers, and seed, which is useful for managing
worker-specific configurations in parallel processing setups.

This module optionally relies on PyTorch's worker information if the `torch`
package is installed and used in a multiprocessing context.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from .typing import Rank
from .utils import is_package_installed


@dataclass(frozen=True)
class WorkerInfo(object):
    """Holds information about the worker in a multiprocessing context."""

    rank: Rank
    """The rank or ID of the worker."""
    num_workers: int
    """The total number of workers."""
    seed: int
    """The seed used for random number generation in this worker."""

    ctx: SimpleNamespace = field(default_factory=SimpleNamespace)
    """A namespace for any additional worker context data."""


_worker_info: None | WorkerInfo = None


def get_worker_info() -> None | WorkerInfo:
    """Retrieves the current worker's information.

    This function will attempt to retrieve worker information from PyTorch
    if the 'torch' package is installed and the worker is part of a multiprocessing
    setup.

    Returns:
        WorkerInfo | None: The worker information, or None if the worker info is not set
        or if not running in a multiprocessing context.
    """
    global _worker_info

    if is_package_installed("torch"):
        from torch.utils.data._utils.worker import get_worker_info as torch_get_worker_info

        info = torch_get_worker_info()

        return (
            None
            if info is None
            else WorkerInfo(
                rank=info.id,
                num_workers=info.num_workers,
                seed=info.seed,
                ctx=SimpleNamespace(dataset=info.dataset),
            )
        )

    return _worker_info


def set_worker_info(rank: Rank, num_workers: int, seed: int, **ctx: Any) -> WorkerInfo:
    """Sets the worker information for the current process.

    Args:
        rank (Rank): The rank or ID of the worker.
        num_workers (int): The total number of workers.
        seed (int): The seed for random number generation in this worker.
        **ctx (Any): Additional context data to be stored in the worker's context (ctx).

    Returns:
        WorkerInfo: The newly created worker information.

    Raises:
        AssertionError: If worker information is already set, preventing reassignment.
    """
    global _worker_info

    # make sure the worker info is not set yet
    assert get_worker_info() is None, "Worker info already set."

    # set worker info
    _worker_info = WorkerInfo(rank, num_workers, seed, SimpleNamespace(**ctx))

    if is_package_installed("torch"):
        import torch.utils.data._utils.worker

        # set pytorch worker info
        torch.utils.data._utils.worker._worker_info = torch.utils.data._utils.worker.WorkerInfo(
            id=rank, num_workers=num_workers, seed=seed, dataset=ctx.get("dataset")
        )

    return _worker_info
