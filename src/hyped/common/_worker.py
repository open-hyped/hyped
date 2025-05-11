"""Module for managing worker information in a multiprocessing context.

This module provides functionality to set and retrieve worker information,
such as rank, number of workers, and seed, which is useful for managing
worker-specific configurations in parallel processing setups.

This module optionally relies on PyTorch's worker information if the `torch`
package is installed and used in a multiprocessing context.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TypeAlias

from .utils import is_package_installed

Rank: TypeAlias = int
"""The multi-processing execution rank.

In distributed or parallel computing, the rank is an integer identifier for a process.
This type alias is typically used to represent the rank of a process in a multi-processing
or distributed environment, where each process is assigned a unique rank.
"""


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


def get_worker_info() -> None | WorkerInfo:
    """Retrieves the current worker's information.

    This function attempts to gather worker-specific information from supported
    multiprocessing libraries, such as PyTorch or Crane. If the worker is not
    part of a multiprocessing setup, or the relevant library is not available,
    the function returns :code:`None`.

    Returns:
        WorkerInfo | None: An instance of :class:`WorkerInfo` containing details
        about the worker (e.g., ID, number of workers, seed, and context), or
        :code:`None` if worker information is unavailable.
    """
    if is_package_installed("torch"):
        from torch.utils.data._utils.worker import get_worker_info as torch_get_worker_info

        info = torch_get_worker_info()

        if info is not None:
            # create worker info from torch worker info
            ctx = SimpleNamespace(dataset=info.dataset)
            return WorkerInfo(info.id, info.num_workers, info.seed, ctx)

    elif is_package_installed("crane"):
        from crane.core.worker import get_worker_info as crane_get_worker_info

        info = crane_get_worker_info()

        if info is not None:
            return WorkerInfo(info.rank, info.num_workers, info.seed, info.ctx)

    return None
