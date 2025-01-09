from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hyped.common._worker import WorkerInfo, get_worker_info
from hyped.common.utils import is_package_installed

if is_package_installed("torch"):
    from torch.utils.data._utils.worker import WorkerInfo as TorchWorkerInfo
else:
    TorchWorkerInfo = MagicMock()
if is_package_installed("crane"):
    from crane.core.worker import WorkerInfo as CraneWorkerInfo
else:
    CraneWorkerInfo = MagicMock()


@pytest.mark.skipif(condition=not is_package_installed("torch"), reason="Torch not installed.")
@pytest.mark.skipif(condition=not is_package_installed("crane"), reason="Crane not installed.")
@pytest.mark.parametrize(
    "installed_packages, torch_info, crane_info, expected_info",
    [
        ({"torch": False, "crane": False}, None, None, None),
        (
            {"torch": True, "crane": False},
            TorchWorkerInfo(id=1, num_workers=2, seed=1337, dataset=None),
            None,
            WorkerInfo(rank=1, num_workers=2, seed=1337, ctx=SimpleNamespace(dataset=None)),
        ),
        (
            {"torch": False, "crane": True},
            None,
            CraneWorkerInfo(rank=1, num_workers=2, seed=1337, ctx=SimpleNamespace()),
            WorkerInfo(rank=1, num_workers=2, seed=1337, ctx=SimpleNamespace()),
        ),
        (
            {"torch": True, "crane": True},
            TorchWorkerInfo(id=1, num_workers=2, seed=1337, dataset=None),
            None,
            WorkerInfo(rank=1, num_workers=2, seed=1337, ctx=SimpleNamespace(dataset=None)),
        ),
        (
            {"torch": True, "crane": True},
            None,
            CraneWorkerInfo(rank=1, num_workers=2, seed=1337, ctx=SimpleNamespace()),
            WorkerInfo(rank=1, num_workers=2, seed=1337, ctx=SimpleNamespace()),
        ),
        (
            # prioritize torch
            {"torch": True, "crane": True},
            TorchWorkerInfo(id=2, num_workers=2, seed=1337, dataset=None),
            CraneWorkerInfo(rank=1, num_workers=2, seed=1337, ctx=SimpleNamespace()),
            WorkerInfo(rank=2, num_workers=2, seed=1337, ctx=SimpleNamespace(dataset=None)),
        ),
    ],
)
def test_get_worker_info(
    installed_packages: dict[str, bool],
    torch_info: None | TorchWorkerInfo,
    crane_info: None | CraneWorkerInfo,
    expected_info: None | WorkerInfo,
) -> None:
    with (
        patch(
            "hyped.common._worker.is_package_installed",
            MagicMock(side_effect=installed_packages.get),
        ),
        patch("torch.utils.data._utils.worker.get_worker_info", MagicMock(return_value=torch_info)),
        patch("crane.core.worker.get_worker_info", MagicMock(return_value=crane_info)),
    ):
        actual_info = get_worker_info()

        if expected_info is not None:
            assert actual_info == expected_info
        else:
            assert actual_info is None
