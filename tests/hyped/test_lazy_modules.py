import pytest

import hyped
import hyped.aggregators._ops
import hyped.processors._ops
from hyped.common.lazy_module import LazyModule


@pytest.mark.parametrize(
    "lazy_module",
    [
        hyped.augmenters,
        hyped.aggregators,
        hyped.aggregators._ops,
        hyped.processors,
        hyped.processors._ops,
        hyped.processors._ops.sequence,
        hyped.io.writers,
    ],
)
def test_lazy_imports(lazy_module):
    assert isinstance(lazy_module, LazyModule)
    for name in lazy_module.__all__:
        try:
            getattr(lazy_module, name)
        except (ImportError, AttributeError):
            pytest.fail(
                f"Cannot import '{name}' from lazy module '{lazy_module}'."
            )
