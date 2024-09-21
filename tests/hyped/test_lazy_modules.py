import pytest

import hyped
import hyped.nodes._ops
from hyped.common.lazy_module import LazyModule


@pytest.mark.parametrize(
    "lazy_module",
    [
        hyped.nodes,
        hyped.nodes._ops,
        hyped.io.writers,
        hyped.ops,
        hyped.core,
    ],
)
def test_lazy_imports(lazy_module):
    assert isinstance(lazy_module, LazyModule)
    for name in lazy_module.__all__:
        try:
            getattr(lazy_module, name)
        except (ImportError, AttributeError):
            pytest.fail(f"Cannot import '{name}' from lazy module '{lazy_module}'.")
