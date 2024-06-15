from typing import Generic, TypeVar

from hyped.base.config import BaseConfig, BaseConfigurable
from hyped.base.generic import solve_typevar

from ..refs.inputs import InputRefs
from ..refs.outputs import OutputRefs
from ..refs.ref import FeatureRef


class BaseNodeConfig(BaseConfig):
    ...


C = TypeVar("C", bound=BaseNodeConfig)
I = TypeVar("I", bound=None | InputRefs)
O = TypeVar("O", bound=OutputRefs)


class BaseNode(BaseConfigurable[C], Generic[C, I, O]):
    def __init__(self, config: None | C = None, **kwargs) -> None:
        super(BaseNode, self).__init__(config, **kwargs)
        # get input and output reference types from typevars
        self._in_refs_type = solve_typevar(type(self), I)
        self._out_refs_type = solve_typevar(type(self), O)
