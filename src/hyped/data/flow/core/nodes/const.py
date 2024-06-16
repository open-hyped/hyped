from typing import Annotated, Any

from datasets import Dataset
from datasets.features.features import FeatureType
from pydantic import model_validator

from hyped.common.feature_checks import raise_object_matches_feature
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef

from .base import BaseNode, BaseNodeConfig


class ConstConfig(BaseNodeConfig):
    value: Any

    ftype: None | FeatureType = None

    @model_validator(mode="after")
    def _validate_feature_type(self):
        if self.ftype is None:
            # infer feature type from value
            ds = Dataset.from_dict({"x": [self.value]})
            self.ftype = ds.features["x"]

        else:
            # make sure feature type aligns with the value
            raise_object_matches_feature(self.value, self.ftype)

        return self


class ConstOutputRefs(OutputRefs):
    value: Annotated[FeatureRef, LambdaOutputFeature(lambda c, _: c.ftype)]


class Const(BaseNode[ConstConfig, None, ConstOutputRefs]):
    def get_const_batch(self, batch_size: int) -> list[Any]:
        return {"value": [self.config.value] * batch_size}

    def to(self, flow: object) -> ConstOutputRefs:
        # add node to flow
        out_features = self._out_refs_type.build_features(self.config, None)
        node_id = flow.add_processor_node(self, None, out_features)
        # return output feature reference
        return self._out_refs_type(flow, node_id, out_features)
