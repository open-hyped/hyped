from __future__ import annotations

from functools import cache
from typing import Any, Callable, Generic, Hashable, Mapping, TypeVar

from datasets.features.features import Features, FeatureType, Sequence
from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Annotated

from hyped.common.feature_checks import check_feature_equals
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import InputRefs
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef

T = TypeVar("T")
U = TypeVar("U")


class NestedContainer(BaseModel, Generic[T]):
    data: T | dict[Hashable, NestedContainer] | list[NestedContainer]

    @field_validator("data", mode="before")
    def _validate_data(cls, x: Any) -> NestedContainer[T]:
        return (
            {
                k: v if isinstance(v, NestedContainer) else cls(data=v)
                for k, v in x.items()
            }
            if isinstance(x, Mapping)
            else [
                v if isinstance(v, NestedContainer) else cls(data=v) for v in x
            ]
            if isinstance(x, list)
            else x
        )

    def flatten(self) -> dict[Hashable, T]:
        # collect all values in the flattened dictionary with
        # the key being the corresponding path
        flattened = {}
        self.map(flattened.__setitem__, None)
        # return the flat dictionary
        return flattened

    def map(
        self,
        f: Callable[[tuple[Hashable | int], T], U],
        target_type: type[U],
        _path: tuple[Hashable | int] = tuple(),
    ) -> NestedContainer[U]:
        if isinstance(self.data, dict):
            return NestedContainer[target_type](
                data={
                    k: v.map(f, target_type, _path=_path + (k,))
                    for k, v in self.data.items()
                }
            )

        if isinstance(self.data, list):
            return NestedContainer[target_type](
                data=[
                    v.map(f, target_type, _path=_path + (i,))
                    for i, v in enumerate(self.data)
                ]
            )

        return NestedContainer[target_type](data=f(_path, self.data))

    def unpack(self) -> dict | list | T:
        if isinstance(self.data, dict):
            return {k: v.unpack() for k, v in self.data.items()}

        if isinstance(self.data, list):
            return [v.unpack() for v in self.data]

        return self.data


class CollectFeaturesConfig(BaseDataProcessorConfig):
    ...


def _path_to_str(path: tuple[Hashable | int]) -> str:
    return ".".join(map(str, path))


class CollectFeaturesInputRefs(InputRefs):
    collection: NestedContainer[FeatureRef]

    @classmethod
    def type_validator(cls) -> None:
        """Validate the type of input references."""
        pass

    @property
    def named_refs(self) -> dict[str, FeatureRef]:
        return {
            _path_to_str(key): ref
            for key, ref in self.collection.flatten().items()
        }


def _infer_feature_type(
    container: NestedContainer[FeatureRef],
) -> Features:
    if isinstance(container.data, dict):
        return Features(
            {k: _infer_feature_type(v) for k, v in container.data.items()}
        )

    if isinstance(container.data, list):
        assert len(container.data) > 0
        # get the feature types of the list items
        item_types = map(_infer_feature_type, container.data)
        f = next(item_types)
        # make sure all feature types align
        for ff in item_types:
            if not check_feature_equals(f, ff):
                raise TypeError(
                    "Expected all items of a sequence to be of the "
                    "same feature type, got %s != %s" % (str(f), str(ff))
                )
        # build sequence feature
        return Sequence(f, length=len(container.data))

    # return the feature type of the referenced feature
    assert isinstance(container.data, FeatureRef)
    return container.data.feature_


class CollectFeaturesOutputRefs(OutputRefs):
    collected: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, inputs: _infer_feature_type(inputs.collection)
        ),
    ]

    @model_validator(mode="after")
    def _validate_output_feature_type(self) -> CollectFeaturesOutputRefs:
        if not isinstance(self.collected.feature_, Features):
            raise TypeError()
        return self


class CollectFeatures(
    BaseDataProcessor[
        CollectFeaturesConfig,
        CollectFeaturesInputRefs,
        CollectFeaturesOutputRefs,
    ]
):
    @cache
    def _lookup(self, io: IOContext) -> NestedContainer[str]:
        def build_none_sample(feature: FeatureType) -> NestedContainer[None]:
            """Build a sample of None values matching the expected structure of the feature."""
            # parse feature dictionary
            if isinstance(feature, (Features, dict)):
                return NestedContainer[None](
                    data={k: build_none_sample(v) for k, v in feature.items()}
                )
            # parse sequence feature
            if isinstance(feature, Sequence):
                assert feature.length >= 0
                return NestedContainer[None](
                    data=[build_none_sample(feature.feature)] * feature.length
                )
            # not a nested feature
            return NestedContainer[None](data=None)

        # build the lookup container by first generating a sample
        # and then replacing the none values with the lookup strings
        # generated from the paths
        container = build_none_sample(io.outputs["collected"])
        container = container.map(lambda path, _: _path_to_str(path), str)

        return container

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        # convert dict of lists to list of dicts
        keys = inputs.keys()
        samples = [dict(zip(keys, values)) for values in zip(*inputs.values())]
        # collect values from each sample
        return {
            "collected": [
                self._lookup(io).map(lambda _, key: sample[key], Any).unpack()
                for sample in samples
            ]
        }
