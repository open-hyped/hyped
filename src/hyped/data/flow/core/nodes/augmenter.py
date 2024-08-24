"""Provides base classes for data augmentation in a data flow graph.

This module defines the base classes for data augmentation tasks within
a data flow graph framework. Data augmenters are responsible for filtering or
generating new data samples from on existing ones. This module includes:

- :code:`BaseDataAugmenterConfig`: A base configuration class for data augmenters.
- :code:`BaseDataAugmenter`: The abstract base class for implementing data augmenters,
  which applies transformations to data samples and tracks their indices.

Usage Example:
    Define a custom data augmenter by subclassing `BaseDataAugmenter`:

    .. code-block:: python

        # Import necessary classes from the module
        from hyped.data.flow.core.nodes.augmenter import (
            BaseDataAugmenter, BaseDataAugmenterConfig
        )
        from hyped.data.flow.core.refs.inputs import (
            InputRefs, CheckFeatureEquals
        )
        from hyped.data.flow.core.refs.outputs import (
            OutputRefs, OutputFeature
        )
        from datasets.features.features import Value
        from typing_extensions import Annotated

        class CustomInputRefs(InputRefs):
            x: Annotated[FeatureRef, CheckFeatureEquals(Value("int32"))]

        class CustomOutputRefs(OutputRefs):
            x: Annotated[FeatureRef, OutputFeature(Value("int32"))]

        class CustomAugmenterConfig(BaseDataAugmenterConfig):
            pass

        class CustomAugmenter(BaseDataAugmenter[CustomAugmenterConfig, InputRefs, OutputRefs]):
            async def process(self, inputs, index, rank, io):
                # Define augmentation logic here
                # In this example each sample is duplicated
                yield sample
                yield sample
"""

from __future__ import annotations

from abc import ABC
from itertools import chain, repeat
from typing import Any, Iterable, TypeVar

from typing_extensions import TypeAlias

from ..refs.inputs import InputRefs
from ..refs.outputs import OutputRefs
from .base import BaseNode, BaseNodeConfig, IOContext

Batch: TypeAlias = dict[str, list[Any]]
Sample: TypeAlias = dict[str, Any]


class BaseDataAugmenterConfig(BaseNodeConfig):
    """Base configuration class for data augmenters.

    This class serves as the base configuration for data augmenters,
    inheriting from :code:`BaseNodeConfig` to provide configuration
    functionality specifically for data augmentation tasks.
    """


C = TypeVar("C", bound=BaseDataAugmenterConfig)
I = TypeVar("I", bound=InputRefs)
O = TypeVar("O", bound=OutputRefs)


class BaseDataAugmenter(BaseNode[C, I, O], ABC):
    """Base class for data augmenters in a data flow graph.

    This class represents a data augmenter node in a data flow graph. Data augmenters
    modify or generate new samples from existing ones, which can include filtering or
    creating new data points. Subclasses of :code:`BaseDataAugmenter` must implement
    either the :code:`process` or the :code:`batch_process` method to define how the
    augmentation is applied to the input data.
    """

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> tuple[Batch, list[int]]:
        """Processes a batch of inputs and returns the batch of outputs along with trace indices.

        This method applies the augmentation process to each sample in the batch and tracks
        the index of the source sample for each output. It returns the augmented batch and
        the indices indicating the source of each output sample.

        Args:
            inputs (Batch): A batch of input samples.
            index (list[int]): A list of indices associated with the input samples.
            rank (int): The rank of the augmenter in a distributed processing setting.
            io (IOContext): Context information for the data augmenter's execution.

        Returns:
            tuple[Batch, list[int]]:
                - Batch: A batch of augmented output samples.
                - list[int]: A list of trace indices corresponding indicating the
                  index of the source sample in the input batch that generated the
                  output sample. Specifically the i-th output sample is generated
                  from the trace_index[i]-th input example.
        """
        # apply process function to each sample in the input batch
        keys = inputs.keys()
        samples = (dict(zip(keys, values)) for values in zip(*inputs.values()))

        # apply the process function to each sample in the batch
        # and for each output track the index of the source sample
        # in the input batch, i.e. the trace index
        trace_and_outputs = chain.from_iterable(
            zip(repeat(j), self.process(sample, i, rank, io))
            for j, (i, sample) in enumerate(zip(index, samples))
        )
        # separate the trace index and the outputs
        trace_index, outputs = zip(*trace_and_outputs)

        # pack output samples to batch format
        batch = {key: [d[key] for d in outputs] for key in io.outputs.keys()}
        return batch, trace_index

    # TODO: support async process functions
    def process(
        self, inputs: Sample, index: int, rank: int, io: IOContext
    ) -> Iterable[Sample]:
        """Defines the augmentation logic to be applied to individual samples.

        This method should be overridden by subclasses to define the augmentation logic.

        Args:
            inputs (Sample): A single input sample.
            index (int): The index associated with the input sample.
            rank (int): The rank of the augmenter in a distributed setting.
            io (IOContext): Context information for the data augmenter's execution.

        Returns:
            Iterable[Sample]: An iterable of augmented output samples, which can be multiple samples
                per input sample.
        """
        raise NotImplementedError()
