"""Provides a mean data aggregator for computing the mean of input features.

The :class:`MeanAggregator` data aggregator calculates the mean of specified input features
over batches of data. It supports a variety of numeric and boolean input types and can be
configured with an initial starting value for the mean calculation. This aggregator is
useful for tasks where an average of certain features is required.
"""

from typing import Annotated

import pyarrow as pa
from datasets import Value
from pydantic import Field
from typing_extensions import Unpack

from hyped.common.feature_checks import SCALAR_TYPES
from hyped.common.typing import Aggregate, Batch, IndexList, Rank
from hyped.core.nodes.aggregator import BaseDataAggregator, BaseDataAggregatorConfig, IOContext
from hyped.core.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.core.refs.outputs import OutputFeature, OutputRefs
from hyped.core.refs.ref import FeatureRef


class SimpleAggregatorInputRefs(InputRefs):
    """A collection of input references for simple aggregators.

    This class defines the expected input feature for simple aggregators such as the
    :class:`SumAggregatorr` and :class:`MeanAggregato`.
    """

    x: Annotated[
        FeatureRef,
        CheckFeatureEquals(SCALAR_TYPES),
    ]
    """
    The input feature reference for the aggregation. Must be a numerical type.
    """


class SimpleAggregatorOutputRefs(OutputRefs):
    """A collection of output references for simple aggregators.

    This class defines the expected output feature for simple aggregators such as
    the :class:`SumAggregatorr` and :class:`MeanAggregato`.
    """

    value: Annotated[FeatureRef, OutputFeature(Value("float64"))]
    """
    The output feature reference representing the aggregated value.
    This value is always of type :code:`float64`.
    """


class MeanAggregatorConfig(BaseDataAggregatorConfig):
    """Configuration for the :class:`MeanAggregator`.

    This class defines the configuration options for the :class:`MeanAggregator`,
    including the starting value for the mean calculation.
    """

    start: float = 0
    """The initial value to start the mean calculation. Defaults to 0."""

    start_count: float = Field(default=0, ge=0)
    """The initial count to start the mean calculation. Defaults to 0."""


class MeanAggregator(
    BaseDataAggregator[MeanAggregatorConfig, SimpleAggregatorInputRefs, SimpleAggregatorOutputRefs]
):
    """A data aggregator that computes the mean of input features.

    This class implements a data aggregator that calculates the mean of the specified
    input feature :code:`x` over batches of data.
    """

    def initialize(self, io: IOContext) -> tuple[Aggregate, float]:
        """Initializes the aggregation with the starting value and a count of 0.

        Args:
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[Aggregate, float]: A tuple containing the starting value and a count of 0.
        """
        return {"value": self.config.start}, self.config.start_count

    async def extract(
        self, inputs: Batch, index: IndexList, rank: Rank, io: IOContext
    ) -> tuple[float, int]:
        """Extracts the sum of the input feature :code:`x` and the count of items in the batch.

        Args:
            inputs (Batch): The batch of input data.
            index (IndexList): The indices of the current batch.
            rank (Rank): The rank of the current process.
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[float, int]: The sum of the input feature :code:`x` and the count
            of items in the batch.
        """
        return pa.compute.sum(inputs["x"]), len(index)

    async def update(
        self, val: float, ctx: tuple[float, int], state: float, io: IOContext
    ) -> tuple[Aggregate, float]:
        """Updates the running mean with the extracted value and count.

        Args:
            val (float): The current running mean.
            ctx (tuple[float, int]): The extracted sum and count from the current batch.
            state (float): The current count of items.
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[Aggregate, float]: The updated running mean and the new count of items.
        """
        ext_val, ext_count = ctx
        return {"value": (val["value"] * state + ext_val) / (state + ext_count)}, (
            state + ext_count
        )

    def call(self, **kwargs: Unpack[SimpleAggregatorInputRefs]) -> SimpleAggregatorOutputRefs:
        """Execute the MeanAggregator to compute the mean value.

        Args:
            x (FeatureRef): The reference to the feature to aggregate.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            SimpleAggregatorOutputRefs: The output references containing the computed mean value.
        """
        return super(MeanAggregator, self).call(**kwargs)


class SumAggregatorConfig(BaseDataAggregatorConfig):
    """Configuration for the :class:`SumAggregator`.

    This class defines the configuration options for the :class:`SumAggregator`,
    including the starting value for the summation.
    """

    start: float = 0
    """The initial value to start the mean calculation. Defaults to 0."""


class SumAggregator(
    BaseDataAggregator[SumAggregatorConfig, SimpleAggregatorInputRefs, SimpleAggregatorOutputRefs]
):
    """A data aggregator that computes the sum of input features.

    This class implements a data aggregator that calculates the sum of the specified
    input feature :code:`x` over batches of data.
    """

    def initialize(self, io: IOContext) -> tuple[Aggregate, None]:
        """Initializes the aggregation with the starting value from the configuration.

        Args:
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[float, None]: A tuple containing the starting value and None for the state.
        """
        return {"value": self.config.start}, None

    async def extract(self, inputs: Batch, index: IndexList, rank: Rank, io: IOContext) -> float:
        """Extracts the sum of the input feature :code:`x` from the batch of data.

        Args:
            inputs (Batch): The batch of input data.
            index (IndexList): The indices of the current batch.
            rank (Rank): The rank of the current process.
            io (IOContext): Context information for the aggregator execution.

        Returns:
            float: The sum of the input feature :code:`x` for the current batch.
        """
        return pa.compute.sum(inputs["x"]).as_py()

    async def update(
        self, val: float, ctx: float, state: None, io: IOContext
    ) -> tuple[Aggregate, None]:
        """Updates the running total with the extracted value.

        Args:
            val (float): The current running total.
            ctx (float): The extracted sum from the current batch.
            state (None): The context, which is not used in this aggregator.
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[dict[str, float], None]: The updated running total and None for the state.
        """
        return {"value": val["value"] + ctx}, None

    def call(self, **kwargs: Unpack[SimpleAggregatorInputRefs]) -> SimpleAggregatorOutputRefs:
        """Execute the SumAggregator to compute the mean value.

        Args:
            x (FeatureRef): The reference to the feature to aggregate.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            SimpleAggregatorOutputRefs: The output references containing the computed mean value.
        """
        return super(SumAggregator, self).call(**kwargs)
