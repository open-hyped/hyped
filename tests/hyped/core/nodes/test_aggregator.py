from inspect import Parameter, Signature, _ParameterKind
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from hyped.core.nodes.aggregator import (
    BaseDataAggregator,
    BaseDataAggregatorConfig,
    DataAggregationManager,
)
from hyped.core.nodes.base import ProcessMode, RunContext
from hyped.core.typing import Bool, Int


class TestDataAggregationManager:
    @patch(
        "hyped.core.nodes.aggregator.mp.Manager",
        MagicMock(return_value=MagicMock(dict=MagicMock(side_effect=dict))),
    )
    @patch("hyped.core.nodes.aggregator.pa.array", MagicMock(side_effect=lambda v, **_: v))
    def test_initialization(self) -> None:
        # create mock aggregators and run contexts
        initial_values = [MagicMock(), MagicMock(), MagicMock()]
        initial_states = [MagicMock(), MagicMock(), MagicMock()]
        mock_aggregators = [
            MagicMock(
                spec=BaseDataAggregator,
                seed=MagicMock(return_value=(initial_values[0], initial_states[0])),
            ),
            MagicMock(
                spec=BaseDataAggregator,
                seed=MagicMock(return_value=(initial_values[1], initial_states[1])),
            ),
            MagicMock(
                spec=BaseDataAggregator,
                seed=MagicMock(return_value=(initial_values[2], initial_states[2])),
            ),
        ]
        mock_run_contexts = [MagicMock(), MagicMock(), MagicMock()]

        # create aggregation manager instance
        aggregation_manager = DataAggregationManager(mock_aggregators, mock_run_contexts)
        # check initial values
        assert aggregation_manager.values_proxy[mock_run_contexts[0].node_id] == [initial_values[0]]
        assert aggregation_manager.values_proxy[mock_run_contexts[1].node_id] == [initial_values[1]]
        assert aggregation_manager.values_proxy[mock_run_contexts[2].node_id] == [initial_values[2]]
        # check initial state
        assert aggregation_manager._state_buffer[mock_run_contexts[0].node_id] == initial_states[0]
        assert aggregation_manager._state_buffer[mock_run_contexts[1].node_id] == initial_states[1]
        assert aggregation_manager._state_buffer[mock_run_contexts[2].node_id] == initial_states[2]

    @pytest.mark.asyncio
    @patch(
        "hyped.core.nodes.aggregator.mp.Manager",
        MagicMock(return_value=MagicMock(dict=MagicMock(side_effect=dict))),
    )
    @patch("hyped.core.nodes.aggregator.pa.array", MagicMock(side_effect=lambda v, **_: v))
    @patch("hyped.core.nodes.aggregator.replace", MagicMock(side_effect=lambda x, **_: x))
    async def test_aggregate(self) -> None:
        mock_value = MagicMock()
        mock_state = MagicMock()
        mock_new_value = MagicMock()
        mock_new_state = MagicMock()

        mock_aggregator = MagicMock(
            spec=BaseDataAggregator,
            seed=MagicMock(return_value=(mock_value, mock_state)),
            update=AsyncMock(return_value=(mock_new_value, mock_new_state)),
        )
        mock_run_context = MagicMock()

        # create the data aggregator
        aggregation_manager = DataAggregationManager([mock_aggregator], [mock_run_context])

        prepared_input_for_extract = {"x": MagicMock()}
        prepared_context_for_extract = MagicMock(node_id=mock_run_context.node_id)
        # create the mock process mode for the extract function
        mock_extract_mode = MagicMock(
            batched=True,
            spec=ProcessMode,
            prepare=MagicMock(
                return_value=[(prepared_context_for_extract, prepared_input_for_extract)]
            ),
        )

        prepared_value_for_update = MagicMock()
        prepared_context_for_update = MagicMock(node_id=mock_run_context.node_id)
        # create the mock process mode for the update function
        mock_update_mode = MagicMock(
            batched=False,
            spec=ProcessMode,
            prepare=MagicMock(
                return_value=[(prepared_context_for_update, {"value": prepared_value_for_update})]
            ),
        )

        with patch("hyped.core.nodes.aggregator.ProcessMode") as mock_process_mode:
            # specify mock process modes for extract and update
            mock_process_mode.from_decorated_fn.side_effect = {
                mock_aggregator.extract: mock_extract_mode,
                mock_aggregator.update: mock_update_mode,
            }.get

            # run the aggregator
            mock_input = {"x": MagicMock()}
            await aggregation_manager.aggregate(mock_aggregator, mock_run_context, mock_input)

            # make sure aggregator extract was called as expected
            mock_extract_mode.prepare.assert_called_once_with(mock_run_context, **mock_input)
            mock_aggregator.extract.assert_called_once_with(
                prepared_context_for_extract, **prepared_input_for_extract
            )

            # make sure the lock for the aggregator was aquired
            aggregation_manager._locks[mock_run_context.node_id].acquire.assert_called_once()

            # check the aggregator update was called as expected
            mock_update_mode.prepare.assert_called_once_with(
                prepared_context_for_extract, value=[mock_value]
            )
            mock_aggregator.update.assert_called_once_with(
                prepared_context_for_update,
                prepared_value_for_update,
                mock_state,
                mock_aggregator.extract.return_value,
            )
            mock_update_mode.finalize.assert_called_once_with(
                prepared_context_for_extract, [mock_new_value]
            )

            # check that the buffers where updated as expected
            assert (
                aggregation_manager.values_proxy[mock_run_context.node_id]
                == mock_update_mode.finalize.return_value
            )
            assert aggregation_manager._state_buffer[mock_run_context.node_id] == mock_new_state

            # make sure the lock for the aggregator was released
            aggregation_manager._locks[mock_run_context.node_id].release.assert_called_once()


class MockConfig(BaseDataAggregatorConfig):
    @classmethod
    @property
    def type_id(self) -> str:
        return str(uuid4())


class TestBaseDataAggregator:
    def test_signature(self):
        class MockDataAggregator(BaseDataAggregator[MockConfig]):
            def seed(self, ctx: RunContext) -> tuple[Int, int]:
                ...

            def extract(self, ctx: RunContext, x: Bool, y: Int) -> object:
                ...

            def update(
                self, ctx: RunContext, val: Int, state: int, extracted: object
            ) -> tuple[Bool, int]:
                ...

        processor: MockDataAggregator = MockDataAggregator()
        assert processor.signature == Signature(
            parameters=[
                Parameter(name="x", kind=_ParameterKind.POSITIONAL_OR_KEYWORD, annotation=Bool),
                Parameter(name="y", kind=_ParameterKind.POSITIONAL_OR_KEYWORD, annotation=Int),
            ],
            return_annotation=Bool,
        )

    def test_init_subclass(self) -> None:
        class MockConfig(BaseDataAggregatorConfig):
            ...

        class MockDataAggregator(BaseDataAggregator[MockConfig]):
            def initialize(self, ctx: RunContext) -> tuple[Int, int]:
                ...

            def extract(self, ctx: RunContext, x: Bool, y: Int) -> object:
                ...

            def update(
                self, ctx: RunContext, val: Int, state: int, extracted: object
            ) -> tuple[Bool, int]:
                ...

        # make sure the process mode default is set
        ProcessMode.from_decorated_fn(MockDataAggregator.extract)
        ProcessMode.from_decorated_fn(MockDataAggregator.update)

        with pytest.raises(TypeError):
            # missing the update function
            class MockDataAggregator(BaseDataAggregator[MockConfig]):
                def initialize(self, ctx: RunContext) -> tuple[Int, int]:
                    ...

                def extract(self, ctx: RunContext, x: Bool, y: Int) -> object:
                    ...

        with pytest.raises(TypeError):
            # extract function missing context input
            class MockDataAggregator(BaseDataAggregator[MockConfig]):
                def initialize(self, ctx: RunContext) -> tuple[Int, int]:
                    ...

                def extract(self, x: Bool, y: Int) -> object:
                    ...

                def update(
                    self, ctx: RunContext, val: Int, state: int, extracted: object
                ) -> tuple[Bool, int]:
                    ...
