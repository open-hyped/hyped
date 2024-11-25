from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyped.core.nodes.aggregator import BaseDataAggregator, DataAggregationManager


class TestDataAggregationManager:
    @patch("hyped.core.nodes.aggregator._manager", MagicMock(dict=MagicMock(side_effect=dict)))
    @patch("hyped.core.nodes.aggregator.pa.array", MagicMock(side_effect=lambda v, **_: v))
    def test_initialization(self) -> None:
        # create mock aggregators and run contexts
        initial_values = [MagicMock(), MagicMock(), MagicMock()]
        initial_states = [MagicMock(), MagicMock(), MagicMock()]
        mock_aggregators = [
            MagicMock(
                spec=BaseDataAggregator,
                initialize=MagicMock(return_value=(initial_values[0], initial_states[0])),
            ),
            MagicMock(
                spec=BaseDataAggregator,
                initialize=MagicMock(return_value=(initial_values[1], initial_states[1])),
            ),
            MagicMock(
                spec=BaseDataAggregator,
                initialize=MagicMock(return_value=(initial_values[2], initial_states[2])),
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
    @patch("hyped.core.nodes.aggregator._manager", MagicMock(dict=MagicMock(side_effect=dict)))
    @patch("hyped.core.nodes.aggregator.pa.array", MagicMock(side_effect=lambda v, **_: v))
    @patch("hyped.core.nodes.aggregator.replace", MagicMock(side_effect=lambda x, **_: x))
    async def test_safe_update(self) -> None:
        mock_value = MagicMock()
        mock_state = MagicMock()
        mock_new_value = MagicMock()
        mock_new_state = MagicMock()

        mock_aggregator = MagicMock(
            spec=BaseDataAggregator,
            initialize=MagicMock(return_value=(mock_value, mock_state)),
            update=AsyncMock(return_value=(mock_new_value, mock_new_state)),
        )
        mock_run_context = MagicMock()

        # create the data aggregator
        aggregation_manager = DataAggregationManager([mock_aggregator], [mock_run_context])

        with patch("hyped.core.nodes.aggregator.ProcessMode") as mock_process_mode:
            prepared_input = MagicMock()
            prepared_context = MagicMock()

            mock_mode = MagicMock(
                prepare=MagicMock(return_value=[(prepared_context, {"value": prepared_input})]),
            )
            mock_process_mode.from_decorated_fn.return_value = mock_mode

            # run the safe update
            mock_extracted = MagicMock()
            await aggregation_manager._safe_update(
                mock_run_context, mock_aggregator, mock_extracted
            )

            # make sure the lock for the aggregator was aquired
            aggregation_manager._locks[mock_run_context.node_id].acquire.assert_called_once()
            # check the aggregator was called as expected
            mock_mode.prepare.assert_called_once_with(mock_run_context, value=[mock_value])
            mock_aggregator.update.assert_called_once_with(
                prepared_context, prepared_input, mock_state, mock_extracted
            )
            mock_mode.finalize.assert_called_once_with(mock_run_context, [mock_new_value])
            # check that the buffers where updated as expected
            assert (
                aggregation_manager.values_proxy[mock_run_context.node_id]
                == mock_mode.finalize.return_value
            )
            assert aggregation_manager._state_buffer[mock_run_context.node_id] == mock_new_state
            # make sure the lock for the aggregator was released
            aggregation_manager._locks[mock_run_context.node_id].release.assert_called_once()
