from unittest.mock import patch

from datasets import Features, Sequence, Value

from hyped import ops
from hyped.core.flow import DataFlow


def test_prfs():
    flow = DataFlow(
        Features(
            {
                "y_true": Sequence(Value("bool"), length=3),
                "y_pred": Sequence(Value("bool"), length=3),
            }
        )
    )

    with (
        patch("hyped.ops.metrics.MultiLabelConfusionMatrix") as mcm_mock,
        patch("hyped.ops.metrics.PrecisionRecallFScoreSupport") as prfs_mock,
    ):
        # call the operator
        ops.precision_recall_fscore_support(
            y_true=flow.src_features.y_true,
            y_pred=flow.src_features.y_pred,
            labels=[0, 1, 2],
            beta=1.0,
            average="micro",
        )
        # assert processors were called
        mcm_mock().call.assert_called_once_with(
            y_true=flow.src_features.y_true,
            y_pred=flow.src_features.y_pred,
        )
        prfs_mock().call.assert_called_once_with(
            confusion_matrix=mcm_mock().call().confusion_matrix
        )
