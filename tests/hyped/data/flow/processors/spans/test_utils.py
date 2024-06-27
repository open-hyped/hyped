import numpy as np

from hyped.data.flow.processors.spans.utils import compute_spans_overlap_matrix


def test_compute_spans_overlap_matrix():
    # test spans
    src_spans = [(0, 4), (5, 8), (15, 21)]
    tgt_spans = [(0, 4), (3, 7)]
    # expected overlap mask
    expected_mask = np.asarray([[True, True], [False, True], [False, False]])
    # test
    assert (
        compute_spans_overlap_matrix(src_spans, tgt_spans) == expected_mask
    ).all()
