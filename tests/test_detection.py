import numpy as np
import pytest

from courtvision.detection import PlayerDetection, filter_on_court


HOMOGRAPHY = np.array(
    [
        [0.1, 0.0, -10.0],
        [0.0, 0.1, -20.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _detection_at_pixel(x: float, y: float) -> PlayerDetection:
    return {
        "bbox": (x - 5.0, y - 10.0, x + 5.0, y),
        "confidence": 0.9,
        "foot_point": (x, y),
    }


def test_filter_on_court_maps_foot_points_and_keeps_court_plus_margin():
    on_court = _detection_at_pixel(280.0, 590.0)  # (18, 39) feet
    within_margin = _detection_at_pixel(90.0, 300.0)  # (-1, 10) feet
    outside_margin = _detection_at_pixel(20.0, 590.0)  # (-8, 39) feet

    filtered = filter_on_court(
        [on_court, within_margin, outside_margin],
        HOMOGRAPHY,
    )

    assert len(filtered) == 2
    assert filtered[0]["court_point"] == pytest.approx((18.0, 39.0))
    assert filtered[1]["court_point"] == pytest.approx((-1.0, 10.0))
    assert filtered[0]["foot_point"] == on_court["foot_point"]
    assert "court_point" not in on_court


def test_filter_on_court_rejects_negative_or_nonfinite_margin():
    with pytest.raises(ValueError, match="margin_ft"):
        filter_on_court([], HOMOGRAPHY, margin_ft=-1)

    with pytest.raises(ValueError, match="margin_ft"):
        filter_on_court([], HOMOGRAPHY, margin_ft=float("nan"))
