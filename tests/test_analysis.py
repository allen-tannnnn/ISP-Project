import numpy as np
import pandas as pd
import pytest

from courtvision.analysis import (
    HEATMAP_X_RANGE,
    HEATMAP_Y_RANGE,
    build_heatmap,
    clean_positions,
    render_heatmap,
)
from courtvision.calibration import court_to_diagram, draw_top_down_court


def _positions(y_values, x_values=None):
    if x_values is None:
        x_values = [18.0] * len(y_values)
    return pd.DataFrame(
        {
            "frame_index": np.arange(len(y_values)) * 3,
            "time_sec": np.arange(len(y_values)) * 0.1,
            "side": ["near"] * len(y_values),
            "confidence": [0.9] * len(y_values),
            "foot_x_px": [100.0] * len(y_values),
            "foot_y_px": [200.0] * len(y_values),
            "court_x_ft": x_values,
            "court_y_ft": y_values,
        }
    )


def test_clean_positions_removes_single_outlier_and_preserves_input():
    positions = _positions([20.0] * 5 + [70.0] + [20.0] * 5)
    original = positions.copy(deep=True)

    cleaned = clean_positions(
        positions,
        frame_step=3,
        fps=30,
        outlier_ft=12,
        window=7,
    )

    assert cleaned.loc[5, "interpolated"]
    assert cleaned.loc[5, "court_y_ft"] == pytest.approx(20.0)
    pd.testing.assert_frame_equal(positions, original)


def test_clean_positions_interpolates_short_gap_but_keeps_long_gap_missing():
    y_values = [20.0, 22.0, np.nan, np.nan, 28.0]
    y_values.extend([30.0] * 3)
    y_values.extend([np.nan] * 20)
    y_values.extend([32.0] * 5)
    cleaned = clean_positions(
        _positions(y_values),
        frame_step=3,
        fps=30,
        max_gap_s=1.0,
        outlier_ft=100,
    )

    assert cleaned.loc[2:3, "court_y_ft"].notna().all()
    assert cleaned.loc[2:3, "interpolated"].all()
    assert cleaned.loc[8:27, "court_y_ft"].isna().all()
    assert not cleaned.loc[8:27, "interpolated"].any()


def test_clean_positions_flags_edge_clipped_rows():
    positions = _positions([20.0])
    positions.loc[0, "foot_y_px"] = 98.0

    cleaned = clean_positions(positions, frame_step=3, fps=30, frame_height=100)

    assert cleaned.loc[0, "edge_clipped"]


def test_build_heatmap_peak_is_near_point_cluster():
    positions = pd.DataFrame(
        {
            "side": ["near"] * 4 + ["far"],
            "court_x_ft": [10.0, 10.2, 9.8, 10.1, 30.0],
            "court_y_ft": [20.0, 20.1, 19.9, 20.2, 70.0],
        }
    )

    heatmap = build_heatmap(positions, sides=("near",), bins_ft=1, blur_ft=1)

    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    court_x = HEATMAP_X_RANGE[0] + (peak_x + 0.5) * (
        HEATMAP_X_RANGE[1] - HEATMAP_X_RANGE[0]
    ) / heatmap.shape[1]
    court_y = HEATMAP_Y_RANGE[1] - (peak_y + 0.5) * (
        HEATMAP_Y_RANGE[1] - HEATMAP_Y_RANGE[0]
    ) / heatmap.shape[0]
    assert court_x == pytest.approx(10.0, abs=1.0)
    assert court_y == pytest.approx(20.0, abs=1.0)
    assert heatmap.max() == pytest.approx(1.0)


def test_render_heatmap_returns_rgb_uint8_image():
    rendered = render_heatmap(np.ones((102, 36), dtype=np.float32))

    assert rendered.shape == (1000, 480, 3)
    assert rendered.dtype == np.uint8


def test_heatmap_peak_at_y_90_renders_at_its_court_coordinate():
    positions = pd.DataFrame(
        {
            "side": ["far"] * 5,
            "court_x_ft": [18.0] * 5,
            "court_y_ft": [90.0] * 5,
        }
    )
    heatmap = build_heatmap(positions, sides=("far",), blur_ft=1)
    rendered = render_heatmap(heatmap)
    base = draw_top_down_court()
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    court_x = HEATMAP_X_RANGE[0] + (peak_x + 0.5) * (
        HEATMAP_X_RANGE[1] - HEATMAP_X_RANGE[0]
    ) / heatmap.shape[1]
    court_y = HEATMAP_Y_RANGE[1] - (peak_y + 0.5) * (
        HEATMAP_Y_RANGE[1] - HEATMAP_Y_RANGE[0]
    ) / heatmap.shape[0]
    pixel_x, pixel_y = court_to_diagram((court_x, court_y))
    x, y = round(pixel_x), round(pixel_y)

    assert court_y == pytest.approx(90.5)
    assert y < round(court_to_diagram((18.0, 78.0))[1])
    assert np.any(rendered[y - 2 : y + 3, x - 2 : x + 3] != base[y - 2 : y + 3, x - 2 : x + 3])
