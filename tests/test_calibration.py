import copy
import json

import cv2
import numpy as np
import pytest

from courtvision.calibration import (
    calibration_from_dict,
    calibration_to_dict,
    compute_homography,
    court_to_diagram,
    court_to_pixel,
    draw_court_overlay,
    draw_top_down_court,
    get_reference_points,
    pixel_to_court,
    reprojection_error,
)

PIXEL_POINTS = np.array(
    [
        (200.0, 700.0),
        (1000.0, 700.0),
        (800.0, 300.0),
        (400.0, 300.0),
    ],
    dtype=np.float32,
)
COURT_POINTS = np.array(
    get_reference_points("singles_corners"),
    dtype=np.float32,
)
HOMOGRAPHY = cv2.getPerspectiveTransform(PIXEL_POINTS, COURT_POINTS)


def test_pixel_points_map_to_their_court_coordinates():
    mapped_points = pixel_to_court(HOMOGRAPHY, PIXEL_POINTS)

    np.testing.assert_allclose(mapped_points, COURT_POINTS, atol=0.01)


@pytest.mark.parametrize(
    "points",
    [
        PIXEL_POINTS[0],
        PIXEL_POINTS,
    ],
)
def test_pixel_court_round_trip(points):
    court_points = pixel_to_court(HOMOGRAPHY, points)
    round_trip = court_to_pixel(HOMOGRAPHY, court_points)

    np.testing.assert_allclose(round_trip, points, atol=0.01)


def test_pixel_quadrilateral_center_maps_inside_the_singles_court():
    pixel_center = np.mean(PIXEL_POINTS, axis=0)

    court_center = pixel_to_court(HOMOGRAPHY, pixel_center)

    assert 4.5 < court_center[0] < 31.5
    assert 0.0 < court_center[1] < 78.0


def test_transform_preserves_single_and_multiple_point_shapes():
    single = pixel_to_court(HOMOGRAPHY, PIXEL_POINTS[0])
    multiple = pixel_to_court(HOMOGRAPHY, PIXEL_POINTS)

    assert single.shape == (2,)
    assert multiple.shape == (4, 2)


def test_reprojection_error_is_near_zero_for_exact_correspondences():
    mean_error, max_error = reprojection_error(
        HOMOGRAPHY,
        PIXEL_POINTS,
        COURT_POINTS,
    )

    assert mean_error == pytest.approx(0.0, abs=0.01)
    assert max_error == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize(
    "pixel_points",
    [
        PIXEL_POINTS[:3],
        np.vstack((PIXEL_POINTS, (300.0, 500.0))),
        np.array(
            [
                (200.0, 700.0),
                (1000.0, 700.0),
                (800.0, 300.0),
                (np.nan, 300.0),
            ]
        ),
        np.array(
            [
                (200.0, 700.0),
                (1000.0, 700.0),
                (800.0, 300.0),
                (1000.0, 700.0),
            ]
        ),
        np.array(
            [
                (200.0, 700.0),
                (600.0, 700.0),
                (1000.0, 700.0),
                (400.0, 300.0),
            ]
        ),
    ],
    ids=("three-points", "five-points", "nan", "duplicate", "collinear"),
)
def test_compute_homography_rejects_invalid_pixel_points(pixel_points):
    with pytest.raises(ValueError):
        compute_homography(pixel_points, COURT_POINTS)


def test_court_to_pixel_rejects_singular_homography():
    with pytest.raises(ValueError, match="inverted"):
        court_to_pixel(np.zeros((3, 3)), (0.0, 0.0))


def test_draw_court_overlay_preserves_input_and_draws_on_copy():
    frame = np.zeros((800, 1200, 3), dtype=np.uint8)
    original = frame.copy()

    overlay = draw_court_overlay(frame, HOMOGRAPHY)

    assert overlay.shape == frame.shape
    np.testing.assert_array_equal(frame, original)
    assert np.any(overlay != frame)


def test_top_down_court_has_expected_rgb_image_shape_and_dtype():
    diagram = draw_top_down_court(size_px=(480, 900))
    empty_points_diagram = draw_top_down_court([], size_px=(480, 900))

    assert diagram.shape == (900, 480, 3)
    assert diagram.dtype == np.uint8
    np.testing.assert_array_equal(empty_points_diagram, diagram)


def test_court_to_diagram_maps_court_corners_with_expected_orientation():
    bottom_left, top_right = court_to_diagram(
        [(0.0, 0.0), (36.0, 78.0)],
        size_px=(480, 900),
    )

    assert bottom_left[0] < top_right[0]
    assert bottom_left[1] > top_right[1]
    assert (top_right[0] - bottom_left[0]) / (
        bottom_left[1] - top_right[1]
    ) == pytest.approx(36 / 78)


def test_top_down_points_only_change_pixels_near_their_dots():
    plain_diagram = draw_top_down_court(size_px=(480, 900))
    point = (18.0, 39.0)
    marked_diagram = draw_top_down_court([point], size_px=(480, 900))
    pixel_x, pixel_y = court_to_diagram(point, size_px=(480, 900))
    rows, columns = np.ogrid[:900, :480]
    outside_dot = (
        (columns - round(pixel_x)) ** 2 + (rows - round(pixel_y)) ** 2 > 9**2
    )

    np.testing.assert_array_equal(
        marked_diagram[outside_dot],
        plain_diagram[outside_dot],
    )
    assert np.any(marked_diagram != plain_diagram)


def test_top_down_court_skips_points_outside_court_and_margin():
    plain_diagram = draw_top_down_court(size_px=(480, 900))

    diagram = draw_top_down_court(
        [(-7.0, 39.0), (43.0, 39.0), (18.0, 85.0)],
        size_px=(480, 900),
    )

    np.testing.assert_array_equal(diagram, plain_diagram)


def _saved_calibration():
    homography = compute_homography(PIXEL_POINTS, COURT_POINTS)
    return calibration_to_dict(
        PIXEL_POINTS,
        COURT_POINTS,
        homography,
        "singles_corners",
        (1280, 720),
        video_name="match.mp4",
    )


def test_calibration_dict_round_trips_through_json():
    saved_data = _saved_calibration()

    loaded_data = calibration_from_dict(json.loads(json.dumps(saved_data)))

    np.testing.assert_allclose(loaded_data["pixel_points"], PIXEL_POINTS)
    np.testing.assert_allclose(loaded_data["court_points"], COURT_POINTS)
    np.testing.assert_allclose(loaded_data["H"], saved_data["homography"])
    assert loaded_data["preset"] == "singles_corners"
    assert loaded_data["frame_width"] == 1280
    assert loaded_data["frame_height"] == 720


@pytest.mark.parametrize(
    "invalid_case",
    ("missing-key", "wrong-version", "mismatched-counts", "nan", "bad-matrix", "inconsistent"),
)
def test_calibration_from_dict_rejects_invalid_data(invalid_case):
    saved_data = copy.deepcopy(_saved_calibration())
    if invalid_case == "missing-key":
        del saved_data["pixel_points"]
    elif invalid_case == "wrong-version":
        saved_data["version"] = 2
    elif invalid_case == "mismatched-counts":
        saved_data["court_points"].pop()
    elif invalid_case == "nan":
        saved_data["pixel_points"][0][0] = float("nan")
    elif invalid_case == "bad-matrix":
        saved_data["homography"] = [[1.0, 0.0], [0.0, 1.0]]
    else:
        saved_data["homography"][0][0] += 1.0

    with pytest.raises(ValueError):
        calibration_from_dict(saved_data)
