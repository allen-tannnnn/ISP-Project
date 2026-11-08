import cv2
import numpy as np
import pytest

from courtvision.calibration import (
    compute_homography,
    court_to_pixel,
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
