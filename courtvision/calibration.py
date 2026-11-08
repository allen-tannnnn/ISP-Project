"""Tennis court geometry and calibration reference points.

Court coordinates are measured in feet. The origin is the near-left doubles
corner; x runs across the 36-foot width and y runs along the 78-foot length.
The doubles outline is 36 by 78 feet, with baselines at y=0 and y=78 and
doubles sidelines at x=0 and x=36. Singles sidelines are 4.5 feet inside the
doubles sidelines, at x=4.5 and x=31.5. Service lines are 21 feet from the
net, at y=18 and y=60, and span the singles width. The center service line
is at x=18 between those service lines. The net is at y=39 across the doubles
width.
"""

from collections.abc import Sequence
from itertools import combinations

import cv2
import numpy as np

Point = tuple[float, float]
LineSegment = tuple[Point, Point]

COURT_LINES: dict[str, LineSegment] = {
    # The doubles outline is formed by the four sidelines and baselines.
    "near_baseline": ((0.0, 0.0), (36.0, 0.0)),
    "far_baseline": ((0.0, 78.0), (36.0, 78.0)),
    "left_doubles_sideline": ((0.0, 0.0), (0.0, 78.0)),
    "right_doubles_sideline": ((36.0, 0.0), (36.0, 78.0)),
    # Singles sidelines are 4.5 feet inside the doubles sidelines.
    "left_singles_sideline": ((4.5, 0.0), (4.5, 78.0)),
    "right_singles_sideline": ((31.5, 0.0), (31.5, 78.0)),
    # Each service line is 21 feet from the net and spans the singles width.
    "near_service_line": ((4.5, 18.0), (31.5, 18.0)),
    "far_service_line": ((4.5, 60.0), (31.5, 60.0)),
    # The center service line divides the service boxes between service lines.
    "center_service_line": ((18.0, 18.0), (18.0, 60.0)),
    # The net is halfway along the 78-foot court and spans the doubles width.
    "net_line": ((0.0, 39.0), (36.0, 39.0)),
}
REFERENCE_POINT_PRESETS: dict[str, tuple[Point, Point, Point, Point]] = {
    "singles_corners": (
        (4.5, 0.0),
        (31.5, 0.0),
        (31.5, 78.0),
        (4.5, 78.0),
    ),
    "doubles_corners": (
        (0.0, 0.0),
        (36.0, 0.0),
        (36.0, 78.0),
        (0.0, 78.0),
    ),
}

DEFAULT_REFERENCE_PRESET = "singles_corners"


def get_reference_points(
    preset_name: str,
) -> tuple[Point, Point, Point, Point]:
    """Return preset points in near-left, near-right, far-right, far-left order.

    Args:
        preset_name: A key in :data:`REFERENCE_POINT_PRESETS`.

    Raises:
        KeyError: If no reference-point preset has that name.
    """
    return REFERENCE_POINT_PRESETS[preset_name]


def _validate_four_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Return four finite 2D points as float64 coordinates."""
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must contain exactly four points, each with two numbers."
        ) from error

    if array.shape != (4, 2):
        raise ValueError(
            f"{name} must contain exactly four points, each with two numbers."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers.")

    if np.unique(array, axis=0).shape[0] != 4:
        raise ValueError(
            f"{name} must form a real four-cornered shape; "
            "two points are identical."
        )

    scale = np.max(np.abs(array))
    normalized = array / scale if scale else array.copy()
    normalized -= normalized.min(axis=0)
    extent = np.max(normalized)
    if extent:
        normalized /= extent

    tolerance = np.finfo(np.float64).eps * 32
    for first, second, third in combinations(normalized, 3):
        first_edge = second - first
        second_edge = third - first
        area_twice = first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
        if abs(area_twice) <= tolerance:
            raise ValueError(
                f"{name} must form a real four-cornered shape; "
                "three points are collinear."
            )
    return array


def _validate_transform_points(
    points: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Normalize one 2D point or a nonempty collection of 2D points."""
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Points must be one (x, y) pair or an array of such pairs."
        ) from error

    if array.ndim == 1 and array.shape == (2,):
        original_shape = array.shape
        array = array.reshape(1, 2)
    elif array.ndim == 2 and array.shape[1] == 2 and array.shape[0] > 0:
        original_shape = array.shape
    else:
        raise ValueError("Points must be one (x, y) pair or an array of such pairs.")
    if not np.isfinite(array).all():
        raise ValueError("Points must contain only finite numbers.")
    return array, original_shape


def _validate_homography(H: np.ndarray) -> np.ndarray:
    """Return a finite 3-by-3 homography matrix as float64."""
    try:
        matrix = np.asarray(H, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("H must be a finite 3-by-3 matrix.") from error
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("H must be a finite 3-by-3 matrix.")
    return matrix


def compute_homography(
    pixel_points: Sequence[Sequence[float]] | np.ndarray,
    court_points: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Compute a matrix mapping four image-pixel points to court feet.

    Both point sets must contain four finite (x, y) pairs in corresponding
    order, and neither set may have duplicate points or collinear triples.
    """
    pixels = _validate_four_points(pixel_points, "pixel_points")
    court = _validate_four_points(court_points, "court_points")

    try:
        matrix = cv2.getPerspectiveTransform(
            pixels.astype(np.float32),
            court.astype(np.float32),
        )
    except cv2.error as error:
        raise ValueError("OpenCV could not compute the homography matrix.") from error
    if matrix is None or np.asarray(matrix).shape != (3, 3):
        raise ValueError("OpenCV did not return a 3-by-3 homography matrix.")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("OpenCV returned a non-finite homography matrix.")
    return matrix


def pixel_to_court(
    H: np.ndarray,
    points: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Map one pixel point or a collection of pixel points to court feet.

    The returned NumPy array has the same point shape as ``points``: ``(2,)``
    for one point or ``(n, 2)`` for multiple points.
    """
    matrix = _validate_homography(H)
    point_array, original_shape = _validate_transform_points(points)
    try:
        transformed = cv2.perspectiveTransform(
            point_array.reshape(1, -1, 2),
            matrix,
        ).reshape(original_shape)
    except cv2.error as error:
        raise ValueError("OpenCV could not transform the pixel points.") from error
    if not np.isfinite(transformed).all():
        raise ValueError("The pixel points could not be mapped to finite court coordinates.")
    return transformed


def court_to_pixel(
    H: np.ndarray,
    points: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Map one court point or a collection of court points to image pixels.

    The returned NumPy array has the same point shape as ``points``. A clear
    ``ValueError`` is raised if ``H`` cannot be inverted.
    """
    matrix = _validate_homography(H)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("H cannot be inverted to map court points to pixels.") from error
    if not np.isfinite(inverse).all():
        raise ValueError("H cannot be inverted to map court points to pixels.")
    point_array, original_shape = _validate_transform_points(points)
    try:
        transformed = cv2.perspectiveTransform(
            point_array.reshape(1, -1, 2),
            inverse,
        ).reshape(original_shape)
    except cv2.error as error:
        raise ValueError("OpenCV could not transform the court points.") from error
    if not np.isfinite(transformed).all():
        raise ValueError("The court points could not be mapped to finite pixel coordinates.")
    return transformed


def reprojection_error(
    H: np.ndarray,
    pixel_points: Sequence[Sequence[float]] | np.ndarray,
    court_points: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[float, float]:
    """Return mean and maximum pixel reprojection errors for corresponding points."""
    pixels, _ = _validate_transform_points(pixel_points)
    court, _ = _validate_transform_points(court_points)
    if pixels.shape != court.shape:
        raise ValueError(
            "pixel_points and court_points must have the same number of 2D points."
        )

    reprojected = court_to_pixel(H, court)
    errors = np.linalg.norm(reprojected - pixels, axis=1)
    return float(np.mean(errors)), float(np.max(errors))


def draw_court_overlay(
    frame_bgr: np.ndarray,
    H: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw projected court lines on a copy of a BGR frame.

    Court line endpoints are mapped from feet to pixels using ``H``. Segments
    with non-finite endpoints or endpoints farther than about five frame
    dimensions beyond the image are skipped. Set ``show_labels`` to ``False``
    to omit the court-line names.
    """
    overlay = frame_bgr.copy()
    height, width = overlay.shape[:2]
    matrix = _validate_homography(H)
    try:
        np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("H cannot be inverted to draw the court overlay.") from error

    for name, endpoints in COURT_LINES.items():
        try:
            pixel_endpoints = [
                court_to_pixel(matrix, endpoint) for endpoint in endpoints
            ]
        except ValueError as error:
            if "could not be mapped to finite pixel coordinates" in str(error):
                continue
            raise

        if not all(np.isfinite(point).all() for point in pixel_endpoints):
            continue
        if any(
            point[0] < -5 * width
            or point[0] > 6 * width
            or point[1] < -5 * height
            or point[1] > 6 * height
            for point in pixel_endpoints
        ):
            continue

        first = tuple(int(round(value)) for value in pixel_endpoints[0])
        second = tuple(int(round(value)) for value in pixel_endpoints[1])
        cv2.line(overlay, first, second, color, thickness, cv2.LINE_AA)

        if show_labels:
            midpoint = (
                round((first[0] + second[0]) / 2),
                round((first[1] + second[1]) / 2),
            )
            cv2.putText(
                overlay,
                name,
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    return overlay
