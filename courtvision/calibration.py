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
from datetime import datetime
from itertools import combinations
from typing import TypedDict

import cv2
import numpy as np

Point = tuple[float, float]
LineSegment = tuple[Point, Point]


class CalibrationData(TypedDict):
    """Validated calibration fields returned when loading a saved record."""

    pixel_points: list[list[float]]
    court_points: list[list[float]]
    H: np.ndarray
    preset: str
    frame_width: int
    frame_height: int


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


def calibration_to_dict(
    pixel_points: Sequence[Sequence[float]] | np.ndarray,
    court_points: Sequence[Sequence[float]] | np.ndarray,
    H: np.ndarray,
    preset_name: str,
    frame_size: tuple[int, int],
    video_name: str | None = None,
) -> dict[str, object]:
    """Create a JSON-ready calibration record with plain Python values.

    ``frame_size`` is ``(width, height)``. The record includes an ISO 8601
    timestamp and the supplied point correspondences and homography.
    """
    try:
        width, height = frame_size
        pixels = np.asarray(pixel_points, dtype=np.float64)
        court = np.asarray(court_points, dtype=np.float64)
        matrix = np.asarray(H, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Calibration data and frame_size have invalid shapes or values.") from error

    if pixels.ndim != 2 or pixels.shape[1:] != (2,):
        raise ValueError("pixel_points must be an array of (x, y) pairs.")
    if court.ndim != 2 or court.shape[1:] != (2,):
        raise ValueError("court_points must be an array of (x, y) pairs.")
    if pixels.shape[0] != court.shape[0] or pixels.shape[0] < 4:
        raise ValueError(
            "pixel_points and court_points must have at least four matching points."
        )
    if not np.isfinite(pixels).all() or not np.isfinite(court).all():
        raise ValueError("Calibration points must contain only finite numbers.")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("H must be a finite 3-by-3 matrix.")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, np.integer))
        or not isinstance(height, (int, np.integer))
        or width <= 0
        or height <= 0
    ):
        raise ValueError("frame_size must contain positive integer width and height.")

    return {
        "version": 1,
        "preset": str(preset_name),
        "pixel_points": [[float(value) for value in point] for point in pixels],
        "court_points": [[float(value) for value in point] for point in court],
        "homography": [
            [float(value) for value in row] for row in matrix
        ],
        "frame_width": int(width),
        "frame_height": int(height),
        "video_name": video_name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def calibration_from_dict(data: object) -> CalibrationData:
    """Validate a saved calibration and return its points, matrix, preset, and size.

    Raises:
        ValueError: If the record is missing fields, malformed, unsupported,
            geometrically invalid, or inconsistent with its stored homography.
    """
    if not isinstance(data, dict):
        raise ValueError("Calibration data must be a JSON object.")

    required_keys = {
        "version",
        "preset",
        "pixel_points",
        "court_points",
        "homography",
        "frame_width",
        "frame_height",
    }
    missing_keys = required_keys.difference(data)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Calibration file is missing required key(s): {missing}.")
    if (
        isinstance(data["version"], bool)
        or not isinstance(data["version"], int)
        or data["version"] != 1
    ):
        raise ValueError("Unsupported calibration version; expected version 1.")
    if not isinstance(data["preset"], str):
        raise ValueError("Calibration preset must be a string.")

    pixel_points = _validate_saved_points(data["pixel_points"], "pixel_points")
    court_points = _validate_saved_points(data["court_points"], "court_points")
    if len(pixel_points) != len(court_points):
        raise ValueError("pixel_points and court_points must have matching point counts.")
    if len(pixel_points) < 4:
        raise ValueError("Calibration must contain at least four corresponding points.")

    stored_homography = _validate_saved_homography(data["homography"])
    frame_width = _validate_frame_dimension(data["frame_width"], "frame_width")
    frame_height = _validate_frame_dimension(data["frame_height"], "frame_height")

    try:
        computed_homography = compute_homography(
            pixel_points[:4],
            court_points[:4],
        )
    except ValueError as error:
        raise ValueError(f"Calibration points are invalid: {error}") from error
    if not np.allclose(
        computed_homography,
        stored_homography,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            "Calibration file is inconsistent: its homography does not match "
            "the stored points."
        )

    return {
        "pixel_points": pixel_points,
        "court_points": court_points,
        "H": stored_homography,
        "preset": data["preset"],
        "frame_width": frame_width,
        "frame_height": frame_height,
    }


def _validate_saved_points(value: object, name: str) -> list[list[float]]:
    """Validate saved JSON point arrays and normalize their values to floats."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of [x, y] pairs.")
    points: list[list[float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{name}[{index}] must be a [x, y] pair.")
        if any(isinstance(coordinate, bool) for coordinate in point):
            raise ValueError(f"{name}[{index}] must contain finite numbers.")
        try:
            normalized = [float(coordinate) for coordinate in point]
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name}[{index}] must contain finite numbers.") from error
        if not np.isfinite(normalized).all():
            raise ValueError(f"{name}[{index}] must contain finite numbers.")
        points.append(normalized)
    return points


def _validate_saved_homography(value: object) -> np.ndarray:
    """Validate a saved JSON homography matrix."""
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("homography must be a finite 3-by-3 matrix.") from error
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("homography must be a finite 3-by-3 matrix.")
    return matrix


def _validate_frame_dimension(value: object, name: str) -> int:
    """Validate a positive integer frame dimension."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


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


def _diagram_dimensions(size_px: int | tuple[int, int]) -> tuple[int, int]:
    """Return diagram width and height after validating the requested size."""
    if isinstance(size_px, int):
        dimensions = (size_px, size_px)
    else:
        dimensions = size_px
    if (
        len(dimensions) != 2
        or any(not isinstance(value, int) or value <= 0 for value in dimensions)
    ):
        raise ValueError("size_px must be a positive integer or a (width, height) pair.")
    return dimensions


def _diagram_layout(
    size_px: int | tuple[int, int],
    margin_ft: float,
) -> tuple[int, int, float, float, float]:
    """Return canvas dimensions, court origin, and pixels-per-foot scale."""
    width, height = _diagram_dimensions(size_px)
    if not np.isfinite(margin_ft) or margin_ft < 0:
        raise ValueError("margin_ft must be a finite nonnegative number.")

    scale = min(
        width / (36.0 + 2.0 * margin_ft),
        height / (78.0 + 2.0 * margin_ft),
    )
    left = (width - (36.0 + 2.0 * margin_ft) * scale) / 2.0 + margin_ft * scale
    top = (height - (78.0 + 2.0 * margin_ft) * scale) / 2.0 + margin_ft * scale
    return width, height, left, top, scale


def court_to_diagram(
    points: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    size_px: int | tuple[int, int] = (480, 900),
    margin_ft: float = 6,
) -> np.ndarray:
    """Convert court feet to diagram pixels, preserving the input point shape.

    ``size_px`` is the output ``(width, height)`` in pixels (or one size for a
    square image). The returned points use top-left-origin image coordinates,
    with the near baseline at the bottom of the diagram.
    """
    point_array, original_shape = _normalize_diagram_points(points)
    width, height, left, top, scale = _diagram_layout(size_px, margin_ft)
    del width, height
    diagram_points = np.empty_like(point_array)
    diagram_points[:, 0] = left + point_array[:, 0] * scale
    diagram_points[:, 1] = top + (78.0 - point_array[:, 1]) * scale
    return diagram_points.reshape(original_shape)


def _normalize_diagram_points(
    points: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Normalize one court point or an array of court points, including empty arrays."""
    try:
        point_array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Points must be (x, y) coordinates.") from error
    if point_array.ndim == 1 and point_array.size == 0:
        return np.empty((0, 2), dtype=np.float64), point_array.shape
    if point_array.ndim == 1 and point_array.shape == (2,):
        return point_array.reshape(1, 2), point_array.shape
    if point_array.ndim == 2 and point_array.shape[1] == 2:
        return point_array, point_array.shape
    raise ValueError("Points must be one (x, y) pair or an array of such pairs.")


def draw_top_down_court(
    points: Sequence[Sequence[float]] | np.ndarray | None = None,
    size_px: int | tuple[int, int] = (480, 900),
    margin_ft: float = 6,
) -> np.ndarray:
    """Render a top-down tennis court and optional court-coordinate points.

    The returned image is RGB, with y=0 at the bottom and x=0 on the left.
    The court keeps its 36:78 feet aspect ratio; ``margin_ft`` adds an equal
    border around the court in feet. Points beyond the court plus margin are
    ignored.
    """
    width, height, left, top, scale = _diagram_layout(size_px, margin_ft)
    image = np.full((height, width, 3), (18, 24, 30), dtype=np.uint8)

    court_left = round(left)
    court_right = round(left + 36.0 * scale)
    court_top = round(top)
    court_bottom = round(top + 78.0 * scale)
    cv2.rectangle(
        image,
        (court_left, court_top),
        (court_right, court_bottom),
        (38, 76, 65),
        thickness=cv2.FILLED,
    )

    outer_lines = {
        "near_baseline",
        "far_baseline",
        "left_doubles_sideline",
        "right_doubles_sideline",
    }
    for name, endpoints in COURT_LINES.items():
        diagram_endpoints = court_to_diagram(
            endpoints,
            size_px=size_px,
            margin_ft=margin_ft,
        )
        first = tuple(int(round(value)) for value in diagram_endpoints[0])
        second = tuple(int(round(value)) for value in diagram_endpoints[1])
        cv2.line(
            image,
            first,
            second,
            (235, 239, 235),
            2 if name in outer_lines else 1,
            cv2.LINE_AA,
        )

    if points is not None:
        point_array, _ = _normalize_diagram_points(points)
        diagram_points = court_to_diagram(
            point_array,
            size_px=size_px,
            margin_ft=margin_ft,
        )
        for court_point, diagram_point in zip(point_array, diagram_points):
            x, y = court_point
            if not np.isfinite(court_point).all() or not (
                -margin_ft <= x <= 36.0 + margin_ft
                and -margin_ft <= y <= 78.0 + margin_ft
            ):
                continue
            center = tuple(int(round(value)) for value in diagram_point)
            cv2.circle(image, center, 8, (12, 19, 21), thickness=cv2.FILLED)
            cv2.circle(image, center, 5, (255, 220, 20), thickness=cv2.FILLED)

    return image
