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
