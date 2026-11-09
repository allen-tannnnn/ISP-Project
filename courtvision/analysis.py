"""Position cleaning and court heatmap rendering."""

from collections.abc import Sequence

import cv2
import numpy as np
import pandas as pd

from courtvision.calibration import court_to_diagram, draw_top_down_court

_POSITION_COLUMNS = ("frame_index", "side", "court_x_ft", "court_y_ft")
HEATMAP_MARGIN_X_FT = 6.0
HEATMAP_MARGIN_Y_FT = 20.0
HEATMAP_X_RANGE = (
    -HEATMAP_MARGIN_X_FT,
    36.0 + HEATMAP_MARGIN_X_FT,
)
HEATMAP_Y_RANGE = (
    -HEATMAP_MARGIN_Y_FT,
    78.0 + HEATMAP_MARGIN_Y_FT,
)


def _positive_finite(name: str, value: float, *, allow_zero: bool = False) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if (
        not np.isfinite(numeric_value)
        or numeric_value < 0
        or (not allow_zero and numeric_value == 0)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number.")
    return numeric_value


def clean_positions(
    df: pd.DataFrame,
    frame_step: int,
    fps: float,
    frame_height: int | None = None,
    max_gap_s: float = 1.0,
    outlier_ft: float = 12.0,
    window: int = 7,
) -> pd.DataFrame:
    """Clean each side's sampled positions without modifying the input."""
    missing_columns = set(_POSITION_COLUMNS).difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Position data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if isinstance(frame_step, bool) or not isinstance(frame_step, int) or frame_step < 1:
        raise ValueError("frame_step must be a positive integer.")
    fps = _positive_finite("fps", fps)
    max_gap_s = _positive_finite("max_gap_s", max_gap_s, allow_zero=True)
    outlier_ft = _positive_finite("outlier_ft", outlier_ft, allow_zero=True)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer.")
    if frame_height is not None and (
        isinstance(frame_height, bool)
        or not isinstance(frame_height, int)
        or frame_height < 1
    ):
        raise ValueError("frame_height must be a positive integer or None.")

    result = df.copy(deep=True)
    if "foot_y_px" in result:
        foot_y = pd.to_numeric(result["foot_y_px"], errors="coerce")
        result["edge_clipped"] = (
            foot_y >= frame_height - 3 if frame_height is not None else False
        )
    else:
        result["edge_clipped"] = False
    result["edge_clipped"] = result["edge_clipped"].fillna(False).astype(bool)
    result["interpolated"] = False

    processed: list[pd.DataFrame] = []
    for side in ("near", "far"):
        side_rows = result.loc[result["side"] == side].copy()
        if side_rows.empty:
            continue
        side_rows["frame_index"] = pd.to_numeric(
            side_rows["frame_index"], errors="raise"
        )
        side_rows = side_rows.sort_values("frame_index", kind="stable")
        first_frame = int(side_rows["frame_index"].min())
        last_frame = int(side_rows["frame_index"].max())
        grid_start = ((first_frame + frame_step - 1) // frame_step) * frame_step
        grid = set(range(grid_start, last_frame + 1, frame_step))
        present = set(side_rows["frame_index"].tolist())
        missing_rows: list[dict[str, object]] = []
        for frame_index in sorted(grid.difference(present)):
            empty_row: dict[str, object] = {
                column: np.nan for column in result.columns
            }
            empty_row["frame_index"] = frame_index
            empty_row["side"] = side
            empty_row["interpolated"] = False
            empty_row["edge_clipped"] = False
            if "time_sec" in result:
                empty_row["time_sec"] = frame_index / fps
            missing_rows.append(empty_row)
        if missing_rows:
            side_rows = pd.concat(
                [side_rows, pd.DataFrame(missing_rows)],
                ignore_index=True,
            )
        side_rows = side_rows.sort_values("frame_index", kind="stable").reset_index(
            drop=True
        )

        court_x = pd.to_numeric(side_rows["court_x_ft"], errors="coerce")
        court_y = pd.to_numeric(side_rows["court_y_ft"], errors="coerce")
        rolling_x = court_x.rolling(window, center=True, min_periods=1).median()
        rolling_y = court_y.rolling(window, center=True, min_periods=1).median()
        outliers = (
            ((court_x - rolling_x).abs() > outlier_ft)
            | ((court_y - rolling_y).abs() > outlier_ft)
        )
        side_rows.loc[outliers, ["court_x_ft", "court_y_ft"]] = np.nan

        missing_positions = (
            side_rows["court_x_ft"].isna() | side_rows["court_y_ft"].isna()
        ).to_numpy()
        gap_limit = max_gap_s * fps / frame_step
        index = 0
        while index < len(side_rows):
            if not missing_positions[index]:
                index += 1
                continue
            gap_start = index
            while index < len(side_rows) and missing_positions[index]:
                index += 1
            gap_end = index
            gap_size = gap_end - gap_start
            if (
                gap_start > 0
                and gap_end < len(side_rows)
                and gap_size <= gap_limit
            ):
                side_rows.loc[
                    gap_start : gap_end - 1, ["court_x_ft", "court_y_ft"]
                ] = np.nan
                for column in ("court_x_ft", "court_y_ft"):
                    values = pd.to_numeric(side_rows[column], errors="coerce")
                    interpolated_values = values.interpolate(
                        method="linear",
                        limit_area="inside",
                    )
                    side_rows.loc[gap_start : gap_end - 1, column] = (
                        interpolated_values.iloc[gap_start:gap_end].to_numpy()
                    )
                side_rows.loc[
                    gap_start : gap_end - 1, "interpolated"
                ] = True

        for column in ("court_x_ft", "court_y_ft"):
            values = pd.to_numeric(side_rows[column], errors="coerce")
            smoothed = values.rolling(3, center=True, min_periods=1).median()
            side_rows[column] = smoothed.where(values.notna())
        processed.append(side_rows)

    untouched = result.loc[~result["side"].isin(("near", "far"))]
    processed.append(untouched)
    if not processed:
        return result
    cleaned = pd.concat(processed, ignore_index=True)
    if "frame_index" in cleaned:
        cleaned = cleaned.sort_values(
            ["frame_index", "side"],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)
    cleaned["interpolated"] = cleaned["interpolated"].fillna(False).astype(bool)
    cleaned["edge_clipped"] = cleaned["edge_clipped"].fillna(False).astype(bool)
    return cleaned


def build_heatmap(
    df: pd.DataFrame,
    sides: Sequence[str] = ("near", "far"),
    bins_ft: float = 1.0,
    blur_ft: float = 2.0,
    x_range: tuple[float, float] = HEATMAP_X_RANGE,
    y_range: tuple[float, float] = HEATMAP_Y_RANGE,
) -> np.ndarray:
    """Build a normalized court occupancy histogram in court coordinates."""
    bins_ft = _positive_finite("bins_ft", bins_ft)
    blur_ft = _positive_finite("blur_ft", blur_ft, allow_zero=True)
    x_min, x_max = _validate_range("x_range", x_range)
    y_min, y_max = _validate_range("y_range", y_range)
    missing_columns = {"side", "court_x_ft", "court_y_ft"}.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Position data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    selected = df.loc[df["side"].isin(sides)]
    x = pd.to_numeric(selected["court_x_ft"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    y = pd.to_numeric(selected["court_y_ft"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    x_bins = max(1, int(np.ceil((x_max - x_min) / bins_ft)))
    y_bins = max(1, int(np.ceil((y_max - y_min) / bins_ft)))
    histogram, y_edges, x_edges = np.histogram2d(
        y,
        x,
        bins=(y_bins, x_bins),
        range=((y_min, y_max), (x_min, x_max)),
    )
    bin_width_x = (x_max - x_min) / x_bins
    bin_width_y = (y_max - y_min) / y_bins
    histogram = histogram.astype(np.float32)
    if blur_ft > 0:
        blurred = cv2.GaussianBlur(
            histogram,
            (0, 0),
            sigmaX=blur_ft / bin_width_x,
            sigmaY=blur_ft / bin_width_y,
        )
    else:
        blurred = histogram
    del y_edges, x_edges
    maximum = float(blurred.max(initial=0.0))
    if maximum > 0:
        blurred /= maximum
    return blurred[::-1].copy()


def render_heatmap(
    heatmap: np.ndarray,
    size_px: tuple[int, int] = (480, 1000),
    margin_x_ft: float = HEATMAP_MARGIN_X_FT,
    margin_y_ft: float = HEATMAP_MARGIN_Y_FT,
    alpha: float = 0.6,
) -> np.ndarray:
    """Render a heatmap over a court diagram, returning an RGB image."""
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("heatmap must be a finite two-dimensional array.")
    if values.size == 0:
        raise ValueError("heatmap must not be empty.")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    margin_x_ft = _positive_finite(
        "margin_x_ft", margin_x_ft, allow_zero=True
    )
    margin_y_ft = _positive_finite(
        "margin_y_ft", margin_y_ft, allow_zero=True
    )

    height, width = values.shape
    x_min, x_max = -margin_x_ft, 36.0 + margin_x_ft
    y_min, y_max = -margin_y_ft, 78.0 + margin_y_ft
    base = draw_top_down_court(
        size_px=size_px,
        margin_x_ft=margin_x_ft,
        margin_y_ft=margin_y_ft,
    )
    output_height, output_width = base.shape[:2]
    diagram_axes = court_to_diagram(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        size_px=size_px,
        margin_x_ft=margin_x_ft,
        margin_y_ft=margin_y_ft,
    )
    origin_x, origin_y = diagram_axes[0]
    scale_x = diagram_axes[1, 0] - origin_x
    scale_y = origin_y - diagram_axes[2, 1]
    pixel_y, pixel_x = np.indices((output_height, output_width), dtype=np.float32)
    court_x = (pixel_x - origin_x) / scale_x
    court_y = (origin_y - pixel_y) / scale_y
    x_bin_width = (x_max - x_min) / width
    y_bin_width = (y_max - y_min) / height
    map_x = ((court_x - x_min) / x_bin_width - 0.5).astype(np.float32)
    map_y = ((y_max - court_y) / y_bin_width - 0.5).astype(np.float32)
    aligned = cv2.remap(
        values,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    color_map = cv2.applyColorMap(
        np.clip(aligned * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    opacity = (np.clip(aligned, 0.0, 1.0) * alpha)[..., np.newaxis]
    blended = (
        base.astype(np.float32) * (1.0 - opacity)
        + color_map.astype(np.float32) * opacity
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _validate_range(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    try:
        lower, upper = (float(value) for value in bounds)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain two finite increasing values.") from error
    if not np.isfinite((lower, upper)).all() or lower >= upper:
        raise ValueError(f"{name} must contain two finite increasing values.")
    return lower, upper
