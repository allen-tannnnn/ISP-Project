"""CourtVision AI Streamlit entry point.

Future responsibility: launch the web UI, accept a tennis match clip,
and orchestrate calibration, detection, tracking, bounce, speed, line
calling, and heatmap modules into a single interactive workflow.
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from courtvision.calibration import (
    DEFAULT_REFERENCE_PRESET,
    REFERENCE_POINT_PRESETS,
    compute_homography,
    draw_court_overlay,
    draw_top_down_court,
    get_reference_points,
    pixel_to_court,
    reprojection_error,
)
from courtvision.video_io import (
    VideoReadError,
    get_frame,
    open_video,
    save_upload,
)

MAX_CLIP_MB = 50
RECOMMENDED_CLIP_MB = 40
MAX_CLIP_SECONDS = 30


@st.cache_data(max_entries=16, show_spinner=False)
def _cached_get_frame(path: str, frame_index: int):
    return get_frame(path, frame_index)


st.set_page_config(
    page_title="CourtVision AI",
    layout="wide",
)

with st.sidebar:
    st.title("CourtVision AI")
    st.caption("Tennis video analysis")

st.title("CourtVision AI")
st.markdown(
    f"Upload a **short** `.mp4` clip (**{RECOMMENDED_CLIP_MB} MB or less recommended**, "
    f"with a {MAX_CLIP_MB} MB upload limit) "
    "from a tennis rally. Longer or heavier files may fail to upload."
)

uploaded_clip = st.file_uploader(
    "Tennis clip",
    type=["mp4"],
    help=(
        f"MP4 only. Keep clips under {RECOMMENDED_CLIP_MB} MB for best results. "
        f"The upload limit is {MAX_CLIP_MB} MB."
    ),
)

if uploaded_clip is None:
    st.info("No clip yet. Use the uploader above to add an `.mp4` file.")
    saved_path = st.session_state.pop("video_path", None)
    if saved_path:
        Path(saved_path).unlink(missing_ok=True)
    st.session_state.pop("video_metadata", None)
    st.session_state.pop("upload_signature", None)
else:
    size_mb = uploaded_clip.size / (1024 * 1024)
    st.success(f"Loaded **{uploaded_clip.name}** ({size_mb:.1f} MB).")

    upload_signature = (
        getattr(uploaded_clip, "file_id", None),
        uploaded_clip.name,
        uploaded_clip.size,
    )
    if st.session_state.get("upload_signature") != upload_signature:
        saved_path = st.session_state.pop("video_path", None)
        if saved_path:
            Path(saved_path).unlink(missing_ok=True)
        st.session_state.pop("video_metadata", None)
        st.session_state.pop("upload_signature", None)
        st.session_state.pop("frame_scrubber", None)

        try:
            with save_upload(uploaded_clip, delete_on_exit=False) as clip_path:
                capture, metadata = open_video(clip_path)
                capture.release()
        except VideoReadError as error:
            st.error(str(error))
        else:
            st.session_state["video_path"] = str(clip_path)
            st.session_state["video_metadata"] = metadata
            st.session_state["upload_signature"] = upload_signature

    video_path = st.session_state.get("video_path")
    metadata = st.session_state.get("video_metadata")
    if video_path is not None and metadata is not None:
        st.subheader("Video details")
        fps_column, resolution_column, duration_column, frames_column = st.columns(4)
        fps_column.metric("Frame rate", f"{metadata.fps:.2f} fps")
        resolution_column.metric(
            "Resolution", f"{metadata.width} × {metadata.height}"
        )
        duration_column.metric("Duration", f"{metadata.duration_sec:.2f} sec")
        frames_column.metric("Total frames", f"{metadata.frame_count:,}")

        if metadata.duration_sec > MAX_CLIP_SECONDS:
            st.warning(
                f"This clip is longer than {MAX_CLIP_SECONDS} seconds. "
                "Longer clips may take more time and memory to process."
            )
        if size_mb > RECOMMENDED_CLIP_MB:
            st.warning(
                f"This file is larger than the recommended {RECOMMENDED_CLIP_MB} MB "
                "for Streamlit Community Cloud and may use more memory."
            )
        if metadata.fps < 30:
            st.info(
                "This clip is below 30 fps. Fast ball movement may be harder to analyze."
            )

        frame_index = st.slider(
            "Frame number",
            min_value=0,
            max_value=metadata.frame_count - 1,
            value=0,
            key="frame_scrubber",
        )
        try:
            frame_bgr = _cached_get_frame(video_path, frame_index)
        except VideoReadError as error:
            st.error(str(error))
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_column, details_column = st.columns([3, 1])
            frame_column.image(frame_rgb, use_container_width=True)
            details_column.metric("Frame", f"{frame_index:,}")
            details_column.metric(
                "Timestamp", f"{frame_index / metadata.fps:.2f} sec"
            )

            st.subheader("Calibration")
            preset_names = list(REFERENCE_POINT_PRESETS)
            selected_preset = st.selectbox(
                "Reference-point preset",
                options=preset_names,
                index=preset_names.index(DEFAULT_REFERENCE_PRESET),
                format_func=lambda name: name.replace("_", " "),
                key="calibration_preset",
            )
            calibration_context = (video_path, frame_index, selected_preset)
            if st.session_state.get("calibration_context") != calibration_context:
                st.session_state["calibration_context"] = calibration_context
                st.session_state["calibration_points"] = []
                st.session_state.pop("calibration_homography", None)
                st.session_state.pop("calibration_error", None)
                st.session_state["click_test_points"] = []

            calibration_points = st.session_state.setdefault(
                "calibration_points", []
            )
            reset_column, instruction_column = st.columns([1, 4])
            if reset_column.button("Reset calibration", key="reset_calibration"):
                calibration_points.clear()
                st.session_state.pop("calibration_homography", None)
                st.session_state.pop("calibration_error", None)
                st.session_state["click_test_points"] = []

            corner_order = ("near-left", "near-right", "far-right", "far-left")
            if len(calibration_points) < len(corner_order):
                next_corner = corner_order[len(calibration_points)]
                instruction_column.info(
                    f"Click the {next_corner} corner for the "
                    f"{selected_preset.replace('_', ' ')} preset."
                )
            else:
                instruction_column.success("All four court corners are selected.")

            original_height, original_width = frame_bgr.shape[:2]
            displayed_width = min(original_width, 800)
            scale = displayed_width / original_width
            displayed_height = max(1, round(original_height * scale))
            if displayed_width == original_width:
                displayed_frame = frame_bgr.copy()
            else:
                displayed_frame = cv2.resize(
                    frame_bgr,
                    (displayed_width, displayed_height),
                    interpolation=cv2.INTER_AREA,
                )

            for point_number, (pixel_x, pixel_y) in enumerate(
                calibration_points, start=1
            ):
                marker_x = round(pixel_x * scale)
                marker_y = round(pixel_y * scale)
                cv2.circle(
                    displayed_frame,
                    (marker_x, marker_y),
                    12,
                    (0, 0, 255),
                    thickness=2,
                )
                cv2.putText(
                    displayed_frame,
                    str(point_number),
                    (marker_x + 14, marker_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            click = streamlit_image_coordinates(
                cv2.cvtColor(displayed_frame, cv2.COLOR_BGR2RGB),
                width=displayed_width,
                height=displayed_height,
                key="calibration_image",
                cursor="crosshair",
            )
            if isinstance(click, dict) and all(
                key in click for key in ("x", "y", "unix_time")
            ):
                click_signature = (click["unix_time"], click["x"], click["y"])
                if click_signature != st.session_state.get(
                    "calibration_last_click"
                ):
                    st.session_state["calibration_last_click"] = click_signature
                    if len(calibration_points) < len(corner_order):
                        calibration_points.append(
                            (
                                float(click["x"]) / scale,
                                float(click["y"]) / scale,
                            )
                        )
                        st.rerun()

            if len(calibration_points) == len(corner_order):
                court_points = get_reference_points(selected_preset)
                try:
                    homography = compute_homography(
                        calibration_points,
                        court_points,
                    )
                except ValueError as error:
                    st.session_state.pop("calibration_homography", None)
                    st.session_state["calibration_error"] = str(error)
                    st.error(str(error))
                else:
                    st.session_state["calibration_homography"] = homography
                    st.session_state.pop("calibration_error", None)
                    mean_error, max_error = reprojection_error(
                        homography,
                        calibration_points,
                        court_points,
                    )
                    with st.expander("Homography matrix"):
                        st.code(np.array2string(homography, precision=6))
                    mean_column, max_column = st.columns(2)
                    mean_column.metric(
                        "Mean reprojection error",
                        f"{mean_error:.2f} px",
                    )
                    max_column.metric(
                        "Maximum reprojection error",
                        f"{max_error:.2f} px",
                    )
                    st.subheader("Calibration check")
                    show_line_labels = st.checkbox(
                        "Show line labels",
                        value=False,
                        key="show_calibration_line_labels",
                    )
                    overlay_bgr = draw_court_overlay(
                        frame_bgr,
                        st.session_state["calibration_homography"],
                        show_labels=show_line_labels,
                    )
                    st.image(
                        cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )
                    st.caption(
                        "The lines should sit on the real court lines, including "
                        "the net, service lines, and doubles sidelines, which you "
                        "did NOT click. If they drift or look twisted, press Reset "
                        "and re-click, in order: near-left, near-right, far-right, "
                        "far-left."
                    )
                    match_status = st.radio(
                        "Do the lines match the real court?",
                        ("Not checked", "Yes", "No"),
                        key="calibration_match_status",
                    )
                    if match_status == "Yes":
                        st.success("The calibration lines match the real court.")
                    elif match_status == "No":
                        st.warning(
                            "Likely causes: wrong click order, imprecise clicks "
                            "(zoom in), or camera lens distortion."
                        )

                    st.subheader(
                        "Click test: where does a point land on the court?"
                    )
                    test_points = st.session_state.setdefault(
                        "click_test_points", []
                    )
                    if st.button("Clear test points", key="clear_test_points"):
                        test_points.clear()

                    test_frame_column, court_column = st.columns(2)
                    test_height, test_width = frame_bgr.shape[:2]
                    test_display_width = min(test_width, 800)
                    test_scale = test_display_width / test_width
                    test_display_height = max(1, round(test_height * test_scale))
                    if test_display_width == test_width:
                        test_display_frame = frame_bgr.copy()
                    else:
                        test_display_frame = cv2.resize(
                            frame_bgr,
                            (test_display_width, test_display_height),
                            interpolation=cv2.INTER_AREA,
                        )

                    with test_frame_column:
                        test_click = streamlit_image_coordinates(
                            cv2.cvtColor(test_display_frame, cv2.COLOR_BGR2RGB),
                            width=test_display_width,
                            height=test_display_height,
                            key="click_test_image",
                            cursor="crosshair",
                        )
                    if isinstance(test_click, dict) and all(
                        key in test_click for key in ("x", "y", "unix_time")
                    ):
                        test_signature = (
                            test_click["unix_time"],
                            test_click["x"],
                            test_click["y"],
                        )
                        if test_signature != st.session_state.get(
                            "click_test_last_click"
                        ):
                            st.session_state["click_test_last_click"] = test_signature
                            original_pixel = (
                                float(test_click["x"]) / test_scale,
                                float(test_click["y"]) / test_scale,
                            )
                            try:
                                court_point = pixel_to_court(
                                    st.session_state["calibration_homography"],
                                    original_pixel,
                                )
                            except ValueError as error:
                                st.error(str(error))
                            else:
                                test_points.append(
                                    (float(court_point[0]), float(court_point[1]))
                                )
                                del test_points[:-5]

                    with court_column:
                        st.image(
                            draw_top_down_court(test_points),
                            use_container_width=True,
                        )
                        if test_points:
                            point_descriptions = []
                            for point_number, (court_x, court_y) in enumerate(
                                test_points,
                                start=1,
                            ):
                                description = (
                                    f"Point {point_number}: "
                                    f"({court_x:.1f}, {court_y:.1f}) ft"
                                )
                                if not (
                                    0.0 <= court_x <= 36.0
                                    and 0.0 <= court_y <= 78.0
                                ):
                                    description += " — off court"
                                point_descriptions.append(description)
                            st.caption("\n\n".join(point_descriptions))

        uploaded_clip.seek(0)
        st.video(uploaded_clip)
