"""CourtVision AI Streamlit entry point.

Future responsibility: launch the web UI, accept a tennis match clip,
and orchestrate calibration, detection, tracking, bounce, speed, line
calling, and heatmap modules into a single interactive workflow.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from courtvision.analysis import (
    HEATMAP_MARGIN_X_FT,
    HEATMAP_MARGIN_Y_FT,
    build_heatmap,
    clean_positions,
    render_heatmap,
)
from courtvision.calibration import (
    DEFAULT_REFERENCE_PRESET,
    REFERENCE_POINT_PRESETS,
    calibration_from_dict,
    calibration_to_dict,
    compute_homography,
    draw_court_overlay,
    draw_top_down_court,
    get_reference_points,
    pixel_to_court,
    reprojection_error,
)
from courtvision.detection import (
    PlayerDetector,
    classify_detections,
    process_video,
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


@st.cache_resource(show_spinner=False)
def _cached_player_detector(
    model_name: str = "yolov8n.pt",
    conf: float = 0.4,
) -> PlayerDetector:
    return PlayerDetector(model_name=model_name, conf=conf)


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
    st.session_state.pop("detections_df", None)
    st.session_state.pop("detections_frames_processed", None)
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
        st.session_state.pop("detections_df", None)
        st.session_state.pop("detections_frames_processed", None)

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
            st.caption(
                "A saved calibration is only valid if the camera has not moved "
                "and the video resolution is the same."
            )
            calibration_upload = st.file_uploader(
                "Load saved calibration (JSON)",
                type=["json"],
                key="calibration_upload",
            )
            if calibration_upload is None:
                st.session_state.pop("calibration_load_signature", None)
                st.session_state.pop("calibration_load_feedback", None)
            else:
                load_signature = (
                    calibration_upload.name,
                    calibration_upload.size,
                )
                if load_signature != st.session_state.get(
                    "calibration_load_signature"
                ):
                    st.session_state["calibration_load_signature"] = load_signature
                    try:
                        saved_data = calibration_from_dict(
                            json.loads(calibration_upload.getvalue())
                        )
                    except json.JSONDecodeError as error:
                        st.session_state["calibration_load_feedback"] = (
                            "error",
                            f"Could not parse calibration JSON: {error}",
                        )
                    except ValueError as error:
                        st.session_state["calibration_load_feedback"] = (
                            "error",
                            str(error),
                        )
                    else:
                        saved_preset = saved_data["preset"]
                        if saved_preset not in REFERENCE_POINT_PRESETS:
                            st.session_state["calibration_load_feedback"] = (
                                "error",
                                f"Unknown calibration preset: {saved_preset}.",
                            )
                        elif (
                            saved_data["frame_width"] != frame_bgr.shape[1]
                            or saved_data["frame_height"] != frame_bgr.shape[0]
                        ):
                            st.session_state["calibration_load_feedback"] = (
                                "warning",
                                "This calibration was made for a different frame "
                                "resolution and may be wrong. It was not loaded.",
                            )
                        else:
                            st.session_state["calibration_points"] = (
                                saved_data["pixel_points"]
                            )
                            st.session_state["calibration_court_points"] = (
                                saved_data["court_points"]
                            )
                            st.session_state["calibration_homography"] = (
                                saved_data["H"]
                            )
                            st.session_state["calibration_preset"] = saved_preset
                            st.session_state["calibration_context"] = (
                                video_path,
                                saved_preset,
                            )
                            st.session_state["click_test_points"] = []
                            st.session_state["calibration_load_feedback"] = (
                                "success",
                                f"Loaded calibration from {calibration_upload.name}.",
                            )

                feedback = st.session_state.get("calibration_load_feedback")
                if feedback is not None:
                    level, message = feedback
                    if level == "success":
                        st.success(message)
                    elif level == "warning":
                        st.warning(message)
                    else:
                        st.error(message)

            preset_names = list(REFERENCE_POINT_PRESETS)
            selected_preset = st.selectbox(
                "Reference-point preset",
                options=preset_names,
                index=preset_names.index(DEFAULT_REFERENCE_PRESET),
                format_func=lambda name: name.replace("_", " "),
                key="calibration_preset",
            )
            calibration_context = (video_path, selected_preset)
            if st.session_state.get("calibration_context") != calibration_context:
                st.session_state["calibration_context"] = calibration_context
                st.session_state["calibration_points"] = []
                st.session_state.pop("calibration_court_points", None)
                st.session_state.pop("calibration_homography", None)
                st.session_state.pop("player_detection_result", None)
                st.session_state.pop("detections_df", None)
                st.session_state.pop("detections_frames_processed", None)
                st.session_state.pop("calibration_frame_index", None)
                st.session_state.pop("calibration_error", None)
                st.session_state["click_test_points"] = []
                feedback = st.session_state.get("calibration_load_feedback")
                if feedback is not None and feedback[0] == "success":
                    st.session_state.pop("calibration_load_feedback", None)

            calibration_points = st.session_state.setdefault(
                "calibration_points", []
            )
            reset_column, instruction_column = st.columns([1, 4])
            if reset_column.button("Reset calibration", key="reset_calibration"):
                calibration_points.clear()
                st.session_state.pop("calibration_court_points", None)
                st.session_state.pop("calibration_homography", None)
                st.session_state.pop("player_detection_result", None)
                st.session_state.pop("detections_df", None)
                st.session_state.pop("detections_frames_processed", None)
                st.session_state.pop("calibration_frame_index", None)
                st.session_state.pop("calibration_error", None)
                st.session_state["click_test_points"] = []
                feedback = st.session_state.get("calibration_load_feedback")
                if feedback is not None and feedback[0] == "success":
                    st.session_state.pop("calibration_load_feedback", None)

            corner_order = ("near-left", "near-right", "far-right", "far-left")
            if len(calibration_points) < len(corner_order):
                next_corner = corner_order[len(calibration_points)]
                instruction_column.info(
                    f"Click the {next_corner} corner for the "
                    f"{selected_preset.replace('_', ' ')} preset."
                )
            else:
                instruction_column.success("All four court corners are selected.")

            calibration_frame_index = st.session_state.get(
                "calibration_frame_index"
            )
            if calibration_frame_index is not None:
                st.caption(f"Calibration frame: {calibration_frame_index}")

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
                        if not calibration_points:
                            st.session_state["calibration_frame_index"] = frame_index
                        calibration_points.append(
                            (
                                float(click["x"]) / scale,
                                float(click["y"]) / scale,
                            )
                        )
                        st.rerun()

            if len(calibration_points) >= len(corner_order):
                court_points = st.session_state.get("calibration_court_points")
                if court_points is None:
                    court_points = get_reference_points(selected_preset)
                try:
                    homography = compute_homography(
                        calibration_points[: len(corner_order)],
                        court_points[: len(corner_order)],
                    )
                except ValueError as error:
                    st.session_state.pop("calibration_homography", None)
                    st.session_state.pop("detections_df", None)
                    st.session_state.pop("detections_frames_processed", None)
                    st.session_state["calibration_error"] = str(error)
                    st.error(str(error))
                else:
                    st.session_state["calibration_court_points"] = court_points
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
                    calibration_document = calibration_to_dict(
                        calibration_points,
                        court_points,
                        homography,
                        selected_preset,
                        (frame_bgr.shape[1], frame_bgr.shape[0]),
                        video_name=uploaded_clip.name,
                    )
                    st.download_button(
                        "Download calibration (JSON)",
                        data=json.dumps(calibration_document, indent=2),
                        file_name="courtvision_calibration.json",
                        mime="application/json",
                        key="download_calibration",
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
                            draw_top_down_court(
                                test_points,
                                margin_x_ft=HEATMAP_MARGIN_X_FT,
                                margin_y_ft=HEATMAP_MARGIN_Y_FT,
                            ),
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

                    if st.session_state.get("calibration_homography") is not None:
                        st.subheader("Player detection test")
                        if st.button(
                            "Detect players on this frame",
                            key="detect_players_on_frame",
                        ):
                            st.session_state.pop("player_detection_result", None)
                            try:
                                with st.spinner("Running YOLO..."):
                                    detections = (
                                        _cached_player_detector().detect_players(
                                            frame_bgr
                                        )
                                    )
                                (
                                    court_detections,
                                    rejected_detections,
                                ) = classify_detections(
                                    detections,
                                    st.session_state["calibration_homography"],
                                )
                                detection_frame = frame_bgr.copy()
                                detected_court_points = []
                                for detection in court_detections:
                                    x1, y1, x2, y2 = (
                                        round(value)
                                        for value in detection["bbox"]
                                    )
                                    foot_x, foot_y = (
                                        round(value)
                                        for value in detection["foot_point"]
                                    )
                                    cv2.rectangle(
                                        detection_frame,
                                        (x1, y1),
                                        (x2, y2),
                                        (0, 255, 0),
                                        2,
                                    )
                                    cv2.circle(
                                        detection_frame,
                                        (foot_x, foot_y),
                                        5,
                                        (0, 0, 255),
                                        -1,
                                    )
                                    cv2.putText(
                                        detection_frame,
                                        f"{detection['confidence']:.2f}",
                                        (x1, max(0, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 0),
                                        2,
                                    )
                                    detected_court_points.append(
                                        detection["court_point"]
                                    )
                                for detection in rejected_detections:
                                    x1, y1, x2, y2 = (
                                        round(value)
                                        for value in detection["bbox"]
                                    )
                                    court_x, court_y = detection["court_point"]
                                    cv2.rectangle(
                                        detection_frame,
                                        (x1, y1),
                                        (x2, y2),
                                        (0, 0, 255),
                                        2,
                                    )
                                    cv2.putText(
                                        detection_frame,
                                        f"({court_x:.1f}, {court_y:.1f}) "
                                        f"{detection['reason']}",
                                        (x1, max(0, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 0, 255),
                                        2,
                                    )
                                st.session_state["player_detection_result"] = {
                                    "detection_frame": detection_frame,
                                    "court_points": detected_court_points,
                                    "total_count": len(detections),
                                    "kept_count": len(court_detections),
                                    "rejected_count": len(rejected_detections),
                                    "frame_index": frame_index,
                                }
                            except Exception as error:
                                st.error(f"Player detection failed: {error}")

                        detection_result = st.session_state.get(
                            "player_detection_result"
                        )
                        if (
                            detection_result is not None
                            and detection_result["frame_index"] == frame_index
                        ):
                            st.caption(
                                f"YOLO found {detection_result['total_count']} people; "
                                f"{detection_result['kept_count']} kept on court; "
                                f"{detection_result['rejected_count']} rejected."
                            )
                            detection_column, detection_court_column = st.columns(2)
                            with detection_column:
                                st.image(
                                    cv2.cvtColor(
                                        detection_result["detection_frame"],
                                        cv2.COLOR_BGR2RGB,
                                    ),
                                    use_container_width=True,
                                )
                            with detection_court_column:
                                st.image(
                                    draw_top_down_court(
                                        detection_result["court_points"],
                                        margin_x_ft=HEATMAP_MARGIN_X_FT,
                                        margin_y_ft=HEATMAP_MARGIN_Y_FT,
                                    ),
                                    use_container_width=True,
                                )

                    st.subheader("Analyze full clip")
                    frame_step = st.selectbox(
                        "Frame step",
                        options=(2, 3, 5, 10),
                        index=2,
                        key="full_clip_frame_step",
                    )
                    match_type = st.radio(
                        "Match type",
                        ("Singles", "Doubles"),
                        key="full_clip_match_type",
                    )
                    if st.button("Analyze clip", key="analyze_full_clip"):
                        st.session_state.pop("detections_df", None)
                        st.session_state.pop("detections_frames_processed", None)
                        progress_bar = st.progress(0.0)
                        progress_state = {"done": 0}

                        def update_progress(done: int, total: int) -> None:
                            progress_state["done"] = done
                            progress_bar.progress(
                                done / total if total else 0.0
                            )

                        try:
                            with st.spinner("Analyzing full clip..."):
                                st.session_state["detections_df"] = process_video(
                                    video_path,
                                    st.session_state["calibration_homography"],
                                    _cached_player_detector(),
                                    step=frame_step,
                                    max_players=2 if match_type == "Singles" else 4,
                                    progress_callback=update_progress,
                                )
                            st.session_state["detections_frames_processed"] = (
                                progress_state["done"]
                            )
                        except (VideoReadError, RuntimeError, ValueError) as error:
                            st.error(f"Full clip analysis failed: {error}")

                    detections_df = st.session_state.get("detections_df")
                    if detections_df is not None:
                        detected_sides = detections_df.groupby("frame_index")[
                            "side"
                        ].nunique()
                        frames_with_both_players = int(
                            (detected_sides == 2).sum()
                        )
                        st.caption(
                            f"{len(detections_df):,} detections across "
                            f"{st.session_state.get('detections_frames_processed', 0):,} "
                            f"frames processed; both players detected in "
                            f"{frames_with_both_players:,} frames."
                        )
                        st.download_button(
                            "Download detections (CSV)",
                            data=detections_df.to_csv(index=False),
                            file_name="courtvision_detections.csv",
                            mime="text/csv",
                            key="download_detections_csv",
                        )

        uploaded_clip.seek(0)
        st.video(uploaded_clip)

st.subheader("Heatmap")
detections_df = st.session_state.get("detections_df")
if detections_df is None:
    detections_upload = st.file_uploader(
        "Load detections CSV",
        type=["csv"],
        key="load_detections_csv",
    )
    if detections_upload is not None:
        expected_columns = {
            "frame_index",
            "time_sec",
            "side",
            "confidence",
            "foot_x_px",
            "foot_y_px",
            "court_x_ft",
            "court_y_ft",
        }
        try:
            detections_df = pd.read_csv(detections_upload)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read detections CSV: {error}")
            detections_df = None
        if detections_df is not None:
            missing_columns = expected_columns.difference(detections_df.columns)
            if missing_columns:
                st.error(
                    "Detections CSV is missing expected columns: "
                    + ", ".join(sorted(missing_columns))
                )
                detections_df = None

if detections_df is not None:
    side_selection = st.radio(
        "Heatmap side",
        ("Near", "Far", "Both"),
        horizontal=True,
        key="heatmap_side",
    )
    clean_enabled = st.checkbox(
        "Clean outliers and fill short gaps",
        value=True,
        key="heatmap_clean_positions",
    )
    blur_ft = st.slider(
        "Blur (ft)",
        min_value=1,
        max_value=6,
        value=2,
        key="heatmap_blur_ft",
    )
    side_names = {
        "Near": ("near",),
        "Far": ("far",),
        "Both": ("near", "far"),
    }[side_selection]

    video_metadata = st.session_state.get("video_metadata")
    frame_height = getattr(video_metadata, "height", None)
    fps = getattr(video_metadata, "fps", None)
    frame_indices = pd.to_numeric(
        detections_df["frame_index"], errors="coerce"
    ).dropna()
    time_values = pd.to_numeric(
        detections_df["time_sec"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    index_values = pd.to_numeric(
        detections_df["frame_index"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    valid_time = (
        np.isfinite(time_values)
        & (time_values > 0)
        & np.isfinite(index_values)
        & (index_values > 0)
    )
    if fps is None and np.any(valid_time):
        fps = float(np.median(index_values[valid_time] / time_values[valid_time]))
    if fps is None or not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    unique_indices = np.sort(frame_indices.unique())
    frame_differences = np.diff(unique_indices)
    frame_differences = frame_differences[frame_differences > 0]
    frame_step = int(st.session_state.get("full_clip_frame_step", 1))
    if frame_differences.size:
        frame_step = int(np.min(frame_differences))

    rows_removed_as_outliers = 0
    rows_interpolated = 0
    positions_for_heatmap = detections_df
    try:
        if clean_enabled:
            for side_name in side_names:
                side_rows = detections_df.loc[
                    detections_df["side"] == side_name
                ].sort_values("frame_index", kind="stable")
                x_values = pd.to_numeric(side_rows["court_x_ft"], errors="coerce")
                y_values = pd.to_numeric(side_rows["court_y_ft"], errors="coerce")
                median_x = x_values.rolling(7, center=True, min_periods=1).median()
                median_y = y_values.rolling(7, center=True, min_periods=1).median()
                outliers = (
                    ((x_values - median_x).abs() > 12.0)
                    | ((y_values - median_y).abs() > 12.0)
                )
                rows_removed_as_outliers += int(outliers.sum())
            positions_for_heatmap = clean_positions(
                detections_df,
                frame_step=frame_step,
                fps=fps,
                frame_height=frame_height,
            )
            rows_interpolated = int(
                positions_for_heatmap.loc[
                    positions_for_heatmap["side"].isin(side_names),
                    "interpolated",
                ].sum()
            )
        heatmap_data = build_heatmap(
            positions_for_heatmap,
            sides=side_names,
            blur_ft=blur_ft,
        )
        heatmap_image = render_heatmap(
            heatmap_data,
            margin_x_ft=HEATMAP_MARGIN_X_FT,
            margin_y_ft=HEATMAP_MARGIN_Y_FT,
        )
    except (TypeError, ValueError) as error:
        st.error(f"Could not generate heatmap: {error}")
    else:
        selected_positions = positions_for_heatmap.loc[
            positions_for_heatmap["side"].isin(side_names)
        ]
        finite_positions = np.isfinite(
            pd.to_numeric(selected_positions["court_x_ft"], errors="coerce")
        ) & np.isfinite(
            pd.to_numeric(selected_positions["court_y_ft"], errors="coerce")
        )
        rows_used = int(finite_positions.sum())
        heatmap_column, metrics_column = st.columns([3, 1])
        heatmap_column.image(heatmap_image, use_container_width=True)
        metrics_column.metric("Rows used", f"{rows_used:,}")
        metrics_column.metric(
            "Rows removed as outliers",
            f"{rows_removed_as_outliers:,}",
        )
        metrics_column.metric("Rows interpolated", f"{rows_interpolated:,}")
        st.caption("Far-side positions are less precise than near-side ones.")
