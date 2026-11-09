"""Player detection test UI."""

import cv2
import streamlit as st

from courtvision.analysis import HEATMAP_MARGIN_X_FT, HEATMAP_MARGIN_Y_FT
from courtvision.calibration import draw_top_down_court
from courtvision.detection import PlayerDetector, classify_detections


@st.cache_resource(show_spinner=False)
def _cached_player_detector(
    model_name: str = "yolov8n.pt",
    conf: float = 0.4,
) -> PlayerDetector:
    return PlayerDetector(model_name=model_name, conf=conf)


def render_detection_test(
    frame_bgr,
    frame_index: int,
    homography,
) -> None:
    """Render the player detection test and its results for one frame."""
    if homography is None:
        return

    st.subheader("Player detection test")
    if st.button(
        "Detect players on this frame",
        key="detect_players_on_frame",
    ):
        st.session_state.pop("player_detection_result", None)
        try:
            with st.spinner("Running YOLO..."):
                detections = _cached_player_detector().detect_players(frame_bgr)
            court_detections, rejected_detections = classify_detections(
                detections,
                homography,
            )
            detection_frame = frame_bgr.copy()
            detected_court_points = []
            for detection in court_detections:
                x1, y1, x2, y2 = (round(value) for value in detection["bbox"])
                foot_x, foot_y = (
                    round(value) for value in detection["foot_point"]
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
                detected_court_points.append(detection["court_point"])
            for detection in rejected_detections:
                x1, y1, x2, y2 = (round(value) for value in detection["bbox"])
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

    detection_result = st.session_state.get("player_detection_result")
    if (
        detection_result is None
        or detection_result["frame_index"] != frame_index
    ):
        return

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
