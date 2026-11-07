"""CourtVision AI Streamlit entry point.

Future responsibility: launch the web UI, accept a tennis match clip,
and orchestrate calibration, detection, tracking, bounce, speed, line
calling, and heatmap modules into a single interactive workflow.
"""

from pathlib import Path

import cv2
import streamlit as st

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

        uploaded_clip.seek(0)
        st.video(uploaded_clip)
