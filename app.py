"""CourtVision AI Streamlit entry point.

Future responsibility: launch the web UI, accept a tennis match clip,
and orchestrate calibration, detection, tracking, bounce, speed, line
calling, and heatmap modules into a single interactive workflow.
"""

import streamlit as st

MAX_CLIP_MB = 50

st.set_page_config(
    page_title="CourtVision AI",
    layout="wide",
)

with st.sidebar:
    st.title("CourtVision AI")
    st.caption("Tennis video analysis")

st.title("CourtVision AI")
st.markdown(
    f"Upload a **short** `.mp4` clip (**{MAX_CLIP_MB} MB or less**) "
    "from a tennis rally. Longer or heavier files may fail to upload."
)

uploaded_clip = st.file_uploader(
    "Tennis clip",
    type=["mp4"],
    help=f"MP4 only. Keep clips under {MAX_CLIP_MB} MB.",
)

if uploaded_clip is None:
    st.info("No clip yet. Use the uploader above to add an `.mp4` file.")
else:
    size_mb = uploaded_clip.size / (1024 * 1024)
    if size_mb > MAX_CLIP_MB:
        st.error(
            f"This file is {size_mb:.1f} MB. Please upload a short clip "
            f"under {MAX_CLIP_MB} MB."
        )
    else:
        st.success(f"Loaded **{uploaded_clip.name}** ({size_mb:.1f} MB).")
        st.video(uploaded_clip)
