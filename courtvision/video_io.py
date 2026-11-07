"""Video reading helpers.

Saves a Streamlit upload to a temporary file, opens it with OpenCV, and
yields frames one at a time so a CPU-only host never holds the whole clip
in memory.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import cv2
import numpy as np

# Write the upload in 1 MB pieces so we do not copy the whole file twice.
_CHUNK_BYTES = 1024 * 1024


class VideoReadError(Exception):
    """Raised when a clip cannot be opened or has no usable frames."""


@dataclass(frozen=True)
class VideoMeta:
    """Basic facts about an opened video.

    duration_sec is frame_count / fps. Speed and bounce timing later in
    the pipeline depend on this being a real, positive frame rate.
    """

    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


@contextmanager
def save_upload(uploaded_file: BinaryIO) -> Iterator[Path]:
    """Write an uploaded clip to a temp file and delete it afterwards.

    ``uploaded_file`` is anything with ``read()`` and ``seek()``, such as
    a Streamlit ``UploadedFile``. The path is yielded only after the bytes
    are on disk. The temp file is removed when the ``with`` block ends,
    including when an error is raised.

    Raises:
        VideoReadError: the upload was empty.
    """
    name = getattr(uploaded_file, "name", "") or "clip.mp4"
    suffix = Path(name).suffix or ".mp4"

    # delete=False so Windows can reopen the path after we close this handle.
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(handle.name)
    try:
        uploaded_file.seek(0)
        while True:
            chunk = uploaded_file.read(_CHUNK_BYTES)
            if not chunk:
                break
            handle.write(chunk)
        handle.close()

        if tmp_path.stat().st_size == 0:
            raise VideoReadError(
                "The uploaded file is empty. Choose a short .mp4 clip and try again."
            )
        yield tmp_path
    finally:
        handle.close()
        tmp_path.unlink(missing_ok=True)


def open_video(path: str | Path) -> tuple[cv2.VideoCapture, VideoMeta]:
    """Open a video file and return the capture plus its metadata.

    The caller owns the capture and must call ``capture.release()`` when
    finished (``iter_frames`` does not release it).

    Raises:
        VideoReadError: the file is missing, will not open, has no frames,
            uses a codec OpenCV cannot decode, or has no frame rate.
    """
    path = Path(path)
    if not path.is_file():
        raise VideoReadError(f"Video file not found: {path.name}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(
            "OpenCV could not open this file. It may be corrupt or use a "
            "codec this build of OpenCV does not support. Export the clip "
            "as an H.264 .mp4 and try again."
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Some containers report a size but the first frame still will not decode.
    ok, _first = capture.read()
    if not ok:
        capture.release()
        raise VideoReadError(
            "This video has no readable frames. It may be corrupt, empty, "
            "or use an unsupported codec. Export it as a short H.264 .mp4."
        )
    # Put the read head back so the next caller sees frame 0.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise VideoReadError(
            "This video has no usable frames. It may be corrupt or use an "
            "unsupported codec. Export it as a short H.264 .mp4."
        )
    if fps <= 0:
        capture.release()
        raise VideoReadError(
            "This video has no usable frame rate, so duration and ball "
            "speed cannot be computed. Re-export it as a standard .mp4."
        )

    meta = VideoMeta(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=frame_count / fps,
    )
    return capture, meta


def _downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink a frame so its longer side is at most ``max_side`` pixels."""
    height, width = frame.shape[:2]
    longer = max(height, width)
    if longer <= max_side:
        return frame
    scale = max_side / float(longer)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    # INTER_AREA is the usual choice when shrinking photos or video frames.
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def iter_frames(
    capture: cv2.VideoCapture,
    skip_n: int = 1,
    max_side: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` one frame at a time.

    ``skip_n`` keeps every Nth frame. ``1`` means every frame, ``2`` means
    every other frame. The index is the original frame number in the file,
    not the count of frames yielded.

    ``max_side`` downscales a frame only when its longer side is larger
    than that many pixels. Aspect ratio is kept.

    Raises:
        ValueError: ``skip_n`` or ``max_side`` is not a positive integer.
    """
    if skip_n < 1:
        raise ValueError("skip_n must be 1 or greater (1 keeps every frame).")
    if max_side is not None and max_side < 1:
        raise ValueError("max_side must be a positive pixel count.")

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if index % skip_n == 0:
            if max_side is not None:
                frame = _downscale(frame, max_side)
            yield index, frame
        index += 1
