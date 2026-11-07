import cv2
import numpy as np
import pytest

from courtvision.video_io import VideoReadError, get_frame, iter_frames, open_video


@pytest.fixture
def sample_video(tmp_path):
    """Create a small ten-frame MP4 for video I/O tests."""
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.fail("OpenCV could not create the sample MP4 video.")

    try:
        for frame_index in range(10):
            frame = np.full(
                (48, 64, 3),
                (frame_index * 20, 0, 0),
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()

    return video_path


def test_open_video_returns_correct_metadata(sample_video):
    capture, metadata = open_video(sample_video)
    try:
        assert metadata.fps == pytest.approx(10.0)
        assert metadata.frame_count == 10
        assert metadata.width == 64
        assert metadata.height == 48
        assert metadata.duration_sec == pytest.approx(1.0)
    finally:
        capture.release()


def test_iter_frames_respects_skip_parameter(sample_video):
    capture, _ = open_video(sample_video)
    try:
        frames = list(iter_frames(capture, skip_n=2))
    finally:
        capture.release()

    assert [frame_index for frame_index, _ in frames] == [0, 2, 4, 6, 8]


def test_get_frame_returns_requested_frame_shape(sample_video):
    frame = get_frame(sample_video, 3)

    assert frame.shape == (48, 64, 3)


@pytest.mark.parametrize("frame_index", [-1, 10])
def test_get_frame_rejects_negative_and_out_of_range_indexes(
    sample_video, frame_index
):
    with pytest.raises(VideoReadError, match="frame"):
        get_frame(sample_video, frame_index)


def test_open_video_rejects_corrupt_file(tmp_path):
    corrupt_path = tmp_path / "corrupt.mp4"
    corrupt_path.write_bytes(b"not a valid video file")

    with pytest.raises(VideoReadError, match="could not open|readable frames"):
        open_video(corrupt_path)
