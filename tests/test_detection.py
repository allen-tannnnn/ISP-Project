import numpy as np
import pytest

from courtvision.detection import (
    PlayerDetection,
    classify_detections,
    filter_on_court,
    process_video,
    select_players,
)
from courtvision.video_io import VideoMeta


HOMOGRAPHY = np.array(
    [
        [0.1, 0.0, -10.0],
        [0.0, 0.1, -20.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _detection_at_pixel(x: float, y: float) -> PlayerDetection:
    return {
        "bbox": (x - 5.0, y - 10.0, x + 5.0, y),
        "confidence": 0.9,
        "foot_point": (x, y),
    }


def test_filter_on_court_maps_foot_points_and_keeps_court_plus_margin():
    on_court = _detection_at_pixel(280.0, 590.0)  # (18, 39) feet
    within_margin = _detection_at_pixel(90.0, 300.0)  # (-1, 10) feet
    outside_margin = _detection_at_pixel(20.0, 590.0)  # (-8, 39) feet

    filtered = filter_on_court(
        [on_court, within_margin, outside_margin],
        HOMOGRAPHY,
    )

    assert len(filtered) == 2
    assert filtered[0]["court_point"] == pytest.approx((18.0, 39.0))
    assert filtered[1]["court_point"] == pytest.approx((-1.0, 10.0))
    assert filtered[0]["foot_point"] == on_court["foot_point"]
    assert "court_point" not in on_court


def test_filter_on_court_rejects_negative_or_nonfinite_margin():
    with pytest.raises(ValueError, match="margin_ft"):
        filter_on_court([], HOMOGRAPHY, margin_ft=-1)

    with pytest.raises(ValueError, match="margin_ft"):
        filter_on_court([], HOMOGRAPHY, margin_ft=float("nan"))


def test_classify_detections_keeps_detection_15_feet_behind_far_baseline():
    detection = _detection_at_pixel(280.0, 1130.0)  # (18, 93) feet

    kept, rejected = classify_detections([detection], HOMOGRAPHY)

    assert len(kept) == 1
    assert kept[0]["court_point"] == pytest.approx((18.0, 93.0))
    assert not rejected


def test_classify_detections_rejects_detection_beyond_baseline_margin():
    detection = _detection_at_pixel(280.0, 1130.0)  # (18, 93) feet

    kept, rejected = classify_detections(
        [detection],
        HOMOGRAPHY,
        margin_y_ft=6,
    )

    assert not kept
    assert rejected[0]["court_point"] == pytest.approx((18.0, 93.0))
    assert rejected[0]["reason"] == "behind baseline margin"


def test_classify_detections_rejects_detection_eight_feet_outside_sideline():
    detection = _detection_at_pixel(540.0, 590.0)  # (44, 39) feet

    kept, rejected = classify_detections([detection], HOMOGRAPHY)

    assert not kept
    assert rejected[0]["court_point"] == pytest.approx((44.0, 39.0))
    assert rejected[0]["reason"] == "outside sideline margin"


def test_select_players_keeps_highest_confidence_detection_on_each_side():
    detections = [
        {**_detection_at_pixel(280.0, 580.0), "court_point": (18.0, 38.0)},
        {
            **_detection_at_pixel(280.0, 570.0),
            "confidence": 0.8,
            "court_point": (18.0, 37.0),
        },
        {
            **_detection_at_pixel(280.0, 590.0),
            "confidence": 0.7,
            "court_point": (18.0, 39.0),
        },
        {
            **_detection_at_pixel(280.0, 600.0),
            "confidence": 0.6,
            "court_point": (18.0, 40.0),
        },
    ]

    selected = select_players(detections)

    assert selected == [detections[0], detections[2]]


def test_select_players_returns_empty_list_for_no_detections():
    assert select_players([]) == []


def test_process_video_samples_frames_reports_progress_and_releases_capture(
    monkeypatch,
):
    from courtvision import detection

    class Capture:
        released = False

        def release(self):
            self.released = True

    class Detector:
        def detect_players(self, _frame):
            return [
                _detection_at_pixel(280.0, 580.0),
                _detection_at_pixel(280.0, 590.0),
            ]

    capture = Capture()
    metadata = VideoMeta(
        fps=10.0,
        frame_count=5,
        width=1,
        height=1,
        duration_sec=0.5,
    )

    def fake_open_video(_path):
        return capture, metadata

    def fake_iter_frames(_capture, skip_n):
        for frame_index in range(0, 5, skip_n):
            yield frame_index, np.zeros((1, 1, 3), dtype=np.uint8)

    monkeypatch.setattr(detection, "open_video", fake_open_video)
    monkeypatch.setattr(
        detection,
        "iter_frames",
        fake_iter_frames,
    )
    progress = []

    result = process_video(
        "clip.mp4",
        HOMOGRAPHY,
        Detector(),
        step=2,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert list(result.columns) == [
        "frame_index",
        "time_sec",
        "side",
        "confidence",
        "foot_x_px",
        "foot_y_px",
        "court_x_ft",
        "court_y_ft",
    ]
    assert result["frame_index"].tolist() == [0, 0, 2, 2, 4, 4]
    assert result["side"].tolist() == ["near", "far"] * 3
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert capture.released
