"""Player detection and court-position filtering."""

from collections.abc import Sequence
from typing import TypedDict

import numpy as np

from courtvision.calibration import pixel_to_court


class PlayerDetection(TypedDict):
    bbox: tuple[float, float, float, float]
    confidence: float
    foot_point: tuple[float, float]


class CourtPlayerDetection(PlayerDetection):
    court_point: tuple[float, float]


class RejectedCourtPlayerDetection(CourtPlayerDetection):
    reason: str


def _validate_margin(name: str, margin: float) -> float:
    try:
        value = float(margin)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite nonnegative number.") from error
    if isinstance(margin, bool) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number.")
    return value


class PlayerDetector:
    """Detect people in video frames with an Ultralytics YOLO model."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.4):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Player detection requires the Ultralytics package."
            ) from error

        try:
            self.model = YOLO(model_name)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load YOLO model {model_name!r}: {error}"
            ) from error
        self.conf = conf

    def detect_players(self, frame_bgr: np.ndarray) -> list[PlayerDetection]:
        """Return person detections with bbox and foot point in frame pixels."""
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[0] == 0
            or frame_bgr.shape[1] == 0
            or frame_bgr.shape[2] != 3
        ):
            raise ValueError("frame_bgr must be a non-empty BGR image.")

        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            classes=[0],
            imgsz=640,
            device="cpu",
            verbose=False,
        )
        detections: list[PlayerDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(box.conf[0]),
                        "foot_point": ((x1 + x2) / 2.0, y2),
                    }
                )
        return detections


def filter_on_court(
    detections: Sequence[PlayerDetection],
    H: np.ndarray,
    margin_ft: float | None = None,
    margin_x_ft: float = 4,
    margin_y_ft: float = 25,
) -> list[CourtPlayerDetection]:
    """Map player foot points to court feet and discard positions off-court.

    The legacy ``margin_ft`` argument overrides both axis-specific margins.
    """
    if margin_ft is not None:
        margin_ft = _validate_margin("margin_ft", margin_ft)
        margin_x_ft = margin_y_ft = margin_ft
    kept, _ = classify_detections(detections, H, margin_x_ft, margin_y_ft)
    return kept


def classify_detections(
    detections: Sequence[PlayerDetection],
    H: np.ndarray,
    margin_x_ft: float = 4,
    margin_y_ft: float = 25,
) -> tuple[list[CourtPlayerDetection], list[RejectedCourtPlayerDetection]]:
    """Map detections to court feet and classify them against court margins."""
    margin_x_ft = _validate_margin("margin_x_ft", margin_x_ft)
    margin_y_ft = _validate_margin("margin_y_ft", margin_y_ft)

    kept: list[CourtPlayerDetection] = []
    rejected: list[RejectedCourtPlayerDetection] = []
    for detection in detections:
        try:
            court_point = pixel_to_court(H, detection["foot_point"])
        except ValueError:
            continue
        court_x, court_y = float(court_point[0]), float(court_point[1])
        mapped_detection: CourtPlayerDetection = {
            **detection,
            "court_point": (court_x, court_y),
        }
        reason = None
        if court_x < -margin_x_ft or court_x > 36.0 + margin_x_ft:
            reason = "outside sideline margin"
        elif court_y < -margin_y_ft or court_y > 78.0 + margin_y_ft:
            reason = "behind baseline margin"
        if reason is not None:
            rejected.append({**mapped_detection, "reason": reason})
        else:
            kept.append(mapped_detection)
    return kept, rejected
