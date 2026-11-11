"""Shared inference helpers for image, video, and webcam detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "artifacts" / "best.pt"


def load_model() -> YOLO:
    if not WEIGHTS.is_file():
        raise FileNotFoundError("Trained weights are missing. Run scripts/train.py first.")
    return YOLO(str(WEIGHTS))


def predict_frame(model: YOLO, frame: np.ndarray, confidence: float, image_size: int = 320):
    return model.predict(
        source=frame,
        conf=confidence,
        imgsz=image_size,
        device="cpu",
        verbose=False,
    )[0]


def annotate_frame(model: YOLO, frame: np.ndarray, confidence: float) -> tuple[np.ndarray, int]:
    result = predict_frame(model, frame, confidence)
    annotated = result.plot()
    count = len(result.boxes) if result.boxes is not None else 0
    cv2.putText(
        annotated,
        f"Potholes: {count}",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 190, 120),
        2,
        cv2.LINE_AA,
    )
    return annotated, count