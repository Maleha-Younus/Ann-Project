"""Frame-by-frame inference and H.264 encoding for uploaded videos."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import av
import cv2
import numpy as np
from ultralytics import YOLO

MAX_VIDEO_FRAMES = 900


def process_video(
    input_path: Path,
    output_path: Path,
    model: YOLO,
    confidence: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open this video. Try an MP4 or AVI file.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > MAX_VIDEO_FRAMES:
        capture.release()
        raise ValueError(f"This course demo processes up to {MAX_VIDEO_FRAMES} frames per clip. Trim the video and try again.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The uploaded video has invalid frame dimensions.")
    if not np.isfinite(fps) or fps <= 0:
        fps = 20.0

    output_width = width + width % 2
    output_height = height + height % 2
    output_fps = max(1, round(fps))
    original_preview_path = output_path.with_name("original-preview.mp4")
    frame_count = 0
    total_detections = 0
    confidences: list[float] = []
    unique_track_ids: set[int] = set()
    untracked_detections = 0
    started_at = time.perf_counter()

    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []):
        tracker.reset()

    try:
        with av.open(str(output_path), mode="w") as output_container, av.open(str(original_preview_path), mode="w") as preview_container:
            output_stream = output_container.add_stream("libx264", rate=output_fps)
            output_stream.width = output_width
            output_stream.height = output_height
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"preset": "ultrafast", "crf": "24"}
            preview_stream = preview_container.add_stream("libx264", rate=output_fps)
            preview_stream.width = output_width
            preview_stream.height = output_height
            preview_stream.pix_fmt = "yuv420p"
            preview_stream.options = {"preset": "ultrafast", "crf": "24"}

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_count >= MAX_VIDEO_FRAMES:
                    raise ValueError(f"This course demo processes up to {MAX_VIDEO_FRAMES} frames per clip. Trim the video and try again.")

                result = model.track(
                    source=frame,
                    conf=confidence,
                    imgsz=320,
                    device="cpu",
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                )[0]
                boxes = result.boxes
                detections = len(boxes) if boxes is not None else 0
                total_detections += detections
                if boxes is not None and detections:
                    confidences.extend(float(value) for value in boxes.conf.cpu().tolist())
                    track_ids = boxes.id
                    if track_ids is None:
                        untracked_detections += detections
                    else:
                        ids = [int(value) for value in track_ids.cpu().tolist()]
                        unique_track_ids.update(ids)
                        untracked_detections += detections - len(ids)

                annotated = result.plot()
                unique_so_far = len(unique_track_ids)
                cv2.putText(
                    annotated,
                    f"Frame detections: {detections} | Unique tracks: {unique_so_far}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (30, 190, 120),
                    2,
                    cv2.LINE_AA,
                )
                if output_width != width or output_height != height:
                    frame = cv2.copyMakeBorder(
                        frame,
                        0,
                        output_height - height,
                        0,
                        output_width - width,
                        cv2.BORDER_REPLICATE,
                    )
                    annotated = cv2.copyMakeBorder(
                        annotated,
                        0,
                        output_height - height,
                        0,
                        output_width - width,
                        cv2.BORDER_REPLICATE,
                    )
                original_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                for packet in preview_stream.encode(original_frame):
                    preview_container.mux(packet)
                video_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
                for packet in output_stream.encode(video_frame):
                    output_container.mux(packet)

                frame_count += 1
                if progress_callback is not None:
                    progress_callback(frame_count, total_frames)

            for packet in output_stream.encode():
                output_container.mux(packet)
            for packet in preview_stream.encode():
                preview_container.mux(packet)
    finally:
        capture.release()

    if frame_count == 0 or not output_path.is_file():
        raise ValueError("No readable video frames were found.")

    return {
        "frames_processed": frame_count,
        "total_frame_detections": total_detections,
        "unique_tracked_potholes": len(unique_track_ids) if not untracked_detections else None,
        "tracked_pothole_ids": len(unique_track_ids),
        "untracked_detections": untracked_detections,
        "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "processing_time_seconds": time.perf_counter() - started_at,
        "original_preview_path": str(original_preview_path),
    }