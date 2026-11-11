"""Evaluate the trained detector on the untouched test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from ultralytics import YOLO

from src.project_data import DATA_YAML, METADATA, PROCESSED, ROOT

WEIGHTS = ROOT / "artifacts" / "best.pt"
OUTPUT = ROOT / "artifacts"
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.5


def _ground_truth(label_path: Path, image_shape: tuple[int, int]) -> list[list[float]]:
    height, width = image_shape
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 5:
            continue
        _, center_x, center_y, box_width, box_height = values
        boxes.append([
            (center_x - box_width / 2) * width,
            (center_y - box_height / 2) * height,
            (center_x + box_width / 2) * width,
            (center_y + box_height / 2) * height,
        ])
    return boxes


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def _match(predictions: list[dict], truth: list[list[float]]) -> tuple[int, list[int], list[int]]:
    candidates = sorted(
        (
            (_iou(prediction["box"], target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(truth)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    for overlap, prediction_index, target_index in candidates:
        if overlap < IOU_THRESHOLD:
            break
        if prediction_index not in used_predictions and target_index not in used_targets:
            used_predictions.add(prediction_index)
            used_targets.add(target_index)
    return len(used_predictions), sorted(used_predictions), sorted(used_targets)


def main() -> None:
    if not WEIGHTS.is_file():
        raise FileNotFoundError("Trained weights are missing. Run scripts/train.py first.")
    if not DATA_YAML.is_file():
        raise FileNotFoundError("Prepared dataset is missing. Run scripts/prepare_dataset.py first.")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    model = YOLO(str(WEIGHTS))
    validation = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project=str(ROOT / "runs"),
        name="pothole-test-evaluation",
        exist_ok=True,
        verbose=True,
    )
    box_metrics = validation.box
    precision = float(box_metrics.mp)
    recall = float(box_metrics.mr)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    test_images = sorted((PROCESSED / "images" / "test").glob("*"))
    failure_dir = OUTPUT / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    true_positives = false_positives = false_negatives = 0
    failure_records = []
    for image_path in test_images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        label_path = PROCESSED / "labels" / "test" / f"{image_path.stem}.txt"
        truth = _ground_truth(label_path, image.shape[:2])
        result = model.predict(
            source=image,
            conf=CONFIDENCE,
            imgsz=320,
            device="cpu",
            verbose=False,
        )[0]
        predictions = []
        if result.boxes is not None:
            for row in result.boxes.data.cpu().numpy():
                predictions.append({"box": row[:4].astype(float).tolist(), "confidence": float(row[4])})
        matched_count, matched_predictions, matched_truth = _match(predictions, truth)
        true_positives += matched_count
        false_positives += len(predictions) - matched_count
        false_negatives += len(truth) - matched_count
        missed = [box for index, box in enumerate(truth) if index not in matched_truth]
        extra = [box for index, box in enumerate(predictions) if index not in matched_predictions]
        if (missed or extra) and len(failure_records) < 12:
            failure_records.append({
                "image": str(image_path.relative_to(ROOT)),
                "missed_ground_truth_boxes": len(missed),
                "unmatched_predictions": len(extra),
                "missed_boxes_xyxy": missed,
                "unmatched_prediction_boxes": extra,
            })
        if (missed or extra) and len([path for path in failure_dir.glob("*.jpg")]) < 8:
            annotated = image.copy()
            for x1, y1, x2, y2 in missed:
                cv2.rectangle(annotated, (round(x1), round(y1)), (round(x2), round(y2)), (40, 50, 230), 2)
            for prediction in extra:
                x1, y1, x2, y2 = prediction["box"]
                cv2.rectangle(annotated, (round(x1), round(y1)), (round(x2), round(y2)), (0, 170, 240), 2)
            cv2.imwrite(str(failure_dir / f"failure_{len(list(failure_dir.glob('*.jpg'))) + 1:02d}.jpg"), annotated)

    run_directory = Path(validation.save_dir)
    results = {
        "evaluation_split": "test",
        "test_images": len(test_images),
        "confidence_threshold_for_failure_analysis": CONFIDENCE,
        "iou_threshold_for_failure_analysis": IOU_THRESHOLD,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "map50": float(box_metrics.map50),
        "map50_95": float(box_metrics.map),
        "failure_analysis_counts_at_confidence_threshold": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "failure_examples": failure_records,
        "failure_images_directory": "artifacts/failures",
        "confusion_matrix": str((run_directory / "confusion_matrix.png").relative_to(ROOT)),
        "training_graphs": "runs/pothole-yolo-cpu/results.png",
        "evaluation_run": str(run_directory.relative_to(ROOT)),
        "dataset_source": metadata["source"],
        "dataset_license": metadata["license"],
    }
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()