"""Fine-tune a small pretrained YOLO model on CPU-friendly settings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from src.project_data import DATA_YAML, METADATA, ROOT, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if not DATA_YAML.exists():
        prepare_dataset()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata["splits"]["test"]["images"] == 0:
        raise RuntimeError("The prepared held-out test split is empty")

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        project=str(ROOT / "runs"),
        name="pothole-yolo-cpu",
        exist_ok=True,
        patience=5,
        seed=42,
        deterministic=True,
        cache=False,
        plots=True,
        verbose=True,
    )
    run_dir = ROOT / "runs" / "pothole-yolo-cpu"
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not produce best weights at {best_weights}")
    results_csv = run_dir / "results.csv"
    with results_csv.open("r", encoding="utf-8", newline="") as results_file:
        epochs_completed = sum(1 for _ in csv.DictReader(results_file))
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    shutil.copy2(best_weights, artifacts / "best.pt")
    run_metrics = {
        "weights": "artifacts/best.pt",
        "training_run": str(run_dir.relative_to(ROOT)),
        "pretrained_model": "YOLO11n (yolo11n.pt)",
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_completed,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "device": "CPU",
        "training_images": metadata["splits"]["train"]["images"],
        "validation_images": metadata["splits"]["valid"]["images"],
        "test_images": metadata["splits"]["test"]["images"],
    }
    (artifacts / "training_details.json").write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    print(json.dumps(run_metrics, indent=2))


if __name__ == "__main__":
    main()