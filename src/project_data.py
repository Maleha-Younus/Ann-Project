"""Download, verify, and convert the public pothole-mask dataset."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "Cracks-and-Potholes-in-Road-Images.zip"
EXTRACTED = ROOT / "data" / "raw" / "extracted"
PROCESSED = ROOT / "data" / "processed"
DATA_YAML = PROCESSED / "potholes.yaml"
METADATA = PROCESSED / "dataset_metadata.json"
DOWNLOAD_URL = "https://data.mendeley.com/public-files/datasets/t576ydh9v8/files/afc7c028-06e0-475b-b190-e008df681b19/file_downloaded"
EXPECTED_SHA256 = "e59bb8d771e18f059ab78d88e640e83869fc8c8fd808697a0c2c48e16eec4f76"
SOURCE_URL = "https://data.mendeley.com/datasets/t576ydh9v8/4"
LICENSE = "CC BY 4.0"
INNER_ARCHIVE_NAME = "Cracks-and-Potholes-in-Road-Images.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive() -> Path:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        urllib.request.urlretrieve(DOWNLOAD_URL, ARCHIVE)
    actual = sha256_file(ARCHIVE)
    if actual == EXPECTED_SHA256:
        return ARCHIVE

    try:
        with zipfile.ZipFile(ARCHIVE) as wrapper:
            inner_member = next(
                (member for member in wrapper.namelist() if Path(member).name == INNER_ARCHIVE_NAME),
                None,
            )
            if inner_member is None:
                raise ValueError("The downloaded file is neither the verified dataset nor a known Mendeley wrapper")
            verified_archive = ARCHIVE.with_name("verified-source-dataset.zip")
            with wrapper.open(inner_member) as source, verified_archive.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    except zipfile.BadZipFile as error:
        raise ValueError(f"Dataset checksum mismatch: expected {EXPECTED_SHA256}, got {actual}") from error

    inner_hash = sha256_file(verified_archive)
    if inner_hash != EXPECTED_SHA256:
        raise ValueError(f"Dataset checksum mismatch: expected {EXPECTED_SHA256}, got {inner_hash}")
    return verified_archive


def extract_archive(archive_path: Path) -> Path:
    if EXTRACTED.exists() and any(EXTRACTED.rglob("*")):
        return EXTRACTED
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    root = EXTRACTED.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (EXTRACTED / member.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("The dataset archive contains an unsafe path")
        archive.extractall(EXTRACTED)
    return EXTRACTED


def _find_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    image_paths = [
        path for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    masks_by_key: dict[tuple[Path, str], Path] = {}
    for path in image_paths:
        if path.stem.upper().endswith("_POTHOLE"):
            base_name = path.stem[:-len("_POTHOLE")]
            masks_by_key[(path.parent, base_name.casefold())] = path

    pairs: list[tuple[Path, Path]] = []
    for image in image_paths:
        if not image.stem.upper().endswith("_RAW"):
            continue
        base_name = image.stem[:-len("_RAW")]
        mask = masks_by_key.get((image.parent, base_name.casefold()))
        if mask is not None:
            pairs.append((image, mask))
    return pairs


def _mask_boxes(mask_path: Path, image_size: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read pothole mask: {mask_path}")
    width, height = image_size
    if mask.shape[1] != width or mask.shape[0] != height:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for component in range(1, component_count):
        x, y, box_width, box_height, area = stats[component]
        if area < 8 or box_width < 2 or box_height < 2:
            continue
        center_x = (x + box_width / 2) / width
        center_y = (y + box_height / 2) / height
        boxes.append((center_x, center_y, box_width / width, box_height / height))
    return boxes


def prepare_dataset(seed: int = 42) -> dict:
    archive_path = download_archive()
    extracted_root = extract_archive(archive_path)
    pairs = _find_pairs(extracted_root)
    if len(pairs) < 10:
        raise RuntimeError(f"Expected paired RAW/POTHOLE images, found only {len(pairs)}")

    random.Random(seed).shuffle(pairs)
    train_end = int(len(pairs) * 0.70)
    valid_end = int(len(pairs) * 0.85)
    split_pairs = {
        "train": pairs[:train_end],
        "valid": pairs[train_end:valid_end],
        "test": pairs[valid_end:],
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, dict[str, int]] = {}
    samples: list[str] = []
    for split_name, split_items in split_pairs.items():
        image_dir = PROCESSED / "images" / split_name
        label_dir = PROCESSED / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        positive_images = 0
        object_count = 0
        for index, (image_path, mask_path) in enumerate(split_items):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            boxes = _mask_boxes(mask_path, (width, height))
            output_name = f"{split_name}_{index:05d}{image_path.suffix.lower()}"
            output_image = image_dir / output_name
            cv2.imwrite(str(output_image), image)
            label_path = label_dir / f"{Path(output_name).stem}.txt"
            label_path.write_text(
                "".join("0 " + " ".join(f"{value:.6f}" for value in box) + "\n" for box in boxes),
                encoding="utf-8",
            )
            positive_images += int(bool(boxes))
            object_count += len(boxes)
            if len(samples) < 12 and split_name == "train":
                samples.append(str(output_image.relative_to(ROOT)))
        split_counts[split_name] = {
            "images": len(list(image_dir.iterdir())),
            "images_with_potholes": positive_images,
            "pothole_instances": object_count,
        }

    DATA_YAML.write_text(
        "path: " + str(PROCESSED.resolve()).replace("\\", "/") + "\n"
        "train: images/train\nval: images/valid\ntest: images/test\n"
        "names:\n  0: pothole\n",
        encoding="utf-8",
    )
    metadata = {
        "source": SOURCE_URL,
        "license": LICENSE,
        "download_url": DOWNLOAD_URL,
        "sha256": sha256_file(archive_path),
        "annotation_source": "Original binary POTHOLE masks; bounding boxes derived automatically from connected mask regions.",
        "source_image_mask_pairs": len(pairs),
        "split_method": "Deterministic shuffled image-level split (seed 42): 70% train, 15% validation, 15% held-out test.",
        "seed": seed,
        "classes": ["pothole"],
        "splits": split_counts,
        "sample_images": samples,
        "preprocessing": [
            "Read original RAW road images and corresponding POTHOLE masks.",
            "Convert each connected pothole-mask region to a normalized YOLO bounding box.",
            "Resize/letterbox is performed by Ultralytics during training and inference.",
        ],
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_metadata() -> dict | None:
    if not METADATA.exists():
        return None
    return json.loads(METADATA.read_text(encoding="utf-8"))


def split_summary(metadata: dict) -> Counter:
    return Counter({name: values["images"] for name, values in metadata["splits"].items()})