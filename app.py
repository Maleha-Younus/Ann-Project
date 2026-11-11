"""Streamlit presentation and local pothole detection application."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.detector import ROOT, annotate_frame, load_model
from src.project_data import METADATA, PROCESSED, SOURCE_URL, load_metadata
from src.video_pipeline import MAX_VIDEO_FRAMES, process_video
from src.video_sources import MAX_VIDEO_BYTES, VideoURLDownloadError, download_public_video

st.set_page_config(page_title="Pothole Detection | ANN Course Project", page_icon="P", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink: #172923; --muted: #64736d; --green: #16835d; --lime: #d7ed76; --paper: #f5f7f3; --line: #dfe6df; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 2.2rem; font-weight: 800; }
    section[data-testid="stSidebar"] { background: #eaf0e9; border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label { font-weight: 600; }
    div[data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    .eyebrow { color: var(--green); font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }
    .lead { font-size: 1.08rem; line-height: 1.65; color: #43534c; max-width: 850px; }
    .note { border-left: 3px solid var(--green); padding: 10px 14px; background: #edf4ed; border-radius: 0 5px 5px 0; }
    .section-rule { border: 0; border-top: 1px solid var(--line); margin: 1.25rem 0; }
    div.stButton > button { border-radius: 5px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained YOLO model...")
def get_model():
    return load_model()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def page_header(section: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{section}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="lead">{description}</p>', unsafe_allow_html=True)
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)


def _go_to_detection() -> None:
    st.session_state["project_page"] = "AI Pothole Detection"


def render_home() -> None:
    page_header("ANN COURSE PROJECT", "Pothole Detection System", "Test the locally fine-tuned YOLO11n model on a road image, video file, public video URL, or webcam stream.")
    st.button("Start Detection", type="primary", on_click=_go_to_detection)
    first, second = st.columns([1.45, 1], gap="large")
    with first:
        st.subheader("The problem")
        st.write("Road inspections are often visual and time-consuming. Potholes can be missed between inspections, while rough road surfaces and changing light make automatic detection difficult.")
        st.subheader("Project objectives")
        st.markdown("- Detect pothole locations and confidence scores in user-provided road media.\n- Fine-tune a lightweight YOLO detector on a public, licensed dataset and evaluate it on a separate test split.\n- Demonstrate image, video, public URL, and live webcam inference locally.")
    with second:
        st.subheader("Problem definition")
        st.markdown('<div class="note"><b>Input</b>: a road image or video stream.<br><b>Output</b>: pothole bounding boxes, per-box confidence, and a per-frame count.<br><b>Intended user</b>: a student or road-inspection demonstrator reviewing camera footage.</div>', unsafe_allow_html=True)
        st.subheader("Workflow")
        st.write("Public pothole masks → automatic box conversion → YOLO fine-tuning → held-out test evaluation → local detection demo.")
    metadata = load_metadata()
    results = read_json(ROOT / "artifacts" / "results.json")
    cols = st.columns(3)
    cols[0].metric("Dataset images", f"{sum(s['images'] for s in metadata['splits'].values()):,}" if metadata else "Not prepared")
    cols[1].metric("Detection class", "Pothole")
    cols[2].metric("Test mAP50", f"{results['map50']:.3f}" if results else "Not evaluated")


def render_dataset() -> None:
    page_header("SECTION 04", "Dataset & Model", "The model uses only the existing public road-defect dataset and the fine-tuned weights saved in this project.")
    metadata = load_metadata()
    st.markdown("**Source:** [Cracks and Potholes in Road Images, Mendeley Data](" + SOURCE_URL + ")  \n**License:** CC BY 4.0  \n**Classes:** one object-detection class, `pothole`. The source also supplies road and crack masks, which this project does not use.")
    if not metadata:
        st.info("Dataset preparation has not completed yet. Run `python scripts/prepare_dataset.py` to download, verify, and prepare the public dataset.")
        return
    st.caption(f"Archive SHA-256 verified: `{metadata['sha256']}`. Original label source: {metadata['annotation_source']}")
    columns = st.columns(3)
    for column, split in zip(columns, ("train", "valid", "test")):
        values = metadata["splits"][split]
        column.metric(f"{split.title()} images", values["images"])
        column.caption(f"{values['images_with_potholes']} with potholes · {values['pothole_instances']} derived boxes")
    st.subheader("Preprocessing")
    st.markdown("1. Pair each original `RAW` road image with its corresponding binary `POTHOLE` mask.\n2. Convert connected mask regions into normalized YOLO boxes; no new labels are drawn by hand.\n3. Shuffle with seed 42 into 70% train, 15% validation, and 15% held-out test.\n4. Ultralytics resizes and letterboxes images during training and inference.")
    samples = metadata.get("sample_images", [])[:6]
    if samples:
        st.subheader("Training image samples")
        sample_columns = st.columns(min(3, len(samples)))
        for index, relative_path in enumerate(samples):
            path = ROOT / relative_path
            if path.exists():
                sample_columns[index % len(sample_columns)].image(str(path), caption=Path(relative_path).name, use_container_width=True)

    st.subheader("YOLO model and training")
    details = read_json(ROOT / "artifacts" / "training_details.json")
    weights_path = ROOT / "artifacts" / "best.pt"
    st.write("YOLO11n is a one-stage detector: a feature-extracting backbone feeds a multi-scale feature neck and a detection head. The COCO-initialized weights were fine-tuned for the single `pothole` class.")
    if details:
        columns = st.columns(4)
        columns[0].metric("Training device", details["device"])
        columns[1].metric("Epochs completed", details.get("epochs_completed", details["epochs_requested"]))
        columns[2].metric("Image size", f"{details['image_size']} px")
        columns[3].metric("Batch size", details["batch_size"])
        st.caption(f"Weights: `{details['weights']}` · {details['training_images']} training / {details['validation_images']} validation images. Saved weights size: {weights_path.stat().st_size / (1024 * 1024):.1f} MB." if weights_path.exists() else f"Weights: `{details['weights']}`")
    else:
        st.warning("Training details are not available. No model metrics are fabricated.")


def render_results() -> None:
    page_header("SECTION 05", "Evaluation", "Every metric and plot below comes from the actual held-out test evaluation or saved training run.")
    details = read_json(ROOT / "artifacts" / "training_details.json")
    results = read_json(ROOT / "artifacts" / "results.json")
    if not details or not results:
        st.warning("Training and held-out evaluation artifacts are not available yet. No metrics are shown until the real training and test evaluation finish.")
        st.code("python scripts/train.py\npython scripts/evaluate.py", language="powershell")
        return
    st.markdown("**Architecture:** YOLO11n, a one-stage detector with a backbone for feature extraction, a feature pyramid/neck, and a detection head. The pretrained COCO weights are fine-tuned for the single `pothole` class.")
    st.markdown(f"**Run:** {details['device']} · {details.get('epochs_completed', details['epochs_requested'])} completed epochs · {details['image_size']} px · batch {details['batch_size']} · {details['training_images']} train / {details['validation_images']} validation / {results['test_images']} held-out test images.")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Precision", f"{results['precision']:.3f}")
    metric_columns[1].metric("Recall", f"{results['recall']:.3f}")
    metric_columns[2].metric("F1-score", f"{results['f1_score']:.3f}")
    metric_columns[3].metric("mAP50", f"{results['map50']:.3f}")
    metric_columns[4].metric("mAP50–95", f"{results['map50_95']:.3f}")
    st.caption("Precision, recall, and mAP are Ultralytics box metrics computed on the held-out test split. F1 is calculated from the reported precision and recall. Values are fractions, not percentages.")

    run_dir = ROOT / "runs" / "pothole-yolo-cpu"
    graph_path = run_dir / "results.png"
    csv_path = run_dir / "results.csv"
    if graph_path.exists():
        st.subheader("Training and validation curves")
        st.image(str(graph_path), use_container_width=True)
    if csv_path.exists():
        curves = pd.read_csv(csv_path)
        curves.columns = [column.strip() for column in curves.columns]
        metric_columns = [column for column in curves if column in {"train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss"}]
        if metric_columns:
            st.plotly_chart(px.line(curves, x="epoch", y=metric_columns, title="Loss by epoch", labels={"value": "Loss", "variable": "Metric"}), use_container_width=True)
    confusion_path = ROOT / results.get("confusion_matrix", "")
    if confusion_path.exists():
        st.subheader("Held-out test confusion matrix")
        st.image(str(confusion_path), use_container_width=True)

    st.subheader("Failure analysis")
    counts = results["failure_analysis_counts_at_confidence_threshold"]
    st.write(f"At confidence ≥ {results['confidence_threshold_for_failure_analysis']:.2f} and IoU ≥ {results['iou_threshold_for_failure_analysis']:.1f}, the held-out predictions yielded {counts['true_positives']} matched boxes, {counts['false_positives']} unmatched predictions, and {counts['false_negatives']} missed ground-truth boxes.")
    failures = results.get("failure_examples", [])
    if failures:
        st.caption("Examples below are actual held-out test images with misses (red boxes) or unmatched predictions (amber boxes).")
        failure_files = sorted((ROOT / "artifacts" / "failures").glob("*.jpg"))
        if failure_files:
            st.image([str(path) for path in failure_files[:4]], caption=[path.name for path in failure_files[:4]], use_container_width=True)
        st.dataframe(pd.DataFrame([{key: row[key] for key in ("image", "missed_ground_truth_boxes", "unmatched_predictions")} for row in failures]), hide_index=True, use_container_width=True)
    else:
        st.info("No false-positive or false-negative cases were observed at the stated threshold on this test split.")


def _detection_table(result) -> pd.DataFrame:
    rows = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = [round(float(value), 1) for value in box.xyxy[0].cpu().tolist()]
            rows.append({
                "class": "pothole",
                "confidence": round(float(box.conf[0].cpu()), 3),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })
    return pd.DataFrame(rows, columns=["class", "confidence", "x1", "y1", "x2", "y2"])


def _process_video_path(input_path: Path, source_name: str, model, confidence: float) -> dict:
    output_path = input_path.with_name("annotated.mp4")
    progress = st.progress(0.0, text="Preparing video...")

    def update_progress(current: int, total: int) -> None:
        fraction = min(current / total, 1.0) if total > 0 else 0.0
        progress.progress(fraction, text=f"Processing frame {current}" + (f" of {total}" if total else ""))

    statistics = process_video(input_path, output_path, model, confidence, update_progress)
    progress.progress(1.0, text=f"Finished {statistics['frames_processed']} frames")
    original_preview = Path(statistics.pop("original_preview_path"))
    return {
        "media_type": "video",
        "source_name": source_name,
        "original_bytes": original_preview.read_bytes(),
        "annotated_bytes": output_path.read_bytes(),
        "statistics": statistics,
    }


def _render_detection_output(detection: dict, key_prefix: str) -> None:
    statistics = detection["statistics"]
    media_type = detection["media_type"]
    st.subheader("Detection results")
    st.caption(f"Source: {detection['source_name']} · Fine-tuned model: `artifacts/best.pt`")
    columns = st.columns(4)
    if media_type == "image":
        columns[0].metric("Potholes detected", statistics["total_frame_detections"])
        columns[1].metric("Frames processed", statistics["frames_processed"])
    else:
        columns[0].metric("Frame detections", statistics["total_frame_detections"])
        columns[1].metric("Frames processed", statistics["frames_processed"])
    confidence = statistics["average_confidence"]
    columns[2].metric("Average confidence", f"{confidence:.3f}" if confidence is not None else "N/A")
    columns[3].metric("Processing time", f"{statistics['processing_time_seconds']:.2f} s")

    if media_type == "video":
        unique_tracks = statistics["unique_tracked_potholes"]
        if unique_tracks is None:
            st.metric("Unique tracked potholes (lower bound)", statistics["tracked_pothole_ids"])
            st.caption(f"{statistics['untracked_detections']} frame detections did not receive a tracking ID. Frame detections are not unique objects.")
        else:
            st.metric("Unique tracked potholes", unique_tracks)
        left, right = st.columns(2)
        left.video(detection["original_bytes"])
        left.caption("Original content (browser-compatible preview)")
        right.video(detection["annotated_bytes"])
        right.caption("YOLO annotations: pothole boxes, confidence, frame count, and track IDs")
        st.download_button(
            "Download annotated video",
            detection["annotated_bytes"],
            file_name="pothole-detections.mp4",
            mime="video/mp4",
            key=f"download-video-{key_prefix}",
        )
    else:
        left, right = st.columns(2)
        left.image(detection["original_bytes"], caption="Original image", use_container_width=True)
        right.image(detection["annotated_bytes"], caption="Fine-tuned YOLO output", use_container_width=True)
        st.dataframe(pd.DataFrame(detection["detections"], columns=["class", "confidence", "x1", "y1", "x2", "y2"]), hide_index=True, use_container_width=True)
        st.download_button(
            "Download annotated image",
            detection["annotated_bytes"],
            file_name="pothole-detections.jpg",
            mime="image/jpeg",
            key=f"download-image-{key_prefix}",
        )

    if statistics["total_frame_detections"] == 0:
        st.info("No potholes were detected at the selected confidence threshold.")


def _image_detection(uploaded, model, confidence: float) -> dict:
    original_bytes = uploaded.getvalue()
    frame = cv2.imdecode(np.frombuffer(original_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("This image could not be decoded. Please try a JPG or PNG file.")
    started_at = time.perf_counter()
    result = model.predict(source=frame, conf=confidence, imgsz=320, device="cpu", verbose=False)[0]
    elapsed = time.perf_counter() - started_at
    annotated = result.plot()
    encoded, buffer = cv2.imencode(".jpg", annotated)
    if not encoded:
        raise ValueError("OpenCV could not encode the annotated image.")
    count = len(result.boxes) if result.boxes is not None else 0
    confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None and count else []
    rows = _detection_table(result).to_dict("records")
    return {
        "media_type": "image",
        "source_name": uploaded.name,
        "original_bytes": original_bytes,
        "annotated_bytes": buffer.tobytes(),
        "detections": rows,
        "statistics": {
            "frames_processed": 1,
            "total_frame_detections": count,
            "average_confidence": float(np.mean(confidences)) if confidences else None,
            "processing_time_seconds": elapsed,
        },
    }


def render_detection() -> None:
    page_header("SECTION 02", "AI Pothole Detection", "Upload a road image or video, provide a public video URL, or start a live webcam. All detections use the saved fine-tuned YOLO11n weights.")
    try:
        model = get_model()
    except Exception as error:
        st.error(f"The fine-tuned detector could not be loaded: {error}")
        return

    mode = st.radio("Input mode", ["Upload Image", "Upload Video", "Video URL", "Live Webcam"], horizontal=True)
    confidence = st.slider("Minimum confidence", min_value=0.05, max_value=0.90, value=0.25, step=0.05)

    if mode == "Upload Image":
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="detection-image", max_upload_size=20)
        if uploaded and uploaded.size > 20 * 1024 * 1024:
            st.error("Please use an image smaller than 20 MB.")
        elif uploaded and st.button("Detect potholes", type="primary", key="detect-image"):
            st.session_state.pop("detection_result", None)
            try:
                with st.spinner("Running the fine-tuned YOLO model..."):
                    st.session_state["detection_result"] = _image_detection(uploaded, model, confidence)
            except Exception as error:
                st.session_state.pop("detection_result", None)
                st.error(f"Image detection failed: {error}")
        detection = st.session_state.get("detection_result")
        if detection and detection["media_type"] == "image":
            _render_detection_output(detection, "image")

    elif mode == "Upload Video":
        uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"], key="detection-video", max_upload_size=150)
        st.caption(f"Local demo limit: {MAX_VIDEO_BYTES // (1024 * 1024)} MB and {MAX_VIDEO_FRAMES} frames per clip. Processing can take time on a CPU.")
        if uploaded and uploaded.size > MAX_VIDEO_BYTES:
            st.error("This video exceeds the 150 MB local-demo limit.")
        elif uploaded and st.button("Process video", type="primary", key="process-video"):
            st.session_state.pop("detection_result", None)
            original_bytes = uploaded.getvalue()
            try:
                with tempfile.TemporaryDirectory(prefix="pothole-video-") as temporary_directory:
                    input_path = Path(temporary_directory) / f"source{Path(uploaded.name).suffix.lower()}"
                    input_path.write_bytes(original_bytes)
                    with st.spinner("Tracking potholes and encoding the annotated video..."):
                        detection = _process_video_path(input_path, uploaded.name, model, confidence)
                st.session_state["detection_result"] = detection
            except Exception as error:
                st.session_state.pop("detection_result", None)
                st.error(f"Video processing failed: {error}")
        detection = st.session_state.get("detection_result")
        if detection and detection["media_type"] == "video":
            _render_detection_output(detection, "upload")

    elif mode == "Video URL":
        url = st.text_input("Public video URL", placeholder="https://www.youtube.com/watch?v=... or a direct MP4 URL")
        st.caption("Public YouTube URLs and direct supported video links are handled with yt-dlp. Private, restricted, unavailable, local-network, and unsupported URLs produce an error instead of a result.")
        if st.button("Download and process URL", type="primary", key="process-video-url"):
            st.session_state.pop("detection_result", None)
            try:
                with tempfile.TemporaryDirectory(prefix="pothole-url-") as temporary_directory:
                    with st.status("Checking and downloading the public video...", expanded=True) as status:
                        input_path = download_public_video(url, Path(temporary_directory))
                        if input_path.stat().st_size > MAX_VIDEO_BYTES:
                            raise VideoURLDownloadError("The downloaded video exceeds the 150 MB local-demo limit.")
                        status.write("Download complete. Processing with the fine-tuned YOLO model.")
                        detection = _process_video_path(input_path, url, model, confidence)
                        status.update(label="Video processed", state="complete", expanded=False)
                st.session_state["detection_result"] = detection
            except VideoURLDownloadError as error:
                st.session_state.pop("detection_result", None)
                st.error(str(error))
            except Exception as error:
                st.session_state.pop("detection_result", None)
                st.error(f"URL video processing failed: {error}")
        detection = st.session_state.get("detection_result")
        if detection and detection["media_type"] == "video":
            _render_detection_output(detection, "url")

    else:
        try:
            from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

            class LiveProcessor(VideoProcessorBase):
                def recv(self, frame):
                    image = frame.to_ndarray(format="bgr24")
                    annotated, _ = annotate_frame(model, image, confidence)
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            st.markdown('<div class="note">Allow camera access to see confidence-labeled boxes and the current-frame pothole count.</div>', unsafe_allow_html=True)
            context = webrtc_streamer(
                key="pothole-webcam",
                video_processor_factory=LiveProcessor,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={"iceServers": []},
            )
            if context.state.playing:
                st.caption("Live inference is running on the camera frames received by the browser.")
        except Exception as error:
            st.error(f"Webcam streaming could not start: {error}")
            st.caption("Use localhost and allow camera access in your browser.")


def render_detection_results() -> None:
    page_header("SECTION 03", "Detection Results", "The latest output is generated by the fine-tuned model during this session; no sample or placeholder detections are shown.")
    detection = st.session_state.get("detection_result")
    if not detection:
        st.info("No detection has been run in this session. Open AI Pothole Detection, provide an input, and select Process/Detect.")
        return
    _render_detection_output(detection, "results")


def render_about() -> None:
    page_header("SECTION 06", "About", "A focused university ANN project demonstrating transfer learning, held-out evaluation, and local inference.")
    st.subheader("Methodology")
    st.write("A public road-defect dataset provides binary pothole masks. Connected mask regions are converted automatically to bounding boxes. YOLO11n is fine-tuned on the training split, monitored on the validation split, and evaluated once on the separate test split. The saved fine-tuned weights power image, video, public URL, and webcam inference.")
    st.subheader("Limitations")
    st.markdown("- The source images are from Brazilian highways (2014–2017); other regions, cameras, weather, and road types may look different.\n- A bounding box is derived from each connected mask region and does not preserve the exact pothole shape.\n- Counts in video are per-frame detections, not unique tracked potholes.\n- CPU-only inference is slower than a dedicated GPU, especially for webcam streams.\n- The model is a course demonstration, not a road-safety or maintenance decision system.")
    st.subheader("Conclusion")
    st.write("This project connects a public annotated dataset to a fine-tuned YOLO detector and a local interactive demonstration. The held-out metrics and failure examples should be read alongside the dataset size and domain limitations; further data from varied road conditions would be needed before operational use.")


with st.sidebar:
    st.markdown("### ROAD VISION")
    st.caption("ANN course project")
    selected = st.radio("Project pages", ["Home", "AI Pothole Detection", "Detection Results", "Dataset & Model", "Evaluation", "About"], key="project_page", label_visibility="collapsed")
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.caption("YOLO11n · OpenCV · Streamlit")

if selected == "Home":
    render_home()
elif selected == "AI Pothole Detection":
    render_detection()
elif selected == "Detection Results":
    render_detection_results()
elif selected == "Dataset & Model":
    render_dataset()
elif selected == "Evaluation":
    render_results()
else:
    render_about()