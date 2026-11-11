# Pothole Detection System using YOLO

A small university ANN course project. It fine-tunes a lightweight YOLO11n object detector on a public pothole dataset and runs a local Streamlit AI application for images, uploaded video, public video URLs (including YouTube where available), and webcam input.

## Requirements

- Windows 10 or later
- Python 3.10–3.12
- Internet access for the first dependency, dataset, and pretrained-weight downloads
- CPU works; a CUDA-capable NVIDIA GPU is optional and is not required by the supplied scripts

## Setup

Run these commands from the project folder in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

## Dataset and training

The source is [Cracks and Potholes in Road Images, Mendeley Data, version 4](https://data.mendeley.com/datasets/t576ydh9v8/4), licensed CC BY 4.0. It contains 2,235 road images with separate binary road, crack, and pothole masks. The project uses only the pothole masks. It checks the source archive against the published SHA-256 before extraction and does not collect or manually annotate images.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
.\.venv\Scripts\python.exe scripts\train.py
.\.venv\Scripts\python.exe scripts\evaluate.py
```

The preparation step converts connected regions in the source masks to normalized YOLO boxes, then makes a reproducible 70/15/15 train/validation/test image split (seed 42). Training fine-tunes `yolo11n.pt` on CPU at 320 pixels, batch size 4, for 5 epochs by default. Change the training budget with `--epochs`, `--imgsz`, and `--batch` if your hardware permits.

`artifacts/best.pt` is the fine-tuned detector. `artifacts/results.json` contains the real held-out test metrics, confusion-matrix path, and thresholded false-positive/false-negative analysis. Ultralytics writes training curves and evaluation plots under `runs/`. The source archive and prepared images remain in `data/`.

## Run the website

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. The app has six pages: Home, AI Pothole Detection, Detection Results, Dataset & Model, Evaluation, and About. It loads the existing `artifacts/best.pt`; starting the website does not retrain the model.

Image detection accepts JPG, JPEG, and PNG. Video upload accepts MP4, AVI, and MOV, and public HTTP(S) video URLs are downloaded with yt-dlp. YouTube downloads use a video-only format when the provider does not offer a combined stream. Private, restricted, unavailable, unsupported, or local-network URLs show an error instead of fabricated output. Local video input is limited to 150 MB and 900 frames. Annotated videos are encoded as H.264; the original-content preview is also transcoded for browser playback. Video totals distinguish frame detections from ByteTrack IDs, and missing tracking IDs are disclosed. The webcam option needs browser camera permission and should be opened through localhost.

## Project layout

```text
app.py
src/
  detector.py
  project_data.py
  video_pipeline.py
  video_sources.py
scripts/
  prepare_dataset.py
  train.py
  evaluate.py
data/
  raw/
  processed/
artifacts/
requirements.txt
README.md
```

## Academic citation

Passos, B. T., Cassaniga, M. J., Fernandes, A. M. R., Medeiros, K. B., & Comunello, E. (2020). *Cracks and Potholes in Road Images*, Mendeley Data, V4. https://doi.org/10.17632/t576ydh9v8.4

YOLO11 is provided by [Ultralytics](https://docs.ultralytics.com/models/yolo11/). Review the dataset and software licenses and cite the source dataset in any report or presentation.