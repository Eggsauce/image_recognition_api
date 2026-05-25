# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the API server (from src/)
cd src && uv run python main.py
# or with uvicorn directly:
cd src && uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Train the YOLO model (from project root)
uv run python src/training.py
```

## Environment Variables

The API requires these environment variables (use a `.env` file in the project root):

- `WEIGHTS_PATH` — path to the `.pt` YOLO weights file (required)
- `CLASSNAME_PATH` — path to an `obj.names` file (optional; model-embedded names are preferred)
- `YOLO_API_LOGLEVEL` — logging level, defaults to `INFO`

## Architecture

**FastAPI + YOLOv11 object detection service** that accepts base64-encoded images and returns bounding box detections.

- `src/main.py` — FastAPI app entry point; loads `.env`, mounts the detection router
- `src/routers/detection.py` — single POST endpoint `/detection/image-data`; caches `YoloDetector` instances by `(weights_path, classname_path, device)` using a thread-safe dict; runs inference in `asyncio.to_thread` to stay non-blocking
- `src/detector.py` — `YoloDetector` wraps Ultralytics YOLO; outputs both pixel (`bbox_pixel`) and normalized (`bbox_normalized`) coordinates per detection
- `src/models/api.py` — Pydantic request model `DetectionImageData`
- `config.yaml` — YOLO dataset config pointing at `cvat/` training images; defines 7 UI element classes (login fields, buttons, security elements)
- `src/training.py` — standalone script to train a new `yolo11n` model using `config.yaml`

### Detection endpoint behavior

`POST /detection/image-data` applies these filters in order after inference:
1. Class filter via `target_class_name` query param or `class_name_path` body field
2. Minimum confidence filter (`min_score`; defaults to 0.25 when a class filter is active)
3. Deduplication: keeps only the highest-confidence detection per class
4. If `return_top=true`, returns only the single highest-confidence detection overall

Set `debug=true` in the request body to receive raw model internals in the response.
