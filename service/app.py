"""FastAPI inference service for the exported EcoPlay ONNX classifier."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


MODEL_DIR = Path(os.getenv("ECOPLAY_MODEL_DIR", "models"))
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def load_classifier(model_dir: Path = MODEL_DIR) -> tuple[ort.InferenceSession, dict]:
    model_path = model_dir / "mobilenetv2.onnx"
    metadata_path = model_dir / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Model artifacts not found. Run scripts/train_classifier.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    return session, metadata


def preprocess(image_bytes: bytes, metadata: dict) -> np.ndarray:
    size = int(metadata["image_size"])
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
        width, height = image.size
        scale = 256 / min(width, height)
        resized = image.resize(
            (round(width * scale), round(height * scale)), Image.Resampling.BILINEAR
        )
        left = (resized.width - size) // 2
        top = (resized.height - size) // 2
        image = resized.crop((left, top, left + size, top + size))

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(metadata["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(metadata["normalization"]["std"], dtype=np.float32)
    pixels = (pixels - mean) / std
    return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32)


def create_app() -> FastAPI:
    app = FastAPI(
        title="EcoPlay Image Classification API",
        version="1.0.0",
        description="Classifies recyclable material images with MobileNetV2.",
    )
    try:
        app.state.session, app.state.metadata = load_classifier()
        app.state.load_error = None
    except FileNotFoundError as error:
        app.state.session = None
        app.state.metadata = None
        app.state.load_error = str(error)

    @app.get("/health")
    def health() -> dict[str, str]:
        if app.state.session is None:
            raise HTTPException(status_code=503, detail=app.state.load_error)
        return {"status": "ok", "model": "MobileNetV2 ONNX"}

    @app.post("/classificar")
    async def classify(image: UploadFile = File(...)) -> dict[str, object]:
        if app.state.session is None:
            raise HTTPException(status_code=503, detail=app.state.load_error)
        content = await image.read(MAX_IMAGE_BYTES + 1)
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 10 MiB limit.")
        if not content:
            raise HTTPException(status_code=400, detail="Image file is empty.")
        try:
            tensor = preprocess(content, app.state.metadata)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail="Invalid image file.") from error

        input_name = app.state.metadata["input_name"]
        output_name = app.state.metadata["output_name"]
        logits = app.state.session.run([output_name], {input_name: tensor})[0][0]
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        class_names = app.state.metadata["class_names"]
        ranked_indices = np.argsort(probabilities)[::-1]
        best_index = int(ranked_indices[0])
        return {
            "classe": class_names[best_index],
            "confianca": float(probabilities[best_index]),
            "probabilidades": {
                class_names[int(index)]: float(probabilities[index])
                for index in ranked_indices
            },
        }

    return app


app = create_app()
