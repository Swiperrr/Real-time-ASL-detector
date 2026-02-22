# api.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, UploadFile, File
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from tensorflow import keras
from typing import List
import uvicorn

app = FastAPI(title="ASL Recognition API")


print("Loading models...")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

yolo_model = YOLO(str(BASE_DIR / "best.pt"))
cnn_model = keras.models.load_model(str(BASE_DIR / "cnn.keras"))

class_names = ['A','B','C','D','E','F','G','H','I','K','L','M',
               'N','O','P','Q','R','S','T','U','V','W','X','Y','del','space']

print("Models loaded!")


def process_hand_region(hand_img, cnn_model):
    try:
        if len(hand_img.shape) == 3:
            gray = cv.cvtColor(hand_img, cv.COLOR_BGR2GRAY)
        else:
            gray = hand_img

        resized = cv.resize(gray, (64, 64))
        normalized = resized.astype('float32') / 255.0

        expected_channels = cnn_model.input_shape[-1]

        if expected_channels == 1:
            batched = np.expand_dims(normalized, axis=0)
            batched = np.expand_dims(batched, axis=-1)

        elif expected_channels == 3:
            rgb = cv.cvtColor(resized, cv.COLOR_GRAY2RGB)
            normalized_rgb = rgb.astype('float32') / 255.0
            batched = np.expand_dims(normalized_rgb, axis=0)

        else:
            batched = np.expand_dims(normalized, axis=0)
            batched = np.expand_dims(batched, axis=-1)

        return batched

    except Exception as e:
        print("Processing error:", e)
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

    rresults = yolo_model(frame, conf=0.3, verbose=False)


    detections = []

    for result in rresults:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            hand_region = frame[y1:y2, x1:x2]

            if hand_region.size == 0:
                continue

            processed = process_hand_region(hand_region, cnn_model)
            if processed is None:
                continue

            preds = cnn_model.predict(processed, verbose=0)

            idx = int(np.argmax(preds[0]))
            conf = float(preds[0][idx])
            letter = class_names[idx]

            detections.append({
                "letter": letter,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

    return {"detections": detections}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
