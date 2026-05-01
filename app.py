import os

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from posture import analyze_posture


app = FastAPI(title="Upright Posture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image payload.")

    array = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unsupported or invalid image data.")

    return analyze_posture(frame)
