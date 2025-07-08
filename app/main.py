from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from detect import detect_boxes
from model import get_trocr
from utils import ocr_single_crop

app = FastAPI(title="Scratch‑Card OCR API")

# Load TrOCR once on startup
tr_model, tr_processor = get_trocr()

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """Nhận duy nhất 1 mã trong ảnh"""
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    text = ocr_single_crop(img, tr_model, tr_processor)
    return {"pin": text}

@app.post("/recognize-multi")
async def recognize_multi(file: UploadFile = File(...)):
    """Nhận nhiều mã: detect → crop → OCR"""
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    boxes = detect_boxes(img)
    pins = []
    for box in boxes:
        crop = img.crop(box)
        pins.append(ocr_single_crop(crop, tr_model, tr_processor))

    return JSONResponse({"pins": pins})
