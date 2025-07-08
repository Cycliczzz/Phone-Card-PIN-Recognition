import torch
from PIL import Image


def ocr_single_crop(img: Image.Image, model, processor) -> str:
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(model.device)
    ids = model.generate(pixel_values, max_new_tokens=25)
    txt = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return txt.strip()