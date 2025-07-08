"""CLI tool: python utils/inference.py path/to/img.jpg"""
import sys
from PIL import Image
from app.model import get_trocr
from app.utils import ocr_single_crop

if __name__ == "__main__":
    model, proc = get_trocr()
    img = Image.open(sys.argv[1]).convert("RGB")
    print(ocr_single_crop(img, model, proc))