"""Generate synthetic scratch‑card images with random PIN text"""
import os, random, string
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT_DIR = "../data/images/train_synth"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
NUM = 1000
os.makedirs(OUT_DIR, exist_ok=True)

for i in range(NUM):
    pin = ''.join(random.choices(string.digits + string.ascii_uppercase, k=15))
    img = Image.new("RGB", (400, 160), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 48)
    draw.text((20, 40), pin, fill=(0, 0, 0), font=font)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    fname = f"synth_{i:05d}.jpg"
    img.save(os.path.join(OUT_DIR, fname))