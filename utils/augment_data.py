import os, glob
from PIL import Image, ImageOps, ImageFilter

DIR = "../data/images/train"
imgs = glob.glob(os.path.join(DIR, "*.jpg"))

for p in imgs:
    img = Image.open(p)
    base = os.path.splitext(os.path.basename(p))[0]
    # Rotate 5°
    img.rotate(5, expand=True).save(os.path.join(DIR, base + "_rot.jpg"))
    # Horizontal flip
    ImageOps.mirror(img).save(os.path.join(DIR, base + "_flip.jpg"))
    # Blur
    img.filter(ImageFilter.BLUR).save(os.path.join(DIR, base + "_blur.jpg"))