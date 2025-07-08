"""Fine‑tune TrOCR with local scratch‑card dataset"""
import os, json
from datasets import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image

DATA_DIR = "../data/images"
LABEL_FILE = "../data/metadata.json"
OUTPUT_DIR = "../models/trocr-finetuned"
PRETRAINED = "microsoft/trocr-base-stage1"
EPOCHS = 5
BATCH = 4

proc = TrOCRProcessor.from_pretrained(PRETRAINED)
model = VisionEncoderDecoderModel.from_pretrained(PRETRAINED)

with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    meta = json.load(f)

dataset = Dataset.from_dict({
    "image": [Image.open(os.path.join(DATA_DIR, m["file"]).convert("RGB")) for m in meta],
    "text": [m["text"] for m in meta]
})

def transform(ex):
    enc = proc(images=ex["image"], text=ex["text"], padding="max_length", truncation=True, return_tensors="pt")
    return {"pixel_values": enc.pixel_values[0], "labels": enc.labels[0]}

dataset = dataset.map(transform)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    learning_rate=5e-5,
    save_strategy="epoch",
    fp16=True,
)

trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=dataset)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
proc.save_pretrained(OUTPUT_DIR)