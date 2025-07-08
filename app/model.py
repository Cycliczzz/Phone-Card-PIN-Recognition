from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch, os

MODEL_DIR = os.getenv("TROCR_DIR", "models/trocr-finetuned")

def get_trocr():
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, processor