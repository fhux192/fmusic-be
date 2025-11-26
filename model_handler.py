import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

from candidates import SONG_CANDIDATES

DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"

MODEL_PATH = "openai/clip-vit-large-patch14"


processor = None
model = None


def load_model():
    global processor, model

    if model is not None:
        return

    print(f"[CLIP Handler] Loading CLIP model on device: {DEVICE}...")
    try:
        processor = CLIPProcessor.from_pretrained(MODEL_PATH)

        model = CLIPModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16
        ).to(DEVICE)

        print("[CLIP Handler] CLIP model is ready.")

    except Exception as e:
        print(f"[CLIP Handler] Failed to load model: {e}")
        processor = None
        model = None


load_model()


def generate_description_from_image(image_path: str) -> str | None:
    if not model or not processor:
        print("[CLIP Handler] Model is not available. Cannot generate description.")
        return None

    try:
        raw_image = Image.open(image_path).convert('RGB')

        inputs = processor(
            text=SONG_CANDIDATES,
            images=raw_image,
            return_tensors="pt",
            padding=True
        ).to(DEVICE, dtype=torch.float16)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        best_index = probs.argmax().item()
        description = SONG_CANDIDATES[best_index]

        print(f"[CLIP Handler] Image classified as: '{description}'")

        return description

    except Exception as e:
        print(f"[CLIP Handler] An error occurred in generate_description_from_image: {e}")
        return None


if __name__ == '__main__':
    dummy_image_path = "test_image.png"
    try:
        Image.new('RGB', (224, 224), color='red').save(dummy_image_path)
        print(f"\n[Test] Created a dummy image at: {dummy_image_path}")

        description = generate_description_from_image(dummy_image_path)

        if description:
            print(f"[Test] Generated Description: '{description}'")
        else:
            print("[Test] Could not generate description.")

        print("\n[Test] Running second time to test cache...")
        description_2 = generate_description_from_image(dummy_image_path)
        if description_2:
            print(f"[Test] Generated Description (2nd run): '{description_2}'")

    finally:
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)
            print(f"[Test] Removed dummy image.")