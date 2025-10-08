#
# import torch
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
#
# def get_device():
#     if torch.backends.mps.is_available():
#         return "mps"
#     elif torch.cuda.is_available():
#         return "cuda"
#     else:
#         return "cpu"
#
# DEVICE = get_device()
# MODEL_PATH = "Salesforce/blip-image-captioning-large"
#
# print(f"Loading BLIP model on device: {DEVICE}...")
# processor = BlipProcessor.from_pretrained(MODEL_PATH)
# model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
# print("BLIP model loaded successfully!")
#
# def generate_description_from_image(image_path):
#     try:
#         raw_image = Image.open(image_path).convert('RGB')
#
#         inputs = processor(raw_image, return_tensors="pt").to(DEVICE)
#         outputs = model.generate(**inputs, max_new_tokens=100)
#         description = processor.decode(outputs[0], skip_special_tokens=True)
#
#         return description
#     except Exception as e:
#         print(f"An error occurred in model_handler: {e}")
#         return None
#

# ---

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import tensorflow as tf

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

DEVICE = get_device()
MODEL_PATH = "Salesforce/blip2-opt-2.7b"

print(f"Loading BLIP-2 model on device: {DEVICE}...")
try:
    processor = Blip2Processor.from_pretrained(MODEL_PATH)
    
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16
    ).to(DEVICE)
    
    print("BLIP-2 model loaded successfully!")

except Exception as e:
    print(f"Failed to load model: {e}")
    processor = None
    model = None

def generate_description_from_image(image_path: str) -> str | None:
    if not model or not processor:
        print("Model is not available. Cannot generate description.")
        return None

    try:
        raw_image = Image.open(image_path).convert('RGB')

        inputs = processor(raw_image, return_tensors="pt").to(DEVICE, dtype=torch.float32)
        
        outputs = model.generate(**inputs, max_new_tokens=100)
        
        description = processor.decode(outputs[0], skip_special_tokens=True)
        
        return description.strip()

    except Exception as e:
        print(f"An error occurred in generate_description_from_image: {e}")
        return None

if __name__ == '__main__':
    dummy_image_path = "test_image.png"
    try:
        Image.new('RGB', (224, 224), color = 'red').save(dummy_image_path)
        print(f"Created a dummy image at: {dummy_image_path}")

        description = generate_description_from_image(dummy_image_path)
        if description:
            print(f"\nGenerated Description: '{description}'")
        else:
            print("\nCould not generate description.")
    finally:
        import os
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)
            print(f"Removed dummy image.")