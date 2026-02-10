import os
import requests
import base64

API_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

payload = {
    "prompt": "professional jewellery product photography, gold diamond ring, isolated product, studio lighting, soft shadows, white or deep blue velvet background, ultra sharp focus, luxury jewellery catalog, no hand, no fingers, no nails, centered composition, macro lens",
    "negative_prompt": "hand, fingers, nails, skin, people, blurry, low quality, distortion, watermark",
    "steps": 30,
    "cfg_scale": 7,
    "denoising_strength": 0.55,
    "sampler_name": "DPM++ 2M Karras"
}

for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    print("Processing:", file)

    payload["init_images"] = [encode_image(os.path.join(INPUT_DIR, file))]

    response = requests.post(API_URL, json=payload).json()
    image = base64.b64decode(response["images"][0])

    with open(os.path.join(OUTPUT_DIR, file), "wb") as f:
        f.write(image)

print("DONE ✅")
