import os
import base64
import io
import time
import json
import requests
from PIL import Image

API_URL = "http://20.245.200.125:8000/v1/chat/completions"
MODELS_URL = "http://20.245.200.125:8000/v1/models"
IMAGE_DIR = "test_images"
PROMPT = "Transcribe all text from the image exactly as it appears in its original language. Do not translate. Do not add extra comments or explanations."

# Detect model name
model_name = ""
try:
    print(f"Fetching models from {MODELS_URL}...")
    r = requests.get(MODELS_URL, timeout=10)
    if r.status_code == 200:
        models = r.json().get("data", [])
        if models:
            model_name = models[0].get("id", "")
            print(f"Auto-detected model: {model_name}")
except Exception as e:
    print(f"Failed to auto-detect model: {e}")

images = [
    {"file": "01_english_clear.png", "lang": "English", "quality": "Clear"},
    {"file": "02_english_blurry.png", "lang": "English", "quality": "Blurry"},
    {"file": "03_hindi_clear.png", "lang": "Hindi", "quality": "Clear"},
    {"file": "04_hindi_poor.png", "lang": "Hindi", "quality": "Poor"},
    {"file": "05_chinese_clear.png", "lang": "Chinese", "quality": "Clear"},
    {"file": "06_arabic_poor.png", "lang": "Arabic", "quality": "Poor"},
    {"file": "07_receipt.png", "lang": "English", "quality": "OCR (Receipt)"},
    {"file": "handwritten.png", "lang": "English", "quality": "OCR (Handwritten)"},
]

def image_to_base64_url(path):
    with open(path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode('utf-8')
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def clean_output(raw):
    if raw is None:
        return None, ""
    raw = str(raw).replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if "<think>" in raw and "</think>" in raw:
        parts = raw.split("</think>")
        think = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return think, answer
    return None, raw

results = []

for idx, img_info in enumerate(images):
    file_path = os.path.join(IMAGE_DIR, img_info["file"])
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    print(f"[{idx+1}/{len(images)}] Processing {img_info['file']} ({img_info['lang']}, {img_info['quality']})...")
    
    # Build payload
    img_url = image_to_base64_url(file_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": PROMPT}
            ]
        }
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.1,
        "stream": False
    }
    
    t0 = time.time()
    try:
        r = requests.post(API_URL, json=payload, timeout=120)
        latency = time.time() - t0
        
        if r.status_code == 200:
            res_json = r.json()
            message = res_json.get("choices", [{}])[0].get("message", {})
            raw_content = message.get("content", "")
            reasoning_content = message.get("reasoning_content", None)
            
            if raw_content is None or raw_content == "":
                print(f"Warning: content is empty or None! Raw message: {message}")
                
            think, answer = clean_output(raw_content)
            if reasoning_content and not think:
                think = reasoning_content
                
            result = {
                "file": img_info["file"],
                "lang": img_info["lang"],
                "quality": img_info["quality"],
                "success": True,
                "latency_seconds": round(latency, 2),
                "think": think,
                "transcription": answer,
                "error": None
            }
            print(f"Success in {result['latency_seconds']}s")
        else:
            result = {
                "file": img_info["file"],
                "lang": img_info["lang"],
                "quality": img_info["quality"],
                "success": False,
                "latency_seconds": round(latency, 2),
                "think": None,
                "transcription": None,
                "error": f"HTTP {r.status_code}: {r.text[:200]}"
            }
            print(f"HTTP Error {r.status_code}")
    except Exception as e:
        latency = time.time() - t0
        result = {
            "file": img_info["file"],
            "lang": img_info["lang"],
            "quality": img_info["quality"],
            "success": False,
            "latency_seconds": round(latency, 2),
            "think": None,
            "transcription": None,
            "error": str(e)
        }
        print(f"Request Exception: {e}")
        
    results.append(result)

# Save results
out_path = "api_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Saved results to {out_path}")
