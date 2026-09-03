import base64
import requests
from typing import Optional

def decrypt_handwriting_with_vision(image_path: str, model: str = "gemma4:e4b-it-qat") -> Optional[str]:
    """
    Decrypts messy cursive / scribbled handwriting using local Vision AI via Ollama.
    """
    try:
        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": model,
            "prompt": "Transcribe every handwritten item or text in this image accurately line by line into a clean list. Fix spelling and abbreviations contextually:",
            "images": [b64_img],
            "stream": False
        }
        res = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=90)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
    except Exception:
        pass
    return None
