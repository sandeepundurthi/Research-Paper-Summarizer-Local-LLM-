import json
import requests
from typing import Optional, Dict, Any


class OllamaClient:
    """
    Minimal Ollama client using the local HTTP API.
    Default URL works with Ollama Desktop on macOS.
    """
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, system: str, prompt: str, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get("response", "")

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """
        Robust-ish JSON extraction: find the first {...} block.
        If model prints extra text, this still often works.
        """
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output.")
        block = text[start:end + 1]
        return json.loads(block)
