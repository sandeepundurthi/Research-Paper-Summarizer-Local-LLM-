from typing import Dict, Any
from .llm_local import OllamaClient
from .prompts import MAP_SYSTEM, map_prompt


def summarize_chunk(client: OllamaClient, chunk_id: str, text: str) -> Dict[str, Any]:
    raw = client.generate(system=MAP_SYSTEM, prompt=map_prompt(chunk_id, text), temperature=0.2)
    return client.extract_json(raw)
