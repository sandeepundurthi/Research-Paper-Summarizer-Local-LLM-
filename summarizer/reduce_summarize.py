from typing import Dict, Any, List
from .llm_local import OllamaClient
from .prompts import REDUCE_SYSTEM, reduce_prompt


def summarize_paper(client: OllamaClient, mapped: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw = client.generate(system=REDUCE_SYSTEM, prompt=reduce_prompt(mapped), temperature=0.2)
    return client.extract_json(raw)
