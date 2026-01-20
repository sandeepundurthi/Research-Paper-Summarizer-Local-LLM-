MAP_SYSTEM = (
    "You are a careful scientific paper summarizer. "
    "Only write claims that are supported by the provided text. "
    "If something is not explicitly stated, write 'unclear'. "
    "Output ONLY valid JSON."
)

REDUCE_SYSTEM = (
    "You combine multiple chunk-level scientific summaries into one faithful paper summary. "
    "Do not invent details. Prefer exact numbers when present. "
    "Every bullet must include evidence chunk_ids. "
    "Output ONLY valid JSON."
)

def map_prompt(chunk_id: str, chunk_text: str) -> str:
    return f"""
Chunk ID: {chunk_id}

Text:
\"\"\"{chunk_text}\"\"\"

Task:
Extract information ONLY supported by this text. Produce JSON with:
- chunk_id
- section_guess (one of: abstract/introduction/methods/results/discussion/conclusion/other)
- claims: 3-8 bullet strings (scientific, specific, no fluff)
- numbers: list of objects {{ "value": "...", "context": "..." }} (include metrics, datasets, results)
- keywords: 5-12 important terms

Return ONLY JSON.
""".strip()


def reduce_prompt(mapped_json_list: list) -> str:
    return f"""
You will be given chunk summaries (JSON objects). Combine them into a single paper summary JSON.

Chunk summaries:
{mapped_json_list}

Output JSON schema:
{{
  "title_guess": "... or null",
  "one_sentence_summary": "...",
  "contributions": [{{"bullet": "...", "evidence": ["c00","c03"]}}],
  "method_overview": [{{"bullet": "...", "evidence": ["c01"]}}],
  "key_results": [{{"bullet": "... (include numbers)", "evidence": ["c02","c04"]}}],
  "limitations": [{{"bullet": "...", "evidence": ["c05"]}}],
  "future_work": [{{"bullet": "...", "evidence": ["c06"]}}],
  "keywords": ["..."],
  "confidence_score": 0.0
}}

Rules:
- If you can't find a field, keep it empty or null; do NOT invent.
- Evidence must reference chunk_ids that support that bullet.
- confidence_score: 0.2 if weak/unclear, 0.5 moderate, 0.8 strong support.

Return ONLY JSON.
""".strip()
