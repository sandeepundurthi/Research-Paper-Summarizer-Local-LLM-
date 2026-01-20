from typing import List, Dict


def chunk_text_by_words(text: str, chunk_words: int = 1400, overlap_words: int = 200) -> List[Dict]:
    """
    Simple word-based chunker with overlap.
    Returns list of {"chunk_id": "c00", "text": "..."}
    """
    words = text.split()
    chunks = []
    start = 0
    c = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({"chunk_id": f"c{c:02d}", "text": chunk})
        c += 1
        if end == len(words):
            break
        start = max(0, end - overlap_words)

    return chunks
