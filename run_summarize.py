import json
import os
import argparse
from tqdm import tqdm

from summarizer.llm_local import OllamaClient
from summarizer.pdf_extract import extract_pdf_text_pages, clean_text, drop_references_section
from summarizer.chunking import chunk_text_by_words
from summarizer.map_summarize import summarize_chunk
from summarizer.reduce_summarize import summarize_paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name, e.g. llama3.1:8b")
    parser.add_argument("--chunk_words", type=int, default=1400)
    parser.add_argument("--overlap_words", type=int, default=200)
    parser.add_argument("--out", default="outputs/summary.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) Extract + clean
    pages = extract_pdf_text_pages(args.pdf)
    full_text = "\n\n".join([f"[PAGE {p['page']}]\n{p['text']}" for p in pages])
    full_text = clean_text(full_text)
    full_text = drop_references_section(full_text)

    # 2) Chunk
    chunks = chunk_text_by_words(full_text, chunk_words=args.chunk_words, overlap_words=args.overlap_words)

    # 3) Map summarize
    client = OllamaClient(model=args.model)
    mapped = []
    for ch in tqdm(chunks, desc="Summarizing chunks"):
        mapped.append(summarize_chunk(client, ch["chunk_id"], ch["text"]))

    # 4) Reduce summarize
    final_summary = summarize_paper(client, mapped)

    # Save outputs
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    # Also save chunk summaries (helpful for debugging/evidence)
    mapped_path = args.out.replace(".json", ".chunks.json")
    with open(mapped_path, "w", encoding="utf-8") as f:
        json.dump(mapped, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved final summary: {args.out}")
    print(f"✅ Saved chunk summaries: {mapped_path}")


if __name__ == "__main__":
    main()
