import re
import fitz  # PyMuPDF
from typing import List, Dict


def extract_pdf_text_pages(pdf_path: str) -> List[Dict]:
    """
    Returns: list of {"page": int, "text": str}
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def clean_text(text: str) -> str:
    # Basic cleanup: collapse whitespace, remove repeated hyphen line breaks
    text = text.replace("-\n", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_references_section(full_text: str) -> str:
    """
    Heuristic: if there's a References / Bibliography heading, cut it off.
    """
    lower = full_text.lower()
    markers = ["references", "bibliography"]
    cut = None
    for m in markers:
        idx = lower.rfind("\n" + m + "\n")
        if idx != -1:
            cut = idx
            break
    return full_text[:cut].strip() if cut else full_text
