import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import tempfile
import streamlit as st

from summarizer.llm_local import OllamaClient
from summarizer.pdf_extract import extract_pdf_text_pages, clean_text, drop_references_section
from summarizer.chunking import chunk_text_by_words
from summarizer.map_summarize import summarize_chunk
from summarizer.reduce_summarize import summarize_paper


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Research Paper Summarizer (Local LLM)")
st.caption("Faithful research summaries with evidence · Runs fully on your machine")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    model = st.selectbox(
        "LLM Model",
        ["llama3.1:8b", "qwen2.5:7b"],
        index=0
    )

    summary_style = st.selectbox(
        "Summary Style",
        ["Technical (Reviewer)", "Executive (High-level)", "Interview Prep"]
    )

    chunk_words = st.slider("Chunk Size (words)", 800, 2200, 1400, 100)
    overlap_words = st.slider("Chunk Overlap", 50, 400, 200, 50)

    st.markdown("---")
    st.caption("💡 Tip: Smaller chunks = safer summaries")

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader(
    "Upload a research paper PDF",
    type=["pdf"],
    help="Works best with academic papers (arXiv, journals, conferences)"
)

# ---------------- RUN ----------------
if uploaded and st.button("🚀 Generate Summary", use_container_width=True):

    client = OllamaClient(model=model)

    with st.spinner("📄 Extracting text from PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            pdf_path = tmp.name

        pages = extract_pdf_text_pages(pdf_path)
        full_text = "\n\n".join([p["text"] for p in pages])
        full_text = clean_text(full_text)
        full_text = drop_references_section(full_text)

    chunks = chunk_text_by_words(full_text, chunk_words, overlap_words)

    # -------- MAP STEP --------
    mapped = []
    with st.spinner("🧠 Analyzing paper sections..."):
        progress = st.progress(0)
        for i, ch in enumerate(chunks):
            mapped.append(summarize_chunk(client, ch["chunk_id"], ch["text"]))
            progress.progress(int((i + 1) / len(chunks) * 100))

    # -------- REDUCE STEP --------
    with st.spinner("📊 Building final summary..."):
        final_summary = summarize_paper(client, mapped)

    st.success("✅ Summary generated successfully!")

    # ---------------- RESULTS ----------------
    st.markdown("## 🧠 TL;DR")
    st.info(final_summary.get("one_sentence_summary", "N/A"))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📌 Key Contributions")
        for c in final_summary.get("contributions", []):
            st.write("•", c["bullet"])

        st.markdown("### ⚙️ Methods")
        for m in final_summary.get("method_overview", []):
            st.write("•", m["bullet"])

    with col2:
        st.markdown("### 📊 Key Results")
        for r in final_summary.get("key_results", []):
            st.write("•", r["bullet"])

        st.markdown("### ⚠️ Limitations")
        for l in final_summary.get("limitations", []):
            st.write("•", l["bullet"])

    # ---------------- TOGGLES ----------------
    with st.expander("🔍 View Evidence (Chunk-level)"):
        st.json(mapped)

    with st.expander("📄 View Raw JSON Output"):
        st.json(final_summary)

    st.download_button(
        "⬇ Download summary.json",
        data=json.dumps(final_summary, indent=2),
        file_name="summary.json",
        mime="application/json",
        use_container_width=True
    )
