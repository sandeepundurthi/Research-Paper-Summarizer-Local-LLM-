# 📄 Research Paper Summarizer (Local LLM)

A **privacy-preserving research paper summarization system** that runs fully **locally** using large language models (LLMs) via **Ollama**.  
The system generates **faithful, structured summaries** with **evidence-backed citations** to reduce hallucinations.

---

## ✨ Features
- 📑 Upload research paper PDFs
- 🧠 Chunk-based *map-reduce* summarization
- 📊 Structured JSON output:
  - One-sentence summary
  - Contributions
  - Methods
  - Key results (with numbers)
  - Limitations & future work
- 🔍 Evidence tracking via chunk IDs
- 🖥️ Runs fully **locally** (no data leaves your machine)
- 🌐 Streamlit web interface + CLI mode

---

## 🧱 Architecture
PDF → Text Extraction → Chunking
→ Chunk Summaries (Map)
→ Global Summary (Reduce)
→ Structured JSON + Evidence

---

## 🛠️ Tech Stack
- **LLM**: LLaMA-3.1 / Qwen-2.5 (via Ollama)
- **Backend**: Python
- **PDF Parsing**: PyMuPDF
- **UI**: Streamlit
- **Platform**: macOS (Apple Silicon M1/M2)

---

## 🚀 Installation

### 1. Install Ollama
Download and install Ollama for macOS  
Then pull a model:
```bash
ollama pull llama3.1:8b
```

---
## 2. Clone Repository
git clone https://github.com/sandeepundurthi/research-paper-summarizer-llm.git
cd research-paper-summarizer-llm

---
## 3. Set up Python Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---
## Usage
### CLI Mode
python run_summarize.py --pdf "/path/to/research_paper.pdf"

### Outputs:

outputs/summary.json

outputs/summary.chunks.json

---
## Streamlit Web App
streamlit run app/app.py

