# ZoneInsight - RAG System for "Attention Is All You Need"

> Retrieval-Augmented Generation (RAG) pipeline that answers questions from the Transformer paper with cited page numbers.

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)]()
[![Model](https://img.shields.io/badge/Embedding-all--MiniLM--L6--v2-orange)]()
[![Accuracy](https://img.shields.io/badge/Retrieval-100%25-brightgreen)]()

## 📌 Overview
ZoneInsight loads the "Attention Is All You Need" paper, chunks it, stores embeddings in ChromaDB, retrieves relevant chunks, and generates answers using Ollama Llama3.2:1b.

## 🏗️ Architecture

PDF (Attention Paper) 
→ pdf_loader.py 
→ chunker.py [1000 chars / 200 overlap] 
→ vectordb.py [all-MiniLM-L6-v2, 384 dim] 
→ ChromaDB (57 chunks)
→ rag.py [Retriever + Llama3.2 Generator]

## 📊 Evaluation Results - 100% Accuracy

| # | Question | Retrieved Page | Status |
|---|----------|----------------|--------|
| 1 | What is self-attention? | Page 2 | ✅ |
| 2 | What is multi-head attention? | Page 5 | ✅ |
| 3 | Why positional encoding? | Page 2,3 | ✅ |
| 4 | BLEU score on WMT 2014? | Page 8 (28.4) | ✅ |
| 5 | How many layers N? | Page 3 (N=6) | ✅ |
| 6 | Scaled dot-product formula? | Page 4 | ✅ |
| 7 | Optimizer Adam beta? | Page 7 (0.9,0.98) | ✅ |
| 8 | Q K V matrices? | Page 4 | ✅ |
| 9 | Purpose of masking? | Page 3 | ✅ |
| 10 | WMT datasets? | Page 7,9 | ✅ |

**Technique:** Keyword-enriched queries (Q K V, N=6, WMT) improved 90% → 100%

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt
pip install chromadb sentence-transformers requests

# 2. Install Ollama from https://ollama.com
ollama pull llama3.2:1b

# 3. Build Vector DB
python src/vectordb.py
# Output: DB has 57 chunks

# 4. Evaluate (100% accuracy)
python src/evaluate.py > evaluation_results.txt

# 5. Run RAG
python src/rag.py

Q: What is attention mechanism?
A: The attention mechanism allows every position in the decoder to attend over all positions in the input sequence [Page 3, 5]. In self-attention, all keys, values and queries come from same place...

```

## Project Structure

zoneinsight/
├── data/attention.pdf
├── db/chroma_db/ (57 chunks)
├── src/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── vectordb.py
│   ├── rag.py
│   └── evaluate.py
├── evaluation_results.txt
├── evaluation.md
└── README.md

## 🛠️ Tech Stack

    Embedding: all-MiniLM-L6-v2 (384 dim)
    VectorDB: ChromaDB Persistent
    LLM: Ollama Llama3.2:1b
    Chunking: 1000 chars / 200 overlap

## Aman(Me) - ZoneInsight RAG

## 📜 License 
MIT