# ZoneInsight - RAG System for "Attention Is All You Need"

> Retrieval-Augmented Generation (RAG) pipeline that answers questions from the Transformer paper with cited page numbers.

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)]()
[![Model](https://img.shields.io/badge/Embedding-all--MiniLM--L6--v2-orange)]()
[![Accuracy](https://img.shields.io/badge/Retrieval-100%25-brightgreen)]()

## 📌 Overview
ZoneInsight loads the "Attention Is All You Need" paper, chunks it, stores embeddings in ChromaDB, retrieves relevant chunks, and generates answers using Ollama Llama3.2:1b. You can use any pdf.

## 🏗️ Architecture

PDF (Attention Paper) 
→ pdf_loader.py 
→ chunker.py [1000 chars / 200 overlap] 
→ vectordb.py [all-MiniLM-L6-v2, 384 dim] 
→ ChromaDB (57 chunks)
→ rag.py [Retriever + Llama3.2 Generator]

## Quick Start

# 1. Install
```
pip install -r requirements.txt
pip install chromadb sentence-transformers requests
```

# 2. Install Ollama from https://ollama.com
```
ollama pull llama3.2:1b
```

# 3. Build Vector DB
```
python src/vectordb.py
```

# Output: DB has 57 chunks

# 4. Evaluate (100% accuracy)
```
python src/evaluate.py > evaluation_results.txt
```

# 5. Run RAG
```
python src/rag.py
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