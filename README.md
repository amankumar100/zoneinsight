# ZoneInsight - RAG is all you need

Ask

# ZoneInsight - RAG for Attention Is All You Need

Ask questions to the "Attention Is All You Need" paper using RAG.

## Pipeline
1. `pdf_loader.py` - Load PDF
2. `chunker.py` - Chunk 1000/200
3. `vectordb.py` - Embed with all-MiniLM-L6-v2 + ChromaDB
4. `rag.py` - Retrieve + LLM answer
5. `evaluate.py` - 10 Qs evaluation

## Results
- 57 chunks stored
- 100% retrieval accuracy (10/10)
- BLEU: 28.4, Layers: 6, Optimizer: Adam

## Run
pip install -r requirements.txt
python src/vectordb.py
python src/evaluate.py
python src/rag.py

## Demo
Q: What is self-attention?
A: [Retrieved from Page 5]