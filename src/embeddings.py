from pypdf import PdfReader
import os
from sentence_transformers import SentenceTransformer
import numpy as np

def load_pdf(file_path):
    print(f"Loading PDF file: {file_path}")
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        page_data = {
            "page_num": i + 1,
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split())
        }
        pages.append(page_data)
    return pages

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start<len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Move by overlap
    return chunks

# Day 2

def cosine_similarity(vec1, vec2):
    """ How similar are two vectors? 1 =  same, 0 = different """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

if __name__ == "__main__":
    pdf_path = "data/attention.pdf"
    if not os.path.exists(pdf_path):  # now os is used again
        print("ERROR: PDF not found")
        exit(1)

    all_pages = load_pdf(pdf_path)

    # 1. Take Page 1 and chunk it - Day 1
    print(f"\nLoaded {len(all_pages)}")
    page1_chunks = chunk_text(all_pages[0]['text'], chunk_size=500, overlap=50)
    print(f"Page 1 -> {len(page1_chunks)}")

    #2. Load Embedding Model - Day 2
    print("\nLoading embedding model: all-MiniLM-L6-v2 (90, first time will download)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded!")

    # 3. Convert chunks to vectors
    print(f"\nConverting {len(page1_chunks)} chunks to vectors...")
    embeddings = model.encode(page1_chunks)
    print(f"Vectors shape: {embeddings.shape}")
    print(f"First chunk vector (first 10 numbers): {embeddings[0][:10]}")

    # 4. Search by meaning
    query = "What is attention mechanism?"
    print(f"\n--- Semantic Search Demo ---")
    print(f"Query: '{query}'")

    query_vector = model.encode([query])[0]

    # Compare query vector with all chunk vectors
    scores = []
    for i, chunk_vec in enumerate(embeddings):
        score = cosine_similarity(query_vector, chunk_vec)
        scores.append((i, score))

    # Sort by score high to low
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 2 most similar chunks:")
    for idx, score in scores[:2]:
        print(f"\n[Chunk {idx} | Score: {score:.4f}]")
        print(f"{page1_chunks[idx][:300]}...")