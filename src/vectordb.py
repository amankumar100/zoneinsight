from pdf_loader import load_pdf
from chunker import chunk_text
from sentence_transformers import SentenceTransformer
import chromadb
import os

PDF_PATH = "data/attention.pdf"
DB_PATH = "db/chroma_db"
COLLECTION_NAME = "attention_paper"

def build_vector_db():
    print(f"Loading PDF: {PDF_PATH}")
    all_pages = load_pdf(PDF_PATH)
    print(f"Loaded {len(all_pages)} pages")

    # Build chunks correctly using pages list
    all_chunks_dict = chunk_text(all_pages, chunk_size=800, overlap=100)

    all_chunks = [c['text'] for c in all_chunks_dict]
    metadatas = [{"page_num": c['page_num']} for c in all_chunks_dict]
    ids = [f"chunk_{c['chunk_id']}" for c in all_chunks_dict]

    print(f"Total chunks from all pages: {len(all_chunks)}")

    print("\nLoading embedding model: all-MiniLM-L6-v2")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded!")

    print(f"\nConverting {len(all_chunks)} chunks to vectors...")
    embeddings = model.encode(all_chunks).tolist()
    print(f"Vector shape: ({len(embeddings)}, {len(embeddings[0])})")

    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"\nCreating ChromaDB at {DB_PATH}")
    print("Storing chunks + vectors in ChromaDB...")
    collection.add(documents=all_chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"Done! Stored {collection.count()} chunks in ChromaDB")
    return collection, model

if __name__ == "__main__":
    collection, model = build_vector_db()