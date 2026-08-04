from chunker import load_pdf, chunk_text
from sentence_transformers import SentenceTransformer
import chromadb
import os

PDF_PATH = "data/attention.pdf"
DB_PATH = "db/chroma_db"
COLLECTION_NAME = "attention_paper"

def build_vector_db():
    #1. Load PDF all 15 pages
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {PDF_PATH}")
        exit(1)

    print(f"Loading PDF: {PDF_PATH}")
    all_pages = load_pdf(PDF_PATH)
    print(f"Loaded {len(all_pages)} pages")

    # 2. Chunk all pages with metadata
    all_chunks = []
    metadatas = []
    ids = []
    chunk_id = 0

    for page in all_pages:
        page_chunks = chunk_text(page['text'], chunk_size=500, overlap=50)
        for chunk in page_chunks:
            all_chunks.append(chunk)
            metadatas.append({"page_num": page['page_num']})
            ids.append(f"chunk_{chunk_id}")
            chunk_id += 1

    print(f"Total chunks from all pages: {len(all_chunks)}")

    # 3. Load embedding model
    print("\nLoading embedding model: all-MiniLM-L6-v2")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded!")

    # 4. Convert chunks to vectors
    print(f"\nConverting {len(all_chunks)} chunks to vectors...")
    embeddings = model.encode(all_chunks).tolist()
    print(f"Vector shape: ({len(embeddings)}, {len(embeddings[0])})")

    # 5. Create ChromaDB - persistent on disk
    print(f"\nCreating ChromaDB at {DB_PATH}")
    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)

    # Delete old collection if exists (for fresh run)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    # 6. Store in DB
    print("Storing chunks + vectors in ChromaDB...")
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Done! Stored {collection.count()} chunks in ChromaDB")
    return collection, model

def search_db(query, collection, model, n_results=3):
    print(f"\n--- Searching: '{query}' ---")
    query_vector = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results
    )

    print(f"\nTop {n_results} results:")
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        # Chroma uses L2 distance, convert to similarity approx
        
        print(f"\n[Result {i+1} | Page {meta['page_num']} | Distance: {distance:.4f}]")
        print(f"{doc[:350]}...")

if __name__ == "__main__":
    collection, model = build_vector_db()

    # Test searches across ALL 15 pages
    search_db("What is attention mechanism?", collection, model, n_results=2)
    search_db("What is Transformer?", collection, model, n_results=2)
    search_db("What is training done?", collection, model, n_results=2)