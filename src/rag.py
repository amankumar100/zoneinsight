import os
import chromadb
from sentence_transformers import SentenceTransformer
import requests

DB_PATH = "db/chroma_db"
COLLECTION_NAME = "attention_paper"

SYSTEM_PROMPT = """You are a helpful assistant. Use the Context to answer the question.
If context is somewhat related, try to answer. Only say not found if truly unrelated.
Cite Page numbers like [Page X]."""

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)

def get_model():
    print("Loading embedding model...")
    return SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_context(query, collection, model, n_results=5):
    query_vector = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results
    )
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    dists = results['distances'][0]

    context = ""
    sources = []
    print(f"DEBUG distances: {dists}")
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"DEBUG Chunk {i} Page {meta['page_num']} Dist {dists[i]:.3f} -> {doc[:150]}...")
        context += f"\n[Page {meta['page_num']}]: {doc}\n"
        sources.append(meta['page_num'])
    return context, sources, docs

def ask_ollama(question, context):
    url = "http://localhost:11434/api/generate"
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"
    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": 4096}
    }
    response = requests.post(url, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()['response']

if __name__ == "__main__":
    print("--- ZoneInsight RAG Test ---")
    collection = get_collection()
    model = get_model()
    print(f"DB Count: {collection.count()} chunks")
    while True:
        q = input("\nAsk question (or 'exit'): ")
        if q.lower() == 'exit':
            break
        ctx, pages, docs = retrieve_context(q, collection, model, n_results=5)
        print(f"\nRetrieved from Pages: {pages}")
        answer = ask_ollama(q, ctx)
        print(f"\n--- ANSWER ---\n{answer}\n")