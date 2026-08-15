import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# Fix Windows Unicode error
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding!= 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = "db/chroma_db"
COLLECTION_NAME = "attention_paper"

SYSTEM_PROMPT = """You are a helpful assistant. Use the Context to answer the question.
If context is somewhat related, try to answer. Only say not found if truly unrelated.
Cite Page numbers like [Page X]. Keep answer concise."""

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
    # Clean print for Windows
    print(f"\nRetrieved {len(docs)} chunks | Distances: {[round(d,3) for d in dists]}")
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        safe_preview = doc[:120].encode('ascii', 'ignore').decode('ascii')
        print(f" [{i+1}] Page {meta['page_num']} (Dist {dists[i]:.3f}): {safe_preview}...")
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
    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama not running! Run `ollama serve` and `ollama pull llama3.2:1b`"
    except Exception as e:
        return f"ERROR: {str(e)[:200]}"

if __name__ == "__main__":
    print("--- ZoneInsight RAG Test ---")
    collection = get_collection()
    model = get_model()
    print(f"DB Count: {collection.count()} chunks")
    print("Model Ready!\n")

    while True:
        q = input("Ask question (or 'exit'): ")
        if q.lower() == 'exit':
            break
        if not q.strip():
            continue

        ctx, pages, docs = retrieve_context(q, collection, model, n_results=3)
        print(f"\nPages: {pages}")
        print("\n--- Generating Answer ---")
        answer = ask_ollama(q, ctx)
        print(f"\n--- ANSWER ---\n{answer}\n")
        print("="*80)