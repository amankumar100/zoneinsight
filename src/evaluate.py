from sentence_transformers import SentenceTransformer
import chromadb
import os

DB_PATH = "db/chroma_db"
COLLECTION_NAME = "attention_paper"

# Load DB and model
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

def query_db(question, n_results=2):
    q_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=n_results)
    return results

# 10 test questions for Attention Is All You Need
test_questions = [
    "What is self-attention mechanism definition?",
    "What is multi-head attention how it works?",
    "Why do we need positional encoding in Transformer?",
    "What BLEU score did Transformer achieve on WMT 2014?",
    "How many layers N does encoder and decoder have?",
    "What is scaled dot-product attention formula Attention(Q K V)?",
    "What optimizer Adam beta1 beta2 is used for training Transformer?",
    "What are Query Key Value Q K V matrices in attention?",
    "What is purpose of masking in decoder to prevent attending?",
    "What datasets WMT English German French were used?"
]

print("=== ZoneInsight RAG Evaluation ===\n")
print(f"DB has {collection.count()} chunks\n")

for i, q in enumerate(test_questions, 1):
    print(f"{i}. Q: {q}")
    results = query_db(q, n_results=2)
    docs = results['documents'][0]
    meta = results['metadatas'][0]
    for j, (doc, m) in enumerate(zip(docs, meta)):
        print(f" [{j+1}] Page {m['page_num']}: {doc[:200]}...")
    print("-"*80 + "\n")
    
print("Evaluation Done!")