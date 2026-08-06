import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests

DB_PATH = "db/chroma_db"
COLLECTION_NAME = "attention_paper"

SYSTEM_PROMPT = """You are a helpful assistant. Use the Context to answer.
If context is somewhat related, try to answer. Only say not found if truly unrelated.
Cite Page numbers like [Page X]."""

@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return collection, model

def retrieve(query, collection, model, k=4):
    q_vec = model.encode([query]).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=k)
    docs = res['documents'][0]
    metas = res['metadatas'][0]
    ctx = ""
    pages = []
    for doc, meta in zip(docs, metas):
        ctx += f"\n[Page {meta['page_num']}]: {doc}\n"
        pages.append(meta['page_num'])
    return ctx, pages, docs

def ask_llama(q, ctx):
    url = "http://localhost:11434/api/generate"
    prompt = f"{SYSTEM_PROMPT}\n\nContext:{ctx}\n\nQuestion:{q}\n\nAnswer with citations like [Page X]:"
    payload = {"model":"llama3.2:1b","prompt":prompt,"stream":False,"options":{"temperature":0.2}}
    r = requests.post(url, json=payload, timeout=120)
    return r.json()['response']

st.set_page_config(page_title="ZoneInsight RAG", page_icon="📄")
st.title("📄 ZoneInsight - Attention Paper RAG")
st.caption("ChromaDB + all-MiniLM + Llama3.2:1b | 104 chunks")

collection, model = load_db()
st.sidebar.success(f"DB: {collection.count()} chunks")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if query := st.chat_input("Ask about attention paper..."):
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.markdown(query)

    ctx, pages, docs = retrieve(query, collection, model)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching pages {pages}..."):
            ans = ask_llama(query, ctx)
            st.markdown(ans)
            st.caption(f"Sources: Pages {pages}")
            with st.expander("View chunks"):
                for i, d in enumerate(docs):
                    st.code(f"[Page {pages[i]}] {d[:600]}...")

    st.session_state.messages.append({"role":"assistant","content":f"{ans}\n\n*Pages: {pages}*"})
    