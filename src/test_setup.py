import streamlit as st
import chromadb
from dotenv import load_dotenv
import os
import sys

load_dotenv()

print("=== DocuMind Day 0 Check ===")
print("Python version:", sys.version.split()[0])
print("Streamlit version:", st.__version__)
print("ChromaDB version:", chromadb.__version__)
print("ENV file loaded:", "GROQ_API_KEY" in os.environ)
print("Setup complete. Ready for Day 1!")