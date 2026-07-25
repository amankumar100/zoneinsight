from pypdf import PdfReader
import os

def load_pdf(file_path):
    """
    Load a PDF file and returna a list of dicts.
    Each dict = 1 page.
    """
    print(f"Loading PDF file: {file_path}")
    reader = PdfReader(file_path)

    pages = []
    print(f"Number of pages: {len(reader.pages)}")

    for i, page in enumerate(reader.pages):
        text = page.extract_text()


    # Skip empty pages
        if not text:
            continue

    # DICT - Hold structured data for 1 page
        page_data = {
            "page_num": i + 1,
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split())
        }

    # ADD DICT to list
        pages.append(page_data)

    return pages

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into smaller chunks with overlap
    This is REAL chunking used in RAG
    """

    chunks = []
    start = 0
    while start<len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Move by overlap
    return chunks

if __name__=="__main__":
    pdf_path = "data/attention.pdf"

    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF path not found")
        print("Download attention paper to data/attention.pdf")
        exit(1)

    # PART 1 : Load pdf as list of dicts
    all_pages = load_pdf(pdf_path)

    print(f"Loaded {len(all_pages)} pages from PDF sucessfully")
    print(f"Page 1 text: {all_pages[0]['char_count']} chars, {all_pages[0]['word_count']} words")
    print("First 300 chars of page 1---")
    print(all_pages[0]['text'][:300])

    # PART 2 : Chunk text from page 1
    print("\n---- Chunking Demo: Page 1 -> 500-char chunks ---")
    page1_chunks = chunk_text(all_pages[0]['text'], chunk_size=500, overlap=50)
    print(f"Page 1 text chunked into {len(page1_chunks)} chunks")
    print(f"Chunk 1 (first 200 chars): {page1_chunks[0][:200]}")
    print(f"Chunk 2 (first 200 chars): {page1_chunks[1][:200]}")