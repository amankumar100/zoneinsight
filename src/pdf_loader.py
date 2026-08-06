from pypdf import PdfReader
import os

def load_pdf(file_path):
    """
    Load Attention paper PDF and return clean pages.
    This version is for the OFFICIAL clean arxiv PDF.
    Figure 3 & 4 example sentences about law/voting are REAL paper content - do not filter them.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: PDF not found at {file_path}")
        return []

    reader = PdfReader(file_path)
    pages = []

    print(f"Reading {len(reader.pages)} pages from {file_path}")

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        # Skip empty pages
        if not text:
            continue

        text = text.strip()
        if len(text) < 100:
            continue

        # Remove token artifacts like <EOS> <pad>
        text = text.replace("<EOS>", " ").replace("<pad>", " ")

        # Clean extra whitespaces
        text = " ".join(text.split())

        pages.append({
            "page_num": i + 1,
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split())
        })

    print(f"Loaded {len(pages)} clean pages")
    return pages

if __name__ == "__main__":
    pdf_path = "data/attention.pdf"
    all_pages = load_pdf(pdf_path)

    if all_pages:
        print(f"\nTotal pages loaded: {len(all_pages)}")
        print(f"Page 13 preview (Figure 3 example):")
        print(all_pages[12]['text'][:500])