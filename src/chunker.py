def chunk_text(pages, chunk_size=1000, overlap=200):
    """
    pages = list of dicts from load_pdf()
    Each dict has: page_num, text
    Returns: list of chunk dicts
    """
    all_chunks = []

    for page in pages:
        text = page['text']

        # Skip any remaining garbage that passed loader
        if not text or len(text.strip()) < 100:
            continue

        # Extra safety - remove legal footer if still present
        if "The Law will never be perfect" in text:
            text = text.split("The Law will never")[0]

        if len(text.strip()) < 100:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Keep only meaningful chunks
            if len(chunk.strip()) > 200:
                all_chunks.append({
                    'text': chunk.strip(),
                    'page_num': page['page_num'],
                    'chunk_id': len(all_chunks)
                })

            start += chunk_size - overlap

    return all_chunks

# For testing alone
if __name__ == "__main__":
    from pdf_loader import load_pdf
    pages = load_pdf("data/attention.pdf")
    chunks = chunk_text(pages)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk Page {chunks[0]['page_num']}: {chunks[0]['text'][:200]}")