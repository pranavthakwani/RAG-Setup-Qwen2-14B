import os
from pathlib import Path
from tqdm import tqdm

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Disable telemetry
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'true'

# --------------------------
# PDF Loader
# --------------------------
# --------------------------
# PDF Loader with Page Metadata
# --------------------------
def load_pdfs(pdf_dir):
    """Load PDFs, extract text per page, and chunk with metadata."""
    docs = []
    for pdf_path in pdf_dir.glob("**/*.pdf"):
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    # Clean text: remove extra whitespace, line breaks
                    text = ' '.join(text.split())
                    # Chunk the page text
                    chunks = chunk_text(text)
                    for chunk_idx, chunk in enumerate(chunks):
                        docs.append((str(pdf_path), page_num + 1, chunk_idx + 1, chunk))
        except Exception as e:
            print(f"PDF parse error {pdf_path}: {e}")
    return docs

# --------------------------
# Text chunking
# --------------------------
def chunk_text(text, chunk_size=200, overlap=50):
    """Chunk text into overlapping segments for better retrieval."""
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

# --------------------------
# Main ingestion
# --------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    pdf_dir = project_root / "data" / "pdfs"
    db_dir = project_root / "db"
    db_dir.mkdir(exist_ok=True)

    # --------------------------
    # Embedding function
    # --------------------------
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-mpnet-base-v2",  # high-quality embeddings
        device="cpu",  # change to "cuda" if GPU is available
    )

    # --------------------------
    # ChromaDB setup
    # --------------------------
    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
    collection = client.get_or_create_collection(
        name="pdf_knowledge",
        embedding_function=embedder
    )

    # --------------------------
    # Load PDFs
    # --------------------------
    all_docs = load_pdfs(pdf_dir)
    unique_pdfs = len(set(fp for fp, _, _, _ in all_docs))
    print(f"Found {unique_pdfs} PDF files ({len(all_docs)} total chunks)")

    ids, docs, metas = [], [], []

    # --------------------------
    # Process chunks and ingest
    # --------------------------
    for fp, page_num, chunk_idx, chunk in tqdm(all_docs, desc="Processing PDFs"):
        ids.append(f"{fp}-{page_num}-{chunk_idx}")
        docs.append(chunk)
        metas.append({"source": fp, "page": page_num, "chunk": chunk_idx})

        # Batch insert every 200 chunks
        if len(ids) >= 200:
            collection.add(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []

    # Add remaining chunks
    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas)

    print("✅ PDF ingestion complete.")
