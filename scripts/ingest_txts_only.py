#!/usr/bin/env python3
"""
Efficient Text File Ingestion for Kore ChatBot
Focuses specifically on ingesting converted text files from data/txts/
"""

import os
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'true'
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

def chunk_text(text, chunk_size=400, overlap=100):
    """Split text into overlapping chunks."""
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

def load_and_ingest_txts(txt_dir, collection):
    """Load and ingest all text files from txt_dir."""
    txt_path = Path(txt_dir)

    if not txt_path.exists():
        print(f"Directory does not exist: {txt_path}")
        return 0

    # Get all text files
    txt_files = list(txt_path.glob("*.txt"))

    if not txt_files:
        print(f"No text files found in {txt_dir}")
        return 0

    print(f"Found {len(txt_files)} text files to ingest")

    processed_count = 0
    total_chunks = 0

    for txt_file in tqdm(txt_files, desc="Processing text files"):
        try:
            # Read text file
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            if not text.strip():
                continue

            # Chunk the text
            chunks = chunk_text(text)

            if not chunks:
                continue

            # Prepare data for insertion
            ids = [f"{txt_file}-{idx}" for idx in range(len(chunks))]
            docs = chunks
            metas = [{"source": str(txt_file)} for _ in chunks]

            # Insert in smaller batches to avoid memory issues
            batch_size = 20
            for i in range(0, len(ids), batch_size):
                end_idx = min(i + batch_size, len(ids))
                batch_ids = ids[i:end_idx]
                batch_docs = docs[i:end_idx]
                batch_metas = metas[i:end_idx]

                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas
                )

            processed_count += 1
            total_chunks += len(chunks)

        except Exception as e:
            print(f"Error processing {txt_file}: {str(e)}")
            continue

    return processed_count, total_chunks

def main():
    """Main function to run the text file ingestion."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    txt_dir = project_root / "data" / "txts"
    db_dir = project_root / "db"

    print("Starting efficient text file ingestion...")
    print(f"Text directory: {txt_dir}")
    print(f"Database directory: {db_dir}")

    # Set up ChromaDB
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name="kore_knowledge",
        embedding_function=embedder
    )

    # Ingest text files
    processed_files, total_chunks = load_and_ingest_txts(txt_dir, collection)

    print("\nIngestion completed!")
    print(f"Successfully processed: {processed_files} files")
    print(f"Total chunks created: {total_chunks}")
    print(f"Average chunks per file: {total_chunks / processed_files:.1f}" if processed_files > 0 else "No files processed")

    return 0

if __name__ == "__main__":
    exit(main())
