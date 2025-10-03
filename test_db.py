import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path

# Setup
project_root = Path(__file__).resolve().parents[0]
db_dir = project_root / "db"
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu",
)
client = chromadb.PersistentClient(
    path=str(db_dir),
    settings=Settings(anonymized_telemetry=False, is_persistent=True),
)
coll = client.get_or_create_collection("pdf_knowledge", embedding_function=embedder)

# Count documents
count = coll.count()
print(f"Total documents in collection: {count}")

# Test retrieval
test_query = "test query from your PDF content"  # Replace with actual PDF text
results = coll.query(query_texts=[test_query], n_results=5, include=["documents"])
docs = results.get("documents", [[]])[0]
print(f"Found {len(docs)} chunks for test query")
for i, doc in enumerate(docs):
    print(f"Chunk {i+1}: {doc[:100]}...")
