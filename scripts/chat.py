import os
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'true'
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import ollama

COMPANY_NAME = "Kore Mobile"
BOT_NAME = "KoreAssist"
HARD_CODED_KNOWLEDGE = (
    "- Assistant name: KoreAssist\n"
    "- Organization: Kore Mobile\n"
    "- Capabilities: Answers general questions and provides Kore-specific info using retrieved context.\n"
    "- Data sources: Local PDFs, text files, and crawled website pages ingested into the vector store.\n"
    "- Privacy: Runs locally, offline. Do not claim to access the internet or external services during chat.\n"
    "- Grounding: Prefer provided context; if information is missing or unclear, say you don't know.\n"
    "- Style: Be concise, helpful, and cite relevant sources by filename when helpful.\n"
    "- Responses: Do not include introductory greetings like 'Hello! I'm KoreAssist.' unless the user asks about your identity.\n"
)

MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M")
TOP_K = int(os.environ.get("TOP_K", "4"))

# Global variables for caching
_cached_embedder = None
_cached_collection = None

def get_embedder_and_collection():
    """Get or create cached embedder and collection to avoid cold starts."""
    global _cached_embedder, _cached_collection

    if _cached_embedder is None or _cached_collection is None:
        print("🔄 Initializing models (first time only)...")

        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        project_root = Path(__file__).resolve().parents[1]
        db_dir = project_root / "db"

        client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        collection = client.get_or_create_collection("kore_knowledge", embedding_function=embedder)

        _cached_embedder = embedder
        _cached_collection = collection
        print("✅ Models initialized and cached")

    return _cached_embedder, _cached_collection

def retrieve(query, k=TOP_K):
    embedder, collection = get_embedder_and_collection()

    results = collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
    docs = results.get("documents", [[]])[0] if results.get("documents") else []
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    return list(zip(docs, metas))

def build_prompt(user_query, contexts):
    context_text = "\n\n".join([f"Source: {m.get('source','unknown')}\n{d}" for d, m in contexts])
    system = (
        f"You are {BOT_NAME}, a helpful assistant for {COMPANY_NAME}.\n"
        f"Follow these rules strictly:\n{HARD_CODED_KNOWLEDGE}\n"
        "Use only the provided context for Kore-specific details. If unsure, say you don't know.\n"
        "Keep answers concise. Address the user directly."
    )
    prompt = f"""<system>
{system}
</system>

<context>
{context_text}
</context>

<user>
{user_query}
</user>"""
    return prompt


def chat_once(query):
    contexts = retrieve(query, k=TOP_K)
    prompt = build_prompt(query, contexts)

    # Enable streaming for character-by-character output
    response_stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.2,
            "num_ctx": 4096,  # lower (e.g., 2048) for more speed
        },
        stream=True  # Enable streaming
    )
    
    print("\n--- Answer ---")
    full_answer = ""
    for chunk in response_stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)  # Print each character immediately
        full_answer += content
    print("\n--------------\n")  # New line after complete answer
    return full_answer


if __name__ == "__main__":
    print("Type your message here (Ctrl+C to exit):")
    while True:
        try:
            q = input("> ").strip()
            if not q:
                continue
            chat_once(q)
        except KeyboardInterrupt:
            break
