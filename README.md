# Kore Assist: PDF-Based RAG Chatbot for Kore Mobile

## 📋 Project Overview

Kore Assist is a Retrieval-Augmented Generation (RAG) chatbot designed specifically for Kore Mobile. It ingests PDF documents into a vector database, retrieves relevant context based on user queries, and generates accurate, context-grounded responses using a local LLM (Large Language Model). The system ensures no hallucination by strictly adhering to retrieved context and provides natural, conversational interactions.

### 🎯 Key Features

- **PDF Ingestion**: Automatically processes and chunks PDF documents for efficient retrieval.
- **Context-Aware Responses**: Uses ChromaDB for vector similarity search to find relevant chunks.
- **Local LLM Integration**: Runs on Ollama for privacy and offline capability.
- **Multilingual Support**: Optimized for Hindi, English, and Hinglish queries.
- **Real-Time Streaming**: Responses appear word-by-word like ChatGPT for better UX.
- **Error Handling**: Graceful fallbacks for missing context or errors.
- **Production-Ready**: Includes logging, configuration, and scalability features.

## 🏗️ Architecture

1. **Ingestion Pipeline** (`scripts/ingest.py`):

   - Loads PDFs from `data/pdfs/`.
   - Extracts text per page, cleans it, and chunks into 200-word segments with 50-word overlap.
   - Embeds chunks using SentenceTransformers and stores in ChromaDB.
2. **Retrieval & Generation** (`web_chat.py`):

   - Receives user queries via Flask API.
   - Retrieves top-k relevant chunks from ChromaDB.
   - Builds prompts with hard-coded rules and context.
   - Queries local Ollama model for responses.
   - Streams output in real-time.
3. **Frontend** (`templates/index.html`):

   - Simple chat UI with message history.
   - Handles streaming responses for real-time display.
4. **Testing** (`test_db.py`):

   - Script to verify DB contents and retrieval.

## 🛠️ Technologies Used

- **Backend**:

  - Flask: Web framework for API endpoints.
  - Ollama: Local LLM inference (Qwen2.5 14B Instruct Q4_0).
  - ChromaDB: Vector database for context storage and retrieval.
  - SentenceTransformers: Embedding model (`all-mpnet-base-v2`, 768 dimensions).
  - PyPDF: PDF text extraction.
- **Frontend**:

  - HTML/CSS/JavaScript: Chat interface with streaming support.
- **Other**:

  - Python 3.12+: Core language.
  - Logging: For debugging and monitoring.

## 📦 Installation

1. **Prerequisites**:

   - Windows OS (as per user setup).
   - Python 3.12+ installed.
   - Ollama installed and running (`ollama serve`).
   - Git for cloning.
2. **Clone Repository**:

   ```bash
   git clone https://github.com/pranavthakwani/RAG-Setup-Qwen2-14B.git
   cd RAG-Setup-Qwen2-14B
   ```
3. **Install Python Dependencies**:

   ```bash
   pip install flask ollama chromadb sentence-transformers pypdf tqdm
   ```
4. **Install Ollama Model**:

   ```bash
   ollama pull qwen2.5:14b-instruct-q4_0
   ```
5. **Set Environment Variables** (Optional):

   ```bash
   $env:OLLAMA_MODEL = "qwen2.5:14b-instruct-q4_0"
   $env:TOP_K = "20"
   $env:CONTEXT_CHUNK_SIZE = "1000"
   ```
6. **Prepare Data**:

   - Place PDF files in `data/pdfs/` directory.
   - Ensure PDFs are readable and not encrypted.

## 🚀 Usage

1. **Ingest PDFs**:

   ```bash
   python scripts/ingest.py
   ```

   - Processes ~185 PDFs into 7,419 chunks.
   - Time: 30-60 minutes.
   - Output: "✅ PDF ingestion complete."
2. **Run the Chatbot**:

   ```bash
   python web_chat.py
   ```

   - Starts Flask server on `http://127.0.0.1:5000`.
   - Open browser and interact via chat UI.
3. **Test Retrieval**:

   ```bash
   python test_db.py
   ```

   - Verifies DB contents and query matching.

## ⚙️ Configuration

- **Model Settings** (in `web_chat.py`):

  - `MODEL`: Ollama model (default: qwen2.5:14b-instruct-q4_0).
  - `TOP_K`: Number of chunks to retrieve (default: 20).
  - `CONTEXT_CHUNK_SIZE`: Max context length per chunk (default: 1000).
- **Embedding Model**:

  - `sentence-transformers/all-mpnet-base-v2` (768 dimensions, English-focused).
  - For multilingual: Switch to `paraphrase-multilingual-MiniLM-L12-v2` (384 dims) in both scripts.
- **Hard-Coded Knowledge** (in `HARD_CODED_KNOWLEDGE`):

  - Strict rules for context adherence, natural responses, no hallucination.

## 🔧 Technical Details

- **Chunking Strategy**:

  - Size: 200 words per chunk.
  - Overlap: 50 words for continuity.
  - Per-page processing for metadata (source, page, chunk).
- **Embeddings**:

  - Model: `all-mpnet-base-v2`.
  - Dimensions: 768.
  - Device: CPU (switch to "cuda" for GPU).
- **Vector DB**:

  - ChromaDB persistent client.
  - Collection: "pdf_knowledge".
  - Storage: `db/` folder (SQLite file ~80MB for 7,419 chunks).
- **LLM Integration**:

  - Ollama API for streaming.
  - Options: Temperature 0.1, top_k 40, num_ctx 4096, num_predict 512.
  - Streaming: Yields chunks immediately for real-time UI.
- **Prompt Engineering**:

  - System prompt enforces identity as "Kore Assist".
  - Includes context only if available; falls back to general knowledge for identity queries.
  - Rules prevent typo pointing, ensure first-person speech.
- **Error Handling**:

  - Dimension mismatch fixed by matching embedding models.
  - Telemetry errors ignored (ChromaDB warnings).
  - Graceful fallbacks for missing context.
- **Performance**:

  - Ingestion: ~4-5 seconds per PDF.
  - Retrieval: <1 second for top-20 chunks.
  - Response: 2-5 seconds depending on query length.
  - Memory: ~8-10GB RAM for 14B model + DB.

## 🚨 Troubleshooting

- **Embedding Dimension Mismatch**:

  - Ensure ingestion and retrieval use the same embedding model.
  - Re-ingest if switching models.
- **No Context Retrieved**:

  - Check ingestion completion.
  - Test with `test_db.py`.
  - Increase TOP_K if needed.
- **Streaming Issues**:

  - Ensure frontend handles ReadableStream (fixed in `index.html`).
  - Use async/await in JS for promises.
- **Ollama Errors**:

  - Ensure Ollama is running.
  - Pull model: `ollama pull qwen2.5:14b-instruct-q4_0`.
- **DB Issues**:

  - Delete `db/` and re-ingest if corrupted.
  - Check collection name matches ("pdf_knowledge").

## 📈 Scalability & Optimization

- **Batching**: Inserts in batches of 200 chunks to reduce memory.
- **Limits**: Caps context size to prevent token overflow.
- **Multilingual**: Use multilingual embeddings for Hindi/English.
- **GPU**: Enable CUDA for faster embeddings.

## 📝 License

This project is for educational/demonstration purposes. Kore Mobile is a fictional company in this context.

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch.
3. Test changes with PDFs.
4. Submit a PR.

## 📞 Support

For issues, open a GitHub issue with logs and error messages.

---

This README provides a complete guide to setting up and using Kore Assist.
