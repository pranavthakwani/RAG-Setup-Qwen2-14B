import sounddevice as sd
import vosk
import json
import queue
import sys
import subprocess
import pyaudio
import wave
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import ollama

# Vosk model path - update to your downloaded model
MODEL_PATH = "model/vosk-model-small-en-us-0.15"
model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, 16000)

# Audio settings
SAMPLE_RATE = 16000
BLOCKSIZE = 8000
audio_queue = queue.Queue()

# Ollama model
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M")
TOP_K = int(os.environ.get("TOP_K", "4"))

# Hardcoded assistant identity
BOT_NAME = "KoreAssist"
COMPANY_NAME = "Kore Mobile"
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

def retrieve(query, k=TOP_K):
    project_root = Path(__file__).resolve().parents[0]
    db_dir = project_root / "db"

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    coll = client.get_or_create_collection("kore_knowledge", embedding_function=embedder)

    results = coll.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
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

def listen():
    """Capture speech from microphone and return text."""
    print("Listening... Say something!")

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    return text

def speak(text):
    """Convert text to speech using Piper TTS and play audio."""
    print(f"Speaking: {text}")
    
    try:
        # Generate WAV file using Piper
        with open("temp_output.wav", "wb") as f:
            subprocess.run(["./piper/piper.exe", "--model", "en_US-lessac-medium.onnx", "--output_file", "temp_output.wav"], input=text, text=True, check=True)
        
        # Play the WAV file
        play_wav("temp_output.wav")
        
        # Clean up
        os.remove("temp_output.wav")
    except subprocess.CalledProcessError as e:
        print(f"Error with Piper TTS: {e}")
        # Fallback: print the text
        print(f"Would say: {text}")

def play_wav(file_path):
    """Play a WAV file using pyaudio."""
    chunk = 1024
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()

def chatbot_reply(user_text):
    """Get reply from Ollama based on context."""
    contexts = retrieve(user_text, k=TOP_K)
    prompt = build_prompt(user_text, contexts)

    response_stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": 256,
        },
        stream=True
    )
    full_response = ""
    for chunk in response_stream:
        content = chunk["message"]["content"]
        full_response += content
    return full_response.strip()

def main():
    """Main loop: listen, reply, speak."""
    print("Voice Chatbot started. Say 'exit' to quit.")
    while True:
        user_input = listen()
        if not user_input:
            continue
        print(f"You: {user_input}")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        reply = chatbot_reply(user_input)
        print(f"Bot: {reply}")
        speak(reply)

if __name__ == "__main__":
    main()
