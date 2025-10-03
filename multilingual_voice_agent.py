#!/usr/bin/env python3
"""
Multilingual Voice Calling Agent
- Uses Whisper for speech-to-text with automatic language detection
- Supports Hindi, English, Gujarati, and Hinglish
- Integrates with existing Kore chatbot
- Provides real-time voice responses
"""

import os
import sys
import time
import queue
import threading
import subprocess
import tempfile
import wave
from pathlib import Path

import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
import pyaudio
import ffmpeg
from langdetect import detect

# Import existing chatbot functionality
from scripts.chat import retrieve, build_prompt
import ollama

class MultilingualVoiceAgent:
    def __init__(self):
        """Initialize the multilingual voice agent."""
        self.whisper_model = None
        self.tts_engine = None
        self.audio_queue = queue.Queue() 
        self.is_listening = False
        self.is_speaking = False

        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 3  # seconds
        self.chunk_size = self.sample_rate * self.chunk_duration

        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'gu': 'Gujarati'
        }

        # Ollama model
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M")

    def initialize_models(self):
        """Initialize Whisper and TTS models."""
        print("🔄 Initializing Whisper model (medium)...")
        try:
            self.whisper_model = whisper.load_model("medium")
            print("✅ Whisper model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            print("🔄 Falling back to base model...")
            try:
                self.whisper_model = whisper.load_model("base")
                print("✅ Base model loaded successfully")
            except Exception as e2:
                print(f"❌ Failed to load base model: {e2}")
                print("🔄 Falling back to tiny model for testing...")
                self.whisper_model = whisper.load_model("tiny")
                print("✅ Tiny model loaded (fastest for testing)")

        print("🔄 Initializing TTS engine...")
        try:
            self.tts_engine = pyttsx3.init()
            # Set voice properties
            voices = self.tts_engine.getProperty('voices')

            print(f"Available voices: {len(voices)}")
            for i, voice in enumerate(voices):
                print(f"  {i}: {voice.name}")

            # Try to select male Indian voice
            male_voice_selected = False

            # Look for male voices first
            for i, voice in enumerate(voices):
                voice_name = voice.name.lower()
                if any(male_indicator in voice_name for male_indicator in ['david', 'male', 'man', 'पुरुष', 'मर्द']):
                    print(f"✅ Selected male voice: {voice.name}")
                    self.tts_engine.setProperty('voice', voice.id)
                    male_voice_selected = True
                    break

            # If no male voice found, use the first available voice
            if not male_voice_selected and len(voices) > 0:
                print(f"✅ Using default voice: {voices[0].name}")
                self.tts_engine.setProperty('voice', voices[0].id)

            self.tts_engine.setProperty('rate', 180)  # Speed up speech
            print("✅ TTS engine initialized")

        except Exception as e:
            print(f"❌ Failed to initialize TTS: {e}")

    def detect_language(self, text):
        """Detect the language of the given text."""
        try:
            detected_lang = detect(text.lower())
            # Map language codes
            if detected_lang == 'hi':
                return 'hi'
            elif detected_lang in ['en', 'hi']:  # Hinglish often detected as English
                # Simple heuristic: if text contains Hindi words, consider it Hindi
                hindi_words = ['ka', 'ke', 'ki', 'ko', 'se', 'mein', 'me', 'aur', 'ya', 'to', 'hai', 'ho', 'tha', 'the']
                if any(word in text.lower() for word in hindi_words):
                    return 'hi'
                return 'en'
            elif detected_lang == 'gu':
                return 'gu'
            else:
                return 'en'  # Default to English
        except Exception:
            return 'en'  # Default to English if detection fails

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper with language detection."""
        try:
            print("🎯 Transcribing audio...")

            # Check if file exists and has content
            if not os.path.exists(audio_path):
                print(f"❌ Audio file not found: {audio_path}")
                return "", "en"

            file_size = os.path.getsize(audio_path)
            if file_size < 1000:  # Very small file
                print(f"❌ Audio file too small ({file_size} bytes) - no audio data")
                return "", "en"

            result = self.whisper_model.transcribe(
                audio_path,
                language=None,  # Let Whisper auto-detect
                verbose=False
            )

            text = result["text"].strip()

            if not text:
                print("🤐 No speech detected in audio")
                return "", "en"

            detected_lang = self.detect_language(text)
            print(f"📝 Transcribed: '{text}' (Language: {self.supported_languages.get(detected_lang, 'Unknown')})")

            return text, detected_lang

        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return "", "en"

    def get_chatbot_response(self, text, language='en'):
        """Get response from the existing Kore chatbot."""
        try:
            print("🤖 Getting chatbot response...")

            # Use existing retrieve function for context
            contexts = retrieve(text, k=4)

            # Build prompt with language consideration
            if language == 'hi':
                # Add Hindi context to prompt
                system_context = "You are KoreAssist, a helpful assistant for Kore Mobile. Respond in Hindi when the user speaks in Hindi. Use simple, clear Hindi language."
            elif language == 'gu':
                # Add Gujarati context to prompt
                system_context = "You are KoreAssist, a helpful assistant for Kore Mobile. Respond in Gujarati when the user speaks in Gujarati. Use simple, clear Gujarati language."
            else:
                system_context = "You are KoreAssist, a helpful assistant for Kore Mobile."

            # Build full prompt
            context_text = "\n\n".join([f"Source: {m.get('source','unknown')}\n{d}" for d, m in contexts])
            prompt = f"""<system>
{system_context}
Follow these rules strictly:
- Assistant name: KoreAssist
- Organization: Kore Mobile
- Capabilities: Answers general questions and provides Kore-specific info using retrieved context.
- Data sources: Local PDFs, text files, and crawled website pages ingested into the vector store.
- Privacy: Runs locally, offline. Do not claim to access the internet or external services during chat.
- Grounding: Prefer provided context; if information is missing or unclear, say you don't know.
- Style: Be concise, helpful, and cite relevant sources by filename when helpful.
- Responses: Do not include introductory greetings like 'Hello! I'm KoreAssist.' unless the user asks about your identity.
</system>

<context>
{context_text}
</context>

<user>
{text}
</user>"""

            # Get response from Ollama
            response_stream = ollama.chat(
                model=self.ollama_model,
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

        except Exception as e:
            print(f"❌ Chatbot response failed: {e}")
            return "Sorry, I couldn't process your request right now."

    def speak_response(self, text, language='en'):
        """Convert text to speech in the appropriate language."""
        try:
            print(f"🗣️  Speaking response in {self.supported_languages.get(language, 'English')}...")

            if not self.tts_engine:
                print("TTS engine not available, printing text instead")
                print(f"Bot: {text}")
                return

            # Set language-specific voice properties if needed
            if language == 'hi':
                # For Hindi, you might want different voice settings
                pass
            elif language == 'gu':
                # For Gujarati, you might want different voice settings
                pass

            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

        except Exception as e:
            print(f"❌ TTS failed: {e}")
            print(f"Bot: {text}")

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input."""
        if status:
            print(f"Audio status: {status}")

        # Put audio data in queue
        self.audio_queue.put(bytes(indata))

    def record_audio_chunk(self, duration=3):
        """Record an audio chunk for processing."""
        print(f"🎤 Recording {duration}s audio chunk...")

        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=1024,
                dtype='int16',
                channels=1,
                callback=self.audio_callback
            ):
                audio_data = []
                start_time = time.time()

                while time.time() - start_time < duration:
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                        audio_data.append(data)
                    except queue.Empty:
                        continue

                if audio_data:
                    # Combine audio chunks
                    combined_audio = b''.join(audio_data)

                    # Check if we have enough audio data
                    if len(combined_audio) < 1000:  # Less than ~0.03 seconds of audio
                        print("⚠️ Very little audio data captured")
                        return None

                    # Save to temporary WAV file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        with wave.open(temp_file.name, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(self.sample_rate)
                            wav_file.writeframes(combined_audio)

                        return temp_file.name
                else:
                    print("⚠️ No audio data captured")
                    return None

        except Exception as e:
            print(f"❌ Audio recording failed: {e}")
            return None

    def process_audio_chunk(self, audio_path):
        """Process a single audio chunk: transcribe, get response, speak."""
        try:
            # Transcribe audio
            text, detected_language = self.transcribe_audio(audio_path)

            if not text.strip():
                print("🤐 No speech detected")
                return

            print(f"👤 You ({self.supported_languages.get(detected_language, 'Unknown')}): {text}")

            # Get chatbot response
            response = self.get_chatbot_response(text, detected_language)

            print(f"🤖 Bot ({self.supported_languages.get(detected_language, 'English')}): {response}")

            # Speak response
            self.speak_response(response, detected_language)

        except Exception as e:
            print(f"❌ Error processing audio chunk: {e}")
        finally:
            # Clean up temporary file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass

    def run_interactive_mode(self):
        """Run the voice agent in interactive mode."""
        print("🎙️  Multilingual Voice Agent Started!")
        print("Supported languages: Hindi, English, Gujarati, Hinglish")
        print("Say 'exit' or 'quit' to stop")
        print("=" * 50)

        try:
            while True:
                # Record audio chunk
                audio_path = self.record_audio_chunk(self.chunk_duration)

                if audio_path:
                    # Process the chunk
                    self.process_audio_chunk(audio_path)
                else:
                    print("⏳ Waiting for audio input...")

                # Check for exit command (this would need to be detected in the transcribed text)
                # For now, we'll use a simple keyboard interrupt

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error in interactive mode: {e}")

    def run_file_mode(self, audio_file):
        """Process a single audio file."""
        print(f"📁 Processing audio file: {audio_file}")

        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return

        # Convert audio to proper format if needed
        processed_file = self.preprocess_audio(audio_file)

        if processed_file:
            self.process_audio_chunk(processed_file)

    def preprocess_audio(self, input_file):
        """Preprocess audio file to 16kHz mono for Whisper."""
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                output_file = temp_file.name

            # Convert audio using ffmpeg
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(stream, output_file, ar=16000, ac=1, acodec='pcm_s16le')
            ffmpeg.run(stream, quiet=True, overwrite_output=True)

            return output_file

        except Exception as e:
            print(f"❌ Audio preprocessing failed: {e}")
            print("🔄 Using original file (may affect accuracy)")
            return input_file  # Return original file as fallback

def main():
    """Main function."""
    agent = MultilingualVoiceAgent()
    agent.initialize_models()

    if len(sys.argv) > 1:
        # Process audio file
        audio_file = sys.argv[1]
        agent.run_file_mode(audio_file)
    else:
        # Interactive mode
        agent.run_interactive_mode()

if __name__ == "__main__":
    main()
