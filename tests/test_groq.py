"""
Simple Groq API Test
Just tests Groq Whisper - nothing else!
"""

from groq import Groq
import os
import time

# ============= CONFIGURATION =============
AUDIO_FILE = "test_audio.wav"  # Your audio file
GROQ_API_KEY = "your-api-key-here"  # Or set environment variable
# =========================================

# Use environment variable if not set in script
if GROQ_API_KEY == "your-api-key-here":
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    print("❌ ERROR: Set GROQ_API_KEY!")
    print("   Get free key from: https://console.groq.com")
    exit(1)

print("🎤 Testing Groq Whisper API...")
print(f"   Audio file: {AUDIO_FILE}")

try:
    # Create client
    client = Groq(api_key=GROQ_API_KEY)
    
    # Start timer
    start = time.time()
    
    # Transcribe
    with open(AUDIO_FILE, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(AUDIO_FILE, file.read()),
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en"
        )
    
    # End timer
    elapsed = time.time() - start
    
    # Results
    print(f"\n✅ SUCCESS!")
    print(f"⏱️  Time: {elapsed:.2f} seconds")
    print(f"📝 Transcription:")
    print(f"   \"{transcription}\"")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")