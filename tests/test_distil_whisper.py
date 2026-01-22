"""
Simple Distil-Whisper Test
Uses librosa to avoid FFmpeg issues
"""

from transformers import pipeline
import torch
import time
import librosa

# ============= CONFIGURATION =============
AUDIO_FILE = "test2.wav"  # Your audio file
# =========================================

print("🎤 Testing Distil-Whisper...")
print(f"   Audio file: {AUDIO_FILE}")

try:
    # Check GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    if device == "cpu":
        print("   ⚠️  Using CPU - will be slower!")
    
    # Load model
    print("\n⏳ Loading model...")
    start_load = time.time()
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model="distil-whisper/distil-large-v3",
        device=device,
        torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
    )
    
    load_time = time.time() - start_load
    print(f"   Model loaded in {load_time:.2f}s")
    
    # Load audio with librosa (avoids FFmpeg issues)
    print("\n⏳ Loading audio with librosa...")
    audio_array, sr = librosa.load(AUDIO_FILE, sr=16000)
    
    # Transcribe
    print("⏳ Transcribing...")
    start_trans = time.time()
    
    # Pass as dict with 'array' and 'sampling_rate' to avoid file loading
    result = pipe({"array": audio_array, "sampling_rate": sr})
    
    trans_time = time.time() - start_trans
    total_time = time.time() - start_load
    
    transcription = result["text"]
    
    # Results
    print(f"\n✅ SUCCESS!")
    print(f"⏱️  Load time: {load_time:.2f}s")
    print(f"⏱️  Transcribe time: {trans_time:.2f}s")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📝 Transcription:")
    print(f"   \"{transcription}\"")
    
    # Cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n💡 TIP: Install librosa if missing:")
    print("   pip install librosa soundfile")