"""
Simple Faster-Whisper (ONNX) Test
Uses SMALL model to avoid memory issues
"""

from faster_whisper import WhisperModel
import time

# ============= CONFIGURATION =============
AUDIO_FILE = "test2.wav"  # Your audio file
MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large-v3
# =========================================

print("🎤 Testing Faster-Whisper (ONNX)...")
print(f"   Audio file: {AUDIO_FILE}")
print(f"   Model: {MODEL_SIZE}")

try:
    # Check GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except:
        has_gpu = False
    
    device = "cuda" if has_gpu else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"   Device: {device}")
    
    # Load model
    print("\n⏳ Loading model...")
    start_load = time.time()
    
    model = WhisperModel(
        MODEL_SIZE,
        device=device,
        compute_type=compute_type
    )
    
    load_time = time.time() - start_load
    print(f"   Model loaded in {load_time:.2f}s")
    
    # Transcribe
    print("\n⏳ Transcribing...")
    start_trans = time.time()
    
    segments, info = model.transcribe(
        AUDIO_FILE,
        beam_size=5,
        language="en",
        vad_filter=True
    )
    
    # Collect transcription
    transcription = " ".join([seg.text for seg in segments])
    
    trans_time = time.time() - start_trans
    total_time = time.time() - start_load
    
    # Results
    print(f"\n✅ SUCCESS!")
    print(f"⏱️  Load time: {load_time:.2f}s")
    print(f"⏱️  Transcribe time: {trans_time:.2f}s")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🌍 Language detected: {info.language} ({info.language_probability:.1%})")
    print(f"📝 Transcription:")
    print(f"   \"{transcription}\"")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n💡 TIP: If out of memory, use smaller model:")
    print("   MODEL_SIZE = 'tiny'  # or 'base'")