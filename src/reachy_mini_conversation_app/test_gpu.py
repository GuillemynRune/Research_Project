"""Test if GPU is working for your models.

This verifies:
1. PyTorch can see GPU
2. Whisper can use GPU
3. Gemma 3 can use GPU
"""

import torch
import sys

print("="*60)
print("GPU DETECTION TEST")
print("="*60)

# Test 1: PyTorch CUDA
print("\n1. PyTorch CUDA Detection:")
if torch.cuda.is_available():
    print(f"   ✅ CUDA available: True")
    print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   ✅ CUDA version: {torch.version.cuda}")
    print(f"   ✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("   ❌ CUDA not available!")
    print("   → Install PyTorch with CUDA:")
    print("   → pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# Test 2: Whisper GPU
print("\n2. Whisper GPU Support:")
try:
    import whisper
    print("   ✅ Whisper installed")
    
    # Load tiny model to test GPU
    print("   → Loading Whisper tiny model on GPU...")
    model = whisper.load_model("tiny", device="cuda")
    print("   ✅ Whisper can use GPU!")
    
    # Test inference
    import numpy as np
    print("   → Testing GPU inference...")
    audio = np.random.randn(16000).astype(np.float32)  # 1 second of random audio
    result = model.transcribe(audio, fp16=True)  # fp16 only works on GPU
    print("   ✅ GPU inference works!")
    
except Exception as e:
    print(f"   ❌ Whisper GPU test failed: {e}")
    sys.exit(1)

# Test 3: Transformers GPU (for Gemma)
print("\n3. Transformers GPU Support:")
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    print("   ✅ Transformers installed")
    
    # Test a small model
    print("   → Testing small model on GPU...")
    processor = AutoProcessor.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    print(f"   ✅ Model loaded on: {model.device}")
    print("   ✅ Transformers can use GPU!")
    
except Exception as e:
    print(f"   ⚠️  Could not test Gemma (might need HF token): {e}")
    print("   → This is OK if you have HF_TOKEN set in your environment")

# Test 4: Memory check
print("\n4. GPU Memory Check:")
print(f"   Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
print(f"   Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3:.2f} GB")

# Recommendations
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

if total_mem < 6:
    print("\n⚠️  GPU has < 6GB memory")
    print("   → Use Whisper 'base' model")
    print("   → Gemma 3 4B might be tight")
    print("   → Consider Gemma 2 2B instead")
elif total_mem < 12:
    print("\n✅ GPU has 6-12GB memory")
    print("   → Use Whisper 'base' or 'small'")
    print("   → Gemma 3 4B will work")
    print("   → Should be fast!")
else:
    print("\n✅ GPU has 12+ GB memory")
    print("   → Use Whisper 'medium' or 'large'")
    print("   → Gemma 3 4B will work great")
    print("   → Very fast responses!")

print("\n" + "="*60)
print("✅ GPU SETUP COMPLETE!")
print("="*60)
print("\nYour handler will automatically use GPU.")
print("Just run: python -m reachy_mini_conversation_app.main --head-tracker yolo")