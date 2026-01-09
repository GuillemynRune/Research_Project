#!/bin/bash
# Setup script for Reachy Mini with Gemma 3 Vision-Language Model

set -e

echo "=========================================="
echo "Reachy Mini - Gemma 3 VLM Setup"
echo "=========================================="
echo ""

# Check for HuggingFace token
echo "Checking for HuggingFace token..."
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set"
    echo ""
    echo "Gemma 3 requires a HuggingFace account and token."
    echo ""
    echo "Steps:"
    echo "  1. Create account: https://huggingface.co/join"
    echo "  2. Get token: https://huggingface.co/settings/tokens"
    echo "  3. Accept Gemma 3 license: https://huggingface.co/google/gemma-3-4b-it"
    echo ""
    read -p "Enter your HuggingFace token (or press Enter to set later): " token
    
    if [ -n "$token" ]; then
        export HF_TOKEN="$token"
        echo "✓ Token set for this session"
    else
        echo "⚠️  You'll need to set HF_TOKEN before running the app"
    fi
else
    echo "✓ HF_TOKEN found"
fi

echo ""
echo "=========================================="
echo "Installing Python Dependencies"
echo "=========================================="
echo ""

# Install base requirements
if [ -f "requirements.txt" ]; then
    echo "Installing base requirements..."
    pip install -r requirements.txt
fi

# Install Gemma 3 dependencies
echo "Installing Gemma 3 VLM dependencies..."
pip install openai-whisper torch transformers accelerate pillow huggingface-hub

# Ask about 8-bit quantization
echo ""
read -p "Install 8-bit quantization support? (saves 50% VRAM) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing bitsandbytes..."
    pip install bitsandbytes
fi

# Ask about vision extras
echo ""
read -p "Install YOLO face tracking? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing YOLO vision extras..."
    pip install ultralytics supervision
fi

# Ask about TTS
echo ""
echo "TTS Options:"
echo "  1) None (text only)"
echo "  2) Piper TTS (recommended - fast)"
echo "  3) Coqui TTS (more voices)"
echo "  4) Bark (high quality, slow)"
read -p "Select TTS engine (1-4): " tts_choice

case $tts_choice in
    2)
        echo "Installing Piper TTS..."
        pip install piper-tts
        TTS_ENGINE="piper"
        ;;
    3)
        echo "Installing Coqui TTS..."
        pip install TTS
        TTS_ENGINE="coqui"
        ;;
    4)
        echo "Installing Bark..."
        pip install git+https://github.com/suno-ai/bark.git
        TTS_ENGINE="bark"
        ;;
    *)
        TTS_ENGINE="none"
        ;;
esac

echo ""
echo "=========================================="
echo "Downloading Gemma 3 Model"
echo "=========================================="
echo ""

if [ -n "$HF_TOKEN" ]; then
    echo "Pre-downloading Gemma 3 (~9GB)..."
    python3 << EOF
from huggingface_hub import snapshot_download
import os

os.environ['HF_TOKEN'] = '$HF_TOKEN'

try:
    snapshot_download(
        repo_id="google/gemma-3-4b-it",
        cache_dir="./cache",
        token="$HF_TOKEN"
    )
    print("✓ Gemma 3 model downloaded")
except Exception as e:
    print(f"⚠️  Download failed: {e}")
    print("The model will download on first run.")
EOF
else
    echo "⚠️  Skipping download (no HF_TOKEN)"
    echo "Model will download on first run (~9GB)"
fi

echo ""
echo "=========================================="
echo "Downloading Whisper Model"
echo "=========================================="
echo ""

echo "Which Whisper model?"
echo "  1) tiny (39MB, fast, lower quality)"
echo "  2) base (74MB, balanced) [recommended]"
echo "  3) small (244MB, better quality)"
read -p "Select (1-3): " whisper_choice

case $whisper_choice in
    1) WHISPER_MODEL="tiny" ;;
    3) WHISPER_MODEL="small" ;;
    *) WHISPER_MODEL="base" ;;
esac

echo "Pre-downloading Whisper $WHISPER_MODEL model..."
python3 << EOF
import whisper
try:
    whisper.load_model("$WHISPER_MODEL")
    print("✓ Whisper model downloaded")
except Exception as e:
    print(f"⚠️  Download failed: {e}")
EOF

echo ""
echo "=========================================="
echo "Creating Configuration"
echo "=========================================="
echo ""

# Create .env file
cat > .env << EOF
# Model Backend Configuration
USE_LOCAL_MODELS=true

# Local Model Configuration
WHISPER_MODEL=$WHISPER_MODEL
GEMMA_MODEL=google/gemma-3-4b-it

# TTS Configuration
TTS_ENGINE=$TTS_ENGINE

# HuggingFace Configuration
HF_HOME=./cache
HF_TOKEN=$HF_TOKEN

# Custom personality
REACHY_MINI_CUSTOM_PROFILE=
EOF

echo "✓ Configuration file created (.env)"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Gemma 3 VLM: google/gemma-3-4b-it"
echo "  - Whisper: $WHISPER_MODEL"
echo "  - TTS: $TTS_ENGINE"
if [ -n "$HF_TOKEN" ]; then
    echo "  - HF Token: configured"
else
    echo "  - HF Token: NOT SET (required!)"
    echo ""
    echo "⚠️  IMPORTANT: Set your HuggingFace token before running:"
    echo "    export HF_TOKEN='hf_...'"
    echo "    OR add it to .env file"
fi
echo ""
echo "To run the app:"
echo "  python -m reachy_mini_conversation_app.main"
echo ""
echo "With camera and face tracking:"
echo "  python -m reachy_mini_conversation_app.main --head-tracker yolo"
echo ""
echo "With Gradio UI:"
echo "  python -m reachy_mini_conversation_app.main --gradio"
echo ""
echo "For more info, see GEMMA3_SETUP_GUIDE.md"
echo ""