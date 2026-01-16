"""Gemma via Ollama - Much faster than transformers!"""

import asyncio
import json
import re
import time
import httpx
from PIL import Image
import base64
import io
import numpy as np
import cv2

class OllamaGemmaTester:
    """Gemma tester using Ollama API."""
    
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.conversation_history = []
        self.base_url = "http://localhost:11434"
        self.camera = None
        
    async def load_model(self):
        """Check if Ollama is running and model is available."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
                models = response.json().get("models", [])
                
                model_names = [m["name"] for m in models]
                if self.model not in model_names:
                    print(f"⚠️  Model {self.model} not found. Available models:")
                    for name in model_names:
                        print(f"   - {name}")
                    return False
                
                print(f"✓ Ollama running with {self.model}")
                return True
                
            except Exception as e:
                print(f"❌ Ollama not running: {e}")
                return False
    
    def initialize_camera(self):
        """Initialize Reachy camera (index 2)."""
        try:
            self.camera = cv2.VideoCapture(2)  # Reachy camera
            if not self.camera.isOpened():
                print(f"❌ Could not open Reachy camera (index 2)")
                return False
            
            # Set resolution (adjust as needed)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print(f"✓ Reachy camera initialized")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Image.Image | None:
        """Capture a frame from the camera."""
        if self.camera is None or not self.camera.isOpened():
            print("❌ Camera not initialized")
            return None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Failed to capture frame")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            return image
            
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
            return None
    
    def release_camera(self):
        """Release camera resources."""
        if self.camera is not None:
            self.camera.release()
            print("✓ Camera released")
    
    def _extract_tool_calls(self, text: str) -> list:
        """Extract tool calls from response."""
        tool_calls = []
        pattern = r'\[TOOL:(\w+):(.*?)\]'
        matches = re.findall(pattern, text)
        
        for tool_name, args_str in matches:
            try:
                args = json.loads(args_str) if args_str else {}
                tool_calls.append({
                    "tool": tool_name,
                    "arguments": args
                })
            except json.JSONDecodeError:
                pass
        
        return tool_calls
    
    async def generate_response(self, user_message: str, image: Image.Image = None, use_camera: bool = False) -> tuple[str, list]:
        """Generate response using Ollama API.
        
        Args:
            user_message: The text message from user
            image: Optional PIL Image to include
            use_camera: If True, capture from camera (overrides image parameter)
        """
        
        # Capture from camera if requested
        if use_camera:
            image = self.capture_frame()
            if image is None:
                return "I couldn't access the camera.", []
        
        # System prompt - no emojis, enforce tools
        system_prompt = """You are Reachy, a friendly robot assistant. Follow these rules STRICTLY:

1. Your name is Reachy (NOT Gemma)
2. Keep responses SHORT (1-2 sentences max)
3. NO EMOJIS - you're a robot, not a cartoon
4. When asked to do physical actions, you MUST use the tool format
5. Don't describe actions - just use the tool
6. When you see an image, describe what you see concisely

TOOL FORMAT (use EXACTLY this format):
[TOOL:tool_name:{"param":"value"}]

AVAILABLE TOOLS:
- dance: [TOOL:dance:{"move":"random","repeat":1}]
- move_head: [TOOL:move_head:{"direction":"left"}]
- play_emotion: [TOOL:play_emotion:{"emotion":"happy"}]

CORRECT EXAMPLES:
User: "Hello!"
Reachy: "Hi! I'm Reachy! [TOOL:play_emotion:{"emotion":"happy"}]"

User: "Dance for me"
Reachy: "I'd love to dance! [TOOL:dance:{"move":"random","repeat":1}]"

User: "What do you see?" (with image)
Reachy: "I see a person sitting at a desk with a laptop."

WRONG (don't do this):
- Using emojis
- Describing actions instead of using tools
- Saying you're "Gemma"
- Long explanations"""
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 3 pairs)
        history_to_use = self.conversation_history[-6:]
        for msg in history_to_use:
            messages.append(msg)
        
        # Add current user message
        if image is not None:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            messages.append({
                "role": "user",
                "content": user_message,
                "images": [img_str]
            })
        else:
            messages.append({
                "role": "user",
                "content": user_message
            })
        
        # Call Ollama API
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "system": system_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 100,  # Increased for vision responses
                            "top_k": 50,
                            "top_p": 0.9,
                        }
                    }
                )
                
                result = response.json()
                response_text = result["message"]["content"].strip()
                
                # Extract tool calls
                tool_calls = self._extract_tool_calls(response_text)
                
                # Update history
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                return response_text, tool_calls
                
            except Exception as e:
                return f"Error: {e}", []


async def test_text_conversation():
    """Test 1: Text-only conversation."""
    print("\n" + "="*60)
    print("TEST 1: Text Conversation")
    print("="*60)
    
    tester = OllamaGemmaTester(model="gemma3:4b")
    
    if not await tester.load_model():
        print("Model not available!")
        return
    
    test_messages = [
        "Hello! What's your name?",
        "Can you dance for me?",
        "Tell me a joke about robots",
        "Move your head to the left",
        "Hey Richie Danes!",
    ]
    
    for msg in test_messages:
        print(f"\n👤 User: {msg}")
        start = time.time()
        response, tools = await tester.generate_response(msg)
        elapsed = time.time() - start
        print(f"🤖 Reachy: {response}")
        print(f"⏱️  {elapsed:.1f}s")
        
        if tools:
            print(f"🛠️  Tools: {json.dumps(tools, indent=2)}")


async def test_camera_vision():
    """Test 2: Real camera vision test."""
    print("\n" + "="*60)
    print("TEST 2: Camera Vision Test")
    print("="*60)
    
    tester = OllamaGemmaTester(model="gemma3:4b")
    
    if not await tester.load_model():
        print("Model not available!")
        return
    
    # Initialize Reachy camera
    if not tester.initialize_camera():
        print("❌ Could not initialize Reachy camera!")
        return
    
    try:
        vision_questions = [
            "What do you see in front of you?",
            "Describe what's in the image.",
            "What colors do you see?",
            "Is there a person in view?",
        ]
        
        for msg in vision_questions:
            print(f"\n👤 User: {msg}")
            print("📸 Capturing from Reachy camera...")
            
            start = time.time()
            response, tools = await tester.generate_response(msg, use_camera=True)
            elapsed = time.time() - start
            
            print(f"🤖 Reachy: {response}")
            print(f"⏱️  {elapsed:.1f}s")
            
            await asyncio.sleep(1)  # Small delay between captures
    
    finally:
        tester.release_camera()


async def test_synthetic_vision():
    """Test 3: Vision with synthetic image."""
    print("\n" + "="*60)
    print("TEST 3: Synthetic Vision Test")
    print("="*60)
    
    tester = OllamaGemmaTester(model="gemma3:4b")
    
    if not await tester.load_model():
        print("Model not available!")
        return
    
    # Create a test image (colorful pattern)
    width, height = 640, 480
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background gradient
    for y in range(height):
        img_array[y, :] = [y // 2, 128, 255 - y // 2]
    
    # Add shapes
    cv2.rectangle(img_array, (50, 50), (200, 200), (255, 0, 0), -1)  # Blue square
    cv2.circle(img_array, (400, 120), 80, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(img_array, (450, 300), (600, 450), (0, 0, 255), -1)  # Red rectangle
    
    # Convert to PIL Image (RGB)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    
    vision_questions = [
        "What do you see in this image?",
        "What shapes are present?",
        "What colors can you identify?",
    ]
    
    for msg in vision_questions:
        print(f"\n👤 User: {msg}")
        start = time.time()
        response, tools = await tester.generate_response(msg, image=image)
        elapsed = time.time() - start
        print(f"🤖 Reachy: {response}")
        print(f"⏱️  {elapsed:.1f}s")


async def test_interactive():
    """Test 4: Interactive mode with camera option."""
    print("\n" + "="*60)
    print("TEST 4: Interactive Mode")
    print("="*60)
    print("Type your messages (or 'quit' to exit)")
    print("Type 'camera' to use camera for next question")
    print("="*60)
    
    tester = OllamaGemmaTester(model="gemma3:4b")
    
    if not await tester.load_model():
        print("Model not available!")
        return
    
    # Initialize Reachy camera
    camera_available = tester.initialize_camera()
    
    if not camera_available:
        print("⚠️ Camera not available - text-only mode")
    
    try:
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Check if user wants to use camera
            use_camera = user_input.lower() == 'camera'
            
            if use_camera and camera_available:
                follow_up = input("👤 Question with camera: ").strip()
                if not follow_up:
                    continue
                print("📸 Capturing from Reachy camera...")
                response, tools = await tester.generate_response(follow_up, use_camera=True)
            else:
                response, tools = await tester.generate_response(user_input)
            
            print(f"🤖 Reachy: {response}")
            
            if tools:
                print(f"🛠️  Tools: {json.dumps(tools, indent=2)}")
    
    finally:
        if camera_available:
            tester.release_camera()


async def main():
    """Main menu."""
    print("\n" + "="*60)
    print("OLLAMA GEMMA 3 TESTER (with Vision)")
    print("="*60)
    print("\nSelect test:")
    print("1. Text conversation (automatic)")
    print("2. Camera vision test (automatic)")
    print("3. Synthetic vision test (automatic)")
    print("4. Interactive mode (with camera option)")
    print("5. Run all tests")
    
    choice = input("\nChoice (1-5): ").strip()
    
    if choice == "1":
        await test_text_conversation()
    elif choice == "2":
        await test_camera_vision()
    elif choice == "3":
        await test_synthetic_vision()
    elif choice == "4":
        await test_interactive()
    elif choice == "5":
        await test_text_conversation()
        await test_camera_vision()
        await test_synthetic_vision()
        await test_interactive()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")