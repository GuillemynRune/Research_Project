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

class OllamaGemmaTester:
    """Gemma tester using Ollama API."""
    
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.conversation_history = []
        self.base_url = "http://localhost:11434"
        
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
    
    async def generate_response(self, user_message: str, image: Image.Image = None) -> tuple[str, list]:
        """Generate response using Ollama API."""
        
        # System prompt - no emojis, enforce tools
        system_prompt = """You are Reachy, a friendly robot assistant. Follow these rules STRICTLY:

1. Your name is Reachy (NOT Gemma)
2. Keep responses SHORT (1-2 sentences max)
3. NO EMOJIS - you're a robot, not a cartoon
4. When asked to do physical actions, you MUST use the tool format
5. Don't describe actions - just use the tool

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

User: "Look left"  
Reachy: "Looking left now. [TOOL:move_head:{"direction":"left"}]"

User: "Tell me a joke"
Reachy: "Why did the robot cross the playground? To get to the other slide!"

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
                            "num_predict": 75,
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


async def test_vision():
    """Test 2: Vision capabilities."""
    print("\n" + "="*60)
    print("TEST 2: Vision Test")
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
    import cv2
    cv2.rectangle(img_array, (50, 50), (200, 200), (255, 0, 0), -1)  # Blue square
    cv2.circle(img_array, (400, 120), 80, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(img_array, (450, 300), (600, 450), (0, 0, 255), -1)  # Red rectangle
    
    # Convert to PIL Image (RGB)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    
    vision_questions = [
        "What do you see in this image?",
        "What colors are present?",
    ]
    
    for msg in vision_questions:
        print(f"\n👤 User: {msg}")
        start = time.time()
        response, tools = await tester.generate_response(msg, image=image)
        elapsed = time.time() - start
        print(f"🤖 Reachy: {response}")
        print(f"⏱️  {elapsed:.1f}s")


async def test_interactive():
    """Test 3: Interactive mode."""
    print("\n" + "="*60)
    print("TEST 3: Interactive Mode")
    print("="*60)
    print("Type your messages (or 'quit' to exit)")
    print("="*60)
    
    tester = OllamaGemmaTester(model="gemma3:4b")
    
    if not await tester.load_model():
        print("Model not available!")
        return
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        response, tools = await tester.generate_response(user_input)
        print(f"🤖 Reachy: {response}")
        
        if tools:
            print(f"🛠️  Tools: {json.dumps(tools, indent=2)}")


async def main():
    """Main menu."""
    print("\n" + "="*60)
    print("OLLAMA GEMMA 3 TESTER")
    print("="*60)
    print("\nSelect test:")
    print("1. Text conversation (automatic)")
    print("2. Vision test (automatic)")
    print("3. Interactive mode")
    print("4. Run all tests")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        await test_text_conversation()
    elif choice == "2":
        await test_vision()
    elif choice == "3":
        await test_interactive()
    elif choice == "4":
        await test_text_conversation()
        await test_vision()
        await test_interactive()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")