"""Local conversation handler using Whisper (STT) + Gemma 3 (VLM) + pyttsx3 (TTS).

FIXED VERSION with:
- Working audio processing loop
- Real TTS implementation  
- Intent classification
- Task management integration
- Proper tool calling
"""

import json
import asyncio
import logging
from typing import Any, Tuple, Optional
from datetime import datetime

import httpx
import base64
import io

import cv2
import numpy as np
from numpy.typing import NDArray
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from scipy.signal import resample

# Local model imports
import torch
#import whisper
from PIL import Image


logger = logging.getLogger(__name__)

# Audio configuration
INPUT_SAMPLE_RATE = 16000  # Whisper expects 16kHz
OUTPUT_SAMPLE_RATE = 24000  # TTS output rate


class LocalConversationHandler(AsyncStreamHandler):
    """Local conversation handler using Whisper + Gemma + TTS."""

    def __init__(self, deps: Any, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Queues for audio processing
        self.input_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]]]" = asyncio.Queue()
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        # Model references (lazy loaded)
        self.whisper_model: Optional[Any] = None
        self.gemma_processor: Optional[Any] = None
        self.gemma_model: Optional[Any] = None
        self.gemma_model_name = "google/gemma-3-4b-it"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.use_ollama = True
        self.ollama_model = "gemma3:4b"
        self.ollama_base_url = "http://localhost:11434"
        
        # Intent classifier
        from reachy_mini_conversation_app.intent_classifier import IntentClassifier
        self.intent_classifier = IntentClassifier()
        
        # Task manager
        from reachy_mini_conversation_app.task_manager import TaskManager
        self.task_manager = TaskManager()
        
        # Conversation state
        self.conversation_history: list = []
        self.is_processing = False
        self.last_activity_time = asyncio.get_event_loop().time()
        
        # Audio buffering for VADba
        self.audio_buffer: list = []
        self.buffer_duration_s = 0.0
        self.silence_threshold = 0.03  # RMS threshold for silence detection
        self.min_speech_duration = 0.5  # Minimum speech duration in seconds
        self.max_speech_duration = 10.0  # Maximum speech duration
        self.silence_duration_to_stop = 0.6  # Seconds of silence to stop recording
        self._silence_start: Optional[float] = None

        self._models_loaded = False

        # Internal flags
        self._shutdown_requested = False
        self._processing_task: Optional[asyncio.Task] = None

        self._cached_emotion_list: Optional[str] = None

    def copy(self) -> "LocalConversationHandler":
        """Create a copy of the handler."""
        return LocalConversationHandler(self.deps, self.gradio_mode, self.instance_path)

    async def _load_models(self) -> None:
        """Load Whisper and Gemma (either Ollama or transformers)."""
        try:
            # Load Faster-Whisper model
            logger.info("Loading Faster-Whisper model...")
            from faster_whisper import WhisperModel
            self.whisper_model = await asyncio.to_thread(
                WhisperModel,
                "small",
                device="cuda",
                compute_type="float16"
            )
            logger.info("Faster-Whisper model loaded")

            # Check if using Ollama or transformers
            if self.use_ollama:
                logger.info(f"Using Ollama for LLM: {self.ollama_model}")
                # Verify Ollama is running
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{self.ollama_base_url}/api/tags")
                        models = response.json().get("models", [])
                        model_names = [m["name"] for m in models]
                        
                        if self.ollama_model in model_names:
                            logger.info(f"✓ Ollama model {self.ollama_model} available")
                        else:
                            logger.warning(f"⚠️ Model {self.ollama_model} not found in Ollama")
                            logger.info(f"Available models: {model_names}")
                except Exception as e:
                    logger.error(f"❌ Ollama not accessible: {e}")
                    raise  # Don't fallback, just fail

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    async def start_up(self) -> None:
        """Start the handler and load models."""
        await self._load_models()
        
        # Start task manager
        await self.task_manager.start()
        
        self._models_loaded = True
        # Start processing loop
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Local conversation handler started")

    async def _processing_loop(self) -> None:
        """Main processing loop - monitors idle state."""
        while not self._shutdown_requested:
            try:
                # Check for idle behavior
                idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
                if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
                    await self._send_idle_signal()
                    self.last_activity_time = asyncio.get_event_loop().time()
                
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)

    def _is_silence(self, audio_chunk: NDArray[np.int16]) -> bool:
        """Check if audio chunk is silence based on RMS threshold."""
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Lower threshold = more sensitive (triggers on quieter sounds)
        # Higher threshold = less sensitive (only triggers on louder sounds)
        threshold = 0.02  # Adjust this if needed

        is_silent = rms < threshold
        
        # Log occasionally for debugging
        if np.random.random() < 0.01:  # Log 1% of the time
            logger.debug(f"Audio RMS: {rms:.4f} (threshold: {threshold}, silent: {is_silent})")
        
        return is_silent

    async def _process_speech(self) -> None:
        """Process buffered speech."""
        if self.is_processing or not self.audio_buffer:
            return
        
        if not self._models_loaded:
            logger.warning("⚠️ Models not loaded yet, skipping processing")
            self.audio_buffer.clear()
            return

        self.is_processing = True
        
        try:
            # Concatenate audio buffer
            audio_data = np.concatenate(self.audio_buffer)
            self.audio_buffer.clear()
            self.buffer_duration_s = 0.0
            self._silence_start = None

            # CHECK AUDIO LEVEL (prevents hallucinations!)
            audio_float = audio_data.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_float ** 2))
            
            if rms < 0.05:
                logger.info(f"🔇 Audio too quiet (RMS={rms:.4f}), skipping transcription")
                self.is_processing = False
                return
            
            logger.info(f"🎤 Processing audio (RMS={rms:.4f})")

            # Now safe to transcribe
            self.deps.movement_manager.set_listening(True)
            logger.info("Transcribing speech...")
            transcript = await self._transcribe_audio(audio_data)

            if not transcript or len(transcript.strip()) < 3:
                logger.debug("Transcript too short, ignoring")
                self.deps.movement_manager.set_listening(False)
                # After silence, return to callword mode
                self.awaiting_callword = True
                return

            logger.info(f"User said: {transcript}")
            await self.output_queue.put(
                AdditionalOutputs({"role": "user", "content": transcript})
            )

            # Step 1.5: Classify intent
            intent, confidence = self.intent_classifier.classify(transcript)
            logger.info(f"Intent: {intent} (confidence: {confidence:.2f})")

            entities = self.intent_classifier.extract_entities(transcript, intent)
            if entities:
                logger.info(f"Entities: {entities}")

            # Step 2: Check if this needs camera/vision input
            image = None
            needs_vision = intent == 'vision_query'

            if needs_vision and self.deps.camera_worker is not None:
                logger.info("Vision request detected, capturing image...")
                frame = self.deps.camera_worker.get_latest_frame()
                if frame is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb_frame)

                    await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": image})
                    )

            # Step 3: Get LLM response with optional image
            logger.info("Generating response with Gemma 3 VLM...")
            response_text, tool_calls = await self._get_llm_response(transcript, image, intent, entities)

            # Step 4: Emit assistant response FIRST (before executing tools)
            if response_text:
                response_text = self._clean_response_text(response_text)

                logger.info(f"Assistant: {response_text}")
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": response_text})
                )

                # Generate speech (TTS) - say it before doing the action
                await self._generate_speech(response_text)

            # Step 5: Execute tool calls AFTER speaking
            if tool_calls:
                await self._execute_tool_calls(tool_calls)
                # No follow-up response needed since we already spoke

            self.deps.movement_manager.set_listening(False)

        except Exception as e:
            logger.error(f"Error processing speech: {e}", exc_info=True)
            await self.output_queue.put(
                AdditionalOutputs({
                    "role": "assistant",
                    "content": "Sorry, I had trouble processing that. Could you repeat?"
                })
            )
        finally:
            self.is_processing = False
            # After processing, return to callword mode
            self.awaiting_callword = True
    
    def _clean_response_text(self, text: str) -> str:
        """Remove tool call markers from response text."""
        import re
        cleaned = re.sub(r'\[TOOL:\w+:.*?\]', '', text)
        return cleaned.strip()

    async def _transcribe_audio(self, audio_data: NDArray[np.int16]) -> str:
        """Transcribe audio using Faster-Whisper."""
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Normalize audio
        audio_float = audio_float / (np.max(np.abs(audio_float)) + 1e-8)
        
        def transcribe():
            segments, info = self.whisper_model.transcribe(
                audio_float,
                language="en",
                beam_size=3,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
                initial_prompt="Hey Reachy"
            )
            return " ".join([segment.text for segment in segments]).strip()
        
        transcript = await asyncio.to_thread(transcribe)
        
        # Require at least 3 words
        word_count = len(transcript.split())
        if word_count < 3:
            logger.info(f"⚠️ Too short ({word_count} words): '{transcript}'")
            return ""
        
        # WAKE WORD CHECK - Only process if "reachy" is mentioned
        transcript_lower = transcript.lower()
        wake_words = ["reachy", "richie", "reach", "reiki", "ricky", "reichy"]
        has_wake_word = any(word in transcript_lower for word in wake_words)

        if not has_wake_word:
            logger.info(f"⏭️  No wake word: '{transcript}' - IGNORED")
            return ""
        
        logger.info(f"✅ Wake word detected: '{transcript}'")
        return transcript
    
    def _clear_bad_history(self):
        """Clear conversation history if it contains narrative responses."""
        cleaned = []
        for msg in self.conversation_history:
            content = msg.get("content", "")
            # Keep only good responses (no narrative descriptions)
            if not ("(" in content and ")" in content and "He" in content):
                cleaned.append(msg)
        
        if len(cleaned) < len(self.conversation_history):
            logger.info(f"🧹 Cleared {len(self.conversation_history) - len(cleaned)} bad responses from history")
            self.conversation_history = cleaned

    async def _get_llm_response(
        self, 
        user_message: str, 
        image: Optional[Image.Image] = None,
        intent: Optional[str] = None,
        entities: Optional[dict] = None
    ) -> Tuple[str, list]:
        """Get response from Gemma with optional image input."""
        
        # Clear bad history first
        self._clear_bad_history()
        
        # Build system prompt dynamically based on available tools
        system_prompt = self._build_system_prompt()
        
        try:
            # Get response from Ollama
            response_text, tool_calls = await self._get_ollama_response(
                user_message, image, system_prompt
            )
            
            # ENFORCE rules in post-processing
            response_text = self._enforce_response_rules(response_text)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return response_text, tool_calls

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I'm having trouble thinking right now.", []

    def _build_system_prompt(self) -> str:
        """Build system prompt dynamically based on available tools."""
        from reachy_mini_conversation_app.tools.core_tools import ALL_TOOLS
        
        # Get available emotions (cached)
        emotion_list = "cheerful1, enthusiastic1, curious1, attentive1, grateful1"
        
        if self._cached_emotion_list is None:  # Only load once
            try:
                from reachy_mini.motion.recorded_move import RecordedMoves
                recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
                emotions = recorded_moves.list_moves()
                # Pick a few good ones for examples
                good_emotions = [e for e in emotions if any(x in e for x in ['cheerful', 'enthusiastic', 'curious', 'grateful', 'attentive'])]
                if good_emotions:
                    emotion_list = ", ".join(good_emotions[:5])
                self._cached_emotion_list = emotion_list  # Cache it
            except Exception as e:
                logger.warning(f"Could not load emotions dynamically: {e}")
                self._cached_emotion_list = emotion_list  # Use default
        else:
            emotion_list = self._cached_emotion_list  # Use cached version
        
        # Check which tools are available
        has_dance = "dance" in ALL_TOOLS
        has_move_head = "move_head" in ALL_TOOLS
        has_play_emotion = "play_emotion" in ALL_TOOLS
        
        # Build tools section
        tools_section = "AVAILABLE TOOLS:\n"
        
        if has_move_head:
            tools_section += "- move_head: [TOOL:move_head:{\"direction\":\"left\"}]\n"
            tools_section += "  Directions: left, right, up, down, front\n"
        
        if has_play_emotion:
            tools_section += f"- play_emotion: [TOOL:play_emotion:{{\"emotion\":\"{emotion_list.split(',')[0].strip()}\"}}]\n"
            tools_section += f"  Available emotions: {emotion_list}\n"
        
        if has_dance:
            tools_section += "- dance: [TOOL:dance:{\"move\":\"random\",\"repeat\":1}]\n"
        
        # Build examples based on available tools
        examples = []
        
        if has_play_emotion:
            first_emotion = emotion_list.split(',')[0].strip()
            examples.append(f'User: "Hello!"\nReachy: "Hi! I\'m Reachy! [TOOL:play_emotion:{{\"emotion\":\"{first_emotion}\"}}]"')
            examples.append(f'User: "How are you?"\nReachy: "I\'m doing great! [TOOL:play_emotion:{{\"emotion\":\"{first_emotion}\"}}]"')
        
        if has_dance:
            examples.append('User: "Dance for me"\nReachy: "I\'d love to dance! [TOOL:dance:{\\"move\\":\\"random\\",\\"repeat\\":1}]"')
        
        if has_move_head:
            examples.append('User: "Look left"\nReachy: "Looking left now. [TOOL:move_head:{\\"direction\\":\\"left\\"}]"')
        
        examples.append('User: "Tell me a joke"\nReachy: "Why did the robot cross the playground? To get to the other slide!"')
        
        examples_text = "\n\n".join(examples)
        
        return f"""You are Reachy, a friendly robot assistant. Follow these rules STRICTLY:

    1. Your name is Reachy (NOT Gemma)
    2. Keep responses SHORT (1-2 sentences max)
    3. NO EMOJIS - you're a robot, not a cartoon
    4. When asked to do physical actions, you MUST use the tool format
    5. Don't describe actions - just use the tool
    6. When you see an image, describe what you see concisely

    TOOL FORMAT (use EXACTLY this format):
    [TOOL:tool_name:{{"param":"value"}}]

    {tools_section}

    CORRECT EXAMPLES:
    {examples_text}

    WRONG (don't do this):
    - Using emojis
    - Describing actions instead of using tools
    - Saying you're "Gemma"
    - Long explanations
    - Using tools or emotions that don't exist"""
        
        
    async def _get_ollama_response(
        self,
        user_message: str,
        image: Optional[Image.Image],
        system_prompt: str
    ) -> Tuple[str, list]:
        """Get response from Ollama."""
        
        # Build messages - MATCH TEST FILE EXACTLY
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 3 pairs max)
        #history_to_use = self.conversation_history[-6:]
        #for msg in history_to_use:
        #    messages.append(msg)
        
        # Add current user message
        if image is not None:
            # Convert image to base64
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
        
        # Call Ollama API - EXACT SAME AS TEST FILE
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "system": system_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,  # ← MATCH TEST FILE
                            "num_predict": 75,   # ← MATCH TEST FILE
                            "top_k": 50,         # ← MATCH TEST FILE
                            "top_p": 0.9,        # ← MATCH TEST FILE
                        }
                    }
                )

                result = response.json()
                response_text = result["message"]["content"].strip()
                
                # Extract tool calls
                tool_calls = self._extract_tool_calls(response_text)
                
                return response_text, tool_calls
                
            except Exception as e:
                return f"Error: {e}", []
        
    def _enforce_response_rules(self, text: str) -> str:
        """Enforce strict response rules."""
        # Remove emojis
        import re
        text = re.sub(r'[😀-🙏🌀-🗿🚀-🛿☀-➿]', '', text)
        
        # Remove markdown
        text = text.replace('*', '').replace('-', '').replace('#', '')
        
        # If too long, truncate
        words = text.split()
        if len(words) > 15:
            text = ' '.join(words[:15]) + '...'
            logger.warning(f"⚠️ Truncated long response to 15 words")
        
        return text.strip()
        
    
    def _extract_tool_calls(self, text: str) -> list:
        """Extract tool calls from response text."""
        import re
        
        tool_calls = []
        pattern = r'\[TOOL:(\w+):(.*?)\]'
        matches = re.findall(pattern, text)
        
        for tool_name, args_str in matches:
            try:
                args = json.loads(args_str) if args_str else {}
                tool_calls.append({
                    "function": {
                        "name": tool_name,
                        "arguments": args
                    }
                })
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool args: {args_str}")
        
        return tool_calls

    async def _execute_tool_calls(self, tool_calls: list) -> None:
        """Execute tool calls including task management."""
        from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("function", {}).get("arguments", {})
            
            if not tool_name:
                continue

            try:
                # Handle task management tools
                if tool_name == "create_reminder":
                    message = tool_args.get("message", "Reminder")
                    delay = float(tool_args.get("delay_seconds", 60))
                    task_id = self.task_manager.create_reminder(message, delay)
                    result = {"task_id": task_id, "message": f"Reminder set for {delay}s"}
                
                elif tool_name == "create_timer":
                    duration = float(tool_args.get("duration_seconds", 60))
                    task_id = self.task_manager.create_timer(duration)
                    result = {"task_id": task_id, "message": f"Timer set for {duration}s"}
                
                else:
                    # Standard tool dispatch
                    logger.info(f"Executing tool: {tool_name}")
                    args_json = json.dumps(tool_args) if isinstance(tool_args, dict) else tool_args
                    result = await dispatch_tool_call(tool_name, args_json, self.deps)
                
                await self.output_queue.put(
                    AdditionalOutputs({
                        "role": "assistant",
                        "content": json.dumps(result),
                        "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"}
                    })
                )
                
                self.conversation_history.append({
                    "role": "system",
                    "content": f"Tool {tool_name} result: {json.dumps(result)}"
                })

            except Exception as e:
                logger.error(f"Tool execution error ({tool_name}): {e}")

    async def _generate_speech(self, text: str) -> None:
        """Generate speech using pyttsx3 TTS."""
        import pyttsx3
        import tempfile
        import wave
        import os
        
        def generate_audio():
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            with wave.open(temp_path, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
            
            os.remove(temp_path)
            return audio_data
        
        try:
            logger.info(f"[TTS] Speaking: {text}")
            audio_data = await asyncio.to_thread(generate_audio)
            
            # Send to output queue in chunks
            chunk_size = 4800
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await self.output_queue.put((OUTPUT_SAMPLE_RATE, chunk.reshape(1, -1)))
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame and trigger processing when speech detected."""
        if not self._models_loaded:
            return  # Don't buffer if models aren't ready
        
        input_sample_rate, audio_frame = frame

        # Reshape and resample
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        if input_sample_rate != INPUT_SAMPLE_RATE:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * INPUT_SAMPLE_RATE / input_sample_rate)
            )

        audio_frame = audio_to_int16(audio_frame)

        # Check if this chunk has speech
        is_silent = self._is_silence(audio_frame)
        
        # Buffer audio
        self.audio_buffer.append(audio_frame)
        chunk_duration = len(audio_frame) / INPUT_SAMPLE_RATE
        self.buffer_duration_s += chunk_duration

        # Limit buffer size
        if self.buffer_duration_s > self.max_speech_duration:
            removed = self.audio_buffer.pop(0)
            self.buffer_duration_s -= len(removed) / INPUT_SAMPLE_RATE
        
        # Speech detection logic
        if not is_silent:
            # Speech detected - reset silence timer
            self._silence_start = None
            logger.debug("🎤 Speech detected")
        else:
            # Silence detected
            if self.buffer_duration_s >= self.min_speech_duration:
                if self._silence_start is None:
                    self._silence_start = asyncio.get_event_loop().time()
                    logger.debug("🔇 Silence started")
                
                silence_duration = asyncio.get_event_loop().time() - self._silence_start
                
                # Trigger processing after enough silence
                if silence_duration >= self.silence_duration_to_stop and not self.is_processing:
                    logger.info(f"💬 Processing speech ({self.buffer_duration_s:.1f}s)")
                    asyncio.create_task(self._process_speech())

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio or messages."""
        #idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        #if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
        #    await self._send_idle_signal()
        #    self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)  # type: ignore

    async def _send_idle_signal(self) -> None:
        """Trigger idle behavior."""
        logger.info("Idle timeout - triggering idle behavior")
        
        try:
            from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call
            import random
            
            idle_tools = ["dance", "play_emotion", "do_nothing"]
            tool_name = random.choice(idle_tools)
            
            if tool_name == "dance":
                args = json.dumps({"move": "random", "repeat": 1})
            elif tool_name == "play_emotion":
                emotions = ["happy", "curious", "thinking"]
                args = json.dumps({"emotion": random.choice(emotions)})
            else:
                args = json.dumps({"reason": "just chilling"})
            
            await dispatch_tool_call(tool_name, args, self.deps)
            
        except Exception as e:
            logger.error(f"Idle behavior error: {e}")

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        await self.task_manager.stop()

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("Local conversation handler shutdown complete")

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality profile."""
        try:
            from reachy_mini_conversation_app.config import config, set_custom_profile
            
            set_custom_profile(profile)
            self.conversation_history.clear()
            
            logger.info(f"Applied personality: {profile or 'default'}")
            return f"Personality applied: {profile or 'built-in default'}"
            
        except Exception as e:
            logger.error(f"Error applying personality: {e}")
            return f"Failed to apply personality: {e}"

    async def get_available_voices(self) -> list[str]:
        """Get available TTS voices."""
        return ["default"]  # pyttsx3 uses system default