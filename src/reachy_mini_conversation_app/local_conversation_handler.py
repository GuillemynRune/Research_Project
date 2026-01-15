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

import cv2
import numpy as np
from numpy.typing import NDArray
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from scipy.signal import resample

# Local model imports
import torch
#import whisper
from transformers import AutoProcessor, AutoModelForCausalLM
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

    def copy(self) -> "LocalConversationHandler":
        """Create a copy of the handler."""
        return LocalConversationHandler(self.deps, self.gradio_mode, self.instance_path)

    async def _load_models(self) -> None:
        """Load Whisper, Gemma 3 VLM, and initialize TTS."""
        try:
            # Load Faster-Whisper model
            logger.info("Loading Faster-Whisper model...")
            from faster_whisper import WhisperModel
            self.whisper_model = await asyncio.to_thread(
                WhisperModel,
                "base",  # or "small", "medium" - base is fastest
                device="cuda",
                compute_type="float16"
            )
            logger.info("Faster-Whisper model loaded")

            # Load Gemma 3 VLM
            logger.info(f"Loading Gemma 3 VLM: {self.gemma_model_name} on {self.device}")
            
            def load_gemma():
                from huggingface_hub import login
                import os

                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    login(token=hf_token)

                processor = AutoProcessor.from_pretrained(self.gemma_model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.gemma_model_name,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                )
                if self.device == "cpu":
                    model = model.to("cpu")
                return processor, model
            
            self.gemma_processor, self.gemma_model = await asyncio.to_thread(load_gemma)
            logger.info(f"Gemma 3 VLM loaded on {self.device}")

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

            # Step 4: Execute tool calls if any
            if tool_calls:
                await self._execute_tool_calls(tool_calls)

                # Get follow-up response after tool execution
                response_text, _ = await self._get_llm_response(
                    "Based on the tool results, respond to the user.",
                    None,
                    intent,
                    entities
                )

            # Step 5: Emit assistant response
            if response_text:
                response_text = self._clean_response_text(response_text)

                logger.info(f"Assistant: {response_text}")
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": response_text})
                )

                # Step 6: Generate speech (TTS)
                await self._generate_speech(response_text)

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
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
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
        wake_words = ["reachy", "richie", "reach", "reiki", "ricky"]
        has_wake_word = any(word in transcript_lower for word in wake_words)
        
        if not has_wake_word:
            logger.info(f"⏭️  No wake word: '{transcript}' - IGNORED")
            return ""
        
        logger.info(f"✅ Wake word detected: '{transcript}'")
        return transcript

    async def _get_llm_response(
        self, 
        user_message: str, 
        image: Optional[Image.Image] = None,
        intent: Optional[str] = None,
        entities: Optional[dict] = None
    ) -> Tuple[str, list]:
        """Get response from Gemma 3 VLM with optional image input."""
        from reachy_mini_conversation_app.prompts import get_session_instructions
        from reachy_mini_conversation_app.tools.core_tools import get_tool_specs
        
        # Build system prompt with tool definitions
        system_prompt = get_session_instructions()
        
        tool_specs = get_tool_specs()
        if tool_specs:
            tools_description = "\n\nAVAILABLE TOOLS:\n"
            for tool in tool_specs:
                name = tool.get("name", "")
                desc = tool.get("description", "")
                params = tool.get("parameters", {}).get("properties", {})
                
                tools_description += f"\n{name}:\n"
                tools_description += f"  Description: {desc}\n"
                tools_description += f"  Parameters: {list(params.keys())}\n"
            
            tools_description += """
TO USE A TOOL, include in your response:
[TOOL:tool_name:{"param1":"value1","param2":"value2"}]

Example: "I'll dance for you! [TOOL:dance:{"move":"random","repeat":1}]"
"""
            system_prompt += tools_description
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (keep last 20)
        for msg in self.conversation_history[-20:]:
            messages.append(msg)
        
        # Add current user message with optional image
        if image is not None:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_message}
                ]
            })
        else:
            messages.append({"role": "user", "content": user_message})

        try:
            def generate_response():
                # FIXED: Properly handle text-only vs text+image messages
                if image is not None:
                    # For image inputs, use the processor directly
                    text_content = []
                    for msg in messages:
                        if msg["role"] == "system":
                            text_content.append(msg["content"])
                        elif msg["role"] == "user":
                            if isinstance(msg["content"], str):
                                text_content.append(msg["content"])
                            elif isinstance(msg["content"], list):
                                for item in msg["content"]:
                                    if item.get("type") == "text":
                                        text_content.append(item["text"])
                        elif msg["role"] == "assistant":
                            text_content.append(msg["content"])
                    
                    prompt_text = "\n".join(text_content)
                    
                    inputs = self.gemma_processor(
                        text=prompt_text,
                        images=image,
                        return_tensors="pt",
                        padding=True
                    )
                else:
                    # For text-only, use chat template
                    # But extract text content properly
                    text_messages = []
                    for msg in messages:
                        if isinstance(msg.get("content"), str):
                            text_messages.append(msg)
                        elif isinstance(msg.get("content"), list):
                            # Extract text from list content
                            text_parts = [item["text"] for item in msg["content"] if item.get("type") == "text"]
                            if text_parts:
                                text_messages.append({
                                    "role": msg["role"],
                                    "content": " ".join(text_parts)
                                })
                    
                    # Apply chat template to text-only messages
                    prompt_text = self.gemma_processor.tokenizer.apply_chat_template(
                        text_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    inputs = self.gemma_processor(
                        text=prompt_text,
                        return_tensors="pt",
                        padding=True
                    )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.'))
                    
                    gen_kwargs = {
                        **inputs,
                        "max_new_tokens": 200,
                        "temperature": 0.7,
                        "do_sample": True,
                        "pad_token_id": self.gemma_processor.tokenizer.eos_token_id,
                    }
                    
                    # Only add attention_mask if PyTorch >= 2.6
                    if torch_version >= (2, 6, 0):
                        # These features require PyTorch 2.6+
                        pass
                    
                    outputs = self.gemma_model.generate(**gen_kwargs)
                 
                
                response = self.gemma_processor.decode(outputs[0], skip_special_tokens=True)
                return response
            
            full_response = await asyncio.to_thread(generate_response)
            response_text = self._extract_assistant_response(full_response, messages)
            tool_calls = self._extract_tool_calls(response_text)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return response_text, tool_calls

        except Exception as e:
            logger.error(f"Gemma 3 error: {e}")
            return "I'm having trouble thinking right now.", []
    
    def _extract_assistant_response(self, full_text: str, messages: list) -> str:
        """Extract only the new assistant response."""
        last_user_msg = messages[-1]["content"]
        if isinstance(last_user_msg, list):
            last_user_msg = next((item["text"] for item in last_user_msg if item["type"] == "text"), "")
        
        markers = ["assistant\n", "Assistant:", "<assistant>", "\n\n"]
        
        for marker in markers:
            if marker in full_text:
                parts = full_text.split(marker)
                response = parts[-1].strip()
                if response and len(response) > 5:
                    return response
        
        if last_user_msg in full_text:
            idx = full_text.rindex(last_user_msg)
            response = full_text[idx + len(last_user_msg):].strip()
            if response:
                return response
        
        return full_text.strip()
    
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