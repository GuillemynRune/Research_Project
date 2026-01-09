"""Local conversation handler using Whisper (STT) + Gemma (LLM) + TTS.

Replaces OpenAI's realtime API with local models while maintaining
the same interface for compatibility with the existing robot control system.
"""

import json
import base64
import asyncio
import logging
from typing import Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from numpy.typing import NDArray
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from scipy.signal import resample

# Local model imports
import torch
import whisper
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image


logger = logging.getLogger(__name__)

# Audio configuration
INPUT_SAMPLE_RATE = 16000  # Whisper expects 16kHz
OUTPUT_SAMPLE_RATE = 24000  # TTS output rate


class LocalConversationHandler(AsyncStreamHandler):
    """Local conversation handler using Whisper + Gemma + local TTS."""

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
        self.gemma_model_name = "google/gemma-3-4b-it"  # Gemma 3 VLM
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Conversation state
        self.conversation_history: list = []
        self.is_processing = False
        self.last_activity_time = asyncio.get_event_loop().time()
        
        # Audio buffering for VAD
        self.audio_buffer: list = []
        self.buffer_duration_s = 0.0
        self.silence_threshold = 0.02  # RMS threshold for silence detection
        self.min_speech_duration = 0.5  # Minimum speech duration in seconds
        self.max_speech_duration = 10.0  # Maximum speech duration
        self.silence_duration_to_stop = 0.8  # Seconds of silence to stop recording

        # Internal flags
        self._shutdown_requested = False
        self._processing_task: Optional[asyncio.Task] = None

    def copy(self) -> "LocalConversationHandler":
        """Create a copy of the handler."""
        return LocalConversationHandler(self.deps, self.gradio_mode, self.instance_path)

    async def _load_models(self) -> None:
        """Load Whisper and Gemma 3 VLM."""
        try:
            # Load Whisper model (run in thread to avoid blocking)
            logger.info("Loading Whisper model...")
            self.whisper_model = await asyncio.to_thread(
                whisper.load_model, "base"  # Options: tiny, base, small, medium, large
            )
            logger.info("Whisper model loaded")

            # Load Gemma 3 VLM
            logger.info(f"Loading Gemma 3 VLM: {self.gemma_model_name} on {self.device}")
            
            def load_gemma():
                from huggingface_hub import login
                import os

                # Login with token
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
            logger.error("Make sure to: pip install transformers torch pillow")
            raise

    async def start_up(self) -> None:
        """Start the handler and load models."""
        await self._load_models()
        
        # Start processing loop
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Local conversation handler started")

    async def _processing_loop(self) -> None:
        """Main processing loop for handling audio and generating responses."""
        while not self._shutdown_requested:
            try:
                # Check if we have enough audio buffered
                if self.buffer_duration_s < self.min_speech_duration:
                    await asyncio.sleep(0.1)
                    continue

                # Check for silence to detect end of speech
                if await self._detect_end_of_speech():
                    await self._process_speech()
                    
                await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)

    async def _detect_end_of_speech(self) -> bool:
        """Detect if user has stopped speaking based on silence."""
        if len(self.audio_buffer) < 10:
            return False

        # Check last N frames for silence
        recent_frames = self.audio_buffer[-10:]
        silence_count = sum(1 for frame in recent_frames if self._is_silence(frame))
        
        # If most recent frames are silent, consider speech ended
        return silence_count >= 8

    def _is_silence(self, audio_chunk: NDArray[np.int16]) -> bool:
        """Check if audio chunk is silence based on RMS threshold."""
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
        return rms < self.silence_threshold

    async def _process_speech(self) -> None:
        """Process buffered speech through Whisper -> Gemma 3 VLM -> TTS pipeline."""
        if self.is_processing or not self.audio_buffer:
            return

        self.is_processing = True
        
        try:
            # Concatenate audio buffer
            audio_data = np.concatenate(self.audio_buffer)
            self.audio_buffer.clear()
            self.buffer_duration_s = 0.0

            # Inform movement manager that user is speaking
            self.deps.movement_manager.set_listening(True)

            # Step 1: Speech-to-Text with Whisper
            logger.info("Transcribing speech...")
            transcript = await self._transcribe_audio(audio_data)
            
            if not transcript or len(transcript.strip()) < 3:
                logger.debug("Transcript too short, ignoring")
                self.deps.movement_manager.set_listening(False)
                return

            logger.info(f"User said: {transcript}")
            await self.output_queue.put(
                AdditionalOutputs({"role": "user", "content": transcript})
            )

            # Step 2: Check if this needs camera/vision input
            image = None
            needs_vision = any(word in transcript.lower() for word in 
                             ["see", "look", "camera", "what", "show", "picture", "view"])
            
            if needs_vision and self.deps.camera_worker is not None:
                logger.info("Vision request detected, capturing image...")
                frame = self.deps.camera_worker.get_latest_frame()
                if frame is not None:
                    # Convert BGR to RGB and create PIL Image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb_frame)
                    
                    # Show image in UI
                    await self.output_queue.put(
                        AdditionalOutputs({
                            "role": "assistant",
                            "content": image,
                        })
                    )

            # Step 3: Get LLM response with optional image
            logger.info("Generating response with Gemma 3 VLM...")
            response_text, tool_calls = await self._get_llm_response(transcript, image)
            
            # Step 4: Execute tool calls if any
            if tool_calls:
                await self._execute_tool_calls(tool_calls)
                
                # Get follow-up response after tool execution
                # (tool results have been added to conversation)
                response_text, _ = await self._get_llm_response(
                    "Based on the tool results, respond to the user.",
                    None
                )

            # Step 5: Emit assistant response
            if response_text:
                # Clean up any tool call markers from response
                response_text = self._clean_response_text(response_text)
                
                logger.info(f"Assistant: {response_text}")
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": response_text})
                )

                # Step 6: Generate speech (TTS)
                await self._generate_speech(response_text)

            self.deps.movement_manager.set_listening(False)

        except Exception as e:
            logger.error(f"Error processing speech: {e}")
        finally:
            self.is_processing = False
    
    def _clean_response_text(self, text: str) -> str:
        """Remove tool call markers from response text."""
        import re
        # Remove [TOOL:name:{args}] patterns
        cleaned = re.sub(r'\[TOOL:\w+:.*?\]', '', text)
        return cleaned.strip()

    async def _transcribe_audio(self, audio_data: NDArray[np.int16]) -> str:
        """Transcribe audio using Whisper."""
        # Convert to float32 in range [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Whisper expects 16kHz mono audio
        result = await asyncio.to_thread(
            self.whisper_model.transcribe,
            audio_float,
            language="en",
            fp16=False  # Use fp32 for CPU
        )
        
        return result["text"].strip()

    async def _get_llm_response(self, user_message: str, image: Optional[Image.Image] = None) -> Tuple[str, list]:
        """Get response from Gemma 3 VLM with optional image input."""
        from reachy_mini_conversation_app.prompts import get_session_instructions
        
        # Build conversation
        system_prompt = get_session_instructions()
        
        # Gemma 3 uses chat format with optional images
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add conversation history (keep last 10 exchanges)
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
            messages.append({
                "role": "user",
                "content": user_message
            })

        try:
            # Process with Gemma 3
            def generate_response():
                # Apply chat template
                inputs = self.gemma_processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
                
                # Generate response
                with torch.no_grad():
                    outputs = self.gemma_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.gemma_processor.tokenizer.eos_token_id,
                    )
                
                # Decode response
                response = self.gemma_processor.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                return response
            
            # Run generation in thread to not block event loop
            full_response = await asyncio.to_thread(generate_response)
            
            # Extract just the assistant's response (after the last prompt)
            # Gemma includes the full conversation in output, extract new part
            response_text = self._extract_assistant_response(full_response, messages)
            
            # Parse for tool calls (simple pattern matching)
            tool_calls = self._extract_tool_calls(response_text)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return response_text, tool_calls

        except Exception as e:
            logger.error(f"Gemma 3 error: {e}")
            return "I'm having trouble thinking right now.", []
    
    def _extract_assistant_response(self, full_text: str, messages: list) -> str:
        """Extract only the new assistant response from full generated text."""
        # Find the last user message
        last_user_msg = messages[-1]["content"]
        if isinstance(last_user_msg, list):
            last_user_msg = next((item["text"] for item in last_user_msg if item["type"] == "text"), "")
        
        # Split on common assistant markers
        markers = ["assistant\n", "Assistant:", "<assistant>", "\n\n"]
        
        for marker in markers:
            if marker in full_text:
                parts = full_text.split(marker)
                # Get the last part which should be the new response
                response = parts[-1].strip()
                if response and len(response) > 5:
                    return response
        
        # Fallback: try to find text after the user message
        if last_user_msg in full_text:
            idx = full_text.rindex(last_user_msg)
            response = full_text[idx + len(last_user_msg):].strip()
            if response:
                return response
        
        # Last resort: return cleaned full text
        return full_text.strip()
    
    def _extract_tool_calls(self, text: str) -> list:
        """Extract tool calls from response text using pattern matching.
        
        Expected format in response:
        "Let me dance for you! [TOOL:dance:{"move":"random"}]"
        or
        "I'll take a look [TOOL:camera:{"question":"what do you see"}]"
        """
        import re
        
        tool_calls = []
        
        # Pattern: [TOOL:tool_name:{"arg":"value"}]
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
        """Execute tool calls and add results to conversation."""
        from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("function", {}).get("arguments", {})
            
            if not tool_name:
                continue

            try:
                # Execute tool
                logger.info(f"Executing tool: {tool_name}")
                args_json = json.dumps(tool_args) if isinstance(tool_args, dict) else tool_args
                result = await dispatch_tool_call(tool_name, args_json, self.deps)
                
                # Emit tool result to UI
                await self.output_queue.put(
                    AdditionalOutputs({
                        "role": "assistant",
                        "content": json.dumps(result),
                        "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"}
                    })
                )
                
                # Add tool result to conversation for Gemma to see
                self.conversation_history.append({
                    "role": "system",
                    "content": f"Tool {tool_name} result: {json.dumps(result)}"
                })

            except Exception as e:
                logger.error(f"Tool execution error ({tool_name}): {e}")

    async def _generate_speech(self, text: str) -> None:
        """Generate speech using local TTS (placeholder - needs TTS implementation)."""
        # TODO: Integrate local TTS like:
        # - piper-tts
        # - coqui-tts
        # - bark
        
        # For now, log that we would speak
        logger.info(f"[TTS] Would speak: {text}")
        
        # Placeholder: generate silent audio to maintain timing
        # In real implementation, this would be actual TTS output
        duration_s = len(text.split()) * 0.3  # Rough estimate
        samples = int(OUTPUT_SAMPLE_RATE * duration_s)
        silence = np.zeros(samples, dtype=np.int16)
        
        # Send to output queue in chunks
        chunk_size = 4800  # 200ms chunks
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i:i + chunk_size]
            await self.output_queue.put((OUTPUT_SAMPLE_RATE, chunk.reshape(1, -1)))
            await asyncio.sleep(0.05)

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from microphone and buffer it."""
        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample to 16kHz for Whisper
        if input_sample_rate != INPUT_SAMPLE_RATE:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * INPUT_SAMPLE_RATE / input_sample_rate)
            )

        # Cast to int16
        audio_frame = audio_to_int16(audio_frame)

        # Buffer audio
        self.audio_buffer.append(audio_frame)
        chunk_duration = len(audio_frame) / INPUT_SAMPLE_RATE
        self.buffer_duration_s += chunk_duration

        # Limit buffer size
        if self.buffer_duration_s > self.max_speech_duration:
            # Remove oldest chunk
            removed = self.audio_buffer.pop(0)
            self.buffer_duration_s -= len(removed) / INPUT_SAMPLE_RATE

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio or messages to be played/displayed."""
        # Check for idle behavior
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            await self._send_idle_signal()
            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)  # type: ignore

    async def _send_idle_signal(self) -> None:
        """Trigger idle behavior when no activity."""
        logger.info("Idle timeout - triggering idle behavior")
        
        # Use tool to trigger an idle action (dance, emotion, etc.)
        try:
            from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call
            
            # Randomly pick an idle behavior
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

        # Clear queues
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
            
            # Clear conversation history to start fresh with new personality
            self.conversation_history.clear()
            
            logger.info(f"Applied personality: {profile or 'default'}")
            return f"Personality applied: {profile or 'built-in default'}"
            
        except Exception as e:
            logger.error(f"Error applying personality: {e}")
            return f"Failed to apply personality: {e}"

    async def get_available_voices(self) -> list[str]:
        """Get available TTS voices (placeholder)."""
        # TODO: Implement when TTS is added
        return ["default"]