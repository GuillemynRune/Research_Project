"""Local conversation handler using Whisper (STT) + Gemma 3 (VLM) + pyttsx3 & elevenlabs (TTS).

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

import os


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

        self.tts_provider = os.getenv("TTS_PROVIDER", "pyttsx3")  # "pyttsx3" or "elevenlabs"
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel
        self.elevenlabs_model = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
        
        # Validate ElevenLabs setup if selected
        if self.tts_provider == "elevenlabs" and not self.elevenlabs_api_key:
            logger.warning("⚠️ ElevenLabs selected but no API key found. Falling back to pyttsx3")
            self.tts_provider = "pyttsx3"
        
        logger.info(f"TTS Provider: {self.tts_provider}")

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
        self.task_manager = TaskManager(tts_callback=self._speak_notification)

        self.deps.task_manager = self.task_manager
        
        # Conversation state
        self.conversation_history: list = []
        self.is_processing = False
        self.last_activity_time = asyncio.get_event_loop().time()
        
        # Audio buffering for VADba
        self.audio_buffer: list = []
        self.buffer_duration_s = 0.0
        self.silence_threshold = 0.020  # RMS threshold for silence detection
        self.min_speech_duration = 1.0  # Minimum speech duration in seconds
        self.max_speech_duration = 15.0  # Maximum speech duration
        self.silence_duration_to_stop = 1.2  # Seconds of silence to stop recording
        self._silence_start: Optional[float] = None

        self._noise_floor = 0.01  # Estimated background noise level
        self._recent_energy = []  # Track recent energy levels for adaptive threshold
        self._max_recent_samples = 30  # Number of samples to track

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
        """Check if audio chunk is silence using adaptive threshold."""
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Track recent energy levels for adaptive threshold
        self._recent_energy.append(rms)
        if len(self._recent_energy) > self._max_recent_samples:
            self._recent_energy.pop(0)
        
        # Adaptive threshold: noise floor + margin above recent average
        if len(self._recent_energy) >= 5:
            recent_avg = np.mean(self._recent_energy)
            # Update noise floor slowly (low-pass filter)
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * recent_avg
            # Threshold is noise floor + 50% margin
            adaptive_threshold = self._noise_floor * 1.5
        else:
            adaptive_threshold = 0.015  # Default until we have enough samples
        
        # Clamp threshold to reasonable range
        threshold = max(0.008, min(0.025, adaptive_threshold))
        
        is_silent = rms < threshold
        
        # Log occasionally for debugging
        if np.random.random() < 0.02:  # Log 2% of the time
            logger.debug(
                f"Audio RMS: {rms:.4f} | Threshold: {threshold:.4f} | "
                f"Noise floor: {self._noise_floor:.4f} | Silent: {is_silent}"
            )
        
        return is_silent

    async def _process_speech(self) -> None:
        """Process buffered speech with robust error handling."""
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
            
            if rms < 0.012:
                logger.info(f"🔇 Audio too quiet (RMS={rms:.4f}), skipping transcription")
                self.is_processing = False
                return
            
            logger.info(f"🎤 Processing audio (RMS={rms:.4f})")

            # Transcribe
            self.deps.movement_manager.set_listening(True)
            logger.info("Transcribing speech...")
            
            try:
                transcript = await self._transcribe_audio(audio_data)
            except Exception as e:
                logger.error(f"Transcription error: {e}", exc_info=True)
                self.deps.movement_manager.set_listening(False)
                self.is_processing = False
                return

            if not transcript or len(transcript.strip()) < 3:
                logger.debug("Transcript too short or empty, ignoring")
                self.deps.movement_manager.set_listening(False)
                self.is_processing = False
                return

            logger.info(f"User said: {transcript}")
            await self.output_queue.put(
                AdditionalOutputs({"role": "user", "content": transcript})
            )

            # Classify intent
            intent, confidence = self.intent_classifier.classify(transcript)
            logger.info(f"Intent: {intent} (confidence: {confidence:.2f})")

            entities = self.intent_classifier.extract_entities(transcript, intent)
            if entities:
                logger.info(f"Entities: {entities}")

            # Check if needs vision
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

            # Get LLM response
            logger.info("Generating response with Gemma 3 VLM...")
            try:
                response_text, tool_calls = await self._get_llm_response(
                    transcript, image, intent, entities
                )
            except Exception as e:
                logger.error(f"LLM error: {e}", exc_info=True)
                response_text = "Sorry, I had trouble processing that."
                tool_calls = []

            # Clean and validate response
            if response_text:
                response_text = self._clean_response_text(response_text)
                
                # Safety check - ensure response isn't empty after cleaning
                if not response_text or len(response_text.strip()) < 2:
                    logger.warning("Response empty after cleaning, using fallback")
                    response_text = "I'm not sure how to respond to that."

                logger.info(f"Assistant: {response_text}")
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": response_text})
                )

                # Generate speech
                try:
                    await self._generate_speech(response_text)
                except Exception as e:
                    logger.error(f"TTS error: {e}", exc_info=True)
                    # Continue even if TTS fails
            else:
                logger.warning("⚠️ LLM returned empty response, using fallback")
                fallback = "I didn't quite catch that."
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": fallback})
                )
                try:
                    await self._generate_speech(fallback)
                except Exception as e:
                    logger.error(f"TTS error on fallback: {e}")

            # Execute tool calls
            if tool_calls:
                try:
                    await self._execute_tool_calls(tool_calls)
                except Exception as e:
                    logger.error(f"Tool execution error: {e}", exc_info=True)

            self.deps.movement_manager.set_listening(False)

        except Exception as e:
            logger.error(f"Error processing speech: {e}", exc_info=True)
            # Try to send error message to user
            try:
                await self.output_queue.put(
                    AdditionalOutputs({
                        "role": "assistant",
                        "content": "Sorry, I encountered an error. Please try again."
                    })
                )
            except:
                pass
        finally:
            self.is_processing = False
            # After processing, return to callword mode
            self.awaiting_callword = True
    
    def _clean_response_text(self, text: str) -> str:
        """Remove tool call markers from response text."""
        import re
        cleaned = re.sub(r'\[TOOL:\w+:.*?\]', '', text)
        return cleaned.strip()

    def _format_medicine_response(self, medicine_info: str) -> str:
        """Format medicine identification result for speech."""
        import re
        
        # Try to extract structured info
        name_match = re.search(r'\*\*Medicine Name:\*\*\s*(.+?)(?:\n|$)', medicine_info)
        dosage_match = re.search(r'\*\*Dosage.*?:\*\*\s*(.+?)(?:\n|$)', medicine_info)
        timing_match = re.search(r'\*\*Timing.*?:\*\*\s*(.+?)(?:\n|$)', medicine_info)
        
        # Build spoken response
        parts = []
        
        if name_match:
            medicine_name = name_match.group(1).strip()
            parts.append(f"This is {medicine_name}")
        
        if dosage_match:
            dosage = dosage_match.group(1).strip()
            if "None" not in dosage and dosage:
                parts.append(f"Dosage is {dosage}")
        
        if timing_match:
            timing = timing_match.group(1).strip()
            if "None" not in timing and timing:
                parts.append(f"Take {timing}")
        
        # If parsing failed, just clean up the raw text
        if not parts:
            # Remove markdown formatting
            clean_text = re.sub(r'\*\*.*?\*\*', '', medicine_info)
            clean_text = re.sub(r'\d+\.\s+', '', clean_text)  # Remove numbering
            clean_text = clean_text.replace('\n', ' ').strip()
            # Take first sentence
            first_sentence = clean_text.split('.')[0] + '.'
            return first_sentence
        
        return '. '.join(parts) + '.'

    async def _transcribe_audio(self, audio_data: NDArray[np.int16]) -> str:
        """Transcribe audio using Faster-Whisper - OPTIMIZED."""
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Normalize audio
        audio_float = audio_float / (np.max(np.abs(audio_float)) + 1e-8)
        
        def transcribe():
            segments, info = self.whisper_model.transcribe(
                audio_float,
                language="en",
                beam_size=1,          # ← OPTIMIZED (was 3)
                vad_filter=True,
                condition_on_previous_text=False,  # ← Prevents hallucinations
                compression_ratio_threshold=2.4,    # ← Rejects nonsense
                no_speech_threshold=0.6,           # ← Higher = stricter
                temperature=0.0,
            )
            return " ".join([segment.text for segment in segments]).strip()
        
        transcript = await asyncio.to_thread(transcribe)
        
        # Require at least 3 words
        word_count = len(transcript.split())
        if word_count < 3:
            logger.info(f"⚠️ Too short ({word_count} words): '{transcript}'")
            return ""
        
        # WAKE WORD CHECK
        transcript_lower = transcript.lower()
        # Changed from "reachy" to "robot" with common variations
        wake_words = ["robot", "robots", "robo", "roboto"]

        # Find where wake word appears
        wake_word_index = -1
        found_wake_word = None
        for wake_word in wake_words:
            if wake_word in transcript_lower:
                wake_word_index = transcript_lower.find(wake_word)
                found_wake_word = wake_word
                break

        if wake_word_index == -1:
            logger.info(f"⏭️  No wake word: '{transcript}' - IGNORED")
            return ""

        # IMPORTANT: Remove everything BEFORE the wake word
        # This filters out unwanted pre-wake-word audio
        words = transcript.split()
        cleaned_words = []
        found = False

        for word in words:
            if not found and any(ww in word.lower() for ww in wake_words):
                found = True
            if found:
                cleaned_words.append(word)

        cleaned_transcript = " ".join(cleaned_words)

        logger.info(f"✅ Wake word '{found_wake_word}' found")
        logger.info(f"📝 Cleaned: '{cleaned_transcript}'")

        return cleaned_transcript
    
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
        """Get response from Gemma with tool call validation."""
        
        system_prompt = self._build_system_prompt()
        
        try:
            # Get response from Ollama
            response_text, tool_calls = await self._get_ollama_response(
                user_message, image, system_prompt
            )
            
            # VALIDATE TOOL CALLS - Filter out inappropriate ones
            validated_tool_calls = []
            user_lower = user_message.lower()
            
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                
                # Validation rules for each tool
                if tool_name == "identify_medicine":
                    # Only allow if explicitly asking about medicine
                    medicine_keywords = ["medicine", "medication", "pill", "tablet", 
                                    "prescription", "drug", "capsule", "identify"]
                    has_keyword = any(kw in user_lower for kw in medicine_keywords)
                    
                    if has_keyword:
                        validated_tool_calls.append(tool_call)
                        logger.info(f"✅ Medicine identification requested")
                    else:
                        logger.info(f"⏭️  Skipped identify_medicine - not explicitly requested")
                
                elif tool_name == "list_tasks":
                    # Only allow if asking about tasks/timers/reminders
                    task_keywords = ["task", "timer", "reminder", "list", "show", "active"]
                    has_keyword = any(kw in user_lower for kw in task_keywords)
                    
                    if has_keyword:
                        validated_tool_calls.append(tool_call)
                        logger.info(f"✅ List tasks requested")
                    else:
                        logger.info(f"⏭️  Skipped list_tasks - asking general question, not about tasks")
                
                elif tool_name == "create_timer":
                    # Only allow if explicitly setting a timer
                    timer_keywords = ["timer", "set a timer", "countdown"]
                    has_keyword = any(kw in user_lower for kw in timer_keywords)
                    
                    if has_keyword:
                        validated_tool_calls.append(tool_call)
                    else:
                        logger.info(f"⏭️  Skipped create_timer - not a timer request")
                
                elif tool_name == "create_reminder":
                    # Only allow if explicitly setting a reminder
                    reminder_keywords = ["remind", "reminder"]
                    has_keyword = any(kw in user_lower for kw in reminder_keywords)
                    
                    if has_keyword:
                        validated_tool_calls.append(tool_call)
                    else:
                        logger.info(f"⏭️  Skipped create_reminder - not a reminder request")
                
                else:
                    # Allow other tools (move_head, play_emotion, etc.) without strict validation
                    validated_tool_calls.append(tool_call)
            
            # Validate response text
            if not response_text:
                response_text = "Let me help you with that."
            
            # Clean response
            response_text = self._clean_response_text(response_text)
            
            # Final safety check
            if not response_text or len(response_text.strip()) < 2:
                if validated_tool_calls:
                    response_text = "One moment."
                else:
                    response_text = "I'm listening."
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            self._manage_conversation_history()

            return response_text, validated_tool_calls

        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            return "I'm having trouble thinking right now.", []
        
    def _manage_conversation_history(self):
        """Keep conversation history at a reasonable size."""
        MAX_MESSAGES = 20  # Keep last 20 messages (10 user/assistant pairs)
        
        if len(self.conversation_history) > MAX_MESSAGES:
            # Keep only the most recent messages
            self.conversation_history = self.conversation_history[-MAX_MESSAGES:]
            logger.debug(f"Trimmed conversation history to {MAX_MESSAGES} messages")

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
        has_camera = "camera" in ALL_TOOLS
        has_create_timer = "create_timer" in ALL_TOOLS
        has_create_reminder = "create_reminder" in ALL_TOOLS
        has_list_tasks = "list_tasks" in ALL_TOOLS
        has_head_tracking = "head_tracking" in ALL_TOOLS
        
        # Build tools section
        tools_section = "AVAILABLE TOOLS:\n"
        
        if has_move_head:
            tools_section += "- move_head: [TOOL:move_head:{\"direction\":\"left\"}]\n"
            tools_section += "  Directions: left, right, up, down, front\n"
        
        if has_play_emotion:
            first_emotion = emotion_list.split(',')[0].strip()
            tools_section += f"- play_emotion: [TOOL:play_emotion:{{\"emotion\":\"{first_emotion}\"}}]\n"
            tools_section += f"  Available emotions: {emotion_list}\n"
        
        if has_dance:
            tools_section += "- dance: [TOOL:dance:{\"move\":\"random\",\"repeat\":1}]\n"
        
        if has_camera:
            tools_section += "- camera: [TOOL:camera:{\"question\":\"what do you see?\"}]\n"
            tools_section += "  Use to see through the camera\n"

        if "identify_medicine" in ALL_TOOLS:
            tools_section += "- identify_medicine: [TOOL:identify_medicine:{}]\n"
            tools_section += "  Use SPECIFICALLY to identify medicine/medication/pills\n"
            tools_section += "  ALWAYS say something before using this tool\n"
        
        if has_head_tracking:
            tools_section += "- head_tracking: [TOOL:head_tracking:{\"start\":true}]\n"
            tools_section += "  Enable/disable following faces with head\n"
        
        if has_create_timer:
            # CHANGED: Use "duration" and "unit" instead of "duration_seconds"
            tools_section += "- create_timer: [TOOL:create_timer:{\"duration\":10,\"unit\":\"seconds\"}]\n"
            tools_section += "  Create countdown timers. Units: seconds, minutes, hours\n"
        
        if has_create_reminder:
            # CHANGED: Use "delay" and "unit" instead of "delay_seconds"
            tools_section += "- create_reminder: [TOOL:create_reminder:{\"message\":\"task\",\"delay\":5,\"unit\":\"minutes\"}]\n"
            tools_section += "  Set reminders with custom messages. Units: seconds, minutes, hours\n"
        
        if has_list_tasks:
            tools_section += "- list_tasks: [TOOL:list_tasks:{}]\n"
            tools_section += "  Show active timers and reminders\n"
        
        # Build examples based on available tools
        examples = []
        
        # Basic interaction examples
        if has_play_emotion:
            first_emotion = emotion_list.split(',')[0].strip()
            examples.append(f'User: "Hello!"\nrobot: "Hi! I\'m Reachy! [TOOL:play_emotion:{{\"emotion\":\"{first_emotion}\"}}]"')
        
        # Movement examples
        if has_move_head:
            examples.append('User: "Look left"\nrobot: "Looking left now. [TOOL:move_head:{\\"direction\\":\\"left\\"}]"')
        
        if has_dance:
            examples.append('User: "Dance for me"\nrobot: "I\'d love to dance! [TOOL:dance:{\\"move\\":\\"random\\",\\"repeat\\":1}]"')
        
        # Timer/Reminder examples - CHANGED
        if has_create_timer:
            examples.append('User: "Set a timer for 10 seconds"\nrobot: "Timer set! [TOOL:create_timer:{\\"duration\\":10,\\"unit\\":\\"seconds\\"}]"')
            examples.append('User: "Set a timer for 5 minutes"\nrobot: "Timer set for 5 minutes! [TOOL:create_timer:{\\"duration\\":5,\\"unit\\":\\"minutes\\"}]"')
        
        if has_create_reminder:
            examples.append('User: "Remind me to stretch in 10 minutes"\nrobot: "I\'ll remind you to stretch! [TOOL:create_reminder:{\\"message\\":\\"stretch\\",\\"delay\\":10,\\"unit\\":\\"minutes\\"}]"')
            examples.append('User: "Remind me about the meeting in 1 hour"\nrobot: "Got it! [TOOL:create_reminder:{\\"message\\":\\"meeting\\",\\"delay\\":1,\\"unit\\":\\"hours\\"}]"')
        
        # Camera example
        if has_camera:
            examples.append('User: "What do you see?"\nrobot: "Let me look. [TOOL:camera:{\\"question\\":\\"what do you see?\\"}]"')
        if "identify_medicine" in ALL_TOOLS:
            examples.append('User: "Hey robot, identify this medicine"\nrobot: "Let me look at that. [TOOL:identify_medicine:{}]"')
            examples.append('User: "What medication is this?"\nrobot: "Hold it up for me. [TOOL:identify_medicine:{}]"')
        
        # General conversation example
        examples.append('User: "Tell me a joke"\nrobot: "Why did the robot cross the playground? To get to the other slide!"')
        
        examples_text = "\n\n".join(examples)
        
        return f"""You are Reachy, a friendly robot assistant. Follow these rules STRICTLY:

    1. Your name is Reachy (NOT Gemma)
    2. Keep responses SHORT (1-2 sentences max)
    3. NO EMOJIS - you're a robot, not a cartoon
    4. When asked to do physical actions, you MUST use the tool format
    5. When setting timers/reminders, ALWAYS use tools (never just acknowledge verbally)
    6. Don't describe actions - just use the tool
    7. When you see an image, describe what you see concisely

    CRITICAL FORMATTING RULE:
    ALWAYS include text before tool calls!
    CORRECT: "Let me check. [TOOL:identify_medicine:{{}}]"

    TOOL FORMAT (use EXACTLY this format):
    [TOOL:tool_name:{{"param":"value"}}]

    {tools_section}

    TIMER/REMINDER FORMAT:
    - For timers: {{"duration":NUMBER,"unit":"seconds|minutes|hours"}}
    - For reminders: {{"message":"text","delay":NUMBER,"unit":"seconds|minutes|hours"}}

    Examples:
    - 10 seconds: {{"duration":10,"unit":"seconds"}}
    - 5 minutes: {{"duration":5,"unit":"minutes"}}
    - 2 hours: {{"duration":2,"unit":"hours"}}

    CORRECT EXAMPLES:
    {examples_text}

    WRONG (don't do this):
    - Using emojis
    - Describing actions instead of using tools
    - Saying you're "Gemma"
    - Long explanations
    - Using tools or emotions that don't exist
    - Forgetting to use tools for timers/reminders
    - Using "duration_seconds" instead of "duration" and "unit"
    - Wrong units (always specify: "seconds", "minutes", or "hours")"""
        
        
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
        history_to_use = self.conversation_history[-10:]
        for msg in history_to_use:
            messages.append(msg)
        
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
                
                if not response_text:
                    logger.warning("⚠️ LLM returned empty response, using fallback")
                    response_text = "Let me check that for you."

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
                    
                    # Convert to seconds based on unit
                    delay_value = float(tool_args.get("delay", 60))
                    delay_unit = tool_args.get("unit", "seconds").lower()
                    
                    if "minute" in delay_unit:
                        delay_seconds = delay_value * 60
                    elif "hour" in delay_unit:
                        delay_seconds = delay_value * 3600
                    else:  # seconds or default
                        delay_seconds = delay_value
                    
                    task_id = self.task_manager.create_reminder(message, int(delay_seconds))
                    result = {"task_id": task_id, "message": f"Reminder set for {delay_seconds}s"}
                
                elif tool_name == "create_timer":
                    # Convert to seconds based on unit
                    duration_value = float(tool_args.get("duration", 60))
                    duration_unit = tool_args.get("unit", "seconds").lower()
                    
                    if "minute" in duration_unit:
                        duration_seconds = duration_value * 60
                    elif "hour" in duration_unit:
                        duration_seconds = duration_value * 3600
                    else:  # seconds or default
                        duration_seconds = duration_value
                    
                    task_id = self.task_manager.create_timer(int(duration_seconds))
                    result = {"task_id": task_id, "message": f"Timer set for {duration_seconds}s"}
                
                else:
                    # Standard tool dispatch
                    logger.info(f"Executing tool: {tool_name}")
                    args_json = json.dumps(tool_args) if isinstance(tool_args, dict) else tool_args
                    result = await dispatch_tool_call(tool_name, args_json, self.deps)
                
                # ADD THIS: Special handling for identify_medicine
                if tool_name == "identify_medicine" and "medicine_info" in result:
                    medicine_info = result["medicine_info"]
                    
                    # Store for dashboard
                    logger.info("💊 Storing medicine info for dashboard...")
                    if hasattr(self.deps, 'console_stream') and self.deps.console_stream:
                        from datetime import datetime
                        self.deps.console_stream.latest_medicine = {
                            "medicine_info": medicine_info,
                            "timestamp": datetime.now().isoformat()
                        }
                        logger.info("✅ Medicine stored successfully for dashboard")
                    else:
                        logger.warning("⚠️ console_stream not available - medicine won't display on dashboard")
                    
                    # Parse and format the response for speech
                    spoken_response = self._format_medicine_response(medicine_info)
                    
                    # Speak the result
                    logger.info(f"Speaking medicine info: {spoken_response}")
                    await self._generate_speech(spoken_response)
                    
                    # Output to console
                    await self.output_queue.put(
                        AdditionalOutputs({
                            "role": "assistant",
                            "content": spoken_response
                        })
                    )
                    
                    # Also send the raw data
                    await self.output_queue.put(
                        AdditionalOutputs({
                            "role": "assistant",
                            "content": json.dumps({"medicine_info": medicine_info})
                        })
                    )
                
                self.conversation_history.append({
                    "role": "system",
                    "content": f"Tool {tool_name} result: {json.dumps(result)}"
                })

            except Exception as e:
                logger.error(f"Tool execution error ({tool_name}): {e}")


    async def _generate_speech(self, text: str) -> None:
        """Generate speech using configured TTS provider with error handling."""
        
        # Safety check
        if not text or len(text.strip()) < 2:
            logger.warning("⚠️ TTS called with empty/invalid text, skipping")
            return
        
        # Clean text for TTS
        text = text.strip()
        
        try:
            if self.tts_provider == "elevenlabs":
                await self._generate_speech_elevenlabs(text)
            else:
                await self._generate_speech_pyttsx3(text)
        except Exception as e:
            logger.error(f"TTS generation failed: {e}", exc_info=True)

    async def _speak_notification(self, message: str) -> None:
        """Speak a notification when timer/reminder triggers.
        
        This is called by TaskManager when a timer expires or reminder triggers.
        """
        logger.info(f"🔔 Notification: {message}")
        
        # Generate speech for the notification
        await self._generate_speech(message)
        
        # Optionally: Play an emotion to get attention
        try:
            from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call
            import json
            
            # Play an attention-grabbing emotion
            await dispatch_tool_call(
                "play_emotion", 
                json.dumps({"emotion": "curious1"}),  # or "attentive1"
                self.deps
            )
        except Exception as e:
            logger.debug(f"Could not play emotion for notification: {e}")
        
    async def _generate_speech_pyttsx3(self, text: str) -> None:
        """Generate speech using pyttsx3 TTS with robust error handling."""
        import pyttsx3
        import tempfile
        import wave
        import os
        
        if not text or len(text.strip()) < 2:
            logger.warning("Skipping TTS for empty text")
            return
        
        def generate_audio():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                
                engine.save_to_file(text, temp_path)
                engine.runAndWait()
                
                # Verify file was created
                if not os.path.exists(temp_path):
                    raise Exception("TTS file was not created")
                
                with wave.open(temp_path, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                
                os.remove(temp_path)
                return audio_data
                
            except Exception as e:
                logger.error(f"pyttsx3 generation error: {e}")
                raise
        
        try:
            logger.info(f"[TTS/pyttsx3] Speaking: {text[:50]}...")
            audio_data = await asyncio.to_thread(generate_audio)
            
            if audio_data is None or len(audio_data) == 0:
                logger.error("TTS generated empty audio")
                return
            
            # Send to output queue in chunks
            chunk_size = 4800
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await self.output_queue.put((OUTPUT_SAMPLE_RATE, chunk.reshape(1, -1)))
                await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"pyttsx3 TTS error: {e}", exc_info=True)

    async def _generate_speech_elevenlabs(self, text: str) -> None:
        """Generate speech using ElevenLabs TTS."""
        try:
            from elevenlabs import VoiceSettings
            from elevenlabs.client import ElevenLabs
            import io
            import wave
            
            logger.info(f"[TTS/ElevenLabs] Speaking: {text}")
            
            def generate_audio():
                client = ElevenLabs(api_key=self.elevenlabs_api_key)
                
                # Generate audio
                audio_generator = client.text_to_speech.convert(
                    voice_id=self.elevenlabs_voice_id,
                    output_format="mp3_44100_128",
                    text=text,
                    model_id=self.elevenlabs_model,
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75,
                        style=0.0,
                        use_speaker_boost=True,
                    ),
                )
                
                # Collect audio bytes
                audio_bytes = b"".join(audio_generator)
                return audio_bytes
            
            # Generate audio in thread pool
            audio_bytes = await asyncio.to_thread(generate_audio)
            
            # Convert MP3 to PCM audio data
            audio_data = await self._convert_mp3_to_pcm(audio_bytes)
            
            # Send to output queue in chunks
            chunk_size = 4800
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await self.output_queue.put((OUTPUT_SAMPLE_RATE, chunk.reshape(1, -1)))
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            logger.warning("Falling back to pyttsx3")
            await self._generate_speech_pyttsx3(text)

    async def _convert_mp3_to_pcm(self, mp3_bytes: bytes) -> np.ndarray:
        """Convert MP3 bytes to PCM audio data."""
        try:
            from pydub import AudioSegment
            import io
            
            def convert():
                # Load MP3 from bytes
                audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
                
                # Convert to mono if stereo
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                
                # Resample to OUTPUT_SAMPLE_RATE
                audio = audio.set_frame_rate(OUTPUT_SAMPLE_RATE)
                
                # Convert to int16 numpy array
                samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
                return samples
            
            return await asyncio.to_thread(convert)
            
        except ImportError:
            logger.error("pydub not installed. Install with: pip install pydub")
            logger.error("Also requires ffmpeg. Install with: pip install ffmpeg-python")
            raise
        except Exception as e:
            logger.error(f"MP3 conversion error: {e}")
            raise

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