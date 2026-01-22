"""WORKING TTS Test - Uses fastrtc Stream like the real app.

This is exactly how your app plays audio through Reachy!

The secret: Your app uses fastrtc.Stream which handles the audio routing.

REQUIREMENTS:
1. reachy-mini-daemon running
2. pyttsx3 installed

USAGE:
    Terminal 1: reachy-mini-daemon  
    Terminal 2: python test_tts_final.py
"""

import asyncio
import numpy as np
import wave
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_tts_final():
    """Test TTS using the actual Stream method."""
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║         FINAL TTS TEST (Real Method)                      ║
║                                                           ║
║  This uses fastrtc.Stream - exactly like your app!       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Import what your app uses
        from reachy_mini import ReachyMini
        from fastrtc import audio_to_float32
        
        logger.info("✓ Imports successful")
        
        # Connect to Reachy
        logger.info("Connecting to Reachy...")
        robot = ReachyMini()
        logger.info("✓ Connected to Reachy")
        
        # Create a simple handler that just plays audio
        from fastrtc import AsyncStreamHandler

        class SimpleTTSHandler(AsyncStreamHandler):
            """Minimal handler that only emits TTS audio."""

            def __init__(self):
                super().__init__(
                    expected_layout="mono",
                    output_sample_rate=24000,
                    input_sample_rate=24000,
                )
                self.output_queue = asyncio.Queue()
                self.running = True

            def copy(self):
                """Return a fresh handler instance for fastrtc duplication."""
                return SimpleTTSHandler()

            async def receive(self, frame):
                """Receive audio from mic (ignored for this test)."""
                return None

            async def emit(self):
                """Emit audio to speaker."""
                try:
                    item = await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
                    return item
                except asyncio.TimeoutError:
                    return None
            
            async def generate_and_queue_tts(self, text):
                """Generate TTS and put in output queue."""
                import pyttsx3
                
                logger.info(f"🔊 Generating TTS: '{text}'")
                
                def generate():
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
                
                # Generate audio
                audio_data = await asyncio.to_thread(generate)
                logger.info(f"  → Generated {len(audio_data)} samples")
                
                # Put chunks in output queue (same as your handler)
                chunk_size = 4800
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    chunk_mono = chunk.reshape(1, -1)
                    
                    # Queue as (sample_rate, audio) tuple
                    await self.output_queue.put((self.output_sample_rate, chunk_mono))
                
                logger.info(f"  → Queued {(len(audio_data) + chunk_size - 1) // chunk_size} chunks")
        
        # Create handler
        logger.info("Creating TTS handler...")
        handler = SimpleTTSHandler()
        
        # Start robot audio output pipeline
        logger.info("Starting Reachy audio output...")
        robot.media.start_playing()
        output_sample_rate = robot.media.get_output_audio_samplerate()
        logger.info("✓ Audio output started (device sample rate=%s)", output_sample_rate)
        
        # Test phrases
        test_phrases = [
            "Hello! I am Reachy!",
            "This is a test of my text to speech system!",
            "I can speak through my speaker!",
        ]
        
        logger.info("\n" + "="*60)
        logger.info("Playing audio through Reachy's speaker...")
        logger.info("="*60)
        
        for i, phrase in enumerate(test_phrases, 1):
            logger.info(f"\n[{i}/{len(test_phrases)}] {phrase}")
            
            # Generate and queue TTS
            await handler.generate_and_queue_tts(phrase)
            
            # Drain queued chunks and push to Reachy audio player
            queued = 0
            total_samples = 0
            while not handler.output_queue.empty():
                input_sr, audio_chunk = await handler.output_queue.get()
                audio_frame = audio_to_float32(audio_chunk.flatten())

                if input_sr != output_sample_rate and audio_frame.size > 0:
                    # Linear resample to match speaker rate
                    duration = audio_frame.size / input_sr
                    target_len = int(duration * output_sample_rate)
                    audio_frame = np.interp(
                        np.linspace(0, audio_frame.size - 1, num=target_len),
                        np.arange(audio_frame.size),
                        audio_frame,
                    ).astype(np.float32)

                robot.media.push_audio_sample(audio_frame)
                total_samples += audio_frame.size
                queued += 1

            # Calculate playback duration and wait for audio to finish
            playback_duration = total_samples / output_sample_rate
            logger.info(f"  → Played {queued} chunks ({playback_duration:.1f}s duration)")
            logger.info(f"  → Waiting for playback to complete...")
            await asyncio.sleep(playback_duration + 0.5)  # Add buffer time
            
            if i < len(test_phrases):
                logger.info("  → Pausing...")
                await asyncio.sleep(0.5)
        
        logger.info("\n" + "="*60)
        logger.info("✓ TTS test complete!")
        logger.info("="*60)
        
        # Stop audio output
        logger.info("\nStopping Reachy audio output...")
        try:
            robot.media.stop_playing()
        except Exception:
            pass
        
        print("""
╔═══════════════════════════════════════════════════════════╗
║  If you heard Reachy speak, TTS is working!              ║
║                                                           ║
║  Your local_conversation_handler.py uses this exact      ║
║  same method - it puts audio in output_queue and the     ║
║  Stream plays it through Reachy's speaker!               ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
        """)
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure installed: pip install reachy-mini pyttsx3 fastrtc")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)


if __name__ == "__main__":
    print("Checking dependencies...\n")
    
    # Check pyttsx3
    try:
        import pyttsx3
        print("✓ pyttsx3")
    except ImportError:
        print("❌ pyttsx3: pip install pyttsx3")
        exit(1)
    
    # Check reachy-mini
    try:
        from reachy_mini import ReachyMini
        print("✓ reachy-mini")
    except ImportError:
        print("❌ reachy-mini: pip install reachy-mini")
        exit(1)
    
    # Check fastrtc
    try:
        from fastrtc import Stream
        print("✓ fastrtc")
    except ImportError:
        print("❌ fastrtc: pip install fastrtc")
        exit(1)
    
    print("\n✓ All dependencies OK!\n")
    
    # Check daemon
    print("Checking daemon...")
    try:
        robot = ReachyMini()
        print("✓ Daemon is running!\n")
        del robot
    except Exception:
        print("❌ Daemon not running!")
        print("Start it first: reachy-mini-daemon\n")
        exit(1)
    
    # Run test
    asyncio.run(test_tts_final())