"""Test Whisper speech-to-text."""
import whisper
import numpy as np
import sounddevice as sd
import queue
import time

print('Loading Whisper model (base)...')
model = whisper.load_model('base')
print('✓ Model loaded!\n')

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Callback for audio recording."""
    audio_queue.put(indata.copy())

print('='*60)
print('WHISPER SPEECH-TO-TEXT TEST')
print('='*60)
print('\nRecording for 5 seconds...')
print('SPEAK NOW! Say something like:')
print('  "Hello Reachy, can you hear me?"')
print('  "What do you see in front of you?"')
print('  "Dance for me!"')
print('\nRecording...')

# Start recording
stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    callback=audio_callback,
    blocksize=int(16000 * 0.1)
)

stream.start()
time.sleep(5)
stream.stop()
stream.close()

print('\n✓ Recording complete!')
print('Processing audio...')

# Collect all audio chunks
audio_data = []
while not audio_queue.empty():
    audio_data.append(audio_queue.get())

if audio_data:
    # Concatenate and flatten
    audio_np = np.concatenate(audio_data, axis=0).flatten()
    
    # Transcribe
    result = model.transcribe(audio_np, language='en')
    
    print('\n' + '='*60)
    print('TRANSCRIPTION RESULT:')
    print('='*60)
    print(f'"{result["text"]}"')
    print('='*60)
else:
    print('✗ No audio captured!')