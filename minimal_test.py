import logging
import numpy as np
from fastrtc import ReplyOnPause, Stream

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MinimalSTT")

# Ensure fastrtc logs are also visible
logging.getLogger("fastrtc").setLevel(logging.DEBUG)
logging.getLogger("aiortc").setLevel(logging.INFO)

import requests
import os
import io
import scipy.io.wavfile as wavfile

# Note: You need to set SARVAM_API_KEY environment variable in Colab.
# If you don't have it set, we will show a clear error.

def transcribe_audio(audio: tuple[int, np.ndarray]):
    sample_rate, audio_data = audio
    max_amp = np.max(np.abs(audio_data))
    
    logger.info(f"[VAD Triggered] Speech detected! Audio shape: {audio_data.shape}, Max Amp: {max_amp}")
    
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        print(">>> ERROR: SARVAM_API_KEY environment variable is not set. Please set it in your Colab notebook!", flush=True)
        yield (16000, np.zeros(160, dtype=np.int16))
        return

    # Convert audio data to wav file in memory
    try:
        # audio_data comes in as shape (16000, ...) but it's mono so let's flatten it just in case
        audio_flat = audio_data.flatten()
        
        # Save to BytesIO
        wav_io = io.BytesIO()
        wavfile.write(wav_io, sample_rate, audio_flat)
        wav_io.seek(0)
        
        print(">>> Running Speech-to-Text inference using Sarvam AI...", flush=True)
        url = "https://api.sarvam.ai/speech-to-text"
        
        headers = {
            "api-subscription-key": api_key
        }
        
        # language_code 'hi-IN' supports Hindi and English code-mixed by default, but we can also use 'en-IN' for English.
        # We will use 'hi-IN' as it handles Hinglish very well.
        payload = {
            "model": "saaras:v4",
        }
        
        files = {
            "file": ("audio.wav", wav_io, "audio/wav")
        }
        
        response = requests.post(url, headers=headers, data=payload, files=files)
        
        if response.status_code == 200:
            res_json = response.json()
            print(f">>> RAW STT RESULT: {res_json}", flush=True)
            text = res_json.get("transcript", "")
            print(f">>> TRANSCRIBED: {text}", flush=True)
        else:
            print(f">>> ERROR from Sarvam AI: {response.status_code} {response.text}", flush=True)

    except Exception as e:
        print(f">>> Transcription EXCEPTION: {e}", flush=True)
        
    # Yield a tiny silent audio chunk so ReplyOnPause doesn't crash
    silent_audio = np.zeros(160, dtype=np.int16)
    yield (16000, silent_audio)


# ReplyOnPause handles WebRTC and automatically runs Silero VAD. 
# Once you finish a sentence, it passes the audio chunk to `transcribe_audio`.
logger.info("Initializing fastrtc ReplyOnPause stream...")
handler = ReplyOnPause(
    transcribe_audio,
    input_sample_rate=16000,
)

stream = Stream(
    handler=handler,
    modality="audio",
    mode="send",
    concurrency_limit=100,
    time_limit=900,
    rtc_configuration={
        "iceServers": [{"urls": ["turn:global.relay.metered.ca:80"], "username": "8aa37256336b81d19b1b6de6", "credential": "A9Chbapp7I3ExRaA"}]
    }
)

if __name__ == "__main__":
    logger.info("Starting Minimal WebRTC -> VAD -> STT Server on port 8282...")
    stream.ui.launch(server_name="0.0.0.0", server_port=8282)
