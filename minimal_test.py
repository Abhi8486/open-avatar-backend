import logging
import numpy as np
from funasr import AutoModel
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

logger.info("Loading SenseVoice model...")
try:
    model = AutoModel(model="iic/SenseVoiceSmall", disable_update=True)
    logger.info("SenseVoice loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load SenseVoice: {e}")

def transcribe_audio(audio: tuple[int, np.ndarray]):
    sample_rate, audio_data = audio
    max_amp = np.max(np.abs(audio_data))
    
    logger.info(f"[VAD Triggered] Speech detected! Audio shape: {audio_data.shape}, Max Amp: {max_amp}")
    
    # Convert from int16 to float32 (SenseVoice requires float32)
    audio_flat = audio_data.astype(np.float32).flatten() / 32768.0
    
    try:
        print(">>> Running Speech-to-Text inference...", flush=True)
        res = model.generate(
            input=audio_flat,
            cache={},
            language="auto",
            use_itn=True,
        )
        print(f">>> RAW STT RESULT: {res}", flush=True)
        
        if len(res) > 0 and 'text' in res[0]:
            text = res[0]['text']
            print(f">>> TRANSCRIBED: {text}", flush=True)
        else:
            print(">>> WARNING: No 'text' in result!", flush=True)
            
        # Yield a tiny silent audio chunk so ReplyOnPause doesn't crash
        silent_audio = np.zeros(160, dtype=np.int16)
        yield (16000, silent_audio)
    except Exception as e:
        print(f">>> Transcription EXCEPTION: {e}", flush=True)
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
