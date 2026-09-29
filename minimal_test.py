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
    
    # SenseVoice expects mono audio, [1, N] float32 tensor/array
    audio_float = audio_data.astype(np.float32).flatten() / 32768.0
    audio_input = audio_float[np.newaxis, :]
    
    try:
        logger.debug("Running Speech-to-Text inference...")
        res = model.generate(
            input=audio_input,
            cache={},
            language="auto",
            use_itn=True,
        )
        text = res[0]['text']
        logger.info(f">>> TRANSCRIBED: {text}")
        
        # Yield the transcribed text back to the browser UI
        yield text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        yield f"Error: {e}"


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
