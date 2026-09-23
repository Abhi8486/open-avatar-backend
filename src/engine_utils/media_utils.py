

import base64
from io import BytesIO
import os
import time
from typing import Union
import wave

import PIL
from loguru import logger
import numpy as np

from src.engine_utils.directory_info import DirectoryInfo


class AudioTestUtils:

    @staticmethod
    def read_wav_to_bytes(file_path) -> tuple[bytes, int]:
        try:
            # Open WAV file
            with wave.open(file_path, 'rb') as wav_file:
                # Get WAV file parameters
                params = wav_file.getparams()
                logger.info("Channels: {}, Sample Width: {}, Frame Rate: {}, Number of Frames: {}",
                            params.nchannels, params.sampwidth, params.framerate, params.nframes)

                # Read all frames
                frames = wav_file.readframes(params.nframes)
                return frames, params.framerate
        except wave.Error as e:
            logger.info("Error reading WAV file: {}", e)
            return None, None

    @classmethod
    def get_test_audio(cls) -> tuple[bytes, int]:
        audio_path = os.path.join(
            DirectoryInfo.get_project_dir(), "resource", "audio", "ymr_48k.wav"
        )
        return cls.read_wav_to_bytes(audio_path)


class VideoUtils:
    pass


class ImageUtils:
    
    @staticmethod
    def format_image(image: Union[str, np.ndarray]):
        if isinstance(image, np.ndarray):
            return ImageUtils.numpy2base64(image)
        return image
    
    # Note: pay attention to RGB order
    @staticmethod
    def numpy2base64(video_frame, format="JPEG"):
        # if video_frame.dtype != np.uint8:
        #     video_frame = (video_frame * 255).astype(np.uint8)

        # Convert NumPy array to PIL Image object
        image = PIL.Image.fromarray(np.squeeze(video_frame)[..., ::-1])

        # Create an in-memory buffer
        buffered = BytesIO()

        # Save image into the memory buffer
        image.save(buffered, format=format)

        # Get binary data and encode as Base64
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Add Base64 data header (optional)
        data_url = f"data:image/{format.lower()};base64,{base64_image}"
        dump_image = False
        if dump_image:
            from engine_utils.directory_info import DirectoryInfo
            ImageUtils.save_base64_image(base64_image, f"{DirectoryInfo.get_project_dir()}/temp/{time.localtime().tm_min}_{time.localtime().tm_sec}.jpg")
        return data_url
    
    @staticmethod
    def save_base64_image(base64_data, output_path):
        """
        Save Base64 encoded image to a local file.

        :param base64_data: Base64 encoded image string (excluding header)
        :param output_path: Local path to save image (including filename and extension)
        """
        try:
            # Strip potential Base64 header info (e.g. "data:image/png;base64,")
            if ',' in base64_data:
                _, base64_data = base64_data.split(',', 1)
            
            # Decode Base64 data
            image_data = base64.b64decode(base64_data)

            # Write decoded data to file
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            logger.debug(f"Image successfully saved to {output_path}")
        
        except Exception as e:
            logger.debug(f"Error saving image: {e}")
