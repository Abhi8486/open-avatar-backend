"""
Visual Model Abstract Interface

Defines the interface required by visual models, allowing subsequent model substitution.
"""
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, List, Optional
import base64
import os
import re
import threading
import time

import cv2
from loguru import logger
import numpy as np

from handlers.agent.agent_data_models import PerceptionData, EnvironmentEvent


class AsyncPerceptionManager:
    """
    Asynchronous Perception Task Manager
    
    Uses thread pool to concurrently handle LLM requests, employing a "latest-first" policy:
    - If a newer round has already been sent, older rounds are skipped (avoiding blocking)
    - Results are sent immediately without waiting for preceding rounds
    """
    
    def __init__(
        self,
        vision_model: "VisionModelInterface",
        max_workers: int = 3,
        on_result_callback: Optional[Callable[[int, Optional[PerceptionData]], None]] = None,
    ):
        """
        Initialize asynchronous manager
        
        Args:
            vision_model: Vision model instance
            max_workers: Maximum concurrent worker threads
            on_result_callback: Result callback function with parameters (round_id, perception_data)
        """
        self.vision_model = vision_model
        self.max_workers = max_workers
        self.on_result_callback = on_result_callback
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="perception_worker")
        
        # Latest-first policy: track the highest round_id already sent
        self.last_sent_round: int = 0
        self.send_lock = threading.Lock()
        
        # In-flight tasks
        self.pending_futures: Dict[int, Future] = {}
        self.pending_lock = threading.Lock()
        
        # Statistics
        self.total_submitted: int = 0
        self.total_completed: int = 0
        self.total_skipped: int = 0
        self.total_outdated: int = 0
        
        # Running status
        self._running = True
        
        logger.info(f"[AsyncPerceptionManager] Initialized: max_workers={max_workers}, Policy: Latest-first")
    
    @property
    def current_pending_count(self) -> int:
        """Current pending task count"""
        with self.pending_lock:
            return len(self.pending_futures)
    
    def submit_task(self, round_id: int, frames: List[np.ndarray]) -> bool:
        """
        Submit asynchronous perception task
        
        Args:
            round_id: Perception round ID
            frames: Video frame list
            
        Returns:
            bool: Whether successfully submitted (returns False if concurrency limit reached)
        """
        if not self._running:
            logger.warning(f"[AsyncPerceptionManager] [Round-{round_id}] Manager stopped, rejecting task submission")
            return False
        
        with self.pending_lock:
            pending_count = len(self.pending_futures)
            
            # Check concurrency limit
            if pending_count >= self.max_workers:
                logger.warning(
                    f"[AsyncPerceptionManager] [Round-{round_id}] ⚠️ Concurrency limit reached ({pending_count}/{self.max_workers}), skipping round"
                )
                self.total_skipped += 1
                return False
            
            # Submit task to thread pool
            future = self.executor.submit(self._worker, round_id, frames)
            self.pending_futures[round_id] = future
            self.total_submitted += 1
            
            logger.info(
                f"[AsyncPerceptionManager] [Round-{round_id}] 📤 Submitted async task "
                f"(Current concurrency: {pending_count + 1}/{self.max_workers}, Frames: {len(frames)})"
            )
            
            # Add done callback
            future.add_done_callback(lambda f: self._on_future_done(round_id, f))
            
            return True
    
    def _worker(self, round_id: int, frames: List[np.ndarray]) -> Optional[PerceptionData]:
        """
        Worker thread: executes API call
        
        Args:
            round_id: Perception round ID
            frames: Video frame list
            
        Returns:
            Optional[PerceptionData]: Perception data or None
        """
        try:
            logger.debug(f"[AsyncPerceptionManager] [Round-{round_id}] Worker thread execution started")
            result = self.vision_model.generate_perception(frames, round_id=round_id)
            return result
        except Exception as e:
            logger.error(f"[AsyncPerceptionManager] [Round-{round_id}] Worker thread exception: {e}")
            return None
    
    def _on_future_done(self, round_id: int, future: Future):
        """
        Future done callback
        
        Args:
            round_id: Perception round ID
            future: Completed Future object
        """
        # Remove from pending
        with self.pending_lock:
            self.pending_futures.pop(round_id, None)
        
        # Get result
        try:
            result = future.result()
        except Exception as e:
            logger.error(f"[AsyncPerceptionManager] [Round-{round_id}] Error obtaining result: {e}")
            result = None
        
        self.total_completed += 1
        
        # Handle result (latest-first policy)
        self._handle_result(round_id, result)
    
    def _handle_result(self, round_id: int, result: Optional[PerceptionData]):
        """
        Handle result: latest-first policy
        
        - If round_id <= last_sent_round: skip (newer result already sent)
        - Otherwise: send immediately and update last_sent_round
        
        Args:
            round_id: Perception round ID
            result: Perception data
        """
        with self.send_lock:
            # Check if outdated
            if round_id <= self.last_sent_round:
                self.total_outdated += 1
                logger.info(
                    f"[AsyncPerceptionManager] [Round-{round_id}] ⏭️ Skipping outdated result "
                    f"(Already sent: Round-{self.last_sent_round})"
                )
                return
            
            # Even if result is None, update last_sent_round to prevent older rounds from overwriting
            if result is None:
                logger.info(f"[AsyncPerceptionManager] [Round-{round_id}] ⚠️ Result is None, updating round marker")
                self.last_sent_round = round_id
                return
            
            # Send result immediately
            logger.info(
                f"[AsyncPerceptionManager] [Round-{round_id}] ✅ API completed, "
                f"Last sent: Round-{self.last_sent_round}"
            )
            
            if self.on_result_callback:
                try:
                    logger.info(f"[AsyncPerceptionManager] [Round-{round_id}] 📤 Sending result to Manager immediately")
                    self.on_result_callback(round_id, result)
                except Exception as e:
                    logger.error(f"[AsyncPerceptionManager] [Round-{round_id}] Callback execution exception: {e}")
            
            # Update maximum round sent
            self.last_sent_round = round_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self.pending_lock:
            pending = len(self.pending_futures)
        
        return {
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            "total_skipped": self.total_skipped,
            "total_outdated": self.total_outdated,
            "current_pending": pending,
            "last_sent_round": self.last_sent_round,
        }
    
    def shutdown(self, wait: bool = True, timeout: float = 5.0):
        """
        Shut down manager
        
        Args:
            wait: Whether to wait for all tasks to complete
            timeout: Timeout in seconds before cancelling remaining futures
        """
        self._running = False
        
        stats = self.get_stats()
        logger.info(
            f"[AsyncPerceptionManager] Shutting down... "
            f"(submitted: {stats['total_submitted']}, "
            f"completed: {stats['total_completed']}, "
            f"skipped: {stats['total_skipped']}, "
            f"outdated: {stats['total_outdated']}, "
            f"pending: {stats['current_pending']})"
        )
        
        # Cancel all pending tasks
        with self.pending_lock:
            for round_id, future in list(self.pending_futures.items()):
                if not future.done():
                    future.cancel()
                    logger.debug(f"[AsyncPerceptionManager] Cancelled pending task: Round-{round_id}")
        
        # Shut down executor without blocking
        self.executor.shutdown(wait=False, cancel_futures=True)
        logger.info("[AsyncPerceptionManager] Shut down")


class VisionModelInterface(ABC):
    """
    Visual Model Abstract Interface
    
    Subclasses must implement specific visual understanding capabilities.
    """
    
    @abstractmethod
    def generate_perception(self, frames: List[np.ndarray], round_id: int = 0) -> Optional[PerceptionData]:
        """
        Generate layered visual perception data
        
        Args:
            frames: Video frame list (BGR format)
            round_id: Perception round ID, used for log tracking
            
        Returns:
            PerceptionData: Returns layered visual context on success
            None: Returns None on failure
        """
        pass
    
    @abstractmethod
    def detect_events(self, frame: np.ndarray, 
                      previous_frame: Optional[np.ndarray] = None) -> List[EnvironmentEvent]:
        """
        Detect environment events
        
        Args:
            frame: Current frame (BGR format)
            previous_frame: Previous frame (used to detect changes)
            
        Returns:
            List[EnvironmentEvent]: List of detected events
        """
        pass
    
    def warmup(self):
        """Warm up model (optional)"""
        pass
    
    def cleanup(self):
        """Clean up resources (optional)"""
        pass


class MockVisionModel(VisionModelInterface):
    """
    Mock vision model for testing
    
    Returns static perception data without performing actual visual inference
    """
    
    def __init__(self):
        self._frame_count = 0
    
    def generate_perception(self, frames: List[np.ndarray], round_id: int = 0) -> PerceptionData:
        """Return mock perception data"""
        import time
        from handlers.agent.agent_data_models import SceneStructure, UserState
        
        self._frame_count += len(frames)
        logger.info(f"[MockVisionModel] [Round-{round_id}] Generated mock perception data (Frames: {len(frames)})")
        
        return PerceptionData(
            scene_summary="I see the user sitting in front of the computer",
            scene_structure=SceneStructure(
                location="Office",
                people=["User"],
                objects=["Computer", "Keyboard", "Mouse"],
                activities=["Working"],
            ),
            user_state=UserState(
                emotion="neutral",
                gaze="screen",
                posture="sitting",
                action="typing",
            ),
            timestamp=time.time(),
        )
    
    def detect_events(self, frame: np.ndarray,
                      previous_frame: Optional[np.ndarray] = None) -> List[EnvironmentEvent]:
        """Mock event detection, returning no events"""
        return []


class OpenAIVisionModel(VisionModelInterface):
    """
    OpenAI-compatible interface vision model (e.g. qwen-plus-vl)
    """

    DEFAULT_SYSTEM_PROMPT = """You are a visual perception system for an embodied AI agent. You are observing the user in real time via camera.

Important Perspective Guidelines:
- Your perspective is the camera perspective, and the user is facing you.
- You receive a keyframe sequence of a real-time video stream showing changes in the user's state over the past few seconds.
- When you see a person holding a phone toward the screen, this means the user is showing their phone to you, not taking a selfie.
- When you see a person performing an action, describe it as "The user is doing..." rather than "Someone in the video is doing...".
- Please analyze the whole video sequence and describe the user's current state and ongoing activities.

Please describe what you observe from a first-person perspective, strictly using the following XML tag format for output (output ONLY tags, nothing else):

<scene_summary>One sentence describing the current scene (use phrases like "I see the user...")</scene_summary>
<location>Scene location (office/home/outdoor, etc.)</location>
<people>Visible people, comma-separated (e.g. user, colleague in background)</people>
<objects>Visible objects, comma-separated (e.g. laptop, phone, cup)</objects>
<activities>Ongoing activities, comma-separated (e.g. user typing, colleague discussing)</activities>
<emotion>User emotion (focused/happy/confused/sad/angry/surprised/neutral)</emotion>
<gaze>User gaze direction (looking_at_camera/looking_away/looking_down/looking_at_screen)</gaze>
<posture>User posture (sitting/standing/leaning/lying)</posture>
<action>User current action (speaking/typing/holding_phone/gesturing/idle/reading)</action>
<events></events>

Interaction Event Detection Rules (events tag):
- Output events inside <events> ONLY when a clear interaction intent from the user is detected; otherwise keep <events></events> empty.
- Format when events exist:
<events>
<event type="event_type" confidence="0.8">Event description</event>
</events>
- Confidence ranges from 0.0-1.0; only events with confidence >= 0.7 will be processed.
- Event type definitions:
  * waving: User raises hand and waves, indicating greeting or seeking attention
  * showing_object: User proactively holds an object toward the camera to show you
  * asking_for_attention: User performs obvious attention-seeking gestures (e.g., tapping table, beckoning, large waving)
  * leaving: User is leaving the frame
  * arriving: A new person enters the frame

Important Notes:
- Ordinary hand actions (like typing, touching face) do not count as waving
- Only explicit waving toward the camera counts as waving
- If no interaction event is detected, keep the events tag empty: <events></events>"""

    def __init__(
        self,
        model_name: str = "qwen-plus-vl",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        max_frames: int = 4,
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.api_url = api_url
        self.max_frames = max_frames
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        self._client = None
        self.api_timeout = 6.0
        try:
            from openai import OpenAI
            import httpx

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url,
                timeout=httpx.Timeout(self.api_timeout, connect=5.0),
            )
        except Exception:
            self._client = None

    MIN_VIDEO_FRAMES = 4

    def generate_perception(self, frames: List[np.ndarray], round_id: int = 0, debug_save: bool = True) -> Optional[PerceptionData]:
        """
        Generate perception data
        
        Args:
            frames: Video frame list
            round_id: Perception round ID for logging
            debug_save: Whether to save debug frames
            
        Returns:
            PerceptionData: Perception data on success
            None: Returns None on failure, caller should retain previous cache
        """
        round_tag = f"[Round-{round_id}]"
        
        if not frames:
            logger.warning(f"[OpenAI Vision Model] {round_tag} No frames provided")
            return None

        if self._client is None:
            logger.warning(f"[OpenAI Vision Model] {round_tag} Client is None")
            return None

        selected_frames = self._select_frames(frames, self.max_frames)
        
        messages, valid_images = self._build_messages(selected_frames, round_id=round_id, debug_save=debug_save)
        if valid_images == 0:
            logger.warning(f"[OpenAI Vision Model] {round_tag} No valid images to send after processing")
            return None

        try:
            time_start = time.time()
            logger.info(f"[OpenAI Vision Model] {round_tag} 🚀 Starting API call (Timeout: {self.api_timeout}s)...")
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                max_tokens=600,
                timeout=self.api_timeout,
            )
            time_end = time.time()
            logger.info(f"[OpenAI Vision Model] {round_tag} ✅ API call completed, time: {time_end - time_start:.2f}s")
        except Exception as exc:
            time_end = time.time()
            elapsed = time_end - time_start
            if elapsed >= self.api_timeout - 0.5:
                logger.warning(f"[OpenAI Vision Model] {round_tag} ⏱️ API call timeout ({elapsed:.2f}s >= {self.api_timeout}s), skipping round")
            else:
                logger.warning(f"[OpenAI Vision Model] {round_tag} API call failed ({elapsed:.2f}s): {exc}")
            return None

        content = ""
        if response and response.choices:
            raw_content = response.choices[0].message.content
            if isinstance(raw_content, list):
                parts = []
                for item in raw_content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(item))
                content = "".join(parts).strip()
            else:
                content = (raw_content or "").strip()

        logger.info(f"[OpenAI Vision Model] {round_tag} 📝 Response: {content}")
        parsed = self._parse_xml_response(content)
        if not parsed or not parsed.get("scene_summary"):
            logger.warning(f"[OpenAI Vision Model] {round_tag} Empty/invalid XML response, returning None")
            return None
        parsed["timestamp"] = time.time()
        return PerceptionData.from_dict(parsed)

    def detect_events(self, frame: np.ndarray,
                      previous_frame: Optional[np.ndarray] = None) -> List[EnvironmentEvent]:
        return []

    def _select_frames(self, frames: List[np.ndarray], max_frames: int) -> List[np.ndarray]:
        if max_frames <= 0 or len(frames) <= max_frames:
            return frames
        step = max(1, len(frames) // max_frames)
        selected = frames[::step][:max_frames]
        return selected

    def _build_messages(self, frames: List[np.ndarray], round_id: int = 0, debug_save: bool = False) -> tuple[List[Dict[str, Any]], int]:
        round_tag = f"[Round-{round_id}]"
        
        system_prompt = self._system_prompt

        video_frames: List[str] = []
        for frame in frames:
            data_url = self._frame_to_data_url(frame, debug_save_path=None)
            if data_url:
                video_frames.append(data_url)
        
        original_count = len(video_frames)
        
        if original_count == 0:
            logger.warning(f"[OpenAI Vision Model] {round_tag} No valid frames to build video content")
            return [], 0
        
        if original_count < self.MIN_VIDEO_FRAMES:
            logger.info(f"[OpenAI Vision Model] {round_tag} Padding frames: {original_count} -> {self.MIN_VIDEO_FRAMES} (repeating frames)")
            while len(video_frames) < self.MIN_VIDEO_FRAMES:
                video_frames.append(video_frames[len(video_frames) % original_count])
        
        valid_images = len(video_frames)
        
        content: List[Dict[str, Any]] = [
            {
                "type": "video",
                "video": video_frames,
            },
            {
                "type": "text",
                "text": "Please describe the observed user state and scene based on this real-time camera video stream."
            }
        ]
        
        logger.info(f"[OpenAI Vision Model] {round_tag} 📹 Built video message: {valid_images} frames (original: {original_count})")

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ], valid_images

    def _frame_to_data_url(self, frame: np.ndarray, debug_save_path: Optional[str] = None) -> Optional[str]:
        """Convert video frame to data URL using project ImageUtils"""
        try:
            from engine_utils.media_utils import ImageUtils
            
            if not isinstance(frame, np.ndarray):
                logger.warning(f"[OpenAI Vision Model] Frame is not np.ndarray, type: {type(frame)}")
                return None
            
            if frame.size == 0:
                logger.warning("[OpenAI Vision Model] Frame size is 0")
                return None
            
            if debug_save_path:
                try:
                    debug_dir = os.path.dirname(debug_save_path)
                    if debug_dir:
                        os.makedirs(debug_dir, exist_ok=True)
                    data_url = ImageUtils.format_image(frame)
                    ImageUtils.save_base64_image(data_url, debug_save_path)
                    logger.info(f"[OpenAI Vision Model] Debug: saved frame to {debug_save_path}")
                except Exception as save_exc:
                    logger.warning(f"[OpenAI Vision Model] Failed to save debug frame: {save_exc}")

            return ImageUtils.format_image(frame)
            
        except Exception as exc:
            logger.warning(f"Failed to encode frame: {exc}")
            import traceback
            logger.warning(traceback.format_exc())
            return None

    def _parse_xml_response(self, content: str) -> Dict[str, Any]:
        """Extract perception data from XML-tagged VLM output."""
        if not content:
            return {}

        def _extract_tag(text: str, tag: str) -> str:
            m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', text, re.DOTALL)
            return m.group(1).strip() if m else ""

        def _extract_list(text: str, tag: str) -> List[str]:
            raw = _extract_tag(text, tag)
            if not raw:
                return []
            return [item.strip() for item in raw.split(",") if item.strip()]

        scene_summary = _extract_tag(content, "scene_summary")
        if not scene_summary:
            return {}

        detected_events: List[Dict[str, Any]] = []
        events_block = _extract_tag(content, "events")
        if events_block:
            for m in re.finditer(
                r'<event\s+type="([^"]*?)"\s+confidence="([^"]*?)">(.*?)</event>',
                events_block, re.DOTALL,
            ):
                try:
                    conf = float(m.group(2))
                except (ValueError, TypeError):
                    conf = 0.0
                detected_events.append({
                    "event_type": m.group(1),
                    "confidence": conf,
                    "description": m.group(3).strip(),
                })

        return {
            "scene_summary": scene_summary,
            "scene_structure": {
                "location": _extract_tag(content, "location"),
                "people": _extract_list(content, "people"),
                "objects": _extract_list(content, "objects"),
                "activities": _extract_list(content, "activities"),
            },
            "user_state": {
                "emotion": _extract_tag(content, "emotion"),
                "gaze": _extract_tag(content, "gaze"),
                "posture": _extract_tag(content, "posture"),
                "action": _extract_tag(content, "action"),
            },
            "detected_events": detected_events,
        }
