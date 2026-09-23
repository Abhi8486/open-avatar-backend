"""
Data Models for VAD/EOU Analysis Tool

Analysis goals:
1. EOU core function is "goalkeeper" -- preventing VAD from prematurely cutting off users while thinking and speaking
2. Early VAD purpose is fast detection when users genuinely finish speaking to start Q&A early
3. Evaluation targets:
   - Whether EOU correctly prevented premature cutoffs (True Negative)
   - Whether EOU correctly accelerated short sentence responses (True Positive)
   - Whether EOU incorrectly cut off unfinished utterances (False Positive)
   - Whether EOU missed short sentences that should have been accelerated (False Negative)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class VADEvent:
    """VAD Event"""
    timestamp_ms: float              # Relative timestamp (ms)
    event_type: str                  # "speech_start", "early_vad_end", "speech_end", "data"
    sample_id: Optional[int] = None  # Sample ID
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EOUEvent:
    """EOU Event"""
    timestamp_ms: float              # Relative timestamp (ms)
    event_type: str                  # "prediction", "signal_sent"
    prediction: Optional[int] = None # 0=Incomplete, 1=Complete
    probability: Optional[float] = None
    buffer_duration_ms: Optional[float] = None


@dataclass
class ASREvent:
    """ASR Event"""
    timestamp_ms: float              # Relative timestamp (ms)
    event_type: str                  # "started", "completed"
    text: Optional[str] = None
    stream_key: Optional[str] = None
    source: Optional[str] = None     # ASR handler name


@dataclass 
class UtteranceAnalysis:
    """Complete analysis for a single utterance."""
    utterance_id: int
    stream_key: str
    vad_events: List[VADEvent] = field(default_factory=list)
    eou_events: List[EOUEvent] = field(default_factory=list)
    asr_events: List[ASREvent] = field(default_factory=list)
    
    # Timestamps (ms)
    speech_start_ms: Optional[float] = None
    early_vad_end_ms: Optional[float] = None
    speech_end_ms: Optional[float] = None
    asr_completed_ms: Optional[float] = None
    
    # Recognized text
    asr_text: Optional[str] = None
    
    # Computed metrics (ms)
    speech_duration_ms: Optional[float] = None    # Speech duration
    early_to_end_delay_ms: Optional[float] = None  # Duration from early_vad_end to speech_end (user continued speaking duration)
    eou_decision_time_ms: Optional[float] = None  # EOU decision duration (from early_vad_end to signal send)
    total_latency_ms: Optional[float] = None      # VAD->ASR latency (from speech_end to ASR completion, including network roundtrip)
    
    # EOU effects
    eou_triggered: bool = False        # Whether EOU triggered (signal sent)
    eou_effective: bool = False        # Whether EOU was effective (ended speech early)
    eou_prediction_count: int = 0      # EOU prediction count
    eou_final_probability: Optional[float] = None  # Final prediction confidence
    
    # Utterance classification
    utterance_type: str = "unknown"    # "short_complete", "long_thinking", "incomplete"
    is_sentence_complete: bool = False  # Whether sentence is complete
    
    # EOU result classification
    eou_result_type: str = "unknown"   # "accelerated", "protected", "missed", "false_trigger", "reconnected", "normal"
    time_saved_ms: Optional[float] = None  # Time saved by EOU (if triggered)
    
    # Reconnection related
    was_reconnected: bool = False      # Whether utterance triggered reconnection (EOU false prediction auto-corrected)
    cancelled_stream_key: Optional[str] = None  # Cancelled stream key
    reconnect_time_gap_ms: Optional[float] = None  # Time gap during reconnection (ms)
    
    def calculate_metrics(self):
        """Calculate metrics."""
        # Speech duration
        if self.speech_start_ms is not None and self.speech_end_ms is not None:
            self.speech_duration_ms = self.speech_end_ms - self.speech_start_ms
        
        # Time from early_vad_end to speech_end
        if self.early_vad_end_ms is not None and self.speech_end_ms is not None:
            self.early_to_end_delay_ms = self.speech_end_ms - self.early_vad_end_ms
        
        # EOU decision time
        signal_events = [e for e in self.eou_events if e.event_type == "signal_sent"]
        if signal_events and self.early_vad_end_ms is not None:
            self.eou_decision_time_ms = signal_events[0].timestamp_ms - self.early_vad_end_ms
        
        # Total latency
        if self.speech_end_ms is not None and self.asr_completed_ms is not None:
            self.total_latency_ms = self.asr_completed_ms - self.speech_end_ms
        
        # EOU statistics
        self.eou_prediction_count = len([e for e in self.eou_events if e.event_type == "prediction"])
        prediction_events = [e for e in self.eou_events if e.event_type == "signal_sent" and e.probability is not None]
        if prediction_events:
            self.eou_final_probability = prediction_events[-1].probability
        
        # Analyze sentence completeness
        self._analyze_sentence_completeness()
        
        # Classify utterance type
        self._classify_utterance_type()
        
        # Classify EOU result
        self._classify_eou_result()
    
    def _analyze_sentence_completeness(self):
        """Analyze sentence completeness."""
        if not self.asr_text:
            self.is_sentence_complete = False
            return
        
        text = self.asr_text.strip()
        
        # Check if ends with complete sentence punctuation
        complete_endings = ['.', '!', '?', '。', '！', '？']
        incomplete_endings = [',', '，', '、', '...', '…', 'and', 'or', 'but', 'that', 'the', 'a', 'an']
        
        if any(text.endswith(e) for e in complete_endings):
            self.is_sentence_complete = True
        elif any(text.lower().rstrip('.,!? ').endswith(e) for e in incomplete_endings):
            self.is_sentence_complete = False
        else:
            self.is_sentence_complete = len(text.split()) <= 5  # Short phrases may be complete
    
    def _classify_utterance_type(self):
        """Classify utterance type."""
        if self.speech_duration_ms is None:
            self.utterance_type = "unknown"
            return
        
        # Short sentence: under 2 seconds
        if self.speech_duration_ms < 2000:
            if self.is_sentence_complete:
                self.utterance_type = "short_complete"  # Short & complete e.g. "Okay.", "No."
            else:
                self.utterance_type = "short_incomplete"  # Short & incomplete
        else:
            # Long sentence: over 2 seconds
            if self.is_sentence_complete:
                self.utterance_type = "long_complete"  # Long sentence & complete
            else:
                self.utterance_type = "long_thinking"  # Long sentence, likely thinking while speaking
    
    def _classify_eou_result(self):
        """
        Classify EOU result:
        - accelerated: EOU triggered and correctly accelerated response (short/complete sentence)
        - protected: EOU did not trigger, correctly protected long utterance from cutoff
        - missed: EOU should have triggered but missed (short sentence waited too long)
        - false_trigger: EOU should not have triggered but did (may have cut off incomplete sentence)
        - reconnected: EOU false prediction auto-corrected by reconnection mechanism
        - normal: Normal VAD end, no EOU involved
        """
        if self.was_reconnected:
            self.eou_result_type = "reconnected"
            return
        
        if self.eou_triggered:
            if self.is_sentence_complete:
                self.eou_result_type = "accelerated"
                if self.early_to_end_delay_ms is not None:
                    self.time_saved_ms = max(0, 2000 - self.early_to_end_delay_ms)
            else:
                self.eou_result_type = "false_trigger"
        else:
            if self.utterance_type == "short_complete":
                self.eou_result_type = "missed"
            elif self.utterance_type in ["long_thinking", "long_complete"]:
                self.eou_result_type = "protected"
            else:
                self.eou_result_type = "normal"


@dataclass
class SessionAnalysis:
    """Session-level analysis results."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    utterances: List[UtteranceAnalysis] = field(default_factory=list)
    
    # Configuration info
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Summary statistics
    total_utterances: int = 0
    avg_speech_duration_ms: Optional[float] = None
    avg_early_to_end_delay_ms: Optional[float] = None
    avg_eou_decision_time_ms: Optional[float] = None
    avg_total_latency_ms: Optional[float] = None
    
    # EOU performance stats
    eou_trigger_rate: Optional[float] = None
    accelerated_count: int = 0       # EOU correctly accelerated count
    protected_count: int = 0         # EOU correctly protected count
    missed_count: int = 0            # EOU missed count
    false_trigger_count: int = 0     # EOU false trigger count
    reconnected_count: int = 0       # Reconnection corrected count
    normal_count: int = 0            # Normal VAD count
    
    # Response time comparison
    avg_latency_with_eou_ms: Optional[float] = None     # Average latency with EOU acceleration
    avg_latency_without_eou_ms: Optional[float] = None  # Average latency without EOU
    total_time_saved_ms: float = 0   # Total time saved by EOU
    
    # Issues list
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate_summary(self):
        """Calculate summary statistics."""
        self.total_utterances = len(self.utterances)
        
        if self.total_utterances == 0:
            return
        
        # Calculate averages for metrics
        speech_durations = [u.speech_duration_ms for u in self.utterances if u.speech_duration_ms is not None]
        if speech_durations:
            self.avg_speech_duration_ms = sum(speech_durations) / len(speech_durations)
        
        early_delays = [u.early_to_end_delay_ms for u in self.utterances if u.early_to_end_delay_ms is not None]
        if early_delays:
            self.avg_early_to_end_delay_ms = sum(early_delays) / len(early_delays)
        
        eou_decisions = [u.eou_decision_time_ms for u in self.utterances if u.eou_decision_time_ms is not None]
        if eou_decisions:
            self.avg_eou_decision_time_ms = sum(eou_decisions) / len(eou_decisions)
        
        total_latencies = [u.total_latency_ms for u in self.utterances if u.total_latency_ms is not None]
        if total_latencies:
            self.avg_total_latency_ms = sum(total_latencies) / len(total_latencies)
        
        # EOU trigger rate
        eou_triggered_count = len([u for u in self.utterances if u.eou_triggered])
        self.eou_trigger_rate = eou_triggered_count / self.total_utterances
        
        # EOU result classification stats
        self.accelerated_count = len([u for u in self.utterances if u.eou_result_type == "accelerated"])
        self.protected_count = len([u for u in self.utterances if u.eou_result_type == "protected"])
        self.missed_count = len([u for u in self.utterances if u.eou_result_type == "missed"])
        self.false_trigger_count = len([u for u in self.utterances if u.eou_result_type == "false_trigger"])
        self.reconnected_count = len([u for u in self.utterances if u.eou_result_type == "reconnected"])
        self.normal_count = len([u for u in self.utterances if u.eou_result_type == "normal"])
        
        # Calculate total time saved by EOU
        self.total_time_saved_ms = sum(u.time_saved_ms for u in self.utterances if u.time_saved_ms is not None)
        
        # Calculate latency comparison with/without EOU
        eou_latencies = [u.total_latency_ms for u in self.utterances 
                         if u.eou_triggered and u.total_latency_ms is not None]
        if eou_latencies:
            self.avg_latency_with_eou_ms = sum(eou_latencies) / len(eou_latencies)
        
        non_eou_latencies = [u.total_latency_ms for u in self.utterances 
                             if not u.eou_triggered and u.total_latency_ms is not None]
        if non_eou_latencies:
            self.avg_latency_without_eou_ms = sum(non_eou_latencies) / len(non_eou_latencies)
        
        # Detect issues
        self._detect_issues()
    
    def _detect_issues(self):
        """Detect issues."""
        self.issues = []
        
        for utterance in self.utterances:
            # Detect reconnection events (EOU false prediction auto-corrected)
            if utterance.eou_result_type == "reconnected":
                self.issues.append({
                    "type": "reconnected",
                    "severity": "info",
                    "utterance_id": utterance.utterance_id,
                    "detail": f"Reconnection correction: EOU false prediction auto-corrected, cancelled stream {utterance.cancelled_stream_key}",
                    "cancelled_stream_key": utterance.cancelled_stream_key,
                    "time_gap_ms": utterance.reconnect_time_gap_ms
                })
            
            # Detect EOU false trigger (false positive)
            if utterance.eou_result_type == "false_trigger":
                self.issues.append({
                    "type": "eou_false_trigger",
                    "severity": "warning",
                    "utterance_id": utterance.utterance_id,
                    "detail": f"EOU potential false trigger: sentence \"{utterance.asr_text[:30] if utterance.asr_text else ''}...\" appears incomplete",
                    "asr_text": utterance.asr_text,
                    "probability": utterance.eou_final_probability
                })
            
            # Detect EOU missed trigger (false negative) - short sentence but missed
            if utterance.eou_result_type == "missed":
                self.issues.append({
                    "type": "eou_missed",
                    "severity": "info",
                    "utterance_id": utterance.utterance_id,
                    "detail": f"Short sentence \"{utterance.asr_text}\" not accelerated by EOU",
                    "speech_duration_ms": utterance.speech_duration_ms,
                    "early_to_end_delay_ms": utterance.early_to_end_delay_ms
                })
            
            # Detect slow EOU decision
            if utterance.eou_decision_time_ms is not None and utterance.eou_decision_time_ms > 200:
                self.issues.append({
                    "type": "slow_eou_decision",
                    "severity": "info",
                    "utterance_id": utterance.utterance_id,
                    "detail": f"EOU decision time {utterance.eou_decision_time_ms:.0f}ms is relatively long",
                    "value": utterance.eou_decision_time_ms
                })
            
            # Detect high overall latency (only for EOU triggered cases)
            if utterance.eou_triggered and utterance.total_latency_ms is not None:
                if utterance.total_latency_ms > 500:
                    self.issues.append({
                        "type": "high_latency",
                        "severity": "warning",
                        "utterance_id": utterance.utterance_id,
                        "detail": f"Total latency {utterance.total_latency_ms:.0f}ms is high",
                        "value": utterance.total_latency_ms
                    })
