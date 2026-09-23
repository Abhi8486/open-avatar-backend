"""
JSON Exporter

Exports structured VAD/EOU analysis data for AI analysis.

Analysis Objectives:
- Core role of EOU is the "gatekeeper" - preventing VAD from prematurely truncating speech when the user pauses to think while speaking.
- Purpose of Early VAD is to quickly detect when the user has genuinely finished speaking, starting Q&A as early as possible.
"""
import json
from datetime import datetime
from typing import Any, Dict, List

from .analyzer_models import SessionAnalysis, UtteranceAnalysis


def export_json(session: SessionAnalysis, output_path: str):
    """Export JSON analysis data."""
    
    data = {
        "session_id": session.session_id,
        "analysis_timestamp": datetime.now().isoformat(),
        "session_time": {
            "start": session.start_time.isoformat() if session.start_time else None,
            "end": session.end_time.isoformat() if session.end_time else None,
        },
        "config": session.config,
        "summary": _build_summary(session),
        "eou_effect_analysis": _build_eou_effect_analysis(session),
        "utterances": [_build_utterance_data(u) for u in session.utterances],
        "ai_analysis_prompt": _build_ai_prompt(session)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_summary(session: SessionAnalysis) -> Dict[str, Any]:
    """Build summary data."""
    return {
        "total_utterances": session.total_utterances,
        "avg_speech_duration_ms": session.avg_speech_duration_ms,
        "avg_early_to_end_delay_ms": session.avg_early_to_end_delay_ms,
        "avg_eou_decision_time_ms": session.avg_eou_decision_time_ms,
        "avg_total_latency_ms": session.avg_total_latency_ms,
        "eou_trigger_rate": session.eou_trigger_rate,
    }


def _build_eou_effect_analysis(session: SessionAnalysis) -> Dict[str, Any]:
    """Build EOU effect analysis."""
    return {
        "effect_classification": {
            "accelerated": {
                "count": session.accelerated_count,
                "description": "EOU triggered correctly, accelerating response for short/complete sentences",
                "is_good": True
            },
            "protected": {
                "count": session.protected_count,
                "description": "EOU correctly not triggered, protecting long/thinking sentences from truncation",
                "is_good": True
            },
            "reconnected": {
                "count": session.reconnected_count,
                "description": "EOU misjudgment automatically corrected by reconnection mechanism, cancelling incorrect stream and restarting",
                "is_good": True
            },
            "missed": {
                "count": session.missed_count,
                "description": "Short/complete sentence not accelerated by EOU, threshold may need to be lowered",
                "is_good": False
            },
            "false_trigger": {
                "count": session.false_trigger_count,
                "description": "EOU may have falsely triggered, truncating unfinished sentence; threshold may need to be raised",
                "is_good": False
            },
            "normal": {
                "count": session.normal_count,
                "description": "Normal VAD end, no EOU involvement"
            }
        },
        "latency_comparison": {
            "avg_latency_with_eou_ms": session.avg_latency_with_eou_ms,
            "avg_latency_without_eou_ms": session.avg_latency_without_eou_ms,
            "total_time_saved_ms": session.total_time_saved_ms
        },
        "issues": session.issues,
        "issues_by_type": _group_issues_by_type(session.issues)
    }


def _group_issues_by_type(issues: List[Dict[str, Any]]) -> Dict[str, int]:
    """Group and count issues by type."""
    counts = {}
    for issue in issues:
        issue_type = issue.get("type", "unknown")
        counts[issue_type] = counts.get(issue_type, 0) + 1
    return counts


def _build_utterance_data(utterance: UtteranceAnalysis) -> Dict[str, Any]:
    """Build data for a single utterance."""
    return {
        "id": utterance.utterance_id,
        "stream_key": utterance.stream_key,
        "asr_text": utterance.asr_text,
        
        # Utterance classification
        "classification": {
            "utterance_type": utterance.utterance_type,
            "is_sentence_complete": utterance.is_sentence_complete,
            "eou_result_type": utterance.eou_result_type,
        },
        
        # Reconnection info
        "reconnection": {
            "was_reconnected": utterance.was_reconnected,
            "cancelled_stream_key": utterance.cancelled_stream_key,
            "reconnect_time_gap_ms": utterance.reconnect_time_gap_ms,
        },
        
        # Timeline data
        "timeline": {
            "speech_start_ms": utterance.speech_start_ms,
            "early_vad_end_ms": utterance.early_vad_end_ms,
            "speech_end_ms": utterance.speech_end_ms,
            "asr_completed_ms": utterance.asr_completed_ms,
        },
        
        # Calculated metrics
        "metrics": {
            "speech_duration_ms": utterance.speech_duration_ms,
            "early_to_end_delay_ms": utterance.early_to_end_delay_ms,
            "eou_decision_time_ms": utterance.eou_decision_time_ms,
            "total_latency_ms": utterance.total_latency_ms,
            "time_saved_ms": utterance.time_saved_ms,
        },
        
        # EOU analysis
        "eou_analysis": {
            "triggered": utterance.eou_triggered,
            "effective": utterance.eou_effective,
            "prediction_count": utterance.eou_prediction_count,
            "final_probability": utterance.eou_final_probability,
        },
        
        # Detailed events
        "events": {
            "vad": [
                {
                    "timestamp_ms": e.timestamp_ms,
                    "type": e.event_type,
                    "sample_id": e.sample_id,
                    "extra_data": e.extra_data
                }
                for e in utterance.vad_events
            ],
            "eou": [
                {
                    "timestamp_ms": e.timestamp_ms,
                    "type": e.event_type,
                    "prediction": e.prediction,
                    "probability": e.probability,
                    "buffer_duration_ms": e.buffer_duration_ms
                }
                for e in utterance.eou_events
            ],
            "asr": [
                {
                    "timestamp_ms": e.timestamp_ms,
                    "type": e.event_type,
                    "text": e.text,
                    "source": e.source
                }
                for e in utterance.asr_events
            ]
        }
    }


def _build_ai_prompt(session: SessionAnalysis) -> str:
    """Build AI analysis prompt."""
    
    # Extract config info
    vad_config = session.config.get("vad", {})
    eou_config = session.config.get("eou", {})
    
    prompt = f"""The following is analysis data from the VAD/EOU Voice Endpoint Detection System. Please evaluate its performance and provide optimization recommendations.

## System Objectives
- **Early VAD**: Designed to quickly detect when the user has finished speaking and start Q&A as early as possible.
- **EOU (End of Utterance)**: Serves as a "gatekeeper" — preventing VAD from prematurely ending turn while user is mid-thought/pausing, truncating user queries.

## System Configuration
- VAD end_delay: {vad_config.get('end_delay', 'N/A')} samples
- VAD early_end_delay: {vad_config.get('early_end_delay', 'N/A')} samples
- EOU threshold: {eou_config.get('threshold', 'N/A')}
- EOU max_buffer: {eou_config.get('max_buffer_seconds', 'N/A')} seconds

## EOU Effect Statistics
- Total Utterances: {session.total_utterances}
- ✓ Accelerated Response: {session.accelerated_count} times (EOU triggered correctly, accelerating short sentence responses)
- ✓ Protected Long Sentences: {session.protected_count} times (EOU correctly held back, avoiding truncation of thinking user)
- ↻ Reconnection Corrections: {session.reconnected_count} times (EOU misjudgment automatically corrected by reconnection)
- ⚠ Missed Triggers: {session.missed_count} times (short sentence not accelerated by EOU)
- ✗ False Triggers: {session.false_trigger_count} times (might have prematurely truncated unfinished sentence)

## Latency Comparison
- Avg Latency with EOU Acceleration: {f'{session.avg_latency_with_eou_ms:.0f}ms' if session.avg_latency_with_eou_ms else 'N/A'}
- Avg Latency Without EOU: {f'{session.avg_latency_without_eou_ms:.0f}ms' if session.avg_latency_without_eou_ms else 'N/A'}
- Total Time Saved: {session.total_time_saved_ms:.0f}ms

## Detected Issues
"""
    
    if session.issues:
        issues_by_type = _group_issues_by_type(session.issues)
        for issue_type, count in issues_by_type.items():
            prompt += f"- {issue_type}: {count} times\n"
    else:
        prompt += "- No issues detected\n"
    
    prompt += """
## Please Analyze the Following:
1. **EOU Performance Evaluation**: How effective are response acceleration and long-sentence protection? Is the ratio of missed and false triggers acceptable?
2. **Threshold Adjustment Recommendations**:
   - If missed triggers are high (short sentences not accelerated), consider lowering threshold.
   - If false triggers are high (truncating incomplete sentences), consider raising threshold.
3. **Early VAD Configuration**: Is early_end_delay setting appropriate? Can it trigger EOU evaluation in time?
4. **Overall Latency**: Is the EOU acceleration effect significant? Does it achieve the latency reduction goal?
5. **Specific Case Analysis**: Inspect utterances tagged as "missed" or "false_trigger" and analyze root causes.

Please refer to the `utterances` field for detailed event timelines.
"""
    
    return prompt

