"""
HTML Report Generator

Generates visual VAD/EOU analysis reports.

Analysis Objectives:
- Core role of EOU is the "gatekeeper" - preventing VAD from prematurely truncating speech when the user pauses to think while speaking.
- Purpose of Early VAD is to quickly detect when the user has genuinely finished speaking, starting Q&A as early as possible.
"""
import html
from typing import List

from .analyzer_models import SessionAnalysis, UtteranceAnalysis


def generate_html_report(session: SessionAnalysis, output_path: str):
    """Generate HTML report."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VAD/EOU Analysis Report - {html.escape(session.session_id)}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .card {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 24px;
            overflow: hidden;
        }}
        .card-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 24px;
        }}
        .card-header h1 {{
            font-size: 24px;
            font-weight: 600;
        }}
        .card-header h2 {{
            font-size: 18px;
            font-weight: 500;
            opacity: 0.9;
        }}
        .card-body {{
            padding: 24px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-item {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: #667eea;
        }}
        .stat-value.good {{
            color: #10b981;
        }}
        .stat-value.warning {{
            color: #f59e0b;
        }}
        .stat-value.bad {{
            color: #ef4444;
        }}
        .stat-label {{
            font-size: 13px;
            color: #666;
            margin-top: 8px;
        }}
        .timeline {{
            position: relative;
            padding: 20px 0;
        }}
        .timeline-bar {{
            height: 40px;
            background: #e4e8ec;
            border-radius: 8px;
            position: relative;
            margin-bottom: 8px;
            overflow: hidden;
        }}
        .timeline-fill {{
            position: absolute;
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
        }}
        .timeline-marker {{
            position: absolute;
            top: 0;
            height: 100%;
            width: 3px;
            transform: translateX(-50%);
        }}
        .timeline-marker.speech-start {{
            background: #10b981;
        }}
        .timeline-marker.early-vad-end {{
            background: #f59e0b;
        }}
        .timeline-marker.speech-end {{
            background: #ef4444;
        }}
        .timeline-marker.eou-signal {{
            background: #8b5cf6;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        }}
        .timeline-marker.asr-complete {{
            background: #3b82f6;
        }}
        .timeline-legend {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: #666;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}
        .utterance-card {{
            border: 1px solid #e4e8ec;
            border-radius: 12px;
            margin-bottom: 16px;
            overflow: hidden;
        }}
        .utterance-card.accelerated {{
            border-color: #10b981;
            border-width: 2px;
        }}
        .utterance-card.protected {{
            border-color: #3b82f6;
            border-width: 2px;
        }}
        .utterance-card.reconnected {{
            border-color: #4f46e5;
            border-width: 2px;
        }}
        .utterance-card.false-trigger {{
            border-color: #ef4444;
            border-width: 2px;
        }}
        .utterance-header {{
            background: #f8f9fa;
            padding: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #e4e8ec;
        }}
        .utterance-title {{
            font-weight: 600;
            color: #333;
        }}
        .utterance-badges {{
            display: flex;
            gap: 8px;
        }}
        .badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }}
        .badge-success {{
            background: #d1fae5;
            color: #059669;
        }}
        .badge-warning {{
            background: #fef3c7;
            color: #d97706;
        }}
        .badge-info {{
            background: #dbeafe;
            color: #2563eb;
        }}
        .badge-danger {{
            background: #fee2e2;
            color: #dc2626;
        }}
        .badge-secondary {{
            background: #f3f4f6;
            color: #6b7280;
        }}
        .utterance-body {{
            padding: 16px;
        }}
        .asr-text {{
            background: #f8f9fa;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 16px;
            color: #333;
            margin-bottom: 16px;
            border-left: 4px solid #667eea;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
        }}
        .metric-item {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 18px;
            font-weight: 600;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }}
        .issues-list {{
            border-radius: 8px;
            padding: 16px;
        }}
        .issue-item {{
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .issue-item.warning {{
            background: #fef3c7;
            border: 1px solid #fcd34d;
            color: #92400e;
        }}
        .issue-item.info {{
            background: #dbeafe;
            border: 1px solid #93c5fd;
            color: #1e40af;
        }}
        .issue-item:last-child {{
            margin-bottom: 0;
        }}
        .config-section {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }}
        .config-item {{
            font-size: 14px;
        }}
        .config-key {{
            color: #666;
        }}
        .config-value {{
            font-weight: 500;
            color: #333;
        }}
        .effect-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}
        .effect-card {{
            padding: 16px;
            border-radius: 12px;
            text-align: center;
        }}
        .effect-card.accelerated {{
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        }}
        .effect-card.protected {{
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        }}
        .effect-card.missed {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        }}
        .effect-card.false-trigger {{
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        }}
        .effect-card.reconnected {{
            background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        }}
        .effect-count {{
            font-size: 36px;
            font-weight: 700;
        }}
        .effect-label {{
            font-size: 14px;
            margin-top: 4px;
        }}
        .effect-card.accelerated .effect-count {{ color: #059669; }}
        .effect-card.protected .effect-count {{ color: #2563eb; }}
        .effect-card.missed .effect-count {{ color: #d97706; }}
        .effect-card.false-trigger .effect-count {{ color: #dc2626; }}
        .effect-card.reconnected .effect-count {{ color: #4f46e5; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Title Card -->
        <div class="card">
            <div class="card-header">
                <h1>VAD/EOU Analysis Report</h1>
                <h2>Session: {html.escape(session.session_id)}</h2>
            </div>
            <div class="card-body">
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">{session.total_utterances}</div>
                        <div class="stat-label">Total Utterances</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{_format_ms(session.avg_speech_duration_ms)}</div>
                        <div class="stat-label">Avg Speech Duration</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{_format_ms(session.avg_eou_decision_time_ms)}</div>
                        <div class="stat-label">EOU Decision Time</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value good">{_format_ms(session.total_time_saved_ms)}</div>
                        <div class="stat-label">Time Saved by EOU</div>
                    </div>
                </div>
                
                <p style="color: #666; font-size: 14px;">
                    Start Time: {session.start_time.strftime('%Y-%m-%d %H:%M:%S') if session.start_time else 'N/A'}<br>
                    End Time: {session.end_time.strftime('%Y-%m-%d %H:%M:%S') if session.end_time else 'N/A'}
                </p>
            </div>
        </div>
        
        <!-- EOU Effect Summary -->
        {_generate_effect_summary(session)}
        
        <!-- Issues List -->
        {_generate_issues_section(session)}
        
        <!-- Configuration Section -->
        {_generate_config_section(session)}
        
        <!-- Utterance Details -->
        <div class="card">
            <div class="card-header">
                <h2>Utterance Details</h2>
            </div>
            <div class="card-body">
                {_generate_utterances_section(session.utterances)}
            </div>
        </div>
        
        <!-- Timeline Legend -->
        <div class="card">
            <div class="card-body">
                <div class="timeline-legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #10b981;"></div>
                        Speech Start
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #f59e0b;"></div>
                        Early VAD End
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ef4444;"></div>
                        Speech End
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #8b5cf6; border-radius: 50%;"></div>
                        EOU Signal
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #3b82f6;"></div>
                        ASR Complete
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.querySelectorAll('.utterance-card').forEach(card => {{
            card.querySelector('.utterance-header').addEventListener('click', () => {{
                card.classList.toggle('collapsed');
            }});
        }});
    </script>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def _format_ms(value) -> str:
    """Format millisecond values."""
    if value is None:
        return "N/A"
    return f"{value:.0f}ms"


def _format_percent(value) -> str:
    """Format percentage values."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _generate_effect_summary(session: SessionAnalysis) -> str:
    """Generate EOU effect summary HTML."""
    return f"""
        <div class="card">
            <div class="card-header" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                <h2>🎯 EOU Performance Analysis</h2>
            </div>
            <div class="card-body">
                <div class="effect-summary">
                    <div class="effect-card accelerated">
                        <div class="effect-count">{session.accelerated_count}</div>
                        <div class="effect-label">✓ Accelerated Response<br><small>EOU triggered correctly, accelerated short sentence response</small></div>
                    </div>
                    <div class="effect-card protected">
                        <div class="effect-count">{session.protected_count}</div>
                        <div class="effect-label">✓ Protected Long Sentence<br><small>EOU correctly held back, avoiding truncation</small></div>
                    </div>
                    <div class="effect-card reconnected">
                        <div class="effect-count">{session.reconnected_count}</div>
                        <div class="effect-label">↻ Reconnection Correction<br><small>Misjudgment automatically corrected by reconnection</small></div>
                    </div>
                    <div class="effect-card missed">
                        <div class="effect-count">{session.missed_count}</div>
                        <div class="effect-label">⚠ Missed Trigger<br><small>Short sentence not accelerated by EOU</small></div>
                    </div>
                    <div class="effect-card false-trigger">
                        <div class="effect-count">{session.false_trigger_count}</div>
                        <div class="effect-label">✗ False Trigger<br><small>Might have prematurely truncated unfinished sentence</small></div>
                    </div>
                </div>
                
                <div class="stats-grid" style="margin-bottom: 0;">
                    <div class="stat-item">
                        <div class="stat-value">{_format_ms(session.avg_latency_with_eou_ms)}</div>
                        <div class="stat-label">Latency with EOU Acceleration</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{_format_ms(session.avg_latency_without_eou_ms)}</div>
                        <div class="stat-label">Latency Without EOU</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{_format_percent(session.eou_trigger_rate)}</div>
                        <div class="stat-label">EOU Trigger Rate</div>
                    </div>
                </div>
            </div>
        </div>
    """


def _generate_issues_section(session: SessionAnalysis) -> str:
    """Generate issues list section HTML."""
    if not session.issues:
        return """
        <div class="card">
            <div class="card-header" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                <h2>✓ No Issues Detected</h2>
            </div>
            <div class="card-body">
                <p style="color: #666;">VAD/EOU system operating normally, no issues requiring attention were detected.</p>
            </div>
        </div>
        """
    
    issues_html = []
    for issue in session.issues:
        severity = issue.get("severity", "info")
        issues_html.append(
            f'<div class="issue-item {severity}">#{issue.get("utterance_id", "?")} - {html.escape(issue.get("detail", ""))}</div>'
        )
    
    return f"""
        <div class="card">
            <div class="card-header" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
                <h2>⚠️ Requires Attention ({len(session.issues)})</h2>
            </div>
            <div class="card-body">
                <div class="issues-list">
                    {''.join(issues_html)}
                </div>
            </div>
        </div>
    """


def _generate_config_section(session: SessionAnalysis) -> str:
    """Generate configuration section HTML."""
    vad_config = session.config.get("vad", {})
    eou_config = session.config.get("eou", {})
    
    config_items = []
    
    # VAD Configuration
    if vad_config:
        config_items.append(f'<div class="config-item"><span class="config-key">VAD end_delay:</span> <span class="config-value">{vad_config.get("end_delay", "N/A")}</span></div>')
        config_items.append(f'<div class="config-item"><span class="config-key">VAD early_end_delay:</span> <span class="config-value">{vad_config.get("early_end_delay", "N/A")}</span></div>')
    
    # EOU Configuration
    if eou_config:
        config_items.append(f'<div class="config-item"><span class="config-key">EOU threshold:</span> <span class="config-value">{eou_config.get("threshold", "N/A")}</span></div>')
        config_items.append(f'<div class="config-item"><span class="config-key">EOU max_buffer:</span> <span class="config-value">{eou_config.get("max_buffer_seconds", "N/A")}s</span></div>')
    
    if not config_items:
        return ""
    
    return f"""
        <div class="card">
            <div class="card-header" style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);">
                <h2>⚙️ Configuration Information</h2>
            </div>
            <div class="card-body">
                <div class="config-section">
                    <div class="config-grid">
                        {''.join(config_items)}
                    </div>
                </div>
            </div>
        </div>
    """


def _get_result_badge(utterance: UtteranceAnalysis) -> str:
    """Get utterance result badge HTML."""
    result_type = utterance.eou_result_type
    
    if result_type == "accelerated":
        return '<span class="badge badge-success">✓ Accelerated</span>'
    elif result_type == "protected":
        return '<span class="badge badge-info">✓ Protected</span>'
    elif result_type == "reconnected":
        return '<span class="badge badge-info">↻ Reconnected</span>'
    elif result_type == "missed":
        return '<span class="badge badge-warning">⚠ Missed</span>'
    elif result_type == "false_trigger":
        return '<span class="badge badge-danger">✗ False Trigger</span>'
    else:
        return '<span class="badge badge-secondary">Normal</span>'


def _get_type_badge(utterance: UtteranceAnalysis) -> str:
    """Get utterance type badge HTML."""
    utype = utterance.utterance_type
    
    if utype == "short_complete":
        return '<span class="badge badge-info">Short</span>'
    elif utype == "long_thinking":
        return '<span class="badge badge-secondary">Thinking</span>'
    elif utype == "long_complete":
        return '<span class="badge badge-secondary">Long</span>'
    else:
        return ''


def _generate_utterances_section(utterances: List[UtteranceAnalysis]) -> str:
    """Generate utterance details section HTML."""
    if not utterances:
        return "<p style='color: #666;'>No utterance data</p>"
    
    utterances_html = []
    
    for utterance in utterances:
        # Calculate timeline range
        all_times = []
        if utterance.speech_start_ms is not None:
            all_times.append(utterance.speech_start_ms)
        if utterance.speech_end_ms is not None:
            all_times.append(utterance.speech_end_ms)
        if utterance.asr_completed_ms is not None:
            all_times.append(utterance.asr_completed_ms)
        for event in utterance.vad_events:
            all_times.append(event.timestamp_ms)
        for event in utterance.eou_events:
            all_times.append(event.timestamp_ms)
        
        if not all_times:
            continue
        
        min_time = min(all_times)
        max_time = max(all_times)
        time_range = max_time - min_time if max_time > min_time else 1000
        
        # Generate timeline markers
        markers = []
        
        # VAD Event markers
        for event in utterance.vad_events:
            pos = ((event.timestamp_ms - min_time) / time_range) * 100
            marker_class = {
                "speech_start": "speech-start",
                "early_vad_end": "early-vad-end",
                "speech_end": "speech-end"
            }.get(event.event_type, "")
            if marker_class:
                markers.append(f'<div class="timeline-marker {marker_class}" style="left: {pos:.1f}%;"></div>')
        
        # EOU Event markers
        for event in utterance.eou_events:
            if event.event_type == "signal_sent":
                pos = ((event.timestamp_ms - min_time) / time_range) * 100
                markers.append(f'<div class="timeline-marker eou-signal" style="left: {pos:.1f}%;"></div>')
        
        # ASR Complete markers
        if utterance.asr_completed_ms is not None:
            pos = ((utterance.asr_completed_ms - min_time) / time_range) * 100
            markers.append(f'<div class="timeline-marker asr-complete" style="left: {pos:.1f}%;"></div>')
        
        # Speech fill region
        fill_start = 0
        fill_end = 100
        if utterance.speech_start_ms is not None:
            fill_start = ((utterance.speech_start_ms - min_time) / time_range) * 100
        if utterance.speech_end_ms is not None:
            fill_end = ((utterance.speech_end_ms - min_time) / time_range) * 100
        
        # Get badges
        result_badge = _get_result_badge(utterance)
        type_badge = _get_type_badge(utterance)
        
        # ASR text
        asr_text_html = ""
        if utterance.asr_text:
            asr_text_html = f'<div class="asr-text">"{html.escape(utterance.asr_text)}"</div>'
        
        # Card style
        card_class = ""
        if utterance.eou_result_type == "accelerated":
            card_class = "accelerated"
        elif utterance.eou_result_type == "protected":
            card_class = "protected"
        elif utterance.eou_result_type == "reconnected":
            card_class = "reconnected"
        elif utterance.eou_result_type == "false_trigger":
            card_class = "false-trigger"
        
        utterances_html.append(f"""
            <div class="utterance-card {card_class}">
                <div class="utterance-header">
                    <span class="utterance-title">Utterance #{utterance.utterance_id}</span>
                    <div class="utterance-badges">
                        {type_badge}
                        {result_badge}
                    </div>
                </div>
                <div class="utterance-body">
                    {asr_text_html}
                    
                    <div class="timeline">
                        <div class="timeline-bar">
                            <div class="timeline-fill" style="left: {fill_start:.1f}%; width: {fill_end - fill_start:.1f}%;"></div>
                            {''.join(markers)}
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #666;">
                            <span>{min_time:.0f}ms</span>
                            <span>{max_time:.0f}ms</span>
                        </div>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div class="metric-value">{_format_ms(utterance.speech_duration_ms)}</div>
                            <div class="metric-label">Speech Duration</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{_format_ms(utterance.early_to_end_delay_ms)}</div>
                            <div class="metric-label">Early→End</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{_format_ms(utterance.eou_decision_time_ms)}</div>
                            <div class="metric-label">EOU Decision</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{_format_ms(utterance.total_latency_ms)}</div>
                            <div class="metric-label">Total Latency</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{f'{utterance.eou_final_probability:.2f}' if utterance.eou_final_probability else 'N/A'}</div>
                            <div class="metric-label">EOU Confidence</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{_format_ms(utterance.time_saved_ms) if utterance.time_saved_ms else 'N/A'}</div>
                            <div class="metric-label">Time Saved</div>
                        </div>
                    </div>
                </div>
            </div>
        """)
    
    return '\n'.join(utterances_html)

