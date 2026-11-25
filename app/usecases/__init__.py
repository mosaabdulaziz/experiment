"""One module per use case of the camera streams (MR !25 review, 2026-10-01).

StreamManager (app/stream_manager.py) only discovers cameras and starts or
stops each camera's tasks; what each use case DOES lives here, so a change to
one -- fire incidents, say -- doesn't mean editing the class every other use
case lives in.

    vlm_analysis    periodic VLM look at each camera -> comments and alerts
    zone_alerts     restricted/valid zone crossings from the pipeline
    count_alerts    overcrowding and fire/smoke from the pipeline
    fire_incidents  fire/smoke incidents: open, follow-up clips, resolve
    alert_events    saving and announcing one alert (shared)
    descriptions    VLM sentence describing an alert
    recordings      smart-record clips
    chat            frame history the chat looks at
"""
