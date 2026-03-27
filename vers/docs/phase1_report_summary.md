# Phase-1 Report Summary

## Project
Vision-Based Emergency Response System (VERS)

## Core Problem
Most emergency interfaces assume spoken communication, which disadvantages deaf, hard-of-hearing, speech-impaired, and other non-verbal users. VERS addresses that accessibility gap with a vision-based emergency communication workflow that works through gestures, visible facial distress, and structured alerts.

## Real Pipeline
1. **Input Module**: Webcam feed captured with OpenCV.
2. **Preprocessing Module**: Frame flip, resize configuration, and RGB conversion.
3. **Feature Extraction Module**: MediaPipe Hands for 21 hand landmarks and MediaPipe Face Mesh for facial cues.
4. **AI Module**: RandomForest gesture classifier trained on 63 landmark features.
5. **Distress Analysis Module**: Heuristic score based on mouth opening and brow-eye spacing.
6. **Alert Generation Module**: Canonical JSON payload with alert ID, timestamp, gesture, confidence, severity, distress score, and message.
7. **Communication Module**: Streamlit dispatcher dashboard, JSON logs, and a mock Flask `/alert` endpoint.

## Scripts to Mention in Viva
| Slide Theme | Repo Implementation |
|-------------|---------------------|
| Methodology | `src/record_gestures.py`, `src/train_classifier.py`, `src/realtime_vers.py` |
| System Design | `docs/workflow.png` + `src/orchestrate.py` |
| Alert Communication | `src/alert_server.py`, `src/vers_dashboard.py`, `src/web_vers.py` |
| Accessibility Focus | README objectives 6 and 7 |

## Demo Talking Points
- Streamlit is the primary MacBook demo UI.
- Flask remains available as a legacy fallback.
- Existing `data/landmarks.csv` and `models/gesture_classifier.pkl` allow a fast turnkey run.
- `python src/orchestrate.py` is the default demo command.
- `python src/orchestrate.py --force-capture --force-train` demonstrates the full end-to-end pipeline from scratch.

## Limitations and Mitigation
- **Lighting sensitivity**: Demo in an evenly lit room with the hand centered and unobstructed.
- **Single-user focus**: Best results with one visible hand and one visible face.
- **Heuristic distress scoring**: Current distress estimation is explainable and demo-friendly, but not yet a trained emotion model.

## Phase-2 Direction
- Wake gesture and idle mode
- Fused severity from gesture confidence plus distress
- MQTT or IoT transport
- ONNX export for edge deployment
