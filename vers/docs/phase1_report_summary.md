# Phase-1 Report Summary

## Project
Vision-Based Emergency Response System (VERS)

## Core Problem
Most emergency interfaces assume spoken communication, which disadvantages deaf, hard-of-hearing, speech-impaired, and other non-verbal users. VERS addresses that accessibility gap with a vision-based emergency communication workflow that works through gestures, visible facial distress, and structured alerts.

## Real Pipeline
1. **Input Module**: Webcam feed captured with OpenCV.
2. **Preprocessing Module**: Frame flip, resize configuration, and RGB conversion.
3. **Feature Extraction Module**: MediaPipe Hands for 21 hand landmarks and MediaPipe Face Mesh for facial cues.
4. **AI Module**: RandomForest gesture classifier trained on 63 landmark features with data augmentation.
5. **Distress Analysis Module**: Heuristic score based on mouth opening and brow-eye spacing.
6. **Severity Fusion Module**: Fuses gesture confidence (60%) and distress score (40%) into a final severity class.
7. **Alert Generation Module**: Canonical JSON payload with alert ID, timestamp, gesture, confidence, severity, fusion score, distress score, and message.
8. **Communication Module**: Streamlit dispatcher dashboard, JSON logs, and a mock Flask `/alert` endpoint.

## Gesture Mapping
| Gesture | Hand Sign | Severity |
|---------|-----------|----------|
| SOS | Open Hand (5 fingers) | High |
| EMERGENCY | 2 Fingers (V-shape) | Critical |
| ACCIDENT | Fist (all fingers closed) | High |
| MEDICAL | 4 Fingers (thumb folded) | High |
| SAFE | Thumbs Up | Low |

## Scripts to Mention in Viva
| Slide Theme | Repo Implementation |
|-------------|---------------------|
| Methodology | `src/record_gestures.py`, `src/train_classifier.py`, `src/realtime_vers.py` |
| System Design | `docs/workflow.png` + `src/orchestrate.py` |
| Alert Communication | `src/alert_server.py`, `src/vers_dashboard.py`, `src/web_vers.py` |
| Severity Fusion | `src/utils/alert_utils.py` — `calculate_fused_severity()` |
| Accessibility Focus | README objectives 6 and 7 |

## Demo Talking Points
- Streamlit is the primary MacBook demo UI.
- Flask remains available as a legacy fallback.
- Existing `data/landmarks.csv` and `models/gesture_classifier.pkl` allow a fast turnkey run.
- `python src/orchestrate.py` is the default demo command.
- `python src/orchestrate.py --calibrate` demonstrates personalized hand calibration.
- `python src/orchestrate.py --force-capture --force-train` demonstrates the full end-to-end pipeline from scratch.
- Five gesture classes: **SOS**, **EMERGENCY**, **ACCIDENT**, **MEDICAL**, **SAFE**.

## Limitations and Mitigation
- **Lighting sensitivity**: Demo in an evenly lit room with the hand centered and unobstructed.
- **Single-user focus**: Best results with one visible hand and one visible face.
- **Heuristic distress scoring**: Current distress estimation is explainable and demo-friendly, but not yet a trained emotion model.

## Phase-2 Direction
- Wake gesture and idle mode
- Trained emotion model to replace heuristic distress scoring
- MQTT or IoT transport for downstream dispatch
- ONNX export for edge deployment
