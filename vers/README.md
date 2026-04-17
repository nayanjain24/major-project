# Vision-Based Emergency Response System (VERS)

## Abstract
The Vision-Based Emergency Response System (VERS) is a computer-vision-driven solution for non-verbal emergency communication. Using OpenCV, MediaPipe, and machine learning, the system detects hand gestures and estimates facial distress cues in real time, then converts them into structured emergency alerts. The prototype is designed to improve accessibility for deaf, hard-of-hearing, speech-impaired, and other non-verbal users who may be underserved by voice-first SOS systems.

## Problem Statement
Conventional emergency response channels depend heavily on spoken communication. That creates critical accessibility gaps for users who cannot speak, cannot be heard clearly, or must communicate silently during distress. VERS bridges that gap by interpreting gestures and visible distress through a camera feed and producing text-based alerts that responders can understand immediately.

## Objectives
1. Design a vision-based emergency communication system.
2. Detect and recognize sign language gestures using computer vision.
3. Analyze facial expressions to estimate distress and urgency.
4. Convert gestures and expressions into structured emergency alerts.
5. Integrate AI/ML for real-time recognition.
6. Improve accessibility for deaf, hard-of-hearing, and speech-impaired individuals.
7. Deliver an inclusive emergency response workflow suitable for demos and future field adaptation.

## Gesture Mapping
| Gesture | Hand Sign | Severity | Description |
|---------|-----------|----------|-------------|
| **SOS** | ✋ Open Hand (5 fingers) | High | General SOS request |
| **EMERGENCY** | ✌️ V-shape (2 fingers) | Critical | Urgent emergency signal |
| **ACCIDENT** | ✊ Fist (all closed) | High | Accident reported |
| **MEDICAL** | 🖐️ 4 Fingers (thumb folded) | High | Medical assistance needed |
| **SAFE** | 👍 Thumbs Up | Low | Status: safe |

## Architecture Overview
See `docs/workflow.png` for the Phase-1 system pipeline:
- **Input**: Camera feed
- **Preprocessing**: OpenCV frame capture and normalization
- **Feature Extraction**: MediaPipe Hands + Face Mesh
- **AI Module**: Scikit-learn gesture classifier + distress heuristic
- **Severity Fusion**: Weighted combination of gesture confidence (60%) and distress score (40%)
- **Alert Generation**: Canonical JSON payload
- **Communication**: Streamlit dashboard, JSON logs, and mock Flask endpoint

## Methodology to Code Mapping
| Phase-1 Step | Implementation |
|--------------|----------------|
| Camera Input | `src/record_gestures.py`, `src/realtime_vers.py`, `src/vers_dashboard.py` |
| Hand / Face Detection | MediaPipe Hands + Face Mesh in `src/realtime_vers.py` |
| Landmark Extraction | `src/utils/data_utils.py` |
| Gesture Training | `src/train_classifier.py` |
| Distress Analysis | `calc_distress()` in `src/realtime_vers.py` |
| Severity Fusion | `calculate_fused_severity()` in `src/utils/alert_utils.py` |
| Message Generation | `src/utils/alert_utils.py` |
| Alert Transmission | `logs/alerts.log`, `src/alert_server.py`, `src/vers_dashboard.py` |

## Advantages and Known Limitations
**Advantages**
- Real-time local inference with no cloud dependency
- Accessibility-first emergency communication workflow
- Structured JSON alerts suitable for downstream integration
- Demo-ready orchestration for quick MacBook presentations
- Severity fusion combines gesture confidence with facial distress

**Known limitations**
- Lighting quality affects hand and face tracking accuracy
- Current capture pipeline assumes one prominent hand and one face
- Distress scoring is heuristic-based, not a trained emotion model

**Demo mitigation**
- Use a well-lit room with even front lighting
- Keep the active hand centered and fully visible
- Close other apps that may be using the webcam

## Quick Start

```bash
# 1. Set up environment
cd "/Users/nayanjain/Desktop/major project/vers"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Generate data and train (first time only)
python seed_demo_data.py
python src/train_classifier.py

# 3. Launch the demo
python src/orchestrate.py
```

## MacBook Demo Setup

### Prerequisites
- macOS camera permission granted to Terminal or your IDE
- Python 3.9 to 3.11 available locally
- Other camera apps closed before running the demo

### Environment Setup
```bash
cd "/Users/nayanjain/Desktop/major project/vers"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Fast Demo Run
This is the default presentation mode. It reuses the checked-in dataset and trained model if they already exist.
It starts the mock alert server and runs Streamlit as the primary UI (single-camera-owner flow).

```bash
python src/orchestrate.py
```

### Demo Modes
```bash
# Recommended: Streamlit dashboard only
python src/orchestrate.py --mode dashboard

# OpenCV real-time window only
python src/orchestrate.py --mode realtime

# Both at once (may contend for camera)
python src/orchestrate.py --mode hybrid
```

### Calibration (Recommended for Best Accuracy)
Record your own hand signs for each of the 5 gestures:
```bash
python src/orchestrate.py --calibrate
```

### Full From-Scratch Run
Use this when you want to recapture gesture samples and retrain the classifier.

```bash
python src/orchestrate.py --force-capture --force-train
```

### Individual Commands
```bash
python seed_demo_data.py                    # Generate synthetic training data
python src/record_gestures.py --label SOS --samples 200  # Record real gestures
python src/train_classifier.py              # Train the classifier
python src/realtime_vers.py                 # OpenCV real-time demo
python src/alert_server.py                  # Mock alert server
python -m streamlit run src/vers_dashboard.py  # Streamlit dashboard
python web_vers.py                          # Legacy Flask dashboard
```

## Demo Flow
1. Launch `python src/orchestrate.py`.
2. Open the Streamlit dashboard at `http://localhost:8501`.
3. Click **Start Stream** in Streamlit and perform gestures: **SOS**, **EMERGENCY**, **ACCIDENT**, **MEDICAL**, **SAFE**.
4. Show live overlay, recent alerts, and `logs/alerts.log`.
5. If backend webcam access is blocked, use the **Browser Camera Fallback** panel in Streamlit.

The legacy Flask dashboard remains available at `http://localhost:5000` via `python web_vers.py`, but Streamlit is the primary demo surface.

## Running Tests
```bash
# Run the full test suite
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_model.py -v
python -m pytest tests/test_alert_utils.py -v
python -m pytest tests/test_data_utils.py -v
python -m pytest tests/test_realtime_vers.py -v
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `TypeError` when loading model | Run `python seed_demo_data.py && python src/train_classifier.py` to regenerate |
| Camera permission blocked | System Settings → Privacy & Security → Camera → enable for Terminal/IDE |
| Low FPS / laggy video | Close other camera apps, use `--mode dashboard` |
| "No frames delivered" | Restart the app, ensure camera is not in use by another process |
| Model accuracy is low | Run `python src/orchestrate.py --calibrate` with your own hand |

## Expected Output Artifacts
See `docs/expected_output/` for sample deliverables aligned with the Phase-1 presentation:
- `gesture_overlay.png`
- `landmark_extraction.png`
- `distress_scoring.png`
- `json_alert_panel.png`
- `alert_json_example.json`

## Repository Layout
```text
vers/
├── README.md
├── requirements.txt
├── .gitignore
├── seed_demo_data.py
├── docs/
│   ├── phase1_report_summary.md
│   ├── workflow.png
│   └── expected_output/
│       ├── alert_json_example.json
│       ├── distress_scoring.png
│       ├── gesture_overlay.png
│       ├── json_alert_panel.png
│       └── landmark_extraction.png
├── data/
│   ├── landmarks.csv
│   ├── raw/
│   └── processed/
├── logs/
│   ├── alerts.log
│   ├── distress_history.csv
│   └── errors.log
├── models/
│   ├── gesture_classifier.pkl
│   └── reports/
│       ├── classification_report.json
│       └── confusion_matrix.png
├── src/
│   ├── __init__.py
│   ├── alert_server.py
│   ├── orchestrate.py
│   ├── realtime_vers.py
│   ├── record_gestures.py
│   ├── train_classifier.py
│   ├── vers_dashboard.py
│   ├── web_vers.py
│   └── utils/
│       ├── __init__.py
│       ├── alert_utils.py
│       └── data_utils.py
├── templates/
│   └── index.html
├── tests/
│   ├── conftest.py
│   ├── test_alert_utils.py
│   ├── test_data_utils.py
│   ├── test_model.py
│   └── test_realtime_vers.py
├── alert_server.py      (compatibility launcher)
├── realtime_vers.py     (compatibility launcher)
├── record_gestures.py   (compatibility launcher)
├── train_classifier.py  (compatibility launcher)
└── web_vers.py          (compatibility launcher)
```

## Future Scope
- Wake gesture and stateful idle/active pipeline
- Trained emotion model to replace heuristic distress scoring
- MQTT transport for downstream dispatch systems
- ONNX export for edge deployment
