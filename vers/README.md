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

## Architecture Overview
See `docs/workflow.png` for the Phase-1 system pipeline:
- **Input**: Camera feed
- **Preprocessing**: OpenCV frame capture and normalization
- **Feature Extraction**: MediaPipe Hands + Face Mesh
- **AI Module**: Scikit-learn gesture classifier + distress heuristic
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
| Message Generation | `src/utils/alert_utils.py` |
| Alert Transmission | `logs/alerts.log`, `src/alert_server.py`, `src/vers_dashboard.py` |

## Advantages and Known Limitations
**Advantages**
- Real-time local inference with no cloud dependency
- Accessibility-first emergency communication workflow
- Structured JSON alerts suitable for downstream integration
- Demo-ready orchestration for quick MacBook presentations

**Known limitations**
- Lighting quality affects hand and face tracking accuracy
- Current capture pipeline assumes one prominent hand and one face
- Distress scoring is heuristic-based, not a trained emotion model

**Demo mitigation**
- Use a well-lit room with even front lighting
- Keep the active hand centered and fully visible
- Close other apps that may be using the webcam

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

```bash
python src/orchestrate.py
```

### Full From-Scratch Run
Use this when you want to recapture gesture samples and retrain the classifier.

```bash
python src/orchestrate.py --force-capture --force-train
```

### Individual Commands
```bash
python src/record_gestures.py --label HELP --samples 200
python src/train_classifier.py
python src/realtime_vers.py
python src/alert_server.py
python -m streamlit run src/vers_dashboard.py
python web_vers.py
```

## Demo Flow
1. Launch `python src/orchestrate.py`.
2. Open the Streamlit dashboard at `http://localhost:8501`.
3. Perform `HELP`, `MEDICAL`, and `DANGER` gestures in front of the webcam.
4. Show live overlay, recent alerts, and `logs/alerts.log`.
5. Press `q` in the OpenCV window to end the live realtime loop.

The legacy Flask dashboard remains available at `http://localhost:5000` via `python web_vers.py`, but Streamlit is the primary demo surface.

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
│       ├── alert_utils.py
│       └── data_utils.py
├── templates/
│   └── index.html
├── tests/
│   └── test_model.py
├── alert_server.py
├── realtime_vers.py
├── record_gestures.py
├── train_classifier.py
└── web_vers.py
```

## Future Scope
- Wake gesture and stateful pipeline
- Severity fusion heuristics
- MQTT transport for downstream dispatch systems
- ONNX export for edge deployment
