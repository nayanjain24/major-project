# VERS Demo

Vision-Based Emergency Response System (VERS) demo for a laptop webcam.

## What it does

- Detects emergency hand gestures from the webcam using MediaPipe
- Scores facial distress from face landmarks
- Stabilizes predictions over multiple frames
- Generates structured responder alerts
- Saves raised alerts to local JSONL log files

## Demo gestures

- Open palm: `SOS`
- Closed fist: `Medical emergency`
- V sign: `Accident / injury`
- Index finger up: `Security threat`
- Thumbs up: `Safe / resolved`

## Run the main browser demo

```bash
cd "/Users/nayanjain/Documents/major project/blind"
source .venv/bin/activate
streamlit run streamlit_app.py
```

## Run the OpenCV fallback demo

```bash
cd "/Users/nayanjain/Documents/major project/blind"
source .venv/bin/activate
python main.py
```

## Logs

- Streamlit alerts: `runtime/vers_alerts.jsonl`
- Terminal alerts: `runtime/vers_alerts_terminal.jsonl`

## Test

```bash
cd "/Users/nayanjain/Documents/major project/blind"
source .venv/bin/activate
python -m unittest discover -s tests -v
```
