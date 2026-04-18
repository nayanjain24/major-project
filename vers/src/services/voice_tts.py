"""Asynchronous Text-to-Speech voice alert daemon for VERS v2.0.

Runs a dedicated background thread that consumes alert messages from an
internal queue and speaks them aloud using the OS-native speech synthesis
engine (NSSpeechSynthesizer on macOS, SAPI5 on Windows, espeak on Linux)
via the ``pyttsx3`` library.

Design decisions
----------------
- The TTS thread is a *daemon* thread — it dies automatically when the
  main process exits.
- A ``queue.Queue`` decouples the caller from the blocking ``engine.say()``
  call so the vision loop never stalls waiting for speech to finish.
- Duplicate consecutive messages are suppressed within a configurable
  cooldown window to avoid audio spam during continuous detection.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger("vers.services.voice_tts")

# ---------------------------------------------------------------------------
# Module-level singleton — one global TTS engine per process
# ---------------------------------------------------------------------------
_tts_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=32)
_tts_thread: Optional[threading.Thread] = None
_tts_started = False
_start_lock = threading.Lock()


def _tts_worker() -> None:
    """Background worker: pulls messages from the queue and speaks them."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)   # words per minute
        engine.setProperty("volume", 0.9)
        logger.info("TTS engine initialised.")
    except Exception as exc:
        logger.warning("TTS engine unavailable: %s — voice alerts disabled.", exc)
        return

    last_message = ""
    last_time = 0.0
    cooldown = 5.0  # suppress identical messages within 5 seconds

    while True:
        try:
            msg = _tts_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if msg is None:
            # Poison pill — shut down gracefully
            break

        now = time.time()
        if msg == last_message and (now - last_time) < cooldown:
            continue

        try:
            engine.say(msg)
            engine.runAndWait()
            last_message = msg
            last_time = time.time()
        except Exception as exc:
            logger.debug("TTS speech failed: %s", exc)


def start() -> None:
    """Start the background TTS daemon thread (idempotent)."""
    global _tts_thread, _tts_started
    with _start_lock:
        if _tts_started:
            return
        _tts_thread = threading.Thread(target=_tts_worker, daemon=True, name="vers-tts")
        _tts_thread.start()
        _tts_started = True
        logger.info("TTS daemon started.")


def speak(message: str) -> None:
    """Enqueue a message for asynchronous speech synthesis.

    If the queue is full the message is silently dropped (non-blocking).
    """
    if not _tts_started:
        start()
    try:
        _tts_queue.put_nowait(message)
    except queue.Full:
        pass  # Drop—never block the vision loop


def stop() -> None:
    """Send a shutdown signal to the TTS daemon."""
    try:
        _tts_queue.put_nowait(None)
    except queue.Full:
        pass
