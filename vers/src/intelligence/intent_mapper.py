"""NLP-lite intent interpreter for sign language word sequences.

Receives a rolling buffer of predicted sign language words and maps
recognised patterns to structured emergency intents.  This is a rule-based
system (no external NLP model required) designed for the 15-word emergency
vocabulary.

Example mappings::

    ["HELP", "ACCIDENT"]       → ACCIDENT, CRITICAL
    ["MEDICAL", "PAIN"]        → MEDICAL, HIGH
    ["FIRE", "HELP"]           → FIRE, CRITICAL
    ["SAFE"]                   → SAFE, LOW
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("vers.intelligence.intent_mapper")


@dataclass(frozen=True)
class EmergencyIntent:
    """Structured output of the intent interpretation engine."""

    alert_type: str       # ACCIDENT, MEDICAL, FIRE, POLICE, EMERGENCY, SAFE, NONE
    severity: str         # CRITICAL, HIGH, MEDIUM, LOW, NONE
    message: str          # Human-readable description
    source_words: list[str]  # The words that triggered this intent


# ---------------------------------------------------------------------------
# Pattern rules — ordered by priority (first match wins)
# ---------------------------------------------------------------------------
# Each rule is: (required_words_set, alert_type, severity, message_template)

_INTENT_RULES: list[tuple[set[str], str, str, str]] = [
    # Critical emergencies
    ({"FIRE"},                "FIRE",      "CRITICAL", "Fire emergency reported via sign language."),
    ({"ACCIDENT", "HELP"},    "ACCIDENT",  "CRITICAL", "Accident with help request — immediate response needed."),
    ({"ACCIDENT"},            "ACCIDENT",  "HIGH",     "Accident reported via sign language."),
    ({"EMERGENCY", "HELP"},   "EMERGENCY", "CRITICAL", "Emergency with help request detected."),
    ({"EMERGENCY"},           "EMERGENCY", "HIGH",     "Emergency situation indicated."),
    ({"AMBULANCE"},           "MEDICAL",   "CRITICAL", "Ambulance requested via sign language."),
    ({"MEDICAL", "PAIN"},     "MEDICAL",   "HIGH",     "Medical assistance needed — pain indicated."),
    ({"MEDICAL"},             "MEDICAL",   "HIGH",     "Medical assistance requested."),
    ({"POLICE"},              "POLICE",    "HIGH",     "Police assistance requested via sign language."),
    ({"DANGER"},              "EMERGENCY", "HIGH",     "Danger indicated via sign language."),
    ({"HELP"},                "SOS",       "HIGH",     "Help requested via sign language."),
    ({"FALL", "PAIN"},        "MEDICAL",   "HIGH",     "Fall with pain reported."),
    ({"FALL"},                "ACCIDENT",  "MEDIUM",   "Fall incident reported."),
    ({"STOP"},                "EMERGENCY", "MEDIUM",   "Stop signal detected."),
    ({"PAIN"},                "MEDICAL",   "MEDIUM",   "Pain indicated via sign language."),
    # Low severity
    ({"SAFE"},                "SAFE",      "LOW",      "Safe status confirmed via sign language."),
    ({"YES"},                 "CONFIRM",   "LOW",      "Affirmative response received."),
    ({"NO"},                  "DENY",      "LOW",      "Negative response received."),
]


class IntentMapper:
    """Maps rolling sign language word predictions to emergency intents.

    Maintains a short-term word memory (default 5 words) and checks against
    the pattern rules on each new word.

    Parameters
    ----------
    memory_size : int
        Number of recent words to retain for pattern matching.
    """

    def __init__(self, memory_size: int = 5) -> None:
        self._memory: deque[str] = deque(maxlen=memory_size)

    def push_word(self, word: str) -> Optional[EmergencyIntent]:
        """Add a recognised word and attempt to match an intent pattern.

        Returns an ``EmergencyIntent`` if a pattern matches, otherwise ``None``.
        Words equal to ``"NONE"`` are ignored.
        """
        if word == "NONE" or not word:
            return None

        self._memory.append(word)
        current_words = set(self._memory)

        for required, alert_type, severity, message in _INTENT_RULES:
            if required.issubset(current_words):
                intent = EmergencyIntent(
                    alert_type=alert_type,
                    severity=severity,
                    message=message,
                    source_words=list(self._memory),
                )
                logger.info(
                    "Intent matched: %s [%s] from words %s",
                    alert_type, severity, list(self._memory),
                )
                # Clear memory after a match to avoid re-triggering
                self._memory.clear()
                return intent

        return None

    def reset(self) -> None:
        """Clear the word memory."""
        self._memory.clear()

    @property
    def current_words(self) -> list[str]:
        """The current words in memory (for display)."""
        return list(self._memory)
